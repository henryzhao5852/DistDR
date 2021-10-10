#!/usr/bin/env python3
# This source code is largely adapted from DPR (https://github.com/facebookresearch/DPR) repo


"""
 Pipeline to train the reader model on top of the retriever results
"""

import argparse
import collections
import glob
import json
import logging
import os
from collections import defaultdict
from typing import List

import numpy as np
import torch
from tqdm import tqdm
from dpr.data.qa_validation import exact_match_score
from dpr.data.reader_data import ReaderSample, get_best_spans, SpanPrediction, convert_retriever_results
from dpr.models import init_reader_components
from dpr.models.reader import create_reader_input, ReaderBatch, compute_loss
from dpr.options import add_encoder_params, setup_args_gpu, set_seed, add_training_params, \
    add_reader_preprocessing_params, set_encoder_params_from_state, get_encoder_params_state, add_tokenizer_params, \
    print_args
from dpr.utils.data_utils import ShardedDataIterator, read_serialized_data_from_files, Tensorizer
from dpr.utils.model_utils import get_schedule_linear, load_states_from_checkpoint, move_to_device, CheckpointState, \
    get_model_file, setup_for_distributed_mode, get_model_obj

logger = logging.getLogger()
logger.setLevel(logging.INFO)
if (logger.hasHandlers()):
    logger.handlers.clear()
console = logging.StreamHandler()
logger.addHandler(console)

ReaderQuestionPredictions = collections.namedtuple('ReaderQuestionPredictions', ['id', 'predictions', 'gold_answers'])


class ReaderTrainer(object):
    def __init__(self, args):
        self.args = args

        self.shard_id = args.local_rank if args.local_rank != -1 else 0
        self.distributed_factor = args.distributed_world_size or 1

        logger.info("***** Initializing components for training *****")

        model_file = get_model_file(self.args, self.args.checkpoint_file_name)
        saved_state = None
        if model_file:
            saved_state = load_states_from_checkpoint(model_file)
            set_encoder_params_from_state(saved_state.encoder_params, args)

        tensorizer, reader, optimizer = init_reader_components(args.encoder_model_type, args)

        reader, optimizer = setup_for_distributed_mode(reader, optimizer, args.device, args.n_gpu,
                                                       args.local_rank,
                                                       args.fp16,
                                                       args.fp16_opt_level)
        self.reader = reader
        self.optimizer = optimizer
        self.tensorizer = tensorizer
        self.start_epoch = 0
        self.start_batch = 0
        self.scheduler_state = None
        self.best_validation_result = None
        self.best_cp_name = None
        if saved_state:
            self._load_saved_state(saved_state)


    def get_data_iterator(self, path: str, batch_size: int, is_train: bool, shuffle=True,
                          shuffle_seed: int = 0,
                          offset: int = 0, num=0) -> ShardedDataIterator:
        data_files = glob.glob(path)
        logger.info("Data files: %s", data_files)
        if not data_files:
            raise RuntimeError('No Data files found')
        preprocessed_data_files = self._get_preprocessed_files(data_files, is_train)
        #preprocessed_data_files = sorted(preprocessed_data_files)[self.args.idx_num * 4 : (self.args.idx_num + 1) * 4]
        data = read_serialized_data_from_files(preprocessed_data_files)

        iterator = ShardedDataIterator(data, shard_id=self.shard_id,
                                       num_shards=self.distributed_factor,
                                       batch_size=batch_size, shuffle=shuffle, shuffle_seed=shuffle_seed, offset=offset)

        # apply deserialization hook
        iterator.apply(lambda sample: sample.on_deserialize())
        return iterator


    def validate(self):
        logger.info('Validation ...')
        args = self.args
        self.reader.eval()
        data_iterator = self.get_data_iterator(args.dev_file, args.dev_batch_size, False, shuffle=False)

        log_result_step = args.log_batch_step
        all_results = []

        eval_top_docs = args.eval_top_docs
        for i, samples_batch in enumerate(data_iterator.iterate_data()):
            input = create_reader_input(self.tensorizer.get_pad_id(),
                                        samples_batch,
                                        args.passages_per_question_predict,
                                        args.sequence_length,
                                        args.max_n_answers,
                                        is_train=False, shuffle=False)

            input = ReaderBatch(**move_to_device(input._asdict(), args.device))
            attn_mask = self.tensorizer.get_attn_mask(input.input_ids)

            with torch.no_grad():
                start_logits, end_logits, relevance_logits = self.reader(input.input_ids, attn_mask)

            batch_predictions = self._get_best_prediction(start_logits, end_logits, relevance_logits, samples_batch,
                                                          passage_thresholds=eval_top_docs)

            all_results.extend(batch_predictions)

            if (i + 1) % log_result_step == 0:
                logger.info('Eval step: %d ', i)

        if args.prediction_results_file:
            self._save_predictions(args.prediction_results_file, all_results)



    def _load_saved_state(self, saved_state: CheckpointState):
        epoch = saved_state.epoch
        offset = saved_state.offset
        if offset == 0:  # epoch has been completed
            epoch += 1
        logger.info('Loading checkpoint @ batch=%s and epoch=%s', offset, epoch)
        self.start_epoch = epoch
        self.start_batch = offset

        model_to_load = get_model_obj(self.reader)
        if saved_state.model_dict:
            logger.info('Loading model weights from saved state ...')
            model_to_load.load_state_dict(saved_state.model_dict)

        logger.info('Loading saved optimizer state ...')
        if saved_state.optimizer_dict:
            self.optimizer.load_state_dict(saved_state.optimizer_dict)
        self.scheduler_state = saved_state.scheduler_dict

    def _get_best_prediction(self, start_logits, end_logits, relevance_logits,
                             samples_batch: List[ReaderSample], passage_thresholds: List[int] = None) \
            -> List[ReaderQuestionPredictions]:

        args = self.args
        max_answer_length = args.max_answer_length
        questions_num, passages_per_question = relevance_logits.size()

        _, idxs = torch.sort(relevance_logits, dim=1, descending=True, )

        batch_results = []
        #print(questions_num)
        #questions_num = 10
        
        for q in range(questions_num):
            
            sample = samples_batch[q]

            non_empty_passages_num = len(sample.passages)
            nbest = []
            for p in range(passages_per_question):
                #print(p)
                passage_idx = idxs[q, p].item()
                if passage_idx >= non_empty_passages_num:  # empty passage selected, skip
                    continue
                reader_passage = sample.passages[passage_idx]
                sequence_ids = reader_passage.sequence_ids
                sequence_len = sequence_ids.size(0)
                # assuming question & title information is at the beginning of the sequence
                passage_offset = reader_passage.passage_offset

                p_start_logits = start_logits[q, passage_idx].tolist()[passage_offset:sequence_len]
                p_end_logits = end_logits[q, passage_idx].tolist()[passage_offset:sequence_len]

                ctx_ids = sequence_ids.tolist()[passage_offset:]
                best_spans = get_best_spans(self.tensorizer, p_start_logits, p_end_logits, ctx_ids, max_answer_length,
                                            passage_idx, relevance_logits[q, passage_idx].item(), top_spans=5)
                nbest.extend(best_spans)
                if len(nbest) > 0 and not passage_thresholds:
                    break

            if passage_thresholds:
                passage_rank_matches = {}
                for n in passage_thresholds:
                    curr_nbest = [pred for pred in nbest if pred.passage_index < n]
                    passage_rank_matches[n] = curr_nbest
                    #print(len(nbest))
                predictions = passage_rank_matches
            else:
                if len(nbest) == 0:
                    predictions = {passages_per_question: SpanPrediction('', -1, -1, -1, '')}
                else:
                    predictions = {passages_per_question: nbest[0]}
            #print(passage_thresholds)
            
            batch_results.append(ReaderQuestionPredictions(sample.question, predictions, sample.answers))
        return batch_results


    def _get_preprocessed_files(self, data_files: List, is_train: bool, ):

        serialized_files = [file for file in data_files if file.endswith('.pkl')]
        if serialized_files:
            return serialized_files
        assert len(data_files) == 1, 'Only 1 source file pre-processing is supported.'

        # data may have been serialized and cached before, try to find ones from same dir
        def _find_cached_files(path: str):
            dir_path, base_name = os.path.split(path)
            base_name = base_name.replace('.json', '')
            out_file_prefix = os.path.join(dir_path, base_name)
            out_file_pattern = out_file_prefix + '*.pkl'
            return glob.glob(out_file_pattern), out_file_prefix

        serialized_files, out_file_prefix = _find_cached_files(data_files[0])
        if serialized_files:
            logger.info('Found preprocessed files. %s', serialized_files)
            return serialized_files

        gold_passages_src = None
        if self.args.gold_passages_src:
            gold_passages_src = self.args.gold_passages_src if is_train else self.args.gold_passages_src_dev
            assert os.path.exists(gold_passages_src), 'Please specify valid gold_passages_src/gold_passages_src_dev'
        logger.info('Data are not preprocessed for reader training. Start pre-processing ...')

        # start pre-processing and save results
        def _run_preprocessing(tensorizer: Tensorizer):
            # temporarily disable auto-padding to save disk space usage of serialized files
            tensorizer.set_pad_to_max(False)
            serialized_files = convert_retriever_results(is_train, data_files[0], out_file_prefix,
                                                         gold_passages_src,
                                                         self.tensorizer,
                                                         num_workers=self.args.num_workers)
            tensorizer.set_pad_to_max(True)
            return serialized_files

        if self.distributed_factor > 1:
            # only one node in DDP model will do pre-processing
            if self.args.local_rank in [-1, 0]:
                serialized_files = _run_preprocessing(self.tensorizer)
                torch.distributed.barrier()
            else:
                torch.distributed.barrier()
                serialized_files = _find_cached_files(data_files[0])
        else:
            serialized_files = _run_preprocessing(self.tensorizer)

        return serialized_files

    def _save_predictions(self, out_file: str, prediction_results: List[ReaderQuestionPredictions]):
        logger.info('Saving prediction results to  %s', out_file)
        with open(out_file, 'w', encoding="utf-8") as output:
            save_results = []
            for r in tqdm(prediction_results):
                save_results.append({
                    'question': r.id,
                    'gold_answers': r.gold_answers,
                    'top_k': 100,
                    'predictions': [{
                            'text': span_pred.prediction_text,
                            'score': span_pred.span_score,
                            'relevance_score': span_pred.relevance_score,
                            'passage_idx': span_pred.passage_index,
                            'passage': self.tensorizer.to_string(span_pred.passage_token_ids)
                        }
                     for span_pred in r.predictions[100]]
                })
            output.write(json.dumps(save_results, indent=4) + "\n")


def main():
    parser = argparse.ArgumentParser()

    add_encoder_params(parser)
    add_training_params(parser)
    add_tokenizer_params(parser)
    add_reader_preprocessing_params(parser)

    # reader specific params
    parser.add_argument("--max_n_answers", default=10, type=int,
                        help="Max amount of answer spans to marginalize per singe passage")
    parser.add_argument('--passages_per_question', type=int, default=2,
                        help="Total amount of positive and negative passages per question")
    parser.add_argument('--passages_per_question_predict', type=int, default=50,
                        help="Total amount of positive and negative passages per question for evaluation")
    parser.add_argument("--max_answer_length", default=10, type=int,
                        help="The maximum length of an answer that can be generated. This is needed because the start "
                             "and end predictions are not conditioned on one another.")
    parser.add_argument('--eval_top_docs', nargs='+', type=int,
                        help="top retrival passages thresholds to analyze prediction results for")
    parser.add_argument('--checkpoint_file_name', type=str, default='dpr_reader')
    parser.add_argument('--prediction_results_file', type=str, help='path to a file to write prediction results to')

    # training parameters
    parser.add_argument("--eval_step", default=2000, type=int,
                        help="batch steps to run validation and save checkpoint")
    parser.add_argument("--output_dir", default=None, type=str,
                        help="The output directory where the model checkpoints will be written to")

    parser.add_argument('--fully_resumable', action='store_true',
                        help="Enables resumable mode by specifying global step dependent random seed before shuffling "
                             "in-batch data")
    parser.add_argument("--num", default=10, type=int,
                        help="The maximum length of an answer that can be generated. This is needed because the start "
                             "and end predictions are not conditioned on one another.")
    parser.add_argument("--idx_num", default=10, type=int,
                        help="indexs")

    args = parser.parse_args()

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    setup_args_gpu(args)
    set_seed(args)
    print_args(args)
    
    trainer = ReaderTrainer(args)

    trainer.validate()


if __name__ == "__main__":
    main()
