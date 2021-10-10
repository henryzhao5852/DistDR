# DistDR

This repo contains the code to reproduce the results in the our EMNLP paper <b>Distantly-Supervised Dense Retrieval Enables Open-Domain Question Answering without Evidence Annotation</b>. The codebase is largely adapted from [Dense Passage Retrieval](https://github.com/facebookresearch/DPR) git repo. 



## Installation
We use Pytorch 1.7.1 and Huggingface transformers 2.4.0

```bash
git clone git@github.com:henryzhao5852/DistDR.git
cd DistDR
pip install .
```


## Data & Model Checkpoint on NaturalQuestions

We include NQ question-answer pairs for [training set](https://www.dropbox.com/s/kivt92b7pjge7jr/nq-train.qa.csv?dl=0), [dev set](https://www.dropbox.com/s/db6i8totzrro4yl/nq-dev.qa.csv?dl=0) and [test set](https://www.dropbox.com/s/16972vqd59svywi/nq-test.qa.csv?dl=0). We use the same Wikipedia passages as DPR, which could be found [here](https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz).

DistDR retrieval model is [here](https://www.dropbox.com/s/9xe0mgb5hg0mf53/nq_dpr_retrieval_model?dl=0), and reader model is [here](https://www.dropbox.com/s/57smko8qmbpthlz/nq_dpr_reader_model?dl=0). we also include wiki passage embeddings (with four splits), [0](https://www.dropbox.com/s/8dadjf0xbrd85j4/_0?dl=0), [1](https://www.dropbox.com/s/tais0tbzc41w7f3/_1?dl=0), [2](https://www.dropbox.com/s/hztgxmcde7cm60k/_2?dl=0), [3](https://www.dropbox.com/s/akun7cjxfe02ief/_3?dl=0), and model outputs, with [retrieval outputs](https://www.dropbox.com/s/crmse7y3eac1ldc/nq_test_retrieval.json?dl=0) and [reader outputs](https://www.dropbox.com/s/c5kfgzgfz7fg1hh/nq_test_reader.json?dl=0). 

Finally, we include our initial model checkpoint, with [retrieval model](https://www.dropbox.com/s/h8kmiqpbx5kniew/nq_retriever_initial_model.cp?dl=0) and [reader model](https://www.dropbox.com/s/5ngq945ce0avu7y/nq_reader_initial_model?dl=0), the initial distant supervision data is provided by DPR authors [here](https://dl.fbaipublicfiles.com/dpr/data/retriever/nq-old_distant_sup.json.gz).




## DistDR Inference on NaturalQuestions

Step 1: Generate embeddings for wiki passages 

```bash
python generate_dense_embeddings.py \
	--encoder_model_type hf_bert \
	--model_file={path to checkpoint} \
	--pretrained_model_cfg bert-base-uncased \
	--ctx_file={name of the wiki passages} \
	--shard_id={shard_num, 0-based} \
	--num_shards={total number of shards} \
	--out_file={result files location + name PREFX}	\
	--batch_size 512
```

Step 2: Run dense retrieval 


```bash
python nq_dense_retriever.py \
	--model_file ${path to biencoder checkpoint} \
	--ctx_file  {path to wikipedia passages .tsv file} \
	--qa_file {path to .csv QA file} \
	--encoded_ctx_file "{encoded document files glob expression}" \
	--out_file {path to output json file with results} \
	--n-docs 100
```


Step 3 (Optional): Preprocess reader data 


```bash
python preprocess_reader_data.py \
	--retriever_result {path to dense retrieval output file} \
	--do_lower_case \
	--encoder_model_type hf_bert \
	--out_file {path to output data splits} \
```

Step 4: Run reader

```bash
python train_reader.py \
  --encoder_model_type hf_bert \
  --pretrained_model_cfg bert-base-uncased \
  --prediction_results_file={path to a file to write the results to} \
  --eval_top_docs=[10,20,40,50,80,100] \
  --dev_files={path to the retriever results file to evaluate, or path after running step 3 } \
  --model_file= {path to the reader checkpoint} \
  --train.dev_batch_size=80 \
  --passages_per_question_predict=100 \
  --dev_batch_size 8 \
  --encoder.sequence_length 350
```


## DistDR Training on NaturalQuestions

We first describe DistDR for each hard e-step, then m-step. The entire DistDR needs to **repeat the process
for multiple iterations**. 

For e-step, we first run dense retrieval using current model checkpoints.

Step 1: Generate embeddings for wiki passages 

```bash
python generate_dense_embeddings.py \
	--encoder_model_type hf_bert \
	--model_file={path to checkpoint at current step} \
	--pretrained_model_cfg bert-base-uncased \
	--ctx_file={name of the wiki passages} \
	--shard_id={shard_num, 0-based} \
	--num_shards={total number of shards} \
	--out_file={result files location + name PREFX}	\
	--batch_size 512
```

Step 2: Run dense retrieval for training data (dev data is optional if you want to evaluate models at current iteration)


```bash
python nq_dense_retriever.py \
	--model_file ${path to biencoder checkpoint} \
	--ctx_file  {path to wikipedia passages .tsv file} \
	--qa_file {path to .csv QA file} \
	--encoded_ctx_file "{encoded document files glob expression}" \
	--out_file {path to output json file with results} \
	--n-docs 100
```

Step 3: Get candidate positive passages that include gold answers

```bash
python process/nq_positive_filter.py  \
--retrieval_input {path to retrieval output file} \
--out_file {path to output file} \
```


Step 4 filter candidate positive passages by reader 

Note: this step is a bit slow, we suggest splitting the data and run multiple processes to speed up the process.


```bash
python train_reader.py \
  --encoder_model_type hf_bert \
  --pretrained_model_cfg bert-base-uncased \
  --prediction_results_file={path to a file to write the results to} \
  --eval_top_docs=100 \
  --dev_files={path to the retriever results file to evaluate, or path after running step 3 } \
  --model_file= {path to the reader checkpoint} \
  --train.dev_batch_size=80 \
  --passages_per_question_predict=100 \
  --dev_batch_size 8 \
  --encoder.sequence_length 350
```

Step 5 Generate data for m-step 

```bash

python process/nq_e_step_data.py \
--retrieval_input {path to retrieval output file} \
--reader_filter_input {path to reader filtered output file} \
--out_file {path to output file} \

```

For m-step, we train both retrieval and reader models.

For retrieval training (for DistDR, we only train one epoch): 

```bash
python -m torch.distributed.launch \
	--nproc_per_node=4 train_dense_encoder.py \
	--max_grad_norm 2.0 \
	--encoder_model_type hf_bert \
	--pretrained_model_cfg bert-base-uncased \
	--seed 12345 \
	--sequence_length 192 \
	--warmup_steps 1237 \
	--batch_size 4 \
	--do_lower_case \
	--train_file "{glob expression to train files from e-step}" \
	--dev_file {leave it empty for our exp} \
	--output_dir {your output dir} \
	--learning_rate 2e-05 \
	--num_train_epochs 100 \
	--model_file {the retrieval model from last m step}
```

For reader training (for DistDR, we only train one epoch):

Step 1 (Optional): Preprocess reader data for train /dev (from the retrieval results at hard e-step)


```bash
python preprocess_reader_data.py \
	--retriever_result {path to dense retrieval output file from e-step} \
	--do_lower_case \
	--encoder_model_type hf_bert \
	--out_file {path to output data splits} \
	--is_train_set
```

Step 2: Run reader training

```bash
python train_reader.py \
	--seed 42 \
	--learning_rate 1e-5 \
	--eval_step 2000 \
	--do_lower_case \
	--eval_top_docs 50 \
	--encoder_model_type hf_bert \
	--pretrained_model_cfg bert-base-uncased \
	--train_file "{glob expression for train output files from step 1, e.g.,'nq_train*.pkl'}" \
	--dev_file {glob expression for dev output file from step 1} \
	--warmup_steps 0 \
	--sequence_length 350 \
	--batch_size 16 \
	--passages_per_question 24 \
	--num_train_epochs 100000 \
	--dev_batch_size 24 \
	--passages_per_question_predict 50 \
	--output_dir {your save dir path} \
	--gradient_accumulation_steps 4 \
	--checkpoint_file_name {reader checkpoint from last m-step}
```



## Data & Model Checkpoint on HotpotQA


We include HotpotQA question-answer pairs for [training multi-hop subset](https://www.dropbox.com/s/vj8g6vxv6b3ktkb/hotpot_train_subset.json?dl=0) and [dev multi-hop subset](https://www.dropbox.com/s/ssn8rued6bfvam8/hotpot_dev_subset.json?dl=0). We use Wikipedia split provided by HotpotQA dataset (first passage of wiki page), which could be found [here](https://www.dropbox.com/s/q37b3cuaom0pq5k/ori_data.tar.gz?dl=0).

DistDR retrieval model includes two steps, with  [first step](https://www.dropbox.com/s/h6zfrw86i8w0wsv/hotpotqa_retrieval_model_first_step?dl=0) and [second step](https://www.dropbox.com/s/vtrv3y1pz8chgn4/hotpotqa_retrieval_model_sec_step?dl=0).

we include wiki passage embeddings, for [first step](https://www.dropbox.com/s/doaz26o5csnodag/hotpot_emb_first_step.zip?dl=0) and [secnond step](https://www.dropbox.com/s/jbgyokvmtfcjq8w/hotpot_emb_sec_step.zip?dl=0).

and model outputs, with [first step retrieval outputs](https://www.dropbox.com/s/m3w9i7rii0whmwf/hotpotqa_retrieval_out_first.json?dl=0), [second step retrieval outputs](https://www.dropbox.com/s/1gymrokuqdm1e3n/hotpotqa_retrieval_out_second.json?dl=0) and [reader outputs](https://www.dropbox.com/s/urhhso0k6d4cb1y/hotpotqa_reader_output.json?dl=0). 

Finally, we include our initial [retrieval model checkpoint](https://www.dropbox.com/s/qaia41r5yvb3xn4/hotpotqa_retrieval_initial.cp?dl=0), 
and initial [reader filter model checkpoint](https://www.dropbox.com/s/akpjch0u2q7ylqx/hotpotqa_reader_filter_initial.cp?dl=0), we train BeamDR reader from scratch.




## DistDR Inference on HotpotQA



Step 1: Generate embeddings for wiki passages (run twice for both first step and second step retrieval)

```bash
python generate_dense_embeddings.py \
	--encoder_model_type hf_bert \
	--model_file={path to checkpoint} \
	--pretrained_model_cfg bert-base-uncased \
	--ctx_file={name of the wiki passages} \
	--shard_id={shard_num, 0-based} \
	--num_shards={total number of shards} \
	--out_file={result files location + name PREFX}	\
	--batch_size 512
```

Step 2: Run first step dense retrieval (with same parameter as BeamDR).


```bash
python hotpotqa_dense_retriever.py \
	--model_file ${path to first-step retriver checkpoint} \
	--ctx_file  {path to wikipedia passages .tsv file} \
	--qa_file {path to .json QA file} \
	--encoded_ctx_file "{encoded document files glob expression}" \
	--out_file {path to output json file with results} \
	--n-docs 30
```

Step 3: Generate second step queries 

```bash
python hotpot_second_query_generate.py \
	--retriever_result {path to dense retrieval output file} \
	--out_file {path to output data splits} \
	--passage_path {path to passage file}
```


Step 4: Run second step dense retrieval (with same parameter as BeamDR).

```bash
python hotpotqa_dense_retriever.py \
	--model_file ${path to second-step retriever checkpoint} \
	--ctx_file  {path to wikipedia passages .tsv file} \
	--qa_file {path to .json QA file from step 3} \
	--encoded_ctx_file "{encoded document files glob expression}" \
	--out_file {path to output json file with results} \
	--n-docs 100
```

Step 5: Evaluate retrieval outputs

```bash
python hotpot_retrieval_recall.py \
	--first_step_retrieval {path to first step dense retrieval output file} \
	--second_step_retrieval {path to second step dense retrieval output file} \
	--qa_file {path to qa file} \
	--passage_path {path to passage file}
```



For reader, our initial experiments show using implementation from DPR repo underperforms the BeamDR reader. Thus for fair
comparison, we decided to use BeamDR readers with retrieved outputs from DistDR to report reader results (albeit we use DPR reader implementation for reader filter for simpilicity). 
We refer the details to [BeamDR reader repo](https://github.com/henryzhao5852/BeamDR/tree/main/reader) 
and provide our reader models and outputs.



## DistDR Training on HotpotQA

We first describe DistDR for each hard e-step, then m-step. The entire DistDR needs to **repeat the process
for multiple iterations**. 

For e-step, we first run dense retrieval using current model checkpoints.

Step 1: Generate embeddings for wiki passages (both first step and second step)

```bash
python generate_dense_embeddings.py \
	--encoder_model_type hf_bert \
	--model_file={path to checkpoint at current step} \
	--pretrained_model_cfg bert-base-uncased \
	--ctx_file={name of the wiki passages} \
	--shard_id={shard_num, 0-based} \
	--num_shards={total number of shards} \
	--out_file={result files location + name PREFX}	\
	--batch_size 512
```

Step 2: Run first-step dense retrieval for training data (and on dev set)


```bash
python nq_dense_retriever.py \
	--model_file ${path to biencoder checkpoint} \
	--ctx_file  {path to wikipedia passages .tsv file} \
	--qa_file {path to .csv QA file} \
	--encoded_ctx_file "{encoded document files glob expression}" \
	--out_file {path to output json file with results} \
	--n-docs 10
```

Step 3: Generate second step queries 


```bash
python hotpot_second_query_generate.py \
	--retriever_result {path to dense retrieval output file} \
	--out_file {path to output data splits} \
	--passage_path {path to passage file}
```


Step 4: Run second step dense retrieval (with same parameter as BeamDR).

```bash
python hotpotqa_dense_retriever.py \
	--model_file ${path to second-step retriever checkpoint} \
	--ctx_file  {path to wikipedia passages .tsv file} \
	--qa_file {path to .json QA file from step 3} \
	--encoded_ctx_file "{encoded document files glob expression}" \
	--out_file {path to output json file with results} \
	--n-docs 10
```

Step 5: Get candidate chains

```bash
python hotpot_candidate_chains_e_step.py \
	--first_step_retrieval {path to first step dense retrieval output file} \
	--second_step_retrieval {path to second step dense retrieval output file} \
	--qa_file {path to qa file} \
	--passage_path {path to passage file}
	--is_train
```


Step 6: Get candidate positive passages that include gold answers

```bash
python process/hotpot_positive_filter.py  \
--retrieval-input {path to retrieval output file from step 5} \
--out_file {path to output file} \
```


Step 7: filter candidate positive passages by reader 

Note: this step is a bit slow, we suggest splitting the data and run multiple processes to speed up the process.


```bash
python train_reader.py \
  --encoder_model_type hf_bert \
  --pretrained_model_cfg bert-base-uncased \
  --prediction_results_file={path to a file to write the results to} \
  --eval_top_docs=100 \
  --dev_files={path to the retriever results file to evaluate} \
  --model_file= {path to the reader checkpoint} \
  --train.dev_batch_size=80 \
  --passages_per_question_predict=100 \
  --dev_batch_size 8 \
  --encoder.sequence_length 350
```

Step 8: Generate data for m-step 

```bash

python process/hotpot_e_step_data.py \
--retrieval-input {path to retrieval output file} \
--reader-filter-input {path to reader filtered output file} \
--first_step_retrieval {path to first step dense retrieval output file} \
--second_step_retrieval {path to second step dense retrieval output file} \
--qa_file {path to qa file} \
--passage_path {path to passage file} \
--out_first_step {path to first step output file} \
--out_second_step {path to second step output file} \

```

For m-step, we train both retrieval and reader models.

For retrieval training, with separate models for first step and second step (following BeamDR, though it's possible to combine both steps, using same passage encoder), we train DistDR for one step: 

```bash
python -m torch.distributed.launch \
	--nproc_per_node=4 train_dense_encoder.py \
	--max_grad_norm 2.0 \
	--encoder_model_type hf_bert \
	--pretrained_model_cfg bert-base-uncased \
	--seed 12345 \
	--sequence_length 192 \
	--warmup_steps 1237 \
	--batch_size 4 \
	--do_lower_case \
	--train_file "{glob expression to train files from e-step}" \
	--dev_file {leave it empty for our exp} \
	--output_dir {your output dir} \
	--learning_rate 2e-05 \
	--num_train_epochs 100 \
	--model_file {the retrieval model from last m step}
```

For reader training (for DistDR, we only train one epoch):

Step 1 (Optional): Preprocess reader data for train /dev (from the retrieval results at hard e-step)


```bash
python preprocess_reader_data.py \
	--retriever_result {path to dense retrieval output file from e-step} \
	--do_lower_case \
	--encoder_model_type hf_bert \
	--out_file {path to output data splits} \
	--is_train_set
```

Step 2: Update reader for filtering (note that our ablations show that reader filters converge quickly)

```bash
python train_reader.py \
	--seed 42 \
	--learning_rate 1e-5 \
	--eval_step 2000 \
	--do_lower_case \
	--eval_top_docs 50 \
	--encoder_model_type hf_bert \
	--pretrained_model_cfg bert-base-uncased \
	--train_file "{glob expression for train output files from step 5, e.g.,'hotpot_train*.pkl'}" \
	--dev_file {glob expression for dev output file from step 5} \
	--warmup_steps 0 \
	--sequence_length 350 \
	--batch_size 16 \
	--passages_per_question 24 \
	--num_train_epochs 100000 \
	--dev_batch_size 24 \
	--passages_per_question_predict 50 \
	--output_dir {your save dir path} \
	--gradient_accumulation_steps 4 \
	--checkpoint_file_name {reader checkpoint from last m-step}
```



## Citation and Contact
If you find this codebase is useful or use in your work, please cite our paper. 
```
@inproceedings{
zhao2021distdr,
title={Distantly-Supervised Dense Retrieval Enables Open-Domain Question Answering without Evidence Annotation},
author={Chen Zhao and Chenyan Xiong and Jordan Boyd-Graber and Hal {Daum\'{e} III},
booktitle={Emperical Methods in Natural Language Processing},
year={2021}
}
```


We welcome your feedback! If you have questions, suggestions and bug reports, please email chenz@cs.umd.edu.
