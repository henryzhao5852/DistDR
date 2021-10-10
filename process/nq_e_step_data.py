import json
from dpr.data.qa_validation import exact_match_score, _normalize_answer
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--retrieval_input", default=None, type=str)
parser.add_argument("--reader_filter_input", default=None, type=str)
parser.add_argument("--output_file", default=None, type=str)

args = parser.parse_args()


with open(args.retrieval_input, "r") as f:
    predictions = json.load(f)

with open(args.reader_filter_input, "r") as f:
    filtered_predictions= json.load(f)


print(len(filtered_predictions))
idx_dict = dict()

for ii, pred in enumerate(filtered_predictions):
    if pred['question'] in idx_dict:
        pass
    else:
        idx_dict[pred['question']] = ii

instances = list()


print(len(predictions))
for iii, pred in enumerate(predictions):
    question = pred['question']
    answer = pred['answers']


    positive_ctxs = list()
    negative_ctxs = list()
    has_ans = False
    for pp in pred['ctxs'][:100]:
        if pp['has_answer']:
            has_ans = True
            break
    if not has_ans:
        continue
    if question not in idx_dict:
        continue
    
    count = idx_dict[question]
    for pp in pred['ctxs'][:10]:
        if not pp['has_answer']:
            negative_ctxs.append({'title': pp['title'], 'text': pp['text'], 
            'score': -1, 'title_score': -1,  'passage_id': pp['id']})

    all_positives = list()
    for pp in pred['ctxs'][:100]:
        if pp['has_answer']:
            all_positives.append(pp)

    f_pred = filtered_predictions[count]

    assert f_pred['question'] == question
    psg_dict = dict()
    psg_order = list()
    

    for psg in f_pred['predictions']:
        pred_text = psg['text']
        if psg['passage_idx'] >= len(all_positives):
            continue
        if psg['passage_idx'] not in psg_dict:
            psg_order.append(psg['passage_idx'])
            psg_dict[psg['passage_idx']] = {'passage': psg['passage'], 'text': list()}
        psg_dict[psg['passage_idx']]['text'].append(pred_text)

    pos_ids = list()
    for idx in psg_order:
        gold_answers = f_pred['gold_answers']
        sc = list()
        all_spans = psg_dict[idx]['text']
        for spn in all_spans[:1]:
            sc.append(max([exact_match_score(spn, ga) for ga in gold_answers]))
        
        pp = all_positives[idx]
        if max(sc) == 1:
            pos_ids.append(idx)


    for idx in pos_ids[:1]:
        pp = all_positives[idx]
        positive_ctxs.append({'title': pp['title'], 'text': pp['text'], 'score': -1, 'title_score': -1,  'passage_id': pp['id']})


    if len(positive_ctxs) > 0 and len(negative_ctxs) > 0:
        instances.append({'dataset':'nq_e_step', 'qid': count, 'question':  question, 'answers': answer,
        'positive_ctxs': positive_ctxs, 'negative_ctxs': negative_ctxs, 'hard_negative_ctxs': []})


print(count)
print(len(instances))
with open(args.output_file, 'w', encoding='utf-8') as f:
    json.dump(instances, f, indent=2)

    
