import json 
import pickle
import unicodedata
from tqdm import tqdm
import string
from dpr.data.qa_validation import exact_match_score, _normalize_answer
import argparse

def normalize(text):
    return unicodedata.normalize('NFD', text).replace(' ', '_').lower()




parser = argparse.ArgumentParser()
parser.add_argument("--retrieval_input", default=None, type=str)
parser.add_argument("--reader_filter_input", default=None, type=str)
parser.add_argument("--first_step_retrieval", default=None, type=str)
parser.add_argument("--second_step_retrieval", default=None, type=str)
parser.add_argument("--qa_file", default=None, type=str)
parser.add_argument("--passage_path", default=None, type=str)
parser.add_argument("--out_first_step", default=None, type=str)
parser.add_argument("--out_second_step", default=None, type=str)

parser.add_argument("--is_train", action='store_true')

args = parser.parse_args()

with open(args.qa_file, 'r') as fin:
    dataset = json.load(fin)


title2text = dict()
title2idx = dict()

with open(args.passage_path, "r", encoding="utf-8") as tsvfile:
    reader = csv.reader(tsvfile, delimiter='\t', )
    # file format: doc_id, doc_text, title
    for row in reader:
        if row[0] != 'id':
            title2text[normalize(row[2])] = row[1]
            title2idx[normalize(row[2])] = row[0]


instances = []
new_instances = []


idx = 0 


with open(args.first_step_retrieval, 'r') as fin:
    ff_predictions = json.load(fin)

with open(args.second_step_retrieval, 'r') as fin:
    sec_predictions = json.load(fin)


with open(args.retrieval_input, "r") as f:
    predictions = json.load(f)

with open(args.reader_filter_input, "r") as f:
    filtered_predictions= json.load(f)
  

idx_dict = dict()
count = 0 

for ii, pred in enumerate(filtered_predictions):
    if pred['question'] in idx_dict:
        pass
    else:
        idx_dict[pred['question']] = ii





print(len(predictions), count, len(idx_dict))
count = 0 
pos_dict = dict()
hit = 0
f_hit = 0
s_hit = 0
pos_dict1 = dict()
span_dict = dict()

for iii, pred in enumerate(tqdm(predictions)):
    
    question = pred['question']
    answer = pred['answers']


    positive_ctxs = list()
    negative_ctxs = list()
    has_ans = False


    count = idx_dict[question]

    ii = 0
    total_neg = 0 
    total_pos = 0
    all_positives = list()
    for pp in pred['ctxs']:
        if pp['has_answer']:
            all_positives.append(pp)



    f_pred = filtered_predictions[count]

    assert f_pred['question'] == question
    psg_dict = dict()
    psg_order = list()
    

    for psg in f_pred['predictions']:
        #print(psg)
        pred_text = psg['text']
        if psg['passage_idx'] >= len(all_positives):
            print('????????')
            continue
        if psg['passage_idx'] not in psg_dict:
            psg_order.append(psg['passage_idx'])
            psg_dict[psg['passage_idx']] = {'passage': psg['passage'], 'text': list(), 'score':psg['relevance_score'], 'span_score': list() }
        psg_dict[psg['passage_idx']]['text'].append(pred_text)
        psg_dict[psg['passage_idx']]['span_score'].append(psg['score'])

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


    if len(pos_ids) > 0:
        count += 1
    pos_dict1[question] = list()
    for idx in pos_ids[:1]:
        pp = all_positives[idx]
        
        sec_et = pp['title'].split('#######')[1]
        first_et = pp['title'].split('#######')[0]
        
        pos_dict[question] = {'first_et': first_et, 'sec_et': sec_et}


idx = 0  

for data in dataset:
    qid = data['_id']
    question = data['question']

    first_pred = ff_predictions[idx]
    sec_pred = sec_predictions[idx * 10: (idx + 1) * 10]
    idx += 1
    if question not in pos_dict:
        continue
    
    assert first_pred['question'] == question
    f_et = pos_dict[question]['first_et']
    s_et = pos_dict[question]['sec_et']

    sec_idx = -1
    positive_ctxs = list()
    negative_ctxs = list()
    positive_ctxs.append({'title': f_et.replace('_',' '), 'text': ''.join(title2text[f_et]), 'score': -1, 
                            'title_score': -1, 'passage_id': str(title2idx[f_et])})
        
    for iii, ctx in enumerate(first_pred['ctxs'][:10]):
        if normalize(ctx['title']) == normalize(f_et):
            sec_idx = iii
        elif ctx['title'] == s_et:
            continue
        else:
            neg_title = ctx['title']
            negative_ctxs.append({'title': neg_title.replace('_',' '), 'text': ''.join(title2text[neg_title]), 'score': -1, 
                            'title_score': -1, 'passage_id': str(title2idx[neg_title])})
    
    if len(positive_ctxs) >0 and len(negative_ctxs) > 0:
        instances.append({'dataset':'hotpot_dev_first', 'question': data['question'], 'answers': [dev_hops[qid]['first_hop']],
                            'positive_ctxs': positive_ctxs, 'negative_ctxs': negative_ctxs, 'hard_negative_ctxs': []})
    
    
    new_positive_ctxs = list()
    new_negative_ctxs = list()
    new_positive_ctxs.append({'title': s_et.replace('_',' '), 'text': ''.join(title2text[s_et]), 'score': -1, 
                            'title_score': -1, 'passage_id': str(title2idx[s_et])})


    if sec_idx == -1:
        print('????????????')
    else:
        for iii, ctx in enumerate(sec_pred[sec_idx]['ctxs']):
            if ctx['title'] == f_et:
                continue
            elif ctx['title'] == s_et:
                continue
            else:
                neg_title = ctx['title']
                new_negative_ctxs.append({'title': neg_title.replace('_',' '), 'text': ''.join(title2text[neg_title]), 'score': -1, 
                                'title_score': -1, 'passage_id': str(title2idx[neg_title])})
    
    if len(new_positive_ctxs) >0 and len(new_negative_ctxs) > 0:
        qq = data['question'] + ' ' + '[SEP]' + ' ' +  f_et.replace('_', ' ') + ' ' + '[SEP]' + ' ' + ''.join(title2text[f_et])
        new_instances.append({'dataset':'hotpot_dev_sec', 'question': qq, 'answers': [dev_hops[qid]['sec_hop']],
                    'positive_ctxs': new_positive_ctxs, 'negative_ctxs': new_negative_ctxs, 'hard_negative_ctxs': []})




with open(args.out_second_step, 'w', encoding='utf-8') as f:
    json.dump(new_instances, f, indent=2)


with open(args.out_first_step, 'w', encoding='utf-8') as f:
    json.dump(instances, f, indent=2)
   

   