import json 
import pickle
import unicodedata
from tqdm import tqdm
import string
import argparse

def normalize(text):
    return unicodedata.normalize('NFD', text).replace(' ', '_').lower()


parser = argparse.ArgumentParser()
parser.add_argument("--first_step_retrieval", default=None, type=str)
parser.add_argument("--second_step_retrieval", default=None, type=str)
parser.add_argument("--qa_file", default=None, type=str)
parser.add_argument("--passage_path", default=None, type=str)
parser.add_argument("--out_file", default=None, type=str)

parser.add_argument("--is_train", action='store_true')

args = parser.parse_args()



instances = []

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



with open(args.first_step_retrieval, 'r') as fin:
    predictions = json.load(fin)

with open(args.second_step_retrieval, 'r') as fin:
    sec_predictions = json.load(fin)

count = 0
total = 0


instances = list()
for data in tqdm(dataset):
    qid = data['_id']
    
    
    first_pred = predictions[idx]
    sec_pred = sec_predictions[idx * 10: (idx + 1) * 10]
    idx += 1
    has_correct = False
    answer = data['answer'].lower()
    chain_list = list()
    score_list = list()
    first_hop_ets = list()

    for i, first_ctx in enumerate(first_pred['ctxs'][:10]):
        first_title = first_ctx['title']
        first_hop_ets.append(first_title)

        f_p = first_ctx['text'].lower()

        full_ctxs = list()

        sec_rel = sec_pred[i]
        for sec_ctx in sec_rel['ctxs'][:10]:
            sec_title = sec_ctx['title']
            if sec_title == first_title:
                continue
            s_p = sec_ctx['text'].lower()
            #if answer not in f_p and answer in s_p:
            chain_list.append([first_title, sec_title])
            score_list.append(first_ctx['score'] + sec_ctx['score'])
            full_ctxs.append(sec_ctx)

    all_ctxs = list()
    alll_ans = False
    if len(chain_list) > 0:
        total += 1
        sorted_idxs = sorted(range(len(score_list)), reverse=True, key=lambda k: score_list[k])
        if not is_train:
            sorted_idxs = sorted_idxs[:50]

        for iidx in sorted_idxs:
            f_et = chain_list[iidx][0]
            s_et = chain_list[iidx][1]


            full_para = ''.join(title2text[f_et]) + ' ' + ''.join(title2text[s_et])
            all_tt = s_et.replace('_', ' ')

            has_answer = False
            if answer.lower() in full_para.lower():
                
                start = 0 
                while True:
                    pos = full_para.lower().find(answer, start)
                    if pos == -1:
                        break
                    if pos == 0:
                        if pos + len(answer) >= len(full_para) or full_para[pos + len(answer)] == ' ' or full_para[pos + len(answer)] in string.punctuation:
                            has_answer = True
                            break
                        
                    else:
                        if full_para[pos -1] == ' ':
                            #print(s_p, answer, pos, )
                            if pos + len(answer) >= len(full_para) or full_para[pos + len(answer)] == ' ' or full_para[pos + len(answer)] in string.punctuation:
                                has_answer = True
                                break
                    start = pos + 1


            all_ctxs.append({'id':title2idx[s_et], 'title':all_tt, 'score': score_list[iidx], 'text': full_para, 'has_answer': has_answer})



        instances.append({'question':data['question'], 'answers': [data['answer']], 'ctxs': all_ctxs})
    if not alll_ans:
        count += 1
print(count, len(instances))        



with open(args.out_file, 'w', encoding='utf-8') as f:
    json.dump(instances, f, indent=2)
