import json
import unicodedata
from tqdm import tqdm
import pickle 

import argparse

def normalize(text):
    return unicodedata.normalize('NFD', text).replace(' ', '_').lower()



parser = argparse.ArgumentParser()
parser.add_argument("--first_step_retrieval", default=None, type=str)
parser.add_argument("--second_step_retrieval", default=None, type=str)
parser.add_argument("--qa_file", default=None, type=str)
parser.add_argument("--passage_path", default=None, type=str)

args = parser.parse_args()


title2text = dict()
with open(args.passage_path, "r", encoding="utf-8") as tsvfile:
    reader = csv.reader(tsvfile, delimiter='\t', )
    # file format: doc_id, doc_text, title
    for row in reader:
        if row[0] != 'id':
            title2text[normalize(row[2])] = row[1]

with open(args.first_step_retrieval, 'r') as fin:
    first_predictions = json.load(fin)


with open(args.second_step_retrieval, 'r') as fin:
    sec_predictions = json.load(fin)


with open(args.qa_file, 'r') as fin:
    dataset = json.load(fin)

idx = 0 
p_recall = 0 
total = 0
ans_recall = 0 
chain_ct = 0
for data in tqdm(dataset):
    qid = data['_id']
    question = data['question']
    

    first_pred = first_predictions[idx]
    sec_pred = sec_predictions[idx * 30: (idx + 1) * 30]
    idx += 1
    chains = list()
    chain_scores = list()
    for iii, ctx in enumerate(first_pred['ctxs'][:30]):
        for sec_ctx in sec_pred[iii]['ctxs'][:100]:
            if ctx['title'] == sec_ctx['title']:
                continue
            title_list = ctx['title'] + '#######' + sec_ctx['title']
            chains.append(title_list) 
            full_score = float(ctx['score']) + float(sec_ctx['score'])
            chain_scores.append(full_score)


    sorted_idxs = sorted(range(len(chain_scores)), reverse=True, key=lambda k: chain_scores[k])
    final_titles = list()
    for s_idx in sorted_idxs:
        if chains[s_idx] not in final_titles:
            final_titles.append(chains[s_idx])
    
    final_titles = final_titles[:10]



    supp_set = set()
    for supp in data['supporting_facts']:
        title = supp[0]
        supp_set.add(normalize(title))
    #total += 1
    pred = list()
    supp_set = list(supp_set)
    total += 1

    
    for tt in final_titles:
        f_et = tt.split('#######')[0]
        s_et = tt.split('#######')[1]
        if f_et not in pred:
            pred.append(f_et)

        if s_et not in pred:
            pred.append(s_et)
    
    if supp_set[0] in pred or supp_set[1] in pred:
        p_recall += 1

    
    if supp_set[0] in pred and supp_set[1] in pred:
        chain_recall += 1
    
    for tt in final_titles:
        f_et = tt.split('#######')[0]
        s_et = tt.split('#######')[1]
        if data['answer'].lower() in ''.join(title2text[f_et]).lower():
            ans_recall += 1
            break
        if data['answer'].lower() in ''.join(title2text[s_et]).lower():
            ans_recall += 1
            break

print(ans_recall, total, ans_recall / total)
print(p_recall, total, p_recall / total)
print(chain_recall, total, chain_recall / total)
