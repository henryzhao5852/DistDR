import json
import pickle 
import csv


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--retrieval_input", default=None, type=str)
parser.add_argument("--passage_path", default=None, type=str)
parser.add_argument("--output_file", default=None, type=str)
args = parser.parse_args()


with open(args.retrieval_input, 'r') as fin:
    predictions = json.load(fin)


title2text = dict()
with open(args.passage_path, "r", encoding="utf-8") as tsvfile:
    reader = csv.reader(tsvfile, delimiter='\t', )
    # file format: doc_id, doc_text, title
    for row in reader:
        if row[0] != 'id':
            title2text[row[2]] = row[1]


instances = list()
for i, pred in enumerate(predictions):
    question = pred['question']
    answer = pred['answer']
    for ctx in pred['ctxs'][:30]:
        et = ctx['title']
        qq = question + ' ' + '[SEP]' + ' ' +  et.replace('_', ' ') + ' ' + '[SEP]' + ' ' + ''.join(title2text[et])
        instances.append({'dataset':'hotpot_dev_sec', 'question': qq, 'answers': answer,'first_et': [et], 'first_answer':[], 
        'positive_ctxs': [], 'negative_ctxs': [], 'hard_negative_ctxs': []})

print(len(instances))
with open(args.output_file, 'w', encoding='utf-8') as f:
    json.dump(instances, f, indent=2)

