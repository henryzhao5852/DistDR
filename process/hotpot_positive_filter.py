import json
import csv
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--retrieval_input", default=None, type=str)
parser.add_argument("--output_file", default=None, type=str)
args = parser.parse_args()

with open(args.retrieval_input, "r") as f:
    predictions = json.load(f)
print(len(predictions))

new_pred = list()
count = 0 


for pred in predictions:
    new_ctxs = list()
    for pp in pred['ctxs'][:100]:
        if pp['has_answer']:
            new_ctxs.append(pp)
            count += 1
    if len(new_ctxs) > 0:
        pred['ctxs'] = new_ctxs
        new_pred.append(pred)



with open(args.output_file, 'w', encoding='utf-8') as f:
    json.dump(new_pred, f, indent=2)        
