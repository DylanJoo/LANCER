import json
from glob import glob

d = {}
for file in glob('ratings.*bm25*.json'):
    line =  open(file, 'r').readlines()[0]
    item = json.loads(line.strip())
    d[item['id'].replace('testb', 'test')] = item['ratings']

json.dump(d, open('ratings.oracle.bm25.json', 'w'), indent=4)
