import json
from glob import glob

d = {}
for file in glob('ratings.*lsr*.json'):
    line =  open(file, 'r').readlines()[0]
    item = json.loads(line.strip())
    d[item['id'].replace('testb', 'test')] = item['ratings']

json.dump(d, open('ratgins.oracle.lsr.json', 'w'), indent=4)
