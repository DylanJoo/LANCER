import json
from glob import glob

d = {}
for file in glob('duc04-testb-*_subtopics_2.json'):
    line =  open(file, 'r').readlines()[0]
    item = json.loads(line.strip())
    d.update(item)

transformed = {}
for key, value in d.items():
    transformed[key.replace('testb', 'test')] = value

json.dump(transformed, open('combine.json', 'w'), indent=4)
