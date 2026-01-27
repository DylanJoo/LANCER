import json
from glob import glob

d = {}
for file in glob('*_subtopics_2.json'):
    line =  open(file, 'r').readlines()[0]
    item = json.loads(line.strip())
    d.update(item)

json.dump(d, open('combine.json', 'w'), indent=4)
