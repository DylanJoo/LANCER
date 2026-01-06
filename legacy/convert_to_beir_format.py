from tqdm import tqdm
import json

with open('/exp/scale25/neuclir/docs/mlir.mt.jsonl', 'r') as r, open('temp.jsonl', 'w') as w:
    for line in tqdm(r):
        try:
            data = json.loads(line)
            data['_id'] = data.pop('id')
            w.write(json.dumps(data)+'\n')
        except:
            print(line)
