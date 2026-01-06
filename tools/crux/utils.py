import os
import glob
import collections 
import json
from tqdm import tqdm


def load_qrel(path, threshold=1):
    data = defaultdict(list)
    if path is None:
        return None
    with open(path) as f:
        for line in f:
            item = line.strip().split()
            if int(item[3]) >= threshold:
                data[item[0]].append( item[2] )
    return data

def load_judgements(path, report_file=None):
    judgements = defaultdict(lambda: defaultdict(lambda: None))
    with open(path, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            example_id = data['example_id']
            judgements[example_id].update({data['pid']: data['rating']})

    if report_file is not None:
        with open(report_file, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                example_id = data['example_id']
                pid = f"{example_id}:report"
                judgements[example_id].update({pid: data['rating']})

    return judgements
