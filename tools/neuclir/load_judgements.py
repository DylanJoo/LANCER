import glob
import json
import re

from collections import defaultdict

from augmentation.gen_ratings import gen_ratings

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def load_judgements_from_disk(judgements, path, load_andor=False):

    with open(path, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            qid = data['id']
            
            if load_andor:
                for pid, rating in zip(data['docids'], data['ratings_andor']):
                    judgements[qid][pid] = rating
            else:
                for pid, rating in zip(data['docids'], data['ratings']):
                    if 'report' in pid:
                        continue
                    judgements[qid][pid] = rating

    return judgements

def load_judgements(args, rac_data,
                    compute_missing_judgements=False,
                    load_andor=False) -> defaultdict:

    judgements = defaultdict(lambda: defaultdict(lambda: None))
    missing_judgements = defaultdict(lambda: defaultdict(lambda: None))
    if args.crux_artifacts_path:
        judgements = load_judgements_from_disk(judgements, args.crux_artifacts_path, load_andor=load_andor)

        if 'mds' in args.dataset_name:
            return judgements

    for qid in rac_data.keys():
        for pid in rac_data[qid]["docids"]:
            if judgements[qid][pid] is None:
                #logger.warning(f"qid {qid}, pid {pid} is not found in judgements loaded from disk")
                missing_judgements[qid][pid] = True
    
    if compute_missing_judgements and missing_judgements:
        qrel_missing = {qid: list(pid_dict.keys()) for qid, pid_dict in missing_judgements.items()}
        gen_ratings(args, qrel_missing=qrel_missing)

    return judgements

# NOTE: include AND/OR as nuggets
def load_human_judgements(path: str, and_or_subtopics=False):
    judgements = defaultdict(lambda: defaultdict(lambda: []))
    files = [f for f in glob.glob(f'{path}/nuggets_*json')]
    subquestions = {}
    if and_or_subtopics:
        with open(path, 'r') as f:
            for line in f:
                qid, subtopic_id, docid, rating = line.strip().split()
                # assin the corresponding index as relevance=3
                if len(judgements[qid][docid]) < int(subtopic_id)+1:
                    judgements[qid][docid] += [0] * (int(subtopic_id) - len(judgements[qid][docid]) - 1)
                    judgements[qid][docid] += [3]

        # fix the length of all judgements to be the same
        for qid in judgements.keys():
            max_length = max([len(v) for v in judgements[qid].values()])
            for docid in judgements[qid].keys():
                curr_length = len(judgements[qid][docid])
                judgements[qid][docid] += [0] * (max_length - curr_length)
        return judgements

    for file in files:
        match = re.search(r'nuggets_(\d+)\.json$', file)
        qid = str(match.group(1))
        data = json.load(open(file, 'r'))

        subquestions = list(data.keys())
        for idx, subquestion in enumerate(subquestions):
            answers = data[subquestion][1]
            for a in answers.values():
                for docid in a:
                    if not judgements[qid][docid]:
                        judgements[qid][docid] = [0] * len(subquestions)
                    else:
                        judgements[qid][docid][idx] = 3 # all human judgements are ranked with relevance=3

    return judgements
