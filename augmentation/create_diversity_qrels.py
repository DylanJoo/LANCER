import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import os
import re
import argparse
import json
import numpy as np
from copy import copy
import random
from collections import defaultdict
from tqdm import tqdm
from glob import glob
from pyserini.search.lucene import LuceneSearcher

from utils import replace_tags, load_topic_data

def check_newinfo(values, new_values):
    """
    return the value with added or the original value
    """
    mask = (values == 0)
    if (new_values[mask] > values[mask]).any():
        return True, np.max( [values, new_values], 0)
    else:
        return False, values

def get_i_doc(i, psgs_bound):
    for i_doc, bound in enumerate(psgs_bound):
        if i in bound:
            return i_doc

def binarize_amount_rerank_greedy(item_, threshold=3):
    """
    P = binarize(ratings)
    P = rerank(P) ...based on amount then maximum
    P = greedy_select(P)
    """
    item = copy(item_)
    example_id = item['example_id']
    ratings = np.array(item['ratings'])
    passages = item['passages']

    # binariz
    scores = np.zeros_like(ratings).astype(int)
    scores[(ratings >= threshold)] = 1
    answerable = (scores.sum(0) != 0)

    # navigate doc-index of passages (i.e., doci: [psg-i to psg-k])
    end = np.cumsum([len(psgs) for psgs in passages])
    start = np.append([0], end)[:(-1)]
    psgs_in_doc = [range(s, e) for (s, e) in zip(start, end)]

    # greedily include 'useful' passage, iteratively 
    ids = {"documents": [], "useful_passages": [], "redundant_passages": []}
    values = np.zeros(scores.shape[1])
    while (values[answerable]==0).sum() > 0:

        ## rerank the passages with binary + max
        mask = (values == 0)
        bin_counts = np.where(ratings[:, mask] >= threshold, 1, 0).sum(-1) 
        max_scores = np.where(ratings[:, mask] > 0, 1, 0).max(-1)
        psg_scores = bin_counts * 1 + max_scores * 0.1
        i = psg_scores.argmax()

        flag, values = check_newinfo(values, scores[i])
        i_doc = get_i_doc(i, psgs_in_doc)

        # [TODO] the input ranking should match real conditions
        if flag:
            if i_doc not in ids["documents"]:
                ids["documents"].append(i_doc)
            ids["useful_passages"].append( (i_doc, i) )

            # Type1: *hard*/strict duplication (raw rating with separated)
            # for j, r in enumerate(ratings):
            #     if (i_doc, j) in ids['redundant_passages']:
            #         continue
            #     if (i_doc, j) in ids['useful_passages']:
            #         continue
            #     if (ratings[i] > r).all():
            #         ids['redundant_passages'].append( (i_doc, j) )

    if len(ids['useful_passages']) == 0:
        return False

    # Type2: *soft*/general duplication (binarized rating with aggregated)
    max_rating = np.max(ratings[[i for (i_doc, i) in ids['useful_passages']]], axis=0)
    for j, r in enumerate(ratings):
        i_doc = get_i_doc(j, psgs_in_doc)
        if (i_doc, j) in ids['redundant_passages']:
            continue
        if (i_doc, j) in ids['useful_passages']:
            continue
        if (max_rating > r).all():
            ids['redundant_passages'].append( (i_doc, j) )

    return ids

# def minmax_rerank_greedy(scores):
# def reciprocal_rerank_greedy(scores):
def mine_distractor(topic, searcher, k=10, max_docs_return=3, ignored_prefix=""):

    hits = searcher.search(topic, k)
    to_return = []
    for h in hits:
        if ignored_prefix not in h.docid:
            hit_doc_id = h.docid.split(":")[0]
            if hit_doc_id not in to_return:
                to_return.append(hit_doc_id)
            if len(to_return) >= max_docs_return:
                return to_return
    return []

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print output")
    parser.add_argument("--shard_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--ratings_dir", type=str, default=None)
    parser.add_argument("--split", type=str, default='train')
    parser.add_argument("--random_subset", type=int, default=-1)

    # metadata of the training data
    parser.add_argument("--n_max_distractors", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=1)
    parser.add_argument("--doc_lucene_index", type=str)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True) 
    writer = {
        'topics': open(os.path.join(args.output_dir, f'{args.split}_topics_report_request.tsv'), 'w'),
        'questions': open(os.path.join(args.output_dir,  f'{args.split}_topics_exam_questions.jsonl'), 'w'),
        'qrel_p': open(os.path.join(args.output_dir,  f'{args.split}_qrels_oracle_context_pr.txt'), 'w'),
        'judgements': open(os.path.join(args.output_dir,  f'{args.split}_judgements.jsonl'), 'w')
    }

    searcher = None
    if args.n_max_distractors > 0:
        searcher = LuceneSearcher(args.doc_lucene_index)

    ## topic subset
    topics_all = []
    logger.info("load topics ...") 
    for file in tqdm(sorted(glob(os.path.join(args.shard_dir, f"topics-gen/*-{args.split}-*.json")))):
        topic = load_topic_data(file)
        topics_all += topic

    if args.random_subset > 0:
        np.random.seed(args.random_subset)
        selected = np.random.randint(0, len(topics_all), args.random_subset)
        topics_all = {r['example_id']: r['texts'] for i, r in enumerate(topics_all) if i in selected}
    else:
        topics_all = {r['example_id']: r['texts'] for r in topics_all}

    ## determine the informativeness of passages
    for file in glob(os.path.join(args.ratings_dir, f"*-{args.split}-*")):
        with open(file, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                example_id = data['example_id']
                passages = [p for psgs in data['passages'] for p in psgs]
                n_passages = len(passages)
                questions = data['questions']
                ratings = np.array(data['ratings'])

                # sanity check
                if example_id not in topics_all:
                    continue 

                if ratings.shape != (n_passages, len(questions)):
                    logger.warning(f"example id: {example_id} has incorrect number of passages: {n_passages} ({len(questions)} questions).")
                    continue

                ## step1: greedily selection
                ids = binarize_amount_rerank_greedy(data, threshold=args.threshold)
                if ids is False:
                    logger.warning(f"example id: {example_id} has no positive passages with threshold {args.threshold}")
                    continue ### skip this example if no positive passages...

                ## step2: re-organize oracle and positive document/passage ids
                judgements = []
                oracle_docids = [f"{example_id}:{i}" for i in ids['documents']]
                oracle_pos_psgids = [f"{example_id}:{i}#{j}" for (i, j) in ids['useful_passages']]
                oracle_neg_psgids = [f"{example_id}:{i}#{j}" for (i, j) in ids['redundant_passages']]
                oracle_neutral_psgids = []

                j = 0
                for i, plist in enumerate(data['passages']):
                    docid = f"{example_id}:{i}"
                    for passage in plist:
                        psgid = f"{docid}#{j}"

                        ## collect the ratings
                        judgements.append({"example_id": example_id, "pid": psgid, "rating": ratings[j].tolist() })

                        ## oracle psgids = positive && negatuve && neutral
                        if psgid not in oracle_pos_psgids+oracle_neg_psgids:
                            oracle_neutral_psgids.append(f"{docid}#{j}")
                        j += 1

                ## step2: mine distractor (if needed)
                distractor_docids = []
                if searcher is not None:
                    dis_example_doc_ids = mine_distractor(
                        topic=topics_all[data['example_id']],
                        searcher=searcher, 
                        k=10,
                        max_docs_return=args.n_max_distractors,
                        ignored_prefix=data['example_id']
                    )
                    distractor_docids += dis_example_doc_ids

                logger.info(f"#P*: {n_passages} | #P*_+: {len(oracle_pos_psgids)} | #P*_-: {len(oracle_neg_psgids)}")

                ## step3a : creating topics 
                writer['topics'].write(f"{data['example_id']}\t{topics_all[data['example_id']]}\n")

                ## step3b : creating topics exam questions
                writer['questions'].write(json.dumps({
                    'example_id': data['example_id'],
                    'topic': topics_all[data['example_id']],
                    'questions': questions
                })+'\n')

                ## step3c: create judgement cache
                for judgement in judgements:
                    writer['judgements'].write(json.dumps(judgement)+'\n')

                ## step3d : creating qrels 
                for psgid in oracle_pos_psgids:
                    writer['qrel_p'].write(f"{data['example_id']} 0 {psgid} 3\n")

                # less useful contexts (no more answerable questions)
                for psgid in oracle_neutral_psgids:
                    writer['qrel_p'].write(f"{data['example_id']} 0 {psgid} 2\n")

                # useless contexts (no higher-rated answerable questions)
                for psgid in oracle_neg_psgids:
                    writer['qrel_p'].write(f"{data['example_id']} 0 {psgid} 1\n") 

    # write
    for key in writer:
        writer[key].close()

    print('done')
