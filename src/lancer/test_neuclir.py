import json
from collections import defaultdict
from wrapper import rerank
from datasets import load_dataset

## Load Datasets 

# topics (query + additional info need), the first-stage retrieved results, the corpus
topics = {}
with open('/home/dju/lancer-legacy/data/neuclir24-test-request.jsonl', 'r') as f:
    for line in f:
        item = json.loads(line)
        topics[item['request_id']] = item

# runs are provided
runs = {}
with open('/home/dju/lancer-legacy/data/neuclir-runs/bm25-neuclir.run', 'r') as f:
    for line in f:
        qid, _, docid, rank, score, _ = line.strip().split()
        if qid not in runs:
            runs[qid] = {}
        runs[qid][docid] = float(score)

# corpus could be found in the huggingface
corpus = load_dataset('json', data_files='/home/dju/datasets/neuclir1/*.processed_output.jsonl.gz', num_proc=3, split='train')
corpus = {example["id"]: {"title": example["title"], "text": example["text"]} for example in corpus}

# Rerank
reranked_run = rerank(
    runs=runs,
    queries={qid: topics[qid]['problem_statement'] for qid in topics},
    topics=topics,
    corpus=corpus,
    k=100,
    n_subquestions=5,
    aggregation='sum',
    qg_path=None,
    judge_path=None,
    vllm_kwargs={
        'temperature': 0.8,
        'max_tokens': 256,
        'top_p': 1.0,
    }
)

from crux

# Save file
# reranker_name = args.reranker.replace('/', ':')
# reranker_name += ":oracle" if 'human' in args.tag else ""
# reranker_name += f":agg_{args.agg}" if args.agg != 'sum' else ""
# reranker_name += f":nq_{args.n_subquestions}" if args.n_subquestions != 2 else ""
# file_name = f"runs/{args.service_name}+{reranker_name}.run"
# with open(file_name, 'w') as f:
#     for qid in rac_data:
#         for rank, docid in enumerate(rac_data[qid]['docids'], start=1):
#             f.write(f"{qid} Q0 {docid} {rank} {1/rank} {args.reranker}\n")
# print(reranked_run)
