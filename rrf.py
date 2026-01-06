from collections import defaultdict
import json
from crux.tools import load_run_or_qrel

run1 = load_run_or_qrel('qwen3-mds-duc04.+1sq.run', topk=100)
run2 = load_run_or_qrel('qwen3-mds-duc04.+2sq.run', topk=100)


rrf_scores = {}
for qid in run1:
    rrf_scores[qid] = defaultdict(float)
    for rank, docid in enumerate(run1[qid], start=1):
        rrf_scores[qid][docid] += 1/(60+rank)

for qid in run2:
    for rank, docid in enumerate(run2[qid], start=1):
        rrf_scores[qid][docid] += 1/(60+rank)

print(len(rrf_scores[qid]))
with open('qwen-mds-duc04.mq.run', 'w') as f:
    for qid in rrf_scores:
        final_scores = sorted(rrf_scores[qid].items(), key=lambda x: x[1], reverse=True)
        for rank, (docid, score) in enumerate(final_scores, start=1):
            f.write(f"{qid} Q0 {docid} {rank} {score:.6f} RRF\n")
