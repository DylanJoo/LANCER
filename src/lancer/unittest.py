from collections import defaultdict
from wrapper import rerank

runs = {'q1': {'d1': 0.8, 'd2': 0.5}}
queries = {'q1': 'What is AI?'}
corpus = {'d1': 'Artificial Intelligence is ...', 'd2': 'AI stands for ...'}
topics = {'q1': defaultdict(str)}

r = rerank(
    runs=runs,
    queries=queries,
    corpus=corpus,
    k=100,
    n_subquestions=5,
    aggregation='sum',
    topics=topics,
    rerun_qg=True,
    rerun_judge=True,
    qg_path=None,
    judge_path=None,
)
print(r)
