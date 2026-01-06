from collections import defaultdict

import numpy as np

from augmentation.gen_ratings import async_gen_ratings

SUM_SCORES_MEAN = 152.21 # 325.89473
SUM_SCORES_STDEV = 23.74 # 58.24364

sum_scores_list = []
import statistics

async def rerank_with_mcranker(args, ordered_docs, topic, k,
                           docid_to_score=None,
                           combine_with_rrf=True):


    ratings = await async_gen_ratings(args, 
                                      docs_to_rerank=ordered_docs, 
                                      topic=topic,
                                      reranking=True)

    ranked_docs, docid_to_score_rerank = rank_by_sum_of_crux_scores(ratings)

    if combine_with_rrf:
        ranked_docs = rrf_fusion(docid_to_score, docid_to_score_rerank)

    return ranked_docs[:k]

def rank_by_sum_of_crux_scores(ratings):
    sums = [sum(row) for row in ratings["ratings"]]

    indexed_sums = list(enumerate(sums))
    ranked = sorted(indexed_sums, key=lambda x: x[1], reverse=True)
    ranked_indices = [index for index, _ in ranked]
    ranked_docs = [ratings["docids"][i] for i in ranked_indices]

    # create dict of docid -> score
    docid_to_score = dict(zip(ratings["docids"], sums))

    return ranked_docs, docid_to_score

def rrf_fusion(ranked_a, ranked_b, ranked_c=None, k=60, weights=[0.5, 0.5]):
    sorted_ranked_a = sorted(ranked_a.items(), key=lambda item: item[1], reverse=True)
    sorted_ranked_b = sorted(ranked_b.items(), key=lambda item: item[1], reverse=True)
    if ranked_c:
        sorted_ranked_c = sorted(ranked_c.items(), key=lambda item: item[1], reverse=True)
    scores = defaultdict(float)
    for rank, (doc, _) in enumerate(sorted_ranked_a):
        scores[doc] += 2*(weights[0]) / (k + rank + 1)
    for rank, (doc, _) in enumerate(sorted_ranked_b):
        scores[doc] += 2*(weights[1]) / (k + rank + 1)
    if ranked_c:
        for rank, (doc, _) in enumerate(sorted_ranked_c):
            scores[doc] += 2*(weights[2]) / (k + rank + 1)
    rrf_dict = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    sorted_rrf = [docid for docid, _ in rrf_dict]
    return sorted_rrf