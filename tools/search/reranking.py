import os
import json
from collections import defaultdict

import numpy as np

from augmentation.gen_ratings import async_gen_ratings

SUM_SCORES_MEAN = 152.21 # 325.89473
SUM_SCORES_STDEV = 23.74 # 58.24364

sum_scores_list = []
import statistics

async def rerank_with_crux(args, ordered_docs, topic, k,
                           docid_to_score=None,
                           combine_with_rrf=False,
                           get_claims_ratings=False,
                           get_higher_cov_ratings=False):


    # NOTE: search for precomputed rating
    if 'human' in args.tag:
        rating_file = f"/home/hltcoe/jhueiju/temp/{args.dataset_name}/ratings.{topic['request_id']}.{args.service_name}.human.json"
    else:
        rating_file = f"/home/hltcoe/jhueiju/temp/{args.dataset_name}/ratings.{topic['request_id']}.{args.service_name}.n{args.n_subquestions}.json"
        if args.include_original:
            rating_file = f"/home/hltcoe/jhueiju/temp/{args.dataset_name}/ratings.{topic['request_id']}.{args.service_name}.n{args.n_subquestions}+1.json"

    if args.overwrite:
        if os.path.exists(rating_file):
            os.remove(rating_file)

    if os.path.exists(rating_file) and '70B' in args.model:
        with open(rating_file, "r") as f:
            ratings = json.load(f)
    else:
        ratings = await async_gen_ratings(args, 
                                          docs_to_rerank=ordered_docs, 
                                          topic=topic,
                                          reranking=True)
    if (os.path.exists(rating_file) is False) and '70B' in args.model:
        with open(rating_file, "w") as f:
            json.dump(ratings, f)

    if (args.debug) and (args.dataset_name == 'neuclir'):
        # NOTE: debug for alpha ndcg
        ratings_new = defaultdict(list)
        with open('/exp/scale25/neuclir/eval/qrel/neuclir24-test-request.qrel') as f:
            for line in f:
                qid, iteration, docid, score = line.split(' ')
                if qid == topic['request_id']:
                    # only the retrieved
                    # if docid in ratings['docids']:
                    #     ratings_new[docid].append(int(iteration)-1)
                    ratings_new[docid].append(int(iteration)-1)

        # postprocess into ratings
        max_indices = max([max(ratings_new[docid]) for docid in ratings_new])

        for docid in ratings_new:
            indices = ratings_new[docid]
            ratings_new[docid] = [0 for _ in range(max_indices+1)]
            for idx in indices:
                ratings_new[docid][idx] = 3

        ratings = {
            'ratings': [ratings_new[docid] for docid in ratings_new],
            'docids': [docid for docid in ratings_new],
        }

    if (args.debug) and ('mds' in args.dataset_name):
        # NOTE: debug for coverage
        rating_file_oracle = "/exp/scale25/artifacts/crux/temp/neuclir_format/ratings.mds_duc04.jsonl"
        with open(rating_file_oracle, 'r') as f:
            for line in f:
                item = json.loads(line.strip())
                if item['id'] == topic['request_id']:
                    for i, docid in enumerate(item['docids']):
                        if 'report' in docid: # remove the report's answerabiltiy 
                            item['docids'].pop(i)
                            item['ratings'].pop(i)

                    ratings = {'ratings': item['ratings'], 'docids': item['docids']}

    # NOTE: implement aggregation here
    if args.agg.startswith('sum'):
        tau = int(args.agg.split("tau=")[-1]) if 'tau' in args.agg else 1
        ranked_docs, docid_to_score_rerank = rank_by_sum_of_crux_scores(ratings, tau)

    if args.agg.startswith('greedy-sum'):
        tau = int(args.agg.split("tau=")[-1]) if 'tau' in args.agg else 1
        ranked_docs, docid_to_score_rerank = greedy_sum(ratings, tau)

    if args.agg.startswith('greedy-coverage'): # NOTE: Shall we check the logic again?
        tau = int(args.agg.split("tau=")[-1]) if 'tau' in args.agg else 1
        # ranked_docs, docid_to_score_rerank_c = rank_by_maximizing_coverage(ratings)
        ranked_docs, docid_to_score_rerank_c = greedy_coverage(ratings, tau)

    if args.agg.startswith('greedy-alpha'): # NOTE: Shall we check the logic again?
        tau = int(args.agg.split("tau=")[-1]) if 'tau' in args.agg else 1
        ranked_docs, docid_to_score_rerank_c = greedy_alpha(ratings, tau)

    if args.agg.startswith('rrf'): # NOTE: Shall we check the logic again?
        ranked_docs, docid_to_score_rerank_c = rank_by_rrf(ratings)

    #_, docid_to_score_rerank = rank_by_sum_of_high_quality_crux_scores(ratings)
    #ranked_docs, docid_to_score_rerank = rank_by_late_coverage_maximization(ratings, k=5, k_coverage=7)
    """
    top_k_scores = {_k: v for _k, v in docid_to_score_rerank.items() if _k in ranked_docs[:k]}
    sum_top_k_scores = sum(top_k_scores.values())

    sum_scores_list.append(sum_top_k_scores)
    try:
        sum_scores_mean = statistics.mean(sum_scores_list)
        sum_scores_std_dev = statistics.stdev(sum_scores_list)
        print(f"sum_scores_mean: {sum_scores_mean}")
        print(f"sum_scores_std_dev: {sum_scores_std_dev}")
    except:
        pass

    z_score = (sum_top_k_scores - SUM_SCORES_MEAN) / SUM_SCORES_STDEV
    rrf_weights = modulate_weights(z_score)
    """

    # if combine_with_rrf:
    #     ranked_docs = rrf_fusion(docid_to_score, docid_to_score_rerank)
    #                              # ranked_c=docid_to_score_rerank_c)
    #                              # weights=rrf_weights)

    return ranked_docs[:k]

def rank_by_sum_of_crux_scores(ratings, tau=0):
    if tau == 0:
        sums = [sum(row) for row in ratings["ratings"]]
    else:
        sums = []
        for row in ratings["ratings"]:
            score = sum( [r * int(r >= tau) for i, r in enumerate(row)] )
            sums.append(score)
    """
    sums = []
    for row in ratings["ratings"]:
        #if sum(row[2:4]) <= 8:
        #    _sum = sum(row[0:2])
        #else:
        #    _sum = sum(row[0:2]) + 10 # bump up high-quality documents
        _sum = sum(row[0:2])
        quality_multiplier = sum(row[2:4]) / 10
        _sum = _sum * quality_multiplier
        sums.append(_sum)
    """
    indexed_sums = list(enumerate(sums))
    ranked = sorted(indexed_sums, key=lambda x: x[1], reverse=True)
    ranked_indices = [index for index, _ in ranked]
    ranked_docs = [ratings["docids"][i] for i in ranked_indices]

    # create dict of docid -> score
    docid_to_score = dict(zip(ratings["docids"], sums))
    return ranked_docs, docid_to_score

def greedy_sum(ratings, tau=0):
    m, n = len(ratings["ratings"]), len(ratings["ratings"][0])

    # postprocess them
    matrix = []
    sums = []
    for row in ratings["ratings"]:
        row = [r * int(r >= tau) for r in row]
        matrix.append(row)
        sums.append(sum(row))

    ratings["ratings"] = matrix

    remaining_indices = np.argsort(sums).tolist()[::-1]
    reranked_indices = []
    omitted_indices = []
    best_so_far = np.array([0] * n)

    while len(remaining_indices) > 0:

        ## check available docs
        gains = []
        for index in remaining_indices:
            gain = sum([max(b-a, 0) for a, b in zip(best_so_far, ratings["ratings"][index])])
            gains.append(gain)

        ## find the best one
        i = np.argmax(gains)
        index = remaining_indices.pop(i)
        if gains[i] > 0:
            n = len(remaining_indices)
            best_so_far = np.array([max(a, b) for a, b in zip(best_so_far, ratings["ratings"][index])])
            reranked_indices.append(index)
        else:
            omitted_indices.append(index)

    # prepare return
    ranked_docs = [ratings["docids"][i] for i in reranked_indices + omitted_indices]

    # create dict of docid -> score
    docid_to_score = {docid: 1/rank for rank, docid in enumerate(ranked_docs, start=1)}

    return ranked_docs, docid_to_score

def greedy_alpha(ratings, tau=1):
    alpha = 0.5 # the higher the more diverse

    m, n = len(ratings["ratings"]), len(ratings["ratings"][0])

    reranked_indices = []
    selected = [False] * m
    subtopic_gain = [1.0] * n # initialize with zero gain for each subtopic

    for rank in range(m):
        where = -1
        max_score = 0.0

        # looking for the best document to add
        for i in range(m):

            # ignore when i is already selected
            if selected[i]:
                continue

            # calculate the gains
            current_score = 0
            for j in range(n):
                if ratings["ratings"][i][j] >= tau: # in ndeval, we only have binary relevance?
                    current_score += subtopic_gain[j]

            # deal with tie? now we select the earlier one.
            if (where == -1) or (current_score > max_score):
                where = i
                max_score = current_score

        reranked_indices.append(where)
        selected[where] = True

        # update the gain
        for j in range(n):
            if ratings["ratings"][where][j] >= tau:
                subtopic_gain[j] *= (1.0 - alpha)
    
    # prepare return
    ranked_docs = [ratings["docids"][i] for i in reranked_indices]

    # create dict of docid -> score
    docid_to_score = {docid: 1/rank for rank, docid in enumerate(ranked_docs, start=1)}

    return ranked_docs, docid_to_score

def greedy_coverage(ratings, tau=0):
    m, n = len(ratings["ratings"]), len(ratings["ratings"][0])

    # postprocess them
    matrix = []
    sums = []
    for row in ratings["ratings"]:
        row = [int(r >= tau) for r in row]
        matrix.append(row)
        sums.append(sum(row))

    ratings["ratings"] = matrix

    remaining_indices = np.argsort(sums).tolist()[::-1]
    reranked_indices = []
    omitted_indices = []
    best_so_far = np.array([0] * n)

    while len(remaining_indices) > 0:

        ## check available docs
        gains = []
        for index in remaining_indices:
            gain = sum([max(b-a, 0) for a, b in zip(best_so_far, ratings["ratings"][index])])
            gains.append(gain)

        ## find the best one
        i = np.argmax(gains)
        index = remaining_indices.pop(i)
        if gains[i] > 0:
            n = len(remaining_indices)
            best_so_far = np.array([max(a, b) for a, b in zip(best_so_far, ratings["ratings"][index])])
            reranked_indices.append(index)
        else:
            omitted_indices.append(index)

    # prepare return
    ranked_docs = [ratings["docids"][i] for i in reranked_indices + omitted_indices]

    # create dict of docid -> score
    docid_to_score = {docid: 1/rank for rank, docid in enumerate(ranked_docs, start=1)}

    return ranked_docs, docid_to_score
# weights=[0.4, 0.4, 0.1, 0.05, 0.05], [0.7, 0.3]
# def rank_by_sum_of_weighted_crux_scores(ratings, ratings_coverage=None,
#                                         weights=[0.7, 0.3]):
#     if ratings_coverage:
#         sums = []
#         for idx, row in enumerate(ratings["ratings"]):
#             sum_low_granularity = sum(row) / len(row)
#             row_high_granularity = ratings_coverage["ratings"][idx]
#             sum_high_granularity = sum(row_high_granularity) / len(row_high_granularity)
#             weighted_sum = weights[0]*sum_low_granularity + weights[1]*sum_high_granularity
#             sums.append(weighted_sum)
#     else:
#         sums = [sum(v * w for v, w in zip(row, weights)) for row in ratings["ratings"]]
#
#     indexed_sums = list(enumerate(sums))
#     ranked = sorted(indexed_sums, key=lambda x: x[1], reverse=True)
#     ranked_indices = [index for index, _ in ranked]
#     ranked_docs = [ratings["docids"][i] for i in ranked_indices]
#
#     # create dict of docid -> score
#     docid_to_score = dict(zip(ratings["docids"], sums))
#
#     return ranked_docs, docid_to_score

def rank_by_sum_of_high_quality_crux_scores(ratings):
    sums = [sum(val for val in row if val >= 2) for row in ratings["ratings"]]
    indexed_sums = list(enumerate(sums))
    ranked = sorted(indexed_sums, key=lambda x: x[1], reverse=True)
    ranked_indices = [index for index, _ in ranked]
    ranked_docs = [ratings["docids"][i] for i in ranked_indices]

    # create dict of docid -> score
    docid_to_score = dict(zip(ratings["docids"], sums))

    return ranked_docs, docid_to_score

def rank_by_rrf(ratings):
    def reciprocal_rank_fusion(ratings, k=60):
        """
        ratings: numpy array (n_items x n_sources), higher score = better
        k: smoothing constant (default 60, common in IR)
        Returns: fused_scores (n_items,)
        """
        n_items, n_sources = ratings.shape
        
        # Convert scores into ranks (descending, higher = better)
        ranks = np.zeros_like(ratings, dtype=int)
        for s in range(n_sources):
            order = np.argsort(-ratings[:, s])  # sort descending
            ranks[order, s] = np.arange(1, n_items + 1)  # ranks start at 1

        # Apply RRF formula
        fused_scores = np.sum(1.0 / (k + ranks), axis=1)
        return fused_scores
    
    ratings_np = np.array(ratings["ratings"])
    fused = reciprocal_rank_fusion(ratings_np)
    ranked_indices = np.argsort(-fused)

    ranked_docs = [ratings["docids"][i] for i in ranked_indices]

    # create dict of docid -> score
    scores = [float(row) for row in fused]
    docid_to_score = dict(zip(ratings["docids"], scores))

    return ranked_docs, docid_to_score

def rank_by_maximizing_coverage(ratings):
    
    rows = np.array(ratings["ratings"])
    selected_order, remaining_indices = find_docids_with_gains(rows)

    # When there is no more gain to be added, continue with
    # 'rank_by_sum_of_crux_scores' strategy
    while remaining_indices:
        best_row = None
        best_sum_scores = -1

        for i in remaining_indices:
            sum_scores = np.sum(rows[i])
            if sum_scores > best_sum_scores:
                best_sum_scores = sum_scores
                best_row = i

        selected_order.append(best_row)
        remaining_indices.remove(best_row)

    ranked_docs = [ratings["docids"][i] for i in selected_order]

    # create dict of docid -> score
    # assign a score according to position in 'ordered_rows'
    docid_to_score = {docid: len(ranked_docs) - i for i, docid in enumerate(ranked_docs)}

    return ranked_docs, docid_to_score

# def rank_by_late_coverage_maximization(ratings, k, k_coverage):
#
#     rows = np.array(ratings["ratings"])
#     selected_order, _ = find_docids_with_gains(rows)
#     
#     docids_with_gains = [ratings["docids"][i] for i in selected_order]
#
#     ranked_docs_by_sum, docid_to_score = rank_by_sum_of_crux_scores(ratings)
#     # ranked_docs_by_sum = ranked_docs_by_sum[:k] # get top k docs ranked by sum
#
#     # Add docs that add gain, but are not in the top-k documents according to crux sums
#     replacement_candidates = [docid for docid in docids_with_gains if docid not in ranked_docs_by_sum[:k]]
#     num_needed = k_coverage - len(replacement_candidates)
#     ranked_docs = ranked_docs_by_sum[:num_needed] + replacement_candidates + ranked_docs_by_sum[num_needed:]
#     # ranked_docs = ranked_docs[:k]
#
#     # create dict of docid -> score
#     final_docid_to_score = {k: v for k, v in docid_to_score.items() if k in ranked_docs}
#     ranked_docid_to_score = sorted(final_docid_to_score.items(), key=lambda item: item[1], reverse=True)
#     score_at_k_coverage = ranked_docid_to_score[k_coverage-len(replacement_candidates)][1]
#     for docid in final_docid_to_score:
#         if docid in replacement_candidates:
#             final_docid_to_score[docid] = score_at_k_coverage
#
#     return ranked_docs, final_docid_to_score

def rank_by_max_score(ratings):
    maximums = [max(row) for row in ratings["ratings"]]
    indexed_maximums = list(enumerate(maximums))
    ranked = sorted(indexed_maximums, key=lambda x: x[1], reverse=True)
    ranked_indices = [index for index, _ in ranked]
    ranked_docs = [ratings["docids"][i] for i in ranked_indices]

    # create dict of docid -> score
    docid_to_score = dict(zip(ratings["docids"], maximums))

    return ranked_docs, docid_to_score

# NOTE: maybe sort them first is more efficient
def find_docids_with_gains(rows, tau=0):

    # postprocess them
    sums = []
    for row in rows:
        row = [r * int(r >= tau) for r in row]
        sums.append(sum(row))
    remaining_indices = np.argsort(sums).tolist()[::-1]

    # Compute max for each column
    col_max = np.max(rows, axis=0)

    # Track remaining rows and uncovered columns
    selected_order = []
    covered = np.zeros_like(col_max, dtype=bool)
    # remaining_indices = set(range(len(rows))) # this is unordered

    while remaining_indices:
        # Find the row that covers the most new max values
        best_row = None
        best_gain = -1

        for idx, i in enumerate(remaining_indices):
            gain = np.logical_and(rows[i] == col_max, ~covered).sum()
            if gain > best_gain:
                best_gain = gain
                best_row = i
                best_row_idx = idx

        selected_order.append(best_row)
        covered = np.logical_or(covered, rows[best_row] == col_max)
        remaining_indices.pop(best_row_idx)
        # remaining_indices.remove(best_row)

        if np.all(covered):
            break # no more gain can be added

    return selected_order, remaining_indices

# # weights=[0.65, 0.2, 0.15]
# def rrf_fusion(ranked_a, ranked_b, ranked_c=None, k=60, weights=[0, 1]):
#     sorted_ranked_a = sorted(ranked_a.items(), key=lambda item: item[1], reverse=True)
#     sorted_ranked_b = sorted(ranked_b.items(), key=lambda item: item[1], reverse=True)
#     if ranked_c:
#         sorted_ranked_c = sorted(ranked_c.items(), key=lambda item: item[1], reverse=True)
#     scores = defaultdict(float)
#     for rank, (doc, _) in enumerate(sorted_ranked_a):
#         scores[doc] += 2*(weights[0]) / (k + rank + 1)
#     for rank, (doc, _) in enumerate(sorted_ranked_b):
#         scores[doc] += 2*(weights[1]) / (k + rank + 1)
#     if ranked_c:
#         for rank, (doc, _) in enumerate(sorted_ranked_c):
#             scores[doc] += 2*(weights[2]) / (k + rank + 1)
#     rrf_dict = sorted(scores.items(), key=lambda x: x[1], reverse=True)
#     sorted_rrf = [docid for docid, _ in rrf_dict]
#     return sorted_rrf
#
# def modulate_weights(z_score):
#
#     # clip z_score to [-2, 2]
#     z_clipped = max(min(z_score, 2), -2)
#     z_clipped = -z_clipped
#
#     # map z_score in [-2, 2] to a number in [0.5, 1]
#     # mapped = 0.5 + (z_clipped + 2) / 4 * 0.5  # scale from [-2, 2] → [0.5, 1]
#     mapped = 0.7 + (z_clipped) / 2 * 0.1  # scale from [-2, 2] → [0.6, 0.8]
#
#     modulated_weights = [mapped, 1-mapped]
#
#     return modulated_weights
