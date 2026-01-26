import numpy as np

def coverage_based_aggregation(docids, ratings, agg_method):

    if agg_method.startswith('sum'):
        tau = int(agg_method.split("tau=")[-1]) if 'tau' in agg_method else 1
        ranked_docs, docid_to_score_rerank = rank_by_sum_of_crux_scores(docids, ratings, tau)

    if agg_method.startswith('greedy-sum'):
        tau = int(agg_method.split("tau=")[-1]) if 'tau' in agg_method else 1
        ranked_docs, docid_to_score_rerank = greedy_sum(docids, ratings, tau)

    if agg_method.startswith('greedy-coverage'):
        tau = int(agg_method.split("tau=")[-1]) if 'tau' in agg_method else 1
        # ranked_docs, docid_to_score_rerank_c = rank_by_maximizing_coverage(ratings)
        ranked_docs, docid_to_score_rerank = greedy_coverage(docids, ratings, tau)

    if agg_method.startswith('greedy-alpha'): 
        tau = int(agg_method.split("tau=")[-1]) if 'tau' in agg_method else 1
        ranked_docs, docid_to_score_rerank = greedy_alpha(docids, ratings, tau)

    if agg_method.startswith('rrf'): # NOTE: Shall we check the logic again?
        ranked_docs, docid_to_score_rerank = rank_by_rrf(docids, ratings)

    return ranked_docs, docid_to_score_rerank

## The logics of different aggregation methods
def rank_by_sum_of_crux_scores(docids, ratings, tau=0):
    print(ratings)
    if tau == 0:
        sums = [sum(row) for row in ratings]
    else:
        sums = []
        for row in ratings:
            score = sum( [r * int(r >= tau) for i, r in enumerate(row)] )
            sums.append(score)
    indexed_sums = list(enumerate(sums))
    ranked = sorted(indexed_sums, key=lambda x: x[1], reverse=True)
    ranked_indices = [index for index, _ in ranked]
    ranked_docs = [docids[i] for i in ranked_indices]

    # create dict of docid -> score
    docid_to_score = dict(zip(docids, sums))
    return ranked_docs, docid_to_score

def greedy_sum(docids, ratings, tau=0):
    m, n = len(ratings), len(ratings[0])

    # postprocess them
    matrix = []
    sums = []
    for row in ratings:
        row = [r * int(r >= tau) for r in row]
        matrix.append(row)
        sums.append(sum(row))

    ratings = matrix

    remaining_indices = np.argsort(sums).tolist()[::-1]
    reranked_indices = []
    omitted_indices = []
    best_so_far = np.array([0] * n)

    while len(remaining_indices) > 0:

        ## check available docs
        gains = []
        for index in remaining_indices:
            gain = sum([max(b-a, 0) for a, b in zip(best_so_far, ratings[index])])
            gains.append(gain)

        ## find the best one
        i = np.argmax(gains)
        index = remaining_indices.pop(i)
        if gains[i] > 0:
            n = len(remaining_indices)
            best_so_far = np.array([max(a, b) for a, b in zip(best_so_far, ratings[index])])
            reranked_indices.append(index)
        else:
            omitted_indices.append(index)

    # prepare return
    ranked_docs = [docids[i] for i in reranked_indices + omitted_indices]

    # create dict of docid -> score
    docid_to_score = {docid: 1/rank for rank, docid in enumerate(ranked_docs, start=1)}

    return ranked_docs, docid_to_score

def greedy_alpha(docids, ratings, tau=1):
    alpha = 0.5 # the higher the more diverse

    m, n = len(ratings), len(ratings[0])

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
                if ratings[i][j] >= tau: # in ndeval, we only have binary relevance?
                    current_score += subtopic_gain[j]

            # deal with tie? now we select the earlier one.
            if (where == -1) or (current_score > max_score):
                where = i
                max_score = current_score

        reranked_indices.append(where)
        selected[where] = True

        # update the gain
        for j in range(n):
            if ratings[where][j] >= tau:
                subtopic_gain[j] *= (1.0 - alpha)
    
    # prepare return
    ranked_docs = [docids[i] for i in reranked_indices]

    # create dict of docid -> score
    docid_to_score = {docid: 1/rank for rank, docid in enumerate(ranked_docs, start=1)}

    return ranked_docs, docid_to_score

def greedy_coverage(docids, ratings, tau=0):
    m, n = len(ratings), len(ratings[0])

    # postprocess them
    matrix = []
    sums = []
    for row in ratings:
        row = [int(r >= tau) for r in row]
        matrix.append(row)
        sums.append(sum(row))

    ratings = matrix

    remaining_indices = np.argsort(sums).tolist()[::-1]
    reranked_indices = []
    omitted_indices = []
    best_so_far = np.array([0] * n)

    while len(remaining_indices) > 0:

        ## check available docs
        gains = []
        for index in remaining_indices:
            gain = sum([max(b-a, 0) for a, b in zip(best_so_far, ratings[index])])
            gains.append(gain)

        ## find the best one
        i = np.argmax(gains)
        index = remaining_indices.pop(i)
        if gains[i] > 0:
            n = len(remaining_indices)
            best_so_far = np.array([max(a, b) for a, b in zip(best_so_far, ratings[index])])
            reranked_indices.append(index)
        else:
            omitted_indices.append(index)

    # prepare return
    ranked_docs = [docids[i] for i in reranked_indices + omitted_indices]

    # create dict of docid -> score
    docid_to_score = {docid: 1/rank for rank, docid in enumerate(ranked_docs, start=1)}

    return ranked_docs, docid_to_score

def rank_by_sum_of_high_quality_crux_scores(docids, ratings):
    sums = [sum(val for val in row if val >= 2) for row in ratings]
    indexed_sums = list(enumerate(sums))
    ranked = sorted(indexed_sums, key=lambda x: x[1], reverse=True)
    ranked_indices = [index for index, _ in ranked]
    ranked_docs = [docids[i] for i in ranked_indices]

    # create dict of docid -> score
    docid_to_score = dict(zip(docids, sums))

    return ranked_docs, docid_to_score

def rank_by_rrf(docids, ratings):
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
    
    ratings_np = np.array(ratings)
    fused = reciprocal_rank_fusion(ratings_np)
    ranked_indices = np.argsort(-fused)

    ranked_docs = [docids[i] for i in ranked_indices]

    # create dict of docid -> score
    scores = [float(row) for row in fused]
    docid_to_score = dict(zip(docids, scores))

    return ranked_docs, docid_to_score

def rank_by_maximizing_coverage(docids, ratings):
    
    rows = np.array(ratings)
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

    ranked_docs = [docids[i] for i in selected_order]

    # create dict of docid -> score
    # assign a score according to position in 'ordered_rows'
    docid_to_score = {docid: len(ranked_docs) - i for i, docid in enumerate(ranked_docs)}

    return ranked_docs, docid_to_score

def rank_by_max_score(docids, ratings):
    maximums = [max(row) for row in ratings]
    indexed_maximums = list(enumerate(maximums))
    ranked = sorted(indexed_maximums, key=lambda x: x[1], reverse=True)
    ranked_indices = [index for index, _ in ranked]
    ranked_docs = [docids[i] for i in ranked_indices]

    # create dict of docid -> score
    docid_to_score = dict(zip(docids, maximums))

    return ranked_docs, docid_to_score

# NOTE: maybe sort them first is more efficient
def find_docids_with_gains(docids, rows, tau=0):

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

