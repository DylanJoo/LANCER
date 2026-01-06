import re
import json

def deduplicate_and_sort(doc_ids):
    doc_ids = list(set(doc_ids))
    sorted(doc_ids)
    return doc_ids

def check_newinfo(values, new_values):
    mask = (values == 0)
    if (new_values[mask] > values[mask]).any():
        return True, values + new_values
    else:
        return False, values

def replace_tags(sent, tag='q'):
    if tag == 'q':
        sent = re.sub(r"\<q\>|\<\/q\>", "\n", sent)
    if tag == 'p':
        sent = re.sub(r"\<p\>|\<\/p\>", "\n", sent)
    pattern = re.compile(r"\n+")
    sent = re.sub(pattern, '\n', sent)
    return sent

def load_topic_data(path, n=1):
    data = json.load(open(path, 'r'))

    topics = []
    for i, item in enumerate(data['data']):
        example_id = item['example_id']
        outputs = item['output'].strip().split('</r>')[:n]
        outputs = [replace_tags(o, 'r').strip() for o in outputs][0]
        topics.append({"example_id": example_id, "texts": outputs})
    return topics

def get_i_doc(i, psgs_bound):
    for i_doc, bound in enumerate(psgs_bound):
        if i in bound:
            return i_doc

def binary_rerank_greedy(item_, threshold=3):
    item = copy(item_)
    example_id = item['id']
    ratings = np.array(item['ratings'])
    passages = item['passages']
    labels = {"documents": [], "passages": [], "redundant_passages": []}

    # binariz
    scores = np.zeros_like(ratings).astype(int)
    scores[(ratings >= threshold)] = 1
    answerable = (scores.sum(0) != 0)

    ## select the best passages per doc
    end = np.cumsum([len(psgs) for psgs in passages])
    start = np.append([0], end)[:(-1)]
    psgs_in_doc = [range(s, e) for (s, e) in zip(start, end)]

    ## initial values
    values = np.zeros(scores.shape[1])
    while (values[answerable]==0).sum() > 0:

        ## rerank the passages with binary + max
        mask = (values == 0)
        bin_counts = np.where(ratings[:, mask]>=threshold, 1, 0).sum(-1) 
        max_scores = np.where(ratings[:, mask]>0, 1, 0).max(-1)
        psg_scores = bin_counts * 1 + max_scores * 0.1
        i = psg_scores.argmax()

        flag, values = check_newinfo(values, scores[i])
        i_doc = get_i_doc(i, psgs_in_doc)

        if flag:
            if i_doc not in labels["documents"]:
                labels["documents"].append(i_doc)
            labels["passages"].append( (i_doc, i) )

            # strictly duplicated
            for j, r in enumerate(ratings):
                if (ratings[i] > r).all():
                    labels['redundant_passages'].append( (i_doc, j) )

    return labels

def minmax_rerank_greedy(scores):
    pass
def reciprocal_rerank_greedy(scores):
    pass

def mine_distractor(
    query, 
    searcher, 
    k=10,
    max_docs_return=3,
    ignored_prefix=""
):

    hits = searcher.search(query, k)

    hit_doc_ids = []
    ### schema: {example_id}#{n_doc}:{n_claims}
    for h in hits:
        if h.docid.split("#")[0] != ignored_prefix:
            hit_doc_id = h.docid.split(":")[0]
            if hit_doc_id not in hit_doc_ids:
                hit_doc_ids.append(hit_doc_id)
            if len(hit_doc_ids) >= max_docs_return:
                return hit_doc_ids

    return []
