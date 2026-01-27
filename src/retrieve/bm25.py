from pyserini.search.lucene import LuceneSearcher


def bm25_retrieve(
    args, 
    query: str, 
    dataset_name, 
    topic,
    top_k_docs: int = 10, 
    top_k_passages: int = 100,
    **kwargs
) -> dict:

    docs = {}
    with open(f"{dataset_name}.run", 'r') as f:
        for line in f:
            qid, _, docid, rank, score, _ = line.strip().split()
            if qid == topic['request_id']:
                docs[docid] = float(score)
    ordered_docs_by_score = sorted(docs, key=docs.get, reverse=True)

    index_path='/home/hltcoe/jhueiju/temp/neuclir/index/title+text.mlir.mt.lucene'
    searcher = LuceneSearcher(index_path)
    searcher.set_bm25(k1=1.2, b=0.75)

    hits = searcher.search(q=query, k=top_k_passages)
    docs = {hit.docid: hit.score for hit in hits}
    ordered_docs_by_score = sorted(docs, key=docs.get, reverse=True)
