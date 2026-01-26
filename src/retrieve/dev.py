

async def retrieve_with_report_request(
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

    elif dataset_name == 'bm25':
        from pyserini.search.lucene import LuceneSearcher
        index_path='/home/hltcoe/jhueiju/temp/neuclir/index/title+text.mlir.mt.lucene'
        searcher = LuceneSearcher(index_path)
        searcher.set_bm25(k1=1.2, b=0.75)

        hits = searcher.search(q=query, k=top_k_passages)
        docs = {hit.docid: hit.score for hit in hits}
        ordered_docs_by_score = sorted(docs, key=docs.get, reverse=True)
    else:
        docs = await async_search_neuclir(
            args, query, dataset_name, limit=top_k_passages, topic=topic
        )
        docs = docs["result"] # top100 documents
        ordered_docs_by_score = sorted(docs, key=docs.get, reverse=True)

    return {"result": ordered_docs_by_score[:top_k_docs]}
