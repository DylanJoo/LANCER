from tools.neuclir.load_corpus import load_corpus
from reranking.wrapper import AutoLLMReranker

def _autorerank(
        args: Namespace, 
        rac_data: dict, 
        queries: dict, 
        queries_for_search: dict, 
        raw_topics: dict,
        k: int = 100
) -> dict:

    ## prepare reranker
    _, method_name, model_name = args.reranker.split(':')
    reranker = AutoLLMReranker.from_prebuilt(
        method_name, model_name, 
        llm={'max_model_len': 20480, 'backend': 'request'}, 
        max_doc_length=800 if (method_name=='listwise' or method_name=='rankgpt') else 1024, 
    )

    ## prepare data  
    ### covert rac_data to run
    qids, qs = zip(*queries.items())
    run = {}
    for qid in qids:
        run[qid] = {}
        for rank, docid in enumerate(rac_data[qid]['docids'], start=1):
            run[qid][docid] = 1/rank

    queries = {qid: queries_for_search[qid] for qid in qids}
    if args.dataset_name == 'neuclir':
        corpus = load_corpus(args, args.corpus_dir, path_prefix="*.mt.jsonl")
    else:
        corpus = load_corpus(args, args.corpus_dir, path_prefix="*.jsonl")

    ## rerank # it will rank the top100 candidate
    reranked_run = reranker.rerank(run=run, queries=queries, corpus=corpus, query_batch_size=64)

    ## reconnect to rac_data
    for qid in reranked_run:
        original = rac_data[qid]['docids'] 
        rac_data[qid]['docids'] = [docid for docid in reranked_run[qid]]

    return rac_data
