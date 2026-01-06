import asyncio
from argparse import Namespace
import json

from tools.neuclir.create_run_file import create_run_file
from tools.search.search import async_get_content, get_content, retrieve_with_report_request, retrieve_with_subqueries
from tools.neuclir.load_corpus import load_corpus
import pdb


# TODO: move the topk-truncation at the later stage, making it more flexible to different truncation
def load_rac_data(args: Namespace, queries: dict, queries_for_search: dict, raw_topics: dict,
                  k: int = None, collection: str = "neuclir", 
                  retrieval_service_name: str = "plaidx-neuclir") -> dict:
    """Load RAC data
    Args:
        args [Namespace]: args provided as input
        queries [dict]: full queries (query + background)
        queries_for_search [dict]: queries to be used when querying search service
        k [int]: top-k documents to keep
        collection [str]: collection to be used when retrieving document
            content from search service
        retrieval_service_name [str]: retrieval service name
    """
    if not k:
        k = args.top_k

    ## async-retrieval or aysnc-retrievak + crux-reranking
    rac_data = asyncio.run(_async_load_rac_data(args, queries, queries_for_search, raw_topics,
                                                k, collection, retrieval_service_name))

    ## NOTE: the workaround is to separate them -- autoreranking
    if 'autorerank' in args.reranker:
        rac_data = _autorerank(args, rac_data, queries, queries_for_search, raw_topics, k)

    ## output the file
    if args.reranker != 'none':
        reranker_name = args.reranker.replace('/', ':')
        reranker_name += ":oracle" if 'human' in args.tag else ""
        reranker_name += f":agg_{args.agg}" if args.agg != 'sum' else ""
        reranker_name += f":nq_{args.n_subquestions}" if args.n_subquestions != 2 else ""

        file_name = f"runs/{args.service_name}+{reranker_name}.run"
        with open(file_name, 'w') as f:
            for qid in rac_data:
                for rank, docid in enumerate(rac_data[qid]['docids'], start=1):
                    f.write(f"{qid} Q0 {docid} {rank} {1/rank} {args.reranker}\n")

    return rac_data

# NOTE: make this workaround for autoreranker, which only uses the async at the reranking stage.
def _autorerank(args: Namespace, rac_data: dict, 
                queries: dict, queries_for_search: dict, raw_topics: dict,
                k: int = 100) -> dict:

    ## prepare reranker
    from reranking.wrapper import AutoLLMReranker
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

async def _async_load_rac_data(args: Namespace, queries: dict, queries_for_search: dict, raw_topics: dict,
                               k: int = None, collection: str = "neuclir", 
                               retrieval_service_name: str = "plaidx-neuclir") -> dict:
    rac_data = {}
    qids, qs = zip(*queries.items())

    # NOTE: we wont use gptr here so i just commented them
    # if args.gptr_run_file:
    #     # get documents from run file
    #     retrieved = get_retrieved_docs_from_gptr_run_file(args, qids)
    # else:
    if args.search_with_subqueries:
        retrieved = await asyncio.gather(*[retrieve_with_subqueries(
            args, queries_for_search[qid], retrieval_service_name, 
            top_k_docs=k, top_k_passages=args.k_candidates) for qid in qids])
    else: # NOTE: default
        retrieved = await asyncio.gather(*[retrieve_with_report_request(
            args, queries_for_search[qid], retrieval_service_name, 
            [t for t in raw_topics if t["request_id"]==qid][0], # get topic
            top_k_docs=k, top_k_passages=args.k_candidates) for qid in qids])

    if args.reranker == 'none':
        file_name = f"runs/{retrieval_service_name}.run"
        with open(file_name, 'w') as f:
            for qid, q, r in zip(qids, qs, retrieved):
                for rank, docid in enumerate(r['result'], start=1):
                    f.write(f"{qid} Q0 {docid} {rank} {1/rank} {retrieval_service_name}\n")

    if 'autorerank' in args.reranker:
        k = 100 # NOTE: return 100 as the reranking would be done in the next stage (outside this async)

    # NOTE: I also think in the for loop. we should avoid the same variable names: 
    for qid, q, retrieved in zip(qids, qs, retrieved):
        rac_data[qid] = {  # TODO: might be missing some fields
            "qid": qid,
            "topic": q,
            "questions": None,  # list_questions,
            "type": None,  # "vanilla", k) if max_k > 0 else ("oracle", k),
            "docids": list(retrieved["result"])[:k],  # retrieved top k
            "context_list": [], # raw_content,
            "prompt": None,  # template_fn_mapping[template_type](documents),
            "report": None,
            "response": None,
        }

    if args.crux_dir:
        create_run_file(args, queries, rac_data)

    return rac_data
