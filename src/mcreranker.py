import asyncio
from argparse import Namespace
import pdb
import json
from tools.neuclir.create_run_file import create_run_file
from tools.search.search import retrieve_with_report_request, retrieve_with_subqueries

    # if args.reranker == "crux_reranking":
    #     ordered_docs_by_score = await rerank_with_crux(args, ordered_docs_by_score, topic,
    #                                                    docid_to_score=docs,
    #                                                    k=top_k_docs, combine_with_rrf=False)
    # elif args.reranker == "mcranker":
    #     ordered_docs_by_score = await rerank_with_mcranker(args, ordered_docs_by_score, topic,
    #                                                        docid_to_score=docs,
    #                                                        k=top_k_docs, combine_with_rrf=False)
    # else: # neither crux-reranking or mcraker
    #     top_k_docs = top_k_passages

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
    rac_data = asyncio.run(
            _async_load_rac_data(
                args, 
                queries, queries_for_search, raw_topics,
                k, collection, 
                retrieval_service_name
    ))

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

async def _async_load_rac_data(
    args: Namespace, 
    queries: dict, 
    queries_for_search: dict, 
    raw_topics: dict,
    k: int = None, 
    collection: str = "neuclir", 
    retrieval_service_name: str = "plaidx-neuclir"
) -> dict:

    rac_data = {}
    qids, qs = zip(*queries.items())

    retrieved = await asyncio.gather(*[
        retrieve_with_report_request(
            args, 
            queries_for_search[qid], 
            retrieval_service_name, 
            [t for t in raw_topics if t["request_id"]==qid][0], # get topic
            top_k_docs=k, 
            top_k_passages=args.k_candidates
        ) for qid in qids
    ])

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
