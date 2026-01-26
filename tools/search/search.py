from argparse import Namespace
import aiohttp
import asyncio
from collections import defaultdict
import requests
from types import SimpleNamespace
from urllib.parse import urlparse

from augmentation.gen_ratings import gen_ratings
from prompts.crux import prompt_query_generator
from tools.search.mcranker import rerank_with_mcranker
from tools.search.reranking import rerank_with_crux

SCALE_ENDPOINT = "http://10.162.95.158:5000"

# async def async_get_content(collection, doc_id, **kwargs):
#     async with aiohttp.ClientSession() as session:
#         async with session.post(SCALE_ENDPOINT + "/content", json={"collection": collection, "id": doc_id}) as response:
#             doc = await response.json()
#
#     text = doc["text"].replace("\n", "  ")
#     return f"{doc.get('title', '')} {text}".strip()
#
#
# def get_content(collection, doc_id, **kwargs):
#     doc = requests.post(url=SCALE_ENDPOINT + "/content", json={"collection": collection, "id": doc_id}).json()
#     text = doc["text"].replace("\n", "  ")
#     return f"{doc.get('title', '')} {text}".strip()

def search_neuclir(args, query, service_name="plaidx-neuclir", limit=5, **kwargs):
    data = {'service': service_name, 'query': str(query), 'limit': limit, **kwargs}
    return requests.post(args.host+f":{args.port}/query", json=data).json()

async def async_search_neuclir(args, query, service_name="plaidx-neuclir", limit=5, topic=None, **kwargs):
    data = {"service": service_name, "query": str(query), "limit": limit, **kwargs}
    if service_name=="crux-neuclir":
        data.update({"topic": {
            "background": topic["background"],
            "problem_statement": topic["problem_statement"],
            "title": topic["title"],
        }})
    endpoint = args.host if has_port(args.host) else f"{args.host}:{args.port}"
    if "://" not in endpoint:
        endpoint = "http://" + endpoint

    if not kwargs.get("session"):
        nb_retries = 5
        timeout = aiohttp.ClientTimeout(total=120, connect=None, sock_connect=None, sock_read=None)
        while(nb_retries > 0):
            try:
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(endpoint + "/query", json=data) as response:
                        result = await response.json()
                        return result
            except asyncio.TimeoutError:
                print("Request timed out after 120 seconds. Retrying...")
                nb_retries -= 1
    else:
        session = kwargs.get("session")
        data.pop("session")
        async with session.post(endpoint + "/query", json=data) as response:
            result = await response.json()
            return result


async def retrieve_with_subqueries(args: Namespace, query: str, service_name: str, 
                             top_k_docs: int = 20, 
                             k_subqueries: int = 10, 
                             k_search: int = 10, 
                             return_doc_scores: bool = False,
                             **kwargs) -> dict:
    """Retrieve documents using subqueries
        See Eugene's notebook for more details: https://gitlab.hltcoe.jhu.edu/scale25/eugene-bulleted-list-notebook/-/blob/main/Eugene_bulleted_list.ipynb?ref_type=heads
    Args:
        args [Namespace]: args provided to script
        query [str]: query to be used as input of the query generator
        service_name [str]: name of search service
        top_k_docs [int]: top-k documents to keep from retrieval
        k_subqueries [int]: number of subqueries to use for search
        k_search [int]: number of passages to request per subquery search
        return_doc_scores [bool]: return dictionary of scores per document instead of documents list
    """

    # Generate sub-queries with LLM
    from llm.litellm_api import LLM
    llm = LLM(
        model=args.model,
        temperature=0.0,
        top_p=1.0
    )

    prompt = prompt_query_generator(Q=query)
    output = await llm.async_inference(prompt)
    queries = [o for o in output[0].split("\n") if o.strip()][:k_subqueries]

    # Search results for each sub-query
    ret = await asyncio.gather(*[async_search_neuclir(args, query, service_name, limit=k_search, **kwargs) 
                                 for query in queries])

    # Rank documents by score (note: the same document might be returned my multible sub-queries, with different scores)
    doc_scores = defaultdict(float)
    for entry in ret:
        for docid, score in entry["result"].items():
            doc_scores[docid] = max(doc_scores[docid], score)
    if return_doc_scores:
        return doc_scores

    ranked_docids = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    unique_docs = [docid for docid, _ in ranked_docids]

    return {"result": unique_docs[:top_k_docs]}


async def retrieve_with_report_request(args, query: str, service_name, topic,
                                       top_k_docs: int = 10, 
                                       top_k_passages: int = 100,
                                       **kwargs) -> dict:
    """Retrieve documents using a single search query (consisting of the report request)
    Args:
        query [str]: query to be used as input of the query generator
        service_name [str]: service name that will be used when sending queries to search service
        top_k_docs [int]: top-k documents to keep from retrieval
        top_k_passages [int]: top-k passages to retrieve search service
    """
    if args.dataset_name != 'neuclir':
        docs = {}
        with open(f"{service_name}.run", 'r') as f:
            for line in f:
                qid, _, docid, rank, score, _ = line.strip().split()
                if qid == topic['request_id']:
                    docs[docid] = float(score)
        ordered_docs_by_score = sorted(docs, key=docs.get, reverse=True)

    elif service_name == 'bm25':
        from pyserini.search.lucene import LuceneSearcher
        index_path='/home/hltcoe/jhueiju/temp/neuclir/index/title+text.mlir.mt.lucene'
        searcher = LuceneSearcher(index_path)
        searcher.set_bm25(k1=1.2, b=0.75)

        hits = searcher.search(q=query, k=top_k_passages)
        docs = {hit.docid: hit.score for hit in hits}
        ordered_docs_by_score = sorted(docs, key=docs.get, reverse=True)
    else:
        docs = await async_search_neuclir(args, query, service_name, limit=top_k_passages, topic=topic) 
        docs = docs["result"] # top100 documents
        ordered_docs_by_score = sorted(docs, key=docs.get, reverse=True)

    if args.reranker == "crux_reranking":
        ordered_docs_by_score = await rerank_with_crux(args, ordered_docs_by_score, topic,
                                                       docid_to_score=docs,
                                                       k=top_k_docs, combine_with_rrf=False)
    elif args.reranker == "mcranker":
        ordered_docs_by_score = await rerank_with_mcranker(args, ordered_docs_by_score, topic,
                                                           docid_to_score=docs,
                                                           k=top_k_docs, combine_with_rrf=False)
    else: # neither crux-reranking or mcraker
        top_k_docs = top_k_passages

    return {"result": ordered_docs_by_score[:top_k_docs]}

def has_port(url):
    if "://" not in url:
        url = "http://" + url
    parsed = urlparse(url)
    return parsed.port is not None
