import pdb
import json
import requests
from collections import defaultdict
from tools.neuclir.load_corpus import load_corpus
from tools.neuclir.load_judgements import load_human_judgements, load_judgements
from tools.neuclir.ir_utils import (
    load_diversity_qrels,
    load_qrels, 
    load_query, 
)
from types import SimpleNamespace
from evaluation.prejudge.retrieval_augmentation_context import rac_evaluate

############
retrievers = ["bm25-mds-duc04", "lsr-mds-duc04"]

def search_mds(query, searcher, service_name="bm25-mds-duc04", limit=100, **kwargs):

    if 'bm25' in service_name:
        hit = searcher.search(query, k=limit)
        result = {h.docid: h.score for h in hit}

    if 'lsr' in service_name:
        hit = searcher.search(query, k=limit)
        result = {h.docid: h.score for h in hit}

    if 'contriver' in service_name:
        hit = searcher.search(query, k=limit)
        result = {h.docid: h.score for h in hit}

    return result

############
args = SimpleNamespace(
    dataset_name='mds-duc04',
    topics_path='/exp/scale25/artifacts/crux/temp/neuclir_format/topic.mds_duc04.jsonl',
    search_with_subqueries=False,
    input=None,
    corpus_dir='/exp/scale25/artifacts/crux/temp/passages/',
    nuggets_dir='/exp/scale25/',
    qrels=None,
    crux_qrels='/exp/scale25/artifacts/crux/temp/neuclir_format/qrels.mds_duc04.txt',
    crux_artifacts_path='/exp/scale25/artifacts/crux/temp/neuclir_format/ratings.mds_duc04.jsonl',
    n_subquestions=2,
    temperature=0.0,
    top_p=1.0,
    overwrite=False
    # host http://${service_endpoint},
    # llm_base_url http://localhost:8000/v1,
)

# Load data
rac_data = {}
queries = load_query(args)[1]
crux_judgements = load_judgements(args, rac_data, compute_missing_judgements=False)
corpus = load_corpus(args, args.corpus_dir, path_prefix="*.jsonl")
crux_qrels = load_qrels(args.crux_qrels)
writer = None

for retriever in retrievers:

    if ('bm25' in retriever) and args.overwrite:
        from pyserini.search.lucene import LuceneSearcher
        searcher = LuceneSearcher('/exp/scale25/artifacts/crux/temp/bm25.crux.passages.lucene')
        searcher.set_bm25(k1=0.9, b=0.4)
        writer = open(f"{retriever}.run", 'w')

    if ('lsr' in retriever) and args.overwrite:
        from pyserini.search.lucene import LuceneImpactSearcher
        searcher = LuceneImpactSearcher(
            index_dir='/exp/scale25/artifacts/crux/temp/splade-v3.crux.passages.lucene',
            query_encoder="naver/splade-v3",
            min_idf=0,
        )
        searcher.query_encoder.device = 'cuda' 
        searcher.query_encoder.model.to('cuda')
        writer = open(f"{retriever}.run", 'w')

    if ('qwen3' in retriever) and args.overwrite:
        writer = open(f"{retriever}.run", 'w')
        pass

    rac_data = {}
    for qid in queries:

        # retrieval
        if args.overwrite:
            result = search_mds(queries[qid], searcher=searcher, service_name=retriever, limit=100)
            for rank, (docid, score) in enumerate(result.items()):
                writer.write(f"{qid} Q0 {docid} {rank+1} {score} {retriever}\n")
        else:
            result = {}
            with open(f"{retriever}.run", 'r') as f:
                for line in f:
                    qid_, _, docid, rank, score, _ = line.strip().split()
                    if qid_ == qid:
                        result[docid] = float(score)

        run = dict(sorted(result.items(), key=lambda item: item[1], reverse=True))  # sort by score

        #### RERANKING
        # Baseline reranking
        # method_name = 'rankgpt'
        # model_name = 'Qwen/Qwen2.5-7B-Instruct'
        # prepare reranker
        # from reranking.wrapper import AutoLLMReranker
        # reranker = AutoLLMReranker.from_prebuilt(
        #     method_name, model_name, 
        #     llm={'max_model_len': 20480, 'backend': 'vllm'}, 
        #     max_doc_length=768
        # )
        # result = reranker.rerank(run=run, queries=queries, corpus=corpus)

        ## reconnect to rac_data and evaluate
        rac_data[qid] = {  # TODO: might be missing some fields
            "qid": qid,
            "topic": queries[qid],
            "questions": None,  # list_questions,
            "type": None,  # "vanilla", k) if max_k > 0 else ("oracle", k),
            "docids": [docid for docid in run],
            "context_list": [], # raw_content,
            "prompt": None,  # template_fn_mapping[template_type](documents),
            "report": None,
            "response": None,
        }

    # Evalaute
    output_eval = {}
    output_eval["crux_0620"] = rac_evaluate(
        args, crux_qrels,
        crux_judgements,
        rac_data=rac_data,
        diversity_qrels_path=args.crux_qrels,
        tokenizer_name='bert-base-uncased',
        gamma=0.5, tag='experiment',
        dataset_name=args.dataset_name,
    )

    setting = f"## {retriever} | -> none " 
    print(f'## Retrieval | -> Rerank | metrics   | ' + ' |'.join(['A_nDCG', 'Cov', 'P', 'MAP', 'nDCG']))

    all_metric_results = f"{setting} | crux_0620 | "
    for metric in ['alpha_nDCG', 'mean_coverage_at_10', 'P', 'MAP', 'nDCG']:
        num = output_eval['crux_0620'][metric]
        all_metric_results += f"{num:.4f} |"
    print(all_metric_results)

    if writer:
        writer.close()

