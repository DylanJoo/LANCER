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
retrievers = ["qwen3-neuclir", "lsr-neuclir"]
retrievers = ["bm25-neuclir"]

def search(query, service_name="plaidx-neuclir", limit=100, **kwargs):
    if 'bm25' in service_name:
        searcher.set_bm25(k1=1.2, b=0.75)

        hits = searcher.search(q=query, k=limit)
        docs = {hit.docid: hit.score for hit in hits}
        return docs

    data = {'service': service_name, 'query': str(query), 'limit': limit, **kwargs }
    return requests.post("http://10.162.95.158:5000/query", json=data).json()['result']

############
args = SimpleNamespace(
    topics_path='/exp/scale25/neuclir/topics/neuclir24-test-request.jsonl',
    search_with_subqueries=False,
    input=None,
    corpus_dir='/exp/scale25/neuclir/docs',
    nuggets_dir='/exp/scale25/neuclir/eval/nuggets',
    qrels='/exp/scale25/neuclir/eval/qrel/neuclir24-test-request.qrel',
    crux_qrels='/exp/scale25/artifacts/crux/crux-neuclir/crux_llama3.3-70b-instruct.qrel',
    crux_artifacts_path='/exp/scale25/artifacts/crux/crux-neuclir/crux_llama3.3-70b-instruct.jsonl',
    n_subquestions=2,
    dataset_name='neuclir'
)
from pyserini.search.lucene import LuceneSearcher
index_path='/home/hltcoe/jhueiju/temp/neuclir/index/title+text.mlir.mt.lucene'
searcher = LuceneSearcher(index_path)

# Load data
rac_data = {}
queries = load_query(args)[1]
human_judgements = load_human_judgements(args.qrels, True)
crux_judgements = load_judgements(args, rac_data, compute_missing_judgements=False)
# corpus = load_corpus(args, args.corpus_dir, path_prefix="*.mt.jsonl")
qrels = load_qrels(args.qrels)
crux_qrels = load_qrels(args.crux_qrels)


#### Retrieval with multiquery
for retriever in retrievers:

    rac_data = {}
    for qid in queries:
        with open(f'/home/hltcoe/jhueiju/temp/neuclir/{qid}_subtopics_{args.n_subquestions}.json', 'r') as f:
            # subquestions = json.loads(f.readlines()[0])['subquestions']
            subquestions = json.loads(f.readlines()[0])[qid]

        ranking_lists = []
        # add separate
        # for subq in subquestions + [queries[qid]]:

        # concat
        for subq in subquestions:
            subq = queries[qid] + " " + subq
            print(qid, subq)
            result = search(subq, service_name=retriever, limit=100)
            result = {docid: 1/(60+rank) for rank, docid in enumerate(result, start=1)}
            ranking_lists.append(result)

        rrf_scores = defaultdict(float)
        for rank_list in ranking_lists:
            for docid in rank_list:
                rrf_scores[docid] += rank_list[docid]

        final_ranking = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        rac_data[qid] = {
            "qid": qid,
            "topic": queries[qid],
            "questions": None,  # list_questions,
            "type": None,  # "vanilla", k) if max_k > 0 else ("oracle", k),
            "docids": [docid for docid, _ in final_ranking][:20],
            "context_list": [], # raw_content,
            "prompt": None,  # template_fn_mapping[template_type](documents),
            "report": None,
            "response": None,
        }

    # Evalaute
    output_eval = {}
    output_eval["human"] = rac_evaluate(
        args, qrels,
        human_judgements,
        rac_data=rac_data,
        diversity_qrels_path=args.qrels,
        tokenizer_name='bert-base-uncased',
        gamma=0.5, tag='experiment'
    )
    output_eval["crux_0620"] = rac_evaluate(
        args, crux_qrels,
        crux_judgements,
        rac_data=rac_data,
        diversity_qrels_path=args.crux_qrels,
        tokenizer_name='bert-base-uncased',
        gamma=0.5, tag='experiment'
    )

    setting = f"## {retriever} | -> none " 
    print(f'## Retrieval | -> Rerank | metrics   | ' + ' |'.join(['A_nDCG', 'Cov', 'P', 'MAP', 'nDCG']))

    all_metric_results = f"{setting} | Human     | "
    for metric in ['alpha_nDCG', 'mean_coverage_at_10', 'P', 'MAP', 'nDCG']:
        num = output_eval['human'][metric]
        all_metric_results += f"{num:.4f} |"
    print(all_metric_results)

