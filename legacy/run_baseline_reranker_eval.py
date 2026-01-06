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
retrievers = ["plaidx-neuclir", "qwen3-neuclir", "lsr-neuclir-mt", "lsr-neuclir"]

retriever = retrievers[2]

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
    run_path=f'runs/{retriever}.run'
)

# Load data
rac_data = {}
run = defaultdict(dict)
with open(args.run_path, "r") as f:
    for line in f:
        qid, _, docid, rank, score, _ = line.strip().split()
        if int(rank) <= 100:
            run[qid][docid] = float(score)
queries = load_query(args)[1]
human_judgements = load_human_judgements(args.nuggets_dir)
crux_judgements = load_judgements(args, rac_data, compute_missing_judgements=False)
corpus = load_corpus(args, args.corpus_dir, path_prefix="*.mt.jsonl")
qrels = load_qrels(args.qrels)
crux_qrels = load_qrels(args.crux_qrels)

# Baseline reranking
method_name = 'rankgpt'
model_name = 'Qwen/Qwen2.5-7B-Instruct'

## prepare reranker
from reranking.wrapper import AutoLLMReranker
reranker = AutoLLMReranker.from_prebuilt(
    method_name, model_name, 
    llm={'max_model_len': 20480, 'backend': 'vllm'}, 
    max_doc_length=768
)
reranked_run = reranker.rerank(run=run, queries=queries, corpus=corpus)

## reconnect to rac_data and evaluate
for qid in reranked_run:
    rac_data[qid] = {  # TODO: might be missing some fields
        "qid": qid,
        "topic": queries[qid],
        "questions": None,  # list_questions,
        "type": None,  # "vanilla", k) if max_k > 0 else ("oracle", k),
        "docids": [docid for docid in reranked_run[qid]][:10],
        "context_list": [], # raw_content,
        "prompt": None,  # template_fn_mapping[template_type](documents),
        "report": None,
        "response": None,
    }

with open('dev_run.txt', "w") as f:
    for qid in reranked_run:
        for rank, docid in enumerate(reranked_run[qid], start=1):
            score = (1/rank)
            f.write(f"{qid} Q0 {docid} {score} {rank} rel-reranker\n")

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

setting = f"## {retriever[0]} | -> {method_name}{model_name} " 
print(f'## Retrieval | -> Rerank | metrics   | ' + ' |'.join(['A_nDCG', 'Cov', 'P', 'MAP', 'nDCG']))

all_metric_results = f"{setting} | Human     | "
for metric in ['alpha_nDCG', 'mean_coverage_at_10', 'P', 'MAP', 'nDCG']:
    num = output_eval['human'][metric]
    all_metric_results += f"{num:.4f} |"
print(all_metric_results)

all_metric_results = f"{setting} | crux_0620 | "
for metric in ['alpha_nDCG', 'mean_coverage_at_10', 'P', 'MAP', 'nDCG']:
    num = output_eval['crux_0620'][metric]
    all_metric_results += f"{num:.4f} |"
print(all_metric_results)

human_coverage = f"{setting} | Cov@k (Human)     | "
crux_coverage  = f"{setting} | Cov@k (crux_0620) | "
for _k in range(1, 11):
    num = output_eval['human'][f'mean_cov@{_k}']
    human_coverage += f"{num:.4f} |"
    num = output_eval['crux_0620'][f'mean_cov@{_k}']
    crux_coverage += f"{num:.4f} |"

print(human_coverage)
print(crux_coverage)

