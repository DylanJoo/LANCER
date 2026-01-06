import sys
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
from scipy import stats


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
    dataset_name='neuclir',
    input_run_1=sys.argv[1],
    input_run_2=sys.argv[2] if len(sys.argv) > 2 else None,
)

# Load data
rac_data = {}
queries = load_query(args)[1]
human_judgements = load_human_judgements(args.qrels, True)
crux_judgements = load_judgements(args, rac_data, compute_missing_judgements=False)
# corpus = load_corpus(args, args.corpus_dir, path_prefix="*.mt.jsonl")
qrels = load_qrels(args.qrels)
crux_qrels = load_qrels(args.crux_qrels)

############
from crux.tools import load_run_or_qrel

#### Retrieval with multiquery

# load run 1
run1 = load_run_or_qrel(args.input_run_1,  topk=100)
rac_data = {}
for qid in queries:
    final_ranking = sorted(run1[qid].items(), key=lambda x: x[1], reverse=True)
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
output_eval["run1"], output1 = rac_evaluate(
    args, qrels,
    human_judgements,
    rac_data=rac_data,
    diversity_qrels_path=args.qrels,
    tokenizer_name='bert-base-uncased',
    gamma=0.5, tag='experiment',
    return_per_query=True
)

# load run2
run2 = load_run_or_qrel(args.input_run_2, topk=100)
rac_data = {}
for qid in queries:
    final_ranking = sorted(run2[qid].items(), key=lambda x: x[1], reverse=True)
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

output_eval["run2"], output2 = rac_evaluate(
    args, qrels,
    human_judgements,
    rac_data=rac_data,
    diversity_qrels_path=args.qrels,
    tokenizer_name='bert-base-uncased',
    gamma=0.5, tag='experiment',
    return_per_query=True
)

setting = f"## {args.input_run_1} | -> none " 
print(f'## Retrieval | -> Rerank | metrics   | ' + ' |'.join(['A_nDCG', 'Cov', 'P', 'MAP', 'nDCG']))
all_metric_results = f"{setting} | Human     | "
for metric in ['alpha_nDCG', 'mean_coverage_at_10', 'P', 'MAP', 'nDCG']:
    num = output_eval['run1'][metric]
    all_metric_results += f"{num:.4f} |"
print(all_metric_results)

setting = f"## {args.input_run_2} | -> none " 
print(f'## Retrieval | -> Rerank | metrics   | ' + ' |'.join(['A_nDCG', 'Cov', 'P', 'MAP', 'nDCG']))
all_metric_results = f"{setting} | Human     | "

for metric in ['alpha_nDCG', 'mean_coverage_at_10', 'P', 'MAP', 'nDCG']:
    num = output_eval['run2'][metric]
    all_metric_results += f"{num:.4f} |"
print(all_metric_results)

print(f"### {args.input_run_1} > {args.input_run_2} | p-value   | ")
# for metric in ['alpha_nDCG', 'coverage', 'P', 'nDCG']:
for metric in ['nDCG', 'P', 'alpha_nDCG', 'coverage']:
    list1 = output1[metric]
    list2 = output2[metric]
    t_stat, p_value_two_tailed = stats.ttest_rel(list1, list2)
    if t_stat > 0:
        if p_value_two_tailed/2 < 0.1:
            print(f"### {metric}: one-tailed p-value = {p_value_two_tailed/2}")
    else:
        if (1-p_value_two_tailed/2) < 0.1:
            print(f"### {metric}:  one-tailed p-value = {1-p_value_two_tailed/2}")

