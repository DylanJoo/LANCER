import argparse
from tools.neuclir.load_rac_data import load_rac_data
from tools.neuclir.ir_utils import (
    load_diversity_qrels,
    load_qrels, 
    load_query
)
from tools.neuclir.load_judgements import load_human_judgements, load_judgements

        
def main(args):

    ## Load data
    queries = load_query(args.query)
    qrels = load_qrels(args.qrels)
    judgmenets = load_judgements(args.judge)
    ## the document qrels are not available on GRID. download from TREC website:
    # qrels = load_qrels(args.qrels)
    # crux_qrels = load_qrels(args.crux_qrels)
    # crux_andor_qrels = load_qrels(args.crux_andor_qrels if args.crux_andor_qrels is not None else args.crux_qrels)

    # Retrieval. NOTE: we load the run directly
    runs = load_runs(args.output_run)

    # Reranking
    crux_judgements_andor = None

    # We can either 
    # (1) covert the human labels to judgements, or 
    # (2) run CRUX pipeline to generate ratings.
    crux_judgements_andor = load_judgements(args, rac_data, load_andor=True)
    for topicid in crux_judgements.keys():
        crux_judgements_topic = crux_judgements[topicid]
        crux_judgements_andor_topic = crux_judgements_andor[topicid]
        for docid in crux_judgements_topic.keys():
            crux_judgement = crux_judgements_topic[docid]
            crux_judgement_andor = crux_judgements_andor_topic[docid]
            if crux_judgement is None or crux_judgement_andor is None:
                continue
            crux_judgement_andor_min = [min(a, b) for a, b in zip(crux_judgement, crux_judgement_andor)]
            crux_judgements_andor[topicid][docid] = crux_judgement_andor_min


    # TODO: Replace the evaluation pipeline
    # output_eval = {}
    # output_eval["human"] = rac_evaluate(
    #     args, qrels,
    #     human_judgements,
    #     rac_data=rac_data,
    #     diversity_qrels_path=args.qrels,
    #     tokenizer_name='bert-base-uncased',
    #     gamma=0.5, tag='experiment'
    # )
    # output_eval["crux_0620"] = rac_evaluate(
    #     args, crux_qrels,
    #     crux_judgements,
    #     rac_data=rac_data,
    #     diversity_qrels_path=args.crux_qrels,
    #     tokenizer_name='bert-base-uncased',
    #     gamma=0.5, tag='experiment',
    # )

    # TODO: Replace the evluation pipeline
    # if crux_judgements_andor is not None:
    #     output_eval["crux_andor_0620"] = rac_evaluate(
    #         args, crux_andor_qrels,
    #         crux_judgements_andor,
    #         rac_data=rac_data,
    #         diversity_qrels_path=args.crux_andor_qrels if args.crux_andor_qrels is not None else args.crux_qrels,
    #         tokenizer_name='bert-base-uncased',
    #         gamma=0.5, tag='experiment'
    #     )
    #
    # if (args.reranker == 'crux_reranking'):
    #     args.reranker = f"crux_reranking:{args.agg}:{args.n_subquestions}"
    #     if 'human' in args.tag:
    #         args.reranker = f"crux_reranking:{args.agg}:human"
    #
    # setting = f"## {args.service_name} | -> {args.reranker}" 
    # print(f'## Retrieval | -> Rerank | metrics   | ' + ' |'.join(['A_nDCG', 'Cov', 'P', 'MAP', 'nDCG']))
    #
    # if args.dataset_name == 'neuclir':
    #     all_metric_results = f"{setting} | Human     | "
    #     for metric in ['alpha_nDCG', 'mean_coverage_at_10', 'P', 'MAP', 'nDCG']:
    #         num = output_eval['human'][metric]
    #         all_metric_results += f"{num:.4f} |"
    #     print(all_metric_results)
    #
    # all_metric_results = f"{setting} | crux_0620 | "
    # for metric in ['alpha_nDCG', 'mean_coverage_at_10', 'P', 'MAP', 'nDCG']:
    #     num = output_eval['crux_0620'][metric]
    #     all_metric_results += f"{num:.4f} |"
    # print(all_metric_results)
    #
    # # print(f'{setting} | metrics   | ' + ' | '.join([f"Cov@{k}" for k in list(range(1, 11))]) )
    # human_coverage = f"{setting} | Cov@k (Human)     | "
    # crux_coverage  = f"{setting} | Cov@k (crux_0620) | "
    # for _k in range(1, 11):
    #     num = output_eval['human'][f'mean_cov@{_k}']
    #     human_coverage += f"{num:.4f} |"
    #     num = output_eval['crux_0620'][f'mean_cov@{_k}']
    #     crux_coverage += f"{num:.4f} |"
    #
    # if args.dataset_name == 'neuclir':
    #     print(human_coverage)
    # print(crux_coverage)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agg", type=str, default="sum")
    parser.add_argument("--dataset_name", type=str, default="neuclir")
    parser.add_argument("--n_subquestions", type=int, default=2)
    parser.add_argument("--include_original", action="store_true", default=False)
    parser.add_argument("--concat_original", action="store_true", default=False)
    parser.add_argument("--k_candidates", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true", default=False, help="When set to True, existing files will be overwritten")
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--llm_base_url", type=str, default="http://rack7n05:8000/v1", help="LLM base URL")
    parser.add_argument("--crux_dir", type=str, default="", help="Path where CRUX files are saved on disk")
    parser.add_argument("--crux_artifacts_path", type=str, default="/exp/scale25/artifacts/crux/crux-neuclir/crux_llama3.3-70b-instruct.jsonl", help="Path to CRUX artifacts jsonl file")
    parser.add_argument("--topics_path", type=str, default="/exp/scale25/neuclir/topics/neuclir24-test-request.jsonl", help="Path to topics file")
    parser.add_argument("--qrels", type=str, default="/exp/scale25/neuclir/eval/qrel/neuclir24-test-request.qrel", help="Path to qrel file")
    parser.add_argument("--crux_qrels", type=str, default="/exp/scale25/artifacts/crux/crux-neuclir/legacy/crux_llama3.3-70b-instruct.qrel", help="Path to crux qrel file")
    parser.add_argument("--crux_andor_qrels", type=str, default=None, help="Path to crux AND/OR qrel file")
    parser.add_argument("--nuggets_dir", type=str, default="/exp/scale25/neuclir/eval/nuggets", help="Path to qrel file")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.3-70B-Instruct", help="Model to use")
    parser.add_argument("--tag", type=str, default="auto-nuggetizer", help="Tag of run (for saving)") # 'human' if evaluating againt ref nuggets 
    parser.add_argument("--gptr_run_file", type=str, default=None, help="Specify a run file")
    parser.add_argument("--host", default='http://10.162.95.158', type=str)
    parser.add_argument("--port", default='5000', type=str)
    parser.add_argument("--service_name", type=str, help="Retrieval service to use")
    parser.add_argument("--top_k", default=20, type=int)
    parser.add_argument("--search_with_subqueries", default=False, action="store_true", help="When set to True, search will be performed using quer")
    parser.add_argument("--config", type=str, default="configs/scale.llama3-70b.yaml", help="Path to the config file")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shard_size", type=int, default=0)
    parser.add_argument("--input", type=str, default=None, help="Path or directory of generated results")
    parser.add_argument("--split", type=str, default='test', help="Original split of datasets")
    parser.add_argument("--corpus_dir", type=str, default="/exp/scale25/neuclir/docs", help="Path to corpus")
    parser.add_argument("--claims_dir", type=str, default="/exp/scale25/artifacts/decontextualized_corpus/neuclir", help="Path to corpus")
    parser.add_argument("--output_dir", type=str, help="Path to generated results")
    parser.add_argument("--load_mode", type=str, default='api', help="['vllm', 'api', 'litellm']")
    parser.add_argument("--temperature", type=float, default=0.5, help="Temperature for decoding")
    parser.add_argument("--top_p", type=float, default=1.0, help="Nucleus sampling top-p")
    parser.add_argument("--max_new_tokens", type=int, default=3, help="Max number of new tokens to generate in one step")
    parser.add_argument("--generate_additional_subtopics", default=False, action="store_true", help="Will generate subtopics in addition to the reference ones")
    parser.add_argument("--check_overlap", action="store_true", help="When set to True, scores will only be computed for qid's found in qrel file")
    parser.add_argument("--reranker", type=str, default="crux_reranking", help="Reranking method")
    parser.add_argument("--rerank_crux_subquestions", action="store_true", help="When set to True, CRUX reranking will be applied")
    args, _ = parser.parse_known_args()

    # TODO: check what is this?
    # if args.input is None:
    #     args.input = args.crux_dir
    # if args.output_dir is None and args.tag in ["human"]:
    #     args.output_dir = args.crux_dir
    
    rerank(args)
#     --concat_original \
#     --n_subquestions 2 \
#     --agg sum \
#     --host http://${service_endpoint} \
#     --llm_base_url http://localhost:8000/v1 \
#     --temperature 0.0 --top_p 1.0 \
#     --model $MODEL \
#     --reranker crux_reranking \
#     --service_name ${service_name} \
#     --dataset_name neuclir
