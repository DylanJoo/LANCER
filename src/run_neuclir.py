import json
from datasets import load_dataset
import argparse

def main(args):
    ## Load Datasets 
    topics = {}
    with open(args.topic_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            topics[item['request_id']] = item
    queries = {qid: topics[qid]['background'] for qid in topics}

    runs = {}
    with open(args.run_path, 'r') as f:
        for line in f:
            qid, _, docid, rank, score, _ = line.strip().split()
            if int(rank) > args.k:
                continue
            if qid not in runs:
                runs[qid] = {}
            runs[qid][docid] = float(score)

    corpus = {}
    if args.rerun_judge:
        # TODO: Update the path below to your local NeuCLIR1 corpus location.
        # Download from: https://huggingface.co/datasets/neuclir/neuclir1/tree/main/data
        ds = load_dataset('json', data_files='/home/hltcoe/jhueiju/datasets/neuclir1/*.processed_output.jsonl.gz', num_proc=3, split='train')
        corpus = {example["id"]: {"title": example["title"], "text": example["text"]} for example in ds}
        del ds

    ## Reranking
    if args.reranker == 'lancer':
        from lancer import rerank
        reranked_run = rerank(
            runs=runs,
            queries=queries,
            topics=topics,
            corpus=corpus,
            n_subquestions=args.n_subquestions,
            use_oracle=args.use_oracle,
            concat_original=False if args.use_oracle else True,
            aggregation=args.agg_method,
            rerun_qg=args.rerun_qg, qg_path=args.qg_path,
            rerun_judge=args.rerun_judge, judge_path=args.judge_path,
            vllm_kwargs={'max_tokens': 512, 'model_name_or_path': args.model, 'base_url': args.base_url}
        )
        reranker_name = args.reranker
        reranker_name += ":oracle" if args.use_oracle else ""
        reranker_name += f":agg_{args.agg_method}"
        reranker_name += f"nq_{args.n_subquestions}" if args.use_oracle is False else ""

    if 'autorerank' in args.reranker:
        from reranking.wrapper import AutoLLMReranker
        _, method_name, model_name = args.reranker.split(':')
        reranker = AutoLLMReranker.from_prebuilt(
            method_name, model_name, 
            llm={'max_tokens': 3, 'backend': 'request', 'base_url': args.base_url},
            max_doc_length=800 if (method_name=='listwise' or method_name=='rankgpt') else 1024, 
       )
        reranked_run = reranker.rerank(run=runs, queries=queries, corpus=corpus, query_batch_size=64)
        reranker_name = args.reranker.replace('/', '.')

    ## Save file
    reranked_run_path = args.run_path.replace('data/', 'results/')
    reranked_run_path = reranked_run_path.replace('.run', f':{reranker_name}.run')
    with open(reranked_run_path, 'w') as f:
        for qid in reranked_run:
            for rank, (docid, score) in enumerate(reranked_run[qid].items(), start=1):
                f.write(f"{qid} Q0 {docid} {rank} {score} {reranker_name}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--reranker', type=str, default='lancer', help='lancer or autollreranker')
    parser.add_argument('--run_path', type=str)
    parser.add_argument('--topic_path', type=str)
    parser.add_argument("--k", default=100, type=int)

    # parameters for vllm
    parser.add_argument("--base_url", default='http://localhost:8000/v1')
    parser.add_argument("--model", type=str, default='meta-llama/Llama-3.3-70B-Instruct')

    # parameters for lancer
    parser.add_argument('--use_oracle', action='store_true', help='Whether to use oracle sub-questions')
    parser.add_argument('--n_subquestions', type=int, help='Number of sub-questions to generate')
    parser.add_argument('--agg_method', type=str, help='Aggregation method for reranking')

    # parameters for adjuget rerun of the LANCER pipeline
    parser.add_argument('--rerun_qg', action='store_true', default=False)
    parser.add_argument('--rerun_judge', action='store_true', default=False)
    parser.add_argument('--qg_path', type=str, default='temp.qg.json')
    parser.add_argument('--judge_path', type=str, default='temp.judge.json')
    args = parser.parse_args()
    main(args)
