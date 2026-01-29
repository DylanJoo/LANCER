import json
from datasets import load_dataset
import argparse

def main(args):
    ## Load Datasets 
    queries = {}
    with open(args.topic_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            queries[item['id']] = item['request']

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
    train_corpus = load_dataset('DylanJHJ/crux-mds-corpus', split='train')
    test_corpus = load_dataset('DylanJHJ/crux-mds-corpus', split='test')
    corpus.update({example["id"]: {"title": "", "text": example["contents"]} for example in train_corpus})
    corpus.update({example["id"]: {"title": "", "text": example["contents"]} for example in test_corpus})

    ## Reranking
    if args.reranker == 'lancer':
        from lancer import rerank
        reranked_run = rerank(
            runs=runs,
            queries=queries,
            topics=None,
            corpus=corpus,
            n_subquestions=args.n_subquestions,
            concat_original=False if args.use_oracle else True,
            aggregation=args.agg_method,
            rerun_qg=args.rerun_qg, qg_path=args.qg_path,
            rerun_judge=args.rerun_judge, judge_path=args.judge_path,
            vllm_kwargs={'max_tokens': 512, 'model_name_or_path': 'meta-llama/Llama-3.3-70B-Instruct', 'base_url': args.base_url}
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
