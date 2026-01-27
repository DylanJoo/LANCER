import json
from collections import defaultdict
from wrapper import rerank
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
            if qid not in runs:
                runs[qid] = {}
            runs[qid][docid] = float(score)

    corpus = {}
    train_corpus = load_dataset('DylanJHJ/crux-mds-corpus', split='train')
    test_corpus = load_dataset('DylanJHJ/crux-mds-corpus', split='test')
    corpus.update({example["id"]: {"title": "", "text": example["contents"]} for example in train_corpus})
    corpus.update({example["id"]: {"title": "", "text": example["contents"]} for example in test_corpus})

    ## Reranking
    reranked_run = rerank(
        runs=runs,
        queries=queries,
        topics=None,
        corpus=corpus,
        k=100,
        n_subquestions=args.n_subquestions,
        aggregation=args.agg_method,
        qg_path=None,
        judge_path=None,
        vllm_kwargs={'max_tokens': 512}
    )
        # vllm_kwargs={'temperature': 0.8, 'max_tokens': 512, 'top_p': 1.0}

    ## Save file
    reranker_name = args.reranker
    reranker_name += ":oracle" if args.use_oracle else ""
    reranker_name += f":agg_{args.agg_method}"
    reranker_name += f":nq_{args.n_subquestions}"
    reranked_run_path = args.run_path.replace('data/', 'results/')
    reranked_run_path = reranked_run_path.replace('.run', f':{reranker_name}.run')
    with open(reranked_run_path, 'w') as f:
        for qid in reranked_run:
            for rank, (docid, score) in enumerate(reranked_run[qid].items(), start=1):
                print(rank, docid, score)
                f.write(f"{qid} Q0 {docid} {score} {rank} {reranker_name}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--reranker', type=str, default='lancer', help='lancer or autollreranker')
    parser.add_argument('--run_path', type=str)
    parser.add_argument('--topic_path', type=str)
    parser.add_argument('--use_oracle', action='store_true', help='Whether to use oracle sub-questions')
    parser.add_argument('--n_subquestions', type=int, help='Number of sub-questions to generate')
    parser.add_argument('--agg_method', type=str, help='Aggregation method for reranking')
    args = parser.parse_args()
    main(args)
