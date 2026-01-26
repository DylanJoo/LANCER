# import asyncio
# from argparse import Namespace
# from tools.neuclir.create_run_file import create_run_file
# from tools.search.search import retrieve_with_report_request, retrieve_with_subqueries

from lancer.qg import question_generation
from lancer.aj import answerability_judment
from lancer.ca import coverage_based_aggregation

from crux.tools import load_corpus

def rerank(
    run: dict, 
    queries: dict, 
    corpus: dict, 
    k: int = None,
    aggregation: str = 'sum',
    topics: dict = None,
    rerun_qg: bool = False, 
    rerun_judge: bool = False, 
    qg_path: str = None,
    judge_path: str = None,
):

    # Data preparation
    documents_all = {}
    for qid in queries:
        documents_all[qid] = [corpus[docid] for docid in runs[qid]]

    ## 0. Initialize LLM 
    from llm.litellm_api import LLM
    llm = LLM(args)

    ## TODO: add outputing generate subQ 
    ## 1. sub-question generation
    if rerun_qg:
        all_subquestions = question_generation(
            llm=llm
            queries=queries,
            topics=topics,
            n_subquestions-n_questions,
            use_oracle=use_oracle,
            output_path=qg_path
        )
    else: # reuse the generated results
        with open(qg_path, "r") as f:
            all_subquestions = json.loads(f.read())

    ## TODO: add outputing generated ratings
    ## 2. Answerability judgment
    if rerun_judge:
        ratings = answerability_judment(
            llm=llm, 
            subquestions=subquestions,
            documents=documents,
            queries=queries,
            concat_original=concat_original,
            output_path=judge_path
        )
    else: # reuse the generated results
        with open(judge_path, "r") as f:
            ratings = load_ratings(judge_path)

    ## 3. Coverage-based aggregation
    reranked_docs = coverage_based_aggregation(agg_method=aggregation)

    ## 4. Save trec file
    # reranker_name = args.reranker.replace('/', ':')
    # reranker_name += ":oracle" if 'human' in args.tag else ""
    # reranker_name += f":agg_{args.agg}" if args.agg != 'sum' else ""
    # reranker_name += f":nq_{args.n_subquestions}" if args.n_subquestions != 2 else ""
    # file_name = f"runs/{args.service_name}+{reranker_name}.run"
    # with open(file_name, 'w') as f:
    #     for qid in rac_data:
    #         for rank, docid in enumerate(rac_data[qid]['docids'], start=1):
    #             f.write(f"{qid} Q0 {docid} {rank} {1/rank} {args.reranker}\n")
    return reranked_run
