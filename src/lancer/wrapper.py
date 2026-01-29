import json
from lancer.qg import question_generation
from lancer.aj import answerability_judment
from lancer.ca import coverage_based_aggregation
from lancer.llm_request import LLM

def rerank(
    runs: dict, 
    queries: dict, 
    corpus: dict, 
    topics: dict = None,
    n_subquestions: int = 2,
    concat_original: bool = True,
    use_oracle: bool = False,
    aggregation: str = 'sum',
    rerun_qg: bool = True, 
    rerun_judge: bool = True, 
    qg_path: str = None,
    judge_path: str = None,
    vllm_kwargs: dict = {},
):

    # Data preparation
    documents_all = {}
    for qid in queries:
        documents_all[qid] = [corpus[docid] for docid in runs[qid]]

    ## 0. Initialize LLM 
    llm = LLM(api_key='EMPTY', **vllm_kwargs)

    ## 1. sub-question generation
    if rerun_qg:
        all_subquestions = question_generation(
            llm=llm,
            queries=queries,
            topics=topics,
            n_subquestions=n_subquestions,
        )
        json.dump(all_subquestions, open(qg_path, "w"), indent=4)

    else: # reuse the generated results
        with open(qg_path, "r") as f:
            all_subquestions = json.loads(f.read())

    ## 2. Answerability judgment
    if rerun_judge:
        ratings = answerability_judment(
            llm=llm, 
            queries=queries,
            subquestions=all_subquestions,
            documents=documents_all,
            concat_original=concat_original,
        )
        json.dump(ratings, open(judge_path, "w"), indent=4)

    else: # reuse the generated results
        with open(judge_path, "r") as f:
            ratings = json.loads(f.read())

    ## 3. Coverage-based aggregation
    reranked_run = {}
    for qid in queries:
        _, reranked_docs = coverage_based_aggregation(
            docids=[docid for docid in runs[qid]],
            ratings=ratings[qid],
            agg_method=aggregation
        )
        reranked_run[qid] = reranked_docs

    return reranked_run
