from qg import question_generation
from aj import answerability_judment
from ca import coverage_based_aggregation

def rerank(
    runs: dict, 
    queries: dict, 
    corpus: dict, 
    k: int = None,
    n_subquestions: int = 2,
    concat_original: bool = True,
    use_oracle: bool = False,
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
    from llm_empty import LLM
    llm = LLM()

    ## TODO: add outputing generate subQ 
    ## 1. sub-question generation
    if rerun_qg:
        all_subquestions = question_generation(
            llm=llm,
            queries=queries,
            topics=topics,
            n_subquestions=n_subquestions,
            use_oracle=use_oracle,
            # output_path=qg_path
        )
    else: # reuse the generated results
        with open(qg_path, "r") as f:
            all_subquestions = json.loads(f.read())

    ## TODO: add outputing generated ratings
    ## 2. Answerability judgment
    if rerun_judge:
        ratings = answerability_judment(
            llm=llm, 
            queries=queries,
            subquestions=all_subquestions,
            documents=documents_all,
            concat_original=concat_original,
            # output_path=judge_path
        )
    else: # reuse the generated results
        with open(judge_path, "r") as f:
            ratings = load_ratings(judge_path)

    ## 3. Coverage-based aggregation
    reranked_run = {}
    for qid in queries:
        reranked_docs = coverage_based_aggregation(
            docids=[docid for docid in runs[qid]],
            ratings=ratings[qid],
            agg_method=aggregation
        )
        reranked_run[qid] = reranked_docs

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
