import argparse

from evaluation.prejudge.retrieval_augmentation_context import rac_evaluate
from tools.neuclir.load_rac_data import load_rac_data


# Load data (corpus, initial query (report request), sub-questions), 
from tools.neuclir.ir_utils import (
    load_diversity_qrels,
    load_qrels, 
    load_query
)
from tools.neuclir.load_judgements import load_human_judgements, load_judgements

        
def run_rac_nugget_eval(args):

    queries, queries_for_search, raw_topics = load_query(args)

    # QRELS: 
    ## the document qrels are not available on GRID. download from TREC website:
    ## wget https://trec.nist.gov/data/neuclir/2023/neuclir-2023-qrels.final.tar.gz
    qrels = load_qrels(args.qrels)
    crux_qrels = load_qrels(args.crux_qrels)
    crux_andor_qrels = load_qrels(args.crux_andor_qrels if args.crux_andor_qrels is not None else args.crux_qrels)

    # Load RAC data. This includes running search or reading retrieved documents from a local file
    rac_data = load_rac_data(args, queries, queries_for_search, raw_topics,
                               retrieval_service_name=args.service_name)

    # We can either (1) covert the human labels to judgements, or (2) run CRUX pipeline to generate ratings.
    # judgements = load_judgements(args.data.judgement_file) if args.data.judgement_file is not None else None
    # human_judgements = load_human_judgements(args.nuggets_dir)
    human_judgements = load_human_judgements(args.qrels, True)
    crux_judgements = load_judgements(args, rac_data, compute_missing_judgements=False)

    try:
        crux_judgements_andor = load_judgements(args, rac_data, load_andor=True)

        # Take MIN of regular and AND/OR judgements
        # Intuition: If context doesn't answer a question in general, it
        # shouldn't answer a question with an answer better

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

    except:
        crux_judgements_andor = None

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
        gamma=0.5, tag='experiment',
    )

    if crux_judgements_andor is not None:
        output_eval["crux_andor_0620"] = rac_evaluate(
            args, crux_andor_qrels,
            crux_judgements_andor,
            rac_data=rac_data,
            diversity_qrels_path=args.crux_andor_qrels if args.crux_andor_qrels is not None else args.crux_qrels,
            tokenizer_name='bert-base-uncased',
            gamma=0.5, tag='experiment'
        )

    if (args.reranker == 'crux_reranking'):
        args.reranker = f"crux_reranking:{args.agg}:{args.n_subquestions}"
        if 'human' in args.tag:
            args.reranker = f"crux_reranking:{args.agg}:human"

    setting = f"## {args.service_name} | -> {args.reranker}" 
    print(f'## Retrieval | -> Rerank | metrics   | ' + ' |'.join(['A_nDCG', 'Cov', 'P', 'MAP', 'nDCG']))

    if args.dataset_name == 'neuclir':
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

    # print(f'{setting} | metrics   | ' + ' | '.join([f"Cov@{k}" for k in list(range(1, 11))]) )
    human_coverage = f"{setting} | Cov@k (Human)     | "
    crux_coverage  = f"{setting} | Cov@k (crux_0620) | "
    for _k in range(1, 11):
        num = output_eval['human'][f'mean_cov@{_k}']
        human_coverage += f"{num:.4f} |"
        num = output_eval['crux_0620'][f'mean_cov@{_k}']
        crux_coverage += f"{num:.4f} |"

    if args.dataset_name == 'neuclir':
        print(human_coverage)
    print(crux_coverage)

    # if crux_judgements_andor is not None:
    #     print("\n========== NeuCLIR 2024 (ReportGen) - CRUX-0620 (100K qrels) Q/A AND/OR Scores ==========")
    #     if args.crux_andor_qrels is not None:
    #         print(f"A-nDCG@20: {output_eval['crux_andor_0620']['alpha_nDCG']}")
    #     print(f"Cov@10: {output_eval['crux_andor_0620']['mean_coverage_at_10']}")
    #pprint.pprint(output_eval)
