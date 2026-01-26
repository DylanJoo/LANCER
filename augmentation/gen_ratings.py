import asyncio
import argparse
from argparse import Namespace

from collections import defaultdict
import json
import os
import re
import yaml

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import numpy as np
from tqdm import tqdm
from tools.neuclir.ir_utils import (
    load_query,
    load_runs_or_qrels,
    async_load_subtopics,
    load_subtopics,
    load_subtopics_human
)
from tools.neuclir.load_corpus import load_corpus

from prompts.mcranker import prompt_rating_gen_mcranker
from prompts.neuclir import *

# def gen_ratings(args: Namespace, 
#                 qrel_missing: dict = None,
#                 docs_to_rerank: list = None,
#                 topic: dict = None):
#
#     # Load the model or setup the API
#     from llm.litellm_api import LLM
#     llm = LLM(
#         model=args.model,
#         temperature=args.temperature,
#         top_p=args.top_p,
#         max_tokens=args.max_new_tokens,
#     )
#
#     # data/neuclir24-all-request.qrel
#     # [NOTE] Generalize this to other datasets
#     logger.info("loading sub-questions, (qrel's) documents...") 
#     if 'human' in args.tag:
#         if args.generate_additional_subtopics:
#             _, _, raw_topics = load_query(args)
#         questions_all = load_subtopics_human('/exp/scale25/neuclir/eval/nuggets',
#                                              args, 
#                                              raw_topics=raw_topics if args.generate_additional_subtopics else None,
#                                              create_new_subtopics=args.generate_additional_subtopics)
#     else:
#         questions_all = load_subtopics(args, topic=topic)
#         questions_all = {topic["request_id"]: questions_all}
#
#     qrels, documents_all = load_data(args, questions_all, topic,
#                                      qrel_missing=qrel_missing, docs_to_rerank=docs_to_rerank)
#
#     data = process_data(args, qrels, questions_all, documents_all)
#
#     for item in data:
#
#         ## Prompts
#         prompts = []
#         for i, document in enumerate(item['documents']):
#             for j, question in enumerate(item['questions']):
#                 prompt = prompt_rating_gen(question=question, context=document)
#                 prompts.append(prompt)
#
#         outputs = llm.inference_chat(prompts)
#         ratings = [postprocess_output(args, o) for o in outputs]
#         process_ratings(args, ratings, item)
#
#     save_gen_ratings_results(args, data)
#
#     return item

async def async_gen_ratings(args: Namespace, 
                            qrel_missing: dict = None,
                            docs_to_rerank: list = None,
                            topic: dict = None,
                            reranking: bool = False,
                            is_claims: bool = False,
                            use_claims_as_sub_docs: bool = False,
                            higher_cov_ratings: bool = False,
                            mcranker: bool = False):

    # Load the model or setup the API
    from llm.litellm_api import LLM
    llm = LLM(
        args=args,
        model=args.model,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
    )

    # data/neuclir24-all-request.qrel
    # [NOTE] Generalize this to other datasets
    logger.info("loading sub-questions, (qrel's) documents...") 
    if 'human' in args.tag:
        if args.generate_additional_subtopics:
            _, _, raw_topics = load_query(args)
        if args.dataset_name == 'neuclir':
            questions_all = load_subtopics_human('/exp/scale25/neuclir/eval/nuggets',
                                                 args, 
                                                 raw_topics=raw_topics if args.generate_additional_subtopics else None,
                                                 create_new_subtopics=args.generate_additional_subtopics)
            questions_all = {topic["request_id"]: questions_all[topic['request_id']]}
        else:
            with open("/exp/scale25/artifacts/crux/temp/ranking_3/testb_topics.jsonl") as f:
                for line in f:
                    item = json.loads(line)
                    if item['example_id'] == topic['request_id']:
                        questions_all = {item['example_id']: item['questions']}
    else:
        subquestion_file = f"/home/hltcoe/jhueiju/temp/{args.dataset_name}/{topic['request_id']}_subtopics_{args.n_subquestions}.json"
        if args.overwrite:
            if os.path.exists(subquestion_file):
                os.remove(subquestion_file)

        if os.path.exists(subquestion_file) and '70B' in args.model:
            with open(subquestion_file, "r") as f:
                questions_all = json.loads(f.read())
        else:
            # NOTE: add a temporary fix here, change to `ad_original` later.
            if int(args.n_subquestions) > 0:
                questions_all = await async_load_subtopics(args, topic=topic, docs_to_rerank=docs_to_rerank,
                                                           reranking=reranking, is_claims=is_claims,
                                                           higher_cov_ratings=higher_cov_ratings)
                questions_all = {topic["request_id"]: questions_all}
            else:
                questions_all = {topic["request_id"]: []}

        if args.include_original:
            # Directly append
            questions_all[topic["request_id"]].insert(0, topic['problem_statement'])

        if args.concat_original:
            # DIrectly concatenate
            # ps_concatenated = [topic['problem_statement'] + q for q in questions_all[topic["request_id"]]]
            ps_concatenated = [f"{topic['problem_statement']}\n{q}" for q in questions_all[topic["request_id"]]]
            questions_all[topic["request_id"]] = ps_concatenated


        # NOTE: Save only when using 70b
        if (os.path.exists(subquestion_file) is False) and '70B' in args.model:
            with open(subquestion_file, "w") as f:
                f.write(json.dumps(questions_all))

    qrels, documents_all = load_data(args, questions_all, topic,
                                     qrel_missing=qrel_missing, docs_to_rerank=docs_to_rerank,
                                     is_claims=is_claims)

    data = process_data(args, qrels, questions_all, documents_all)

    for item in data:

        ## Prompts
        prompts = []

        # if use_claims_as_sub_docs:
        #     for i, document in enumerate(item['documents']):
        #         for j, question in enumerate(item['questions']):
        #             for k, claim in enumerate(document):
        #                 prompt = prompt_rating_gen_claims_as_subdocs(question=question, context=claim)
        #                 prompts.append(prompt)
        # else:
        for i, document in enumerate(item['documents']):
            for j, question in enumerate(item['questions']):
                if is_claims:
                    prompt = prompt_rating_gen_claims(question=question, context=document)
                elif reranking:
                    if args.reranker == "crux_reranking":
                        # prompt = prompt_rating_gen_reranking(question=question, context=document, topic=topic)
                        prompt = prompt_rating_gen(question=question, context=document)
                    elif args.reranker == "mcranker":
                        prompt = prompt_rating_gen_mcranker(persona_criterias=question, 
                                                            problem_statement=topic["problem_statement"], context=document)
                    else:
                        raise Exception()
                else:
                    prompt = prompt_rating_gen(question=question, context=document)
                prompts.append(prompt)

        outputs = await llm.async_inference_chat(prompts)
        ratings = await asyncio.gather(*[async_postprocess_output(args, o) for o in outputs])

        process_ratings(args, ratings, item, 
                        use_claims_as_sub_docs=use_claims_as_sub_docs)

    save_gen_ratings_results(args, data)

    return item

def load_data(args: Namespace, questions_all: dict, topic: dict,
              qrel_missing: dict = None, docs_to_rerank: list = None,
              is_claims = None):
    qrels = load_runs_or_qrels(args.qrels, topic, threshold=3, 
                               qrel_missing=qrel_missing, docs_to_rerank=docs_to_rerank)
    logger.info("loading corpus...")
    if is_claims: 
        documents_all = load_corpus(args, args.claims_dir, path_prefix="*.mt.jsonl", is_claims=is_claims)
    else:
        if args.dataset_name == 'neuclir':
            documents_all = load_corpus(args, args.corpus_dir, path_prefix="*.mt.jsonl")
        else:
            documents_all = load_corpus(args, args.corpus_dir, path_prefix="*.jsonl")

    # logger.info(f"Total number of Sub-questions: {len(questions_all)}")
    logger.info(f"Total number of passages/documents: {len(documents_all)}")

    # Get subset of questions and documents
    if args.check_overlap:
        questions_all = {k: v for k, v in questions_all.items() if k in qrels}

    return qrels, documents_all

def process_data(args: Namespace, qrels: dict, questions_all: dict, 
                 documents_all: defaultdict):
    
    # Generate prompts
    logger.info("Generating prompts...")
    n_total = 0
    data = []
    for idx, qid in enumerate(tqdm(questions_all)):
        question_list = questions_all[qid]
        document_list = [(docid, documents_all[docid]) for docid in qrels[qid]]

        data.append({
            "id": qid , 
            "shard_id": f"{args.shard}/{args.shard_size}" if args.shard_size else 0,
            "questions": question_list,
            "docids": [docid for docid, _ in document_list],
            "documents": [document for _, document in document_list],
            "ratings": None,  # Placeholder
        })
        n_total += len(document_list) * len(question_list)

    # Start generating with sharding (default: 0 - len(data))
    start = args.shard * (args.shard_size or 0)
    end = start + (args.shard_size or len(data))
    if start >= len(data):
        exit(0)

    data = data[start:end]
    logger.info(f"Start judging {n_total} pairs.")
    return data

def process_ratings(args: Namespace, ratings: list, item: dict, use_claims_as_sub_docs=False):
    nrows, ncols = len(item['documents']), len(item['questions'])
    matrix = np.array(ratings).reshape(nrows, ncols)

    ## Report some details
    num_unanswerable_q = np.sum(np.all(matrix <= 0, axis=0))
    num_failed = np.sum(matrix == -1)
    logger.info(f"Query: {item['id']} | # Unanswerable: {num_unanswerable_q} | # Failed: {num_failed}") 

    logger.info(f" #Doc: {nrows} | #Sub-questions: {ncols}")
    item["ratings"] = matrix.tolist()

def save_gen_ratings_results(args: Namespace, data: list):
    """ Save the result (of nuggets and qrels) """
    if args.output_dir:
        output_dir = os.path.join(args.output_dir, args.dataset_name, args.tag)
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, f'crux_{args.model}.qrels'), "w") as fqrel, \
            open(os.path.join(output_dir, f'crux_{args.model}.jsonl'), "w") as fnugget:
            for item in data:
                fnugget.write(json.dumps({
                    'id': item['id'],
                    'docids': item['docids'],
                    'questions': item['questions'],
                    'ratings': item['ratings']
                })+'\n')
                for i, docid in enumerate(item['docids']):
                    for j, question in enumerate(item['questions']):
                        #if item['ratings'][i][j] <= 0:
                        fqrel.write(f"{item['id']} {(j+1)} {docid} {item['ratings'][i][j]}\n")

        logger.info(f"Saved to {output_dir}/crux_{args.model}.qrels")

def postprocess_output(args, output):
    if output.startswith('Failed'):
        from llm.litellm_api import LLM
        llm = LLM(
            model=args.model,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_new_tokens,
        )
        output = llm.inference_chat(output[1].replace("Failed", ""))

    pattern = re.compile(r"\d|-\d")
    output = re.findall(pattern, output + "-1")[0]
    rating = -1 if len(output) == 0 else int(output)
    return rating

async def async_postprocess_output(args, output) -> int:
    if output.startswith('Failed'):
        from llm.litellm_api import LLM
        llm = LLM(
            model=args.model,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_new_tokens,
        )
        output = await llm.async_inference_chat(output[1].replace("Failed", ""))
        output = output[0] # TODO: verify

    pattern = re.compile(r"\d|-\d")
    output = re.findall(pattern, output + "-1")[0]
    rating = -1 if len(output) == 0 else int(output)
    return rating

def main():
    # see the full arguments in config file
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to the config file")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shard_size", type=int, default=None)
    parser.add_argument("--tag", type=str, help="Tag of run (for saving)") # use shard here

    # Data
    parser.add_argument("--input", type=str, help="Path or directory of generated results")
    parser.add_argument("--dataset_name", type=str, help="Name of the dataset (for saving)")
    parser.add_argument("--split", type=str, default='train', help="Original split of datasets")
    parser.add_argument("--corpus_dir", type=str, help="Path to corpus")
    parser.add_argument("--qrels", type=str, help="Tag of run (for saving)") # use shard here
    parser.add_argument("--output_dir", type=str, help="Path to generated results")

    # Model
    parser.add_argument("--load_mode", type=str, default='no', help="['vllm', 'api', 'litellm']")

    # Decoding
    parser.add_argument("--temperature", type=float, default=0.5, help="Temperature for decoding")
    parser.add_argument("--top_p", type=float, default=1.0, help="Nucleus sampling top-p")
    parser.add_argument("--max_new_tokens", type=int, default=5, help="Max number of new tokens to generate in one step")
    
    # Other parameters
    parser.add_argument("--topics_path", type=str, default="", help="Path to file containing report requests")
    parser.add_argument("--generate_additional_subtopics", default=False, action="store_true", help="Will generate subtopics in addition to the reference ones")
    parser.add_argument("--check_overlap", default=False, action="store_true", help="When set to True, scores will only be computed for qid's found in qrel file")


    args = parser.parse_args()
    config = yaml.safe_load(open(args.config)) if args.config is not None else {}
    parser.set_defaults(**config)
    args = parser.parse_args()

    logger.info(args)
    gen_ratings(args)
        
    
if __name__ == "__main__":
    main()
