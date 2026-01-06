""" NOTE
CRUX: Generate questions from the summary of the report request dataset.
NeuCLIR: (1) Nugget questions [human] (2) Extented questions [Francois] 
Researchy question: (1) Existing questions
"""
import os
import sys
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import re
import yaml
import argparse
import json

import numpy as np
from tqdm import tqdm

from prompts.neuclir import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to the config file")
    parser.add_argument("--shard", type=int, default=0, help="the n-th shard")
    parser.add_argument("--shard_size", type=int, default=None, help="size of one shard")
    parser.add_argument("--output_dir", type=str, help="directory for the output result")

    # Evaluation file is a json file that contains a list of item, each of which contains
    parser.add_argument("--split", type=str, default='train', help="Original split of datasets")

    # Model and name
    parser.add_argument("--dataset_name", type=str, help="Name of the dataset (for saving)")
    parser.add_argument("--tag", type=str, help="Tag of run (for saving)") # use shard here
    parser.add_argument("--load_mode", type=str, default='no', help="['vllm', 'api', 'litellm]")
    parser.add_argument("--model", type=str, help="Model to use")
    parser.add_argument("--model_tag", type=str, help="Tag of run (for saving)") 

    # Decoding
    parser.add_argument("--temperature", type=float, default=0.5, help="Temperature for decoding")
    parser.add_argument("--top_p", type=float, default=1.0, help="Nucleus sampling top-p")
    parser.add_argument("--max_new_tokens", type=int, default=64, help="Max number of new tokens to generate in one step")
    parser.add_argument("--max_length", type=int, default=4096, help="Max length the model can take. Should set properly wrt the model to avoid position overflow.")

    # Configurations
    parser.add_argument("--num_gpus", default=1, type=int)
    parser.add_argument("--ampere_gpu", default=False, action='store_true')

    # Configurations
    parser.add_argument("--source_type", default="report_request", type=str, choices=["report_request", "RAG_draft"])
    parser.add_argument("--source_path", default=None, type=str)

    # Load config
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config)) if args.config is not None else {}
    parser.set_defaults(**config)
    args = parser.parse_args()

    logger.info(f"The configuration is as follow")
    logger.info(config)
        
    # Load the model or setup the API
    from llm.litellm_api import LLM
    llm = LLM(
            model=args.model,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_new_tokens
    )

    np.random.seed(777)
    dataset = [dataset[int(idx)] for idx in qids]
    qids = np.random.choice(len(dataset), args.quick_test, replace=False)
    qids = list(range(len(dataset)))
    logger.info(f"Total number of examples: {len(dataset)}")

    # Generate prompts
    logger.info("Generating prompts...") 
    n_total = 0
    data = []
    for idx, item in enumerate(tqdm(dataset)):
        document_list = item['document']
        summary_text = normalize_text(item['summary'])

        # [TODO] prompt depends on the source type
        if args.source_type == "report_request":
            prompt = prompt_question_gen()
        else:
            prompt = prompt_question_gen(
                INST=instruction_question,
                D=summary_text,
                PREFIX="Questions:\n<q>"
            )

        data.append({
            'example_id': f"{item['mds-source']}-{args.split}-{ids[idx]}", 
            'shard_id': f"{args.shard}-{idx}", 
            'full_text': summary_text,
            'prompt': prompt,
        })
        n_total += len(document_list)
    logger.info(f"Done prompt preparation. Total number of prompts: {len(data)} | {n_total}")

    # Start generation with sharding
    start = args.shard * (args.shard_size or 0)
    end = start + (args.shard_size or len(data))
    if start >= len(data):
        exit(0)

    data = data[start:end]
    for idx, item in enumerate(tqdm(data, total=len(data))):
        prompt = item['prompt']
        prompt_len = len(llm.tokenizer.tokenize(prompt))
        output = llm.inference(prompt, max_tokens=args.max_new_tokens)

        # postprocess for consistent format
        output = output.split('Note:')[0]
        output = output.split('Answer:')[0]
        output = output.split('Instruction:')[0]

        logger.info(f"Example: {item['example_id']} -- {item['shard_id']}")
        logger.info(f"prompt text (length={prompt_len}): {prompt}")
        logger.info(f"Final model output: {output}") 
        item['output'] = output 
        if idx != 0:
            item['prompt'] = ""

    # Save the result
    data = {"args": args.__dict__, "data": data}

    output_dir = os.path.join(args.output_dir, args.tag)
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, f"{args.model_tag}-{args.split}-{args.shard}.json")
    json.dump(data, open(output_file, 'w'), indent=4)

if __name__ == "__main__":
    main()
