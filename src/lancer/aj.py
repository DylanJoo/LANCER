import json
from tqdm import tqdm
import numpy as np
import re

def answerability_judment(
    llm, 
    subquestions,
    documents,
    queries,
    concat_original: bool = True,
):

    # Reset the LLM parameters
    llm.temperature = 0.0
    llm.top_p = 1.0
    llm.max_tokens = 10

    # Concatenate the original query in the begining.
    if concat_original:
        for qid in queries:
            query = queries[qid]
            concatenated = [f"{query}\n{q}" for q in subquestions[qid]]
            subquestions[qid] = concatenated

    ratings = {}
    for qid in tqdm(queries, desc="Answerability Judgment"):
        query = queries[qid]

        prompts = []
        for i, document in enumerate(documents[qid]):
            for j, question in enumerate(subquestions[qid]):
                prompt = prompt_rating_gen(question=question, context=document)
                prompts.append(prompt)
        outputs = llm.generate_ratings(prompts)
        output_ratings = [str2int(o) for o in outputs]
        nrows, ncols = len(documents[qid]), len(subquestions[qid])
        matrix = np.array(output_ratings).reshape(nrows, ncols)
        ratings[qid] = matrix.tolist()

    return ratings

def str2int(output):
    pattern = re.compile(r"\d|-\d")
    output = re.findall(pattern, output + "-1")[0]
    rating = -1 if len(output) == 0 else int(output)
    return rating

def prompt_rating_gen(question="", context="", lang='eng'):

    template = """
    Instruction: Determine whether the question can be answered based on the provided context? Rate the context on a scale from 0 to 5 according to the guideline below. Do not write anything except the rating.

    Guideline: 
    - 5: The context is highly relevant, complete, and accurate to the question.
    - 4: The context is mostly relevant and complete but may have minor gaps or inaccuracies to the question.
    - 3: The context is partially relevant and complete, with noticeable gaps or inaccuracies to the question.
    - 2: The context has limited relevance and completeness, with significant gaps or inaccuracies to the question.
    - 1: The context is minimally relevant or complete, with substantial shortcomings to the question.
    - 0: The context is not relevant or complete at all.
    
    Question: {question}

    Context: {context} 

    Rating:
    """
    p = template.replace("{question}", question).strip()
    if isinstance(context, dict):
        context = context.get('title', '') + ' ' + context.get('text', '')
    p = p.replace("{context}", context).strip()
    return p
