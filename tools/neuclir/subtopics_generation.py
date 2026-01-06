from argparse import Namespace
import json
from types import SimpleNamespace
import re

from prompts.mcranker import prompt_team_recruiting, prompt_criteria_generation
from prompts.neuclir import prompt_subquestion_gen, prompt_complementary_subquestion_gen
from prompts.neuclir import prompt_subquestion_gen_reranking, prompt_subquestion_ranking
from prompts.neuclir import prompt_subquestion_gen_reranking_detailed


def generate_complementary_subtopics(args: Namespace, 
                                     raw_topics: dict, 
                                     ref_subquestions: dict,
                                     k: int = 10,
                                     **kwargs) -> dict:
    """ Generate subquestions complementarily to the reference subquestions
    Args:
        args [Namespace]: args provided to script
        raw_topics [dict]: raw topics as provided in topic file (i.e. example-request.jsonl)
        ref_subquestions [dict]: reference subquestions
        k [int]: number of subquestions to generate
    """

    from llm.litellm_api import LLM
    llm = LLM(
        model=args.model,
        temperature=args.temperature if hasattr(args, 'temperature') else 1.0,
        top_p=args.top_p if hasattr(args, 'top_p') else 0.95,
        max_tokens=512,
    )

    # Generate subtopics with LLM
    subtopics_dict = {}
    prompts = []
    for query in raw_topics:
        prompt = prompt_complementary_subquestion_gen(
            query["background"], query["problem_statement"], 
            query["title"], ref_subquestions[query["request_id"]],
            k=k)
        prompts.append(prompt)

    outputs = llm.inference_chat(prompts)

    for idx, query in enumerate(raw_topics):
        llm_output = outputs[idx]
        pattern = r'<START OF LIST>(.*?)<END OF LIST>'
        match = re.search(pattern, llm_output, flags=re.MULTILINE | re.DOTALL)
        if match:
            extracted = match.group(1).strip()
        else:
            extracted = llm_output

        subtopics = extracted.split("\n")
        suptopics = [s for s in subtopics if (s and s not in ["START OF LIST", "END OF LIST"])]

        suptopics = suptopics[:k]
        subtopics_dict[query["request_id"]] = suptopics
    
    return subtopics_dict


def generate_subtopics(args: Namespace, 
                                     topic: dict, 
                                     k: int = 10,
                                     **kwargs) -> dict:
    """ Generate subquestions complementarily to the reference subquestions
    Args:
        args [Namespace]: args provided to script
        raw_topics [dict]: raw topics as provided in topic file (i.e. example-request.jsonl)
        ref_subquestions [dict]: reference subquestions
        k [int]: number of subquestions to generate
    """

    from llm.litellm_api import LLM
    llm = LLM(
        model=args.model,
        temperature=args.temperature if hasattr(args, 'temperature') else 1.0,
        top_p=args.top_p if hasattr(args, 'top_p') else 0.95,
        max_tokens=512,
    )

    # Generate subtopics with LLM
    prompt = prompt_subquestion_gen(
        topic["background"], topic["problem_statement"], 
        topic["title"], k=k)

    outputs = llm.inference_chat([prompt])

    subtopics = extract_llm_generated_subtopics(outputs[0], k)
    
    return subtopics

# NOTE: only this one is used
async def async_generate_subtopics(args: Namespace, 
                                   topic: dict, 
                                   k: int = 5,
                                   docs_to_rerank: list = None,
                                   reranking: bool = False,
                                   is_claims: bool = False,
                                   higher_cov_ratings: bool = False,
                                   **kwargs) -> dict:
    """ Generate subquestions complementarily to the reference subquestions
    Args:
        args [Namespace]: args provided to script
        raw_topics [dict]: raw topics as provided in topic file (i.e. example-request.jsonl)
        ref_subquestions [dict]: reference subquestions
        k [int]: number of subquestions to generate
    """

    if docs_to_rerank:
        # k = 2 # when reranking, use a limited set of 2 sub-questions
        k = args.n_subquestions
        # if is_claims:
        #     k = 2 # if we're reranking claims, go up to 5 sub-questions
        # if higher_cov_ratings: # NOTE: what is this for?
        #     k = 10

    from llm.litellm_api import LLM
    llm = LLM(
        args=args,
        model=args.model,
        temperature=args.temperature if hasattr(args, 'temperature') else 1.0,
        top_p=args.top_p if hasattr(args, 'top_p') else 0.95,
        max_tokens=512
    )

    # Generate subtopics with LLM
    if reranking:
        if higher_cov_ratings:
            prompt = prompt_subquestion_gen_reranking_detailed(
                topic["background"], topic["problem_statement"], 
                topic["title"], k=k)
        else:
            prompt = prompt_subquestion_gen_reranking(
                topic["background"], topic["problem_statement"], 
                topic["title"], k=k)
    else:
        prompt = prompt_subquestion_gen(
            topic["background"], topic["problem_statement"], 
            topic["title"], k=k)

    outputs = await llm.async_inference_chat([prompt])

    subtopics = extract_llm_generated_subtopics(outputs[0], k)
    
    return subtopics

async def async_rank_subtopics(args: Namespace,
                               subtopics, 
                               topic: dict, 
                               k: int = 5,
                               docs_to_rerank: list = None,
                               **kwargs) -> dict:
    """
    """

    if docs_to_rerank:
        k = 2 # when reranking, use a limited set of 2 sub-questions

    from llm.litellm_api import LLM
    llm = LLM(
        model=args.model,
        temperature=args.temperature if hasattr(args, 'temperature') else 1.0,
        top_p=args.top_p if hasattr(args, 'top_p') else 0.95,
        max_tokens=512,
    )

    # Rank subtopics with LLM
    prompt = prompt_subquestion_ranking(
        subtopics,
        topic["background"], topic["problem_statement"], 
        topic["title"], k=k)

    outputs = await llm.async_inference_chat([prompt])

    subtopics_reranked = extract_llm_generated_subtopics(outputs[0], k)
    
    return subtopics_reranked

# NOTE: this one is used
def extract_llm_generated_subtopics(llm_output, k):
    pattern = r'<START OF LIST>(.*?)<END OF LIST>'
    match = re.search(pattern, llm_output, flags=re.MULTILINE | re.DOTALL)
    if match:
        extracted = match.group(1).strip()
    else:
        extracted = llm_output

    subtopics = extracted.split("\n")
    subtopics = [s for s in subtopics if (s and s not in ["START OF LIST", "END OF LIST"])]

    # suptopics = suptopics[:k]
    subtopics = subtopics[:k]
    #subtopics_dict[query["request_id"]] = suptopics
    return subtopics

async def async_generate_persona_criterias_mcranker(args: Namespace, 
                         topic: dict, 
                         k: int = 5,
                         docs_to_rerank: list = None,
                         reranking: bool = False,
                         is_claims: bool = False,
                         higher_cov_ratings: bool = False,
                         **kwargs) -> dict:
    """ Generate subquestions complementarily to the reference subquestions
    Args:
        args [Namespace]: args provided to script
        raw_topics [dict]: raw topics as provided in topic file (i.e. example-request.jsonl)
        ref_subquestions [dict]: reference subquestions
        k [int]: number of subquestions to generate
    """

    from llm.litellm_api import LLM
    llm = LLM(
        model=args.model,
        temperature=args.temperature if hasattr(args, 'temperature') else 1.0,
        top_p=args.top_p if hasattr(args, 'top_p') else 0.95,
        max_tokens=1024,
    )

    prompt = prompt_team_recruiting(
        topic["background"], topic["problem_statement"], 
        topic["title"], k=k)
    outputs = await llm.async_inference_chat([prompt])
    team_data = extract_llm_generated_team_recruiting(outputs[0])

    prompt = prompt_criteria_generation(
        topic["background"], topic["problem_statement"], 
        topic["title"], team_data)
    outputs = await llm.async_inference_chat([prompt])
    subtopics = extract_llm_criteria_generation(outputs[0])
    
    return subtopics

def extract_llm_generated_team_recruiting(llm_output):
    try:
        team_data = json.loads(llm_output)
        #for member in team_data["team"]:
        #    print(f"{member['id']} - {member['role']}: {member['description']}")
        return team_data
    except json.JSONDecodeError:
        print("Model did not return valid JSON:")
        print(llm_output)

def extract_llm_criteria_generation(llm_output):
    try:
        criteria_data = extract_json(llm_output)
        #for member in team_data["team"]:
        #    print(f"{member['id']} - {member['role']}: {member['description']}")
        return criteria_data["criteria"]
    except json.JSONDecodeError:
        print("Model did not return valid JSON:")
        print(llm_output)

def extract_json(text: str):
    """
    Extract and parse JSON from LLM output, even if wrapped in ```json ... ``` or escaped.
    """
    # 1. Remove code fences like ```json ... ```
    cleaned = re.sub(r"^```(?:json)?", "", text.strip())
    cleaned = re.sub(r"```$", "", cleaned.strip())
    
    # 2. Decode escaped sequences (\n, \")
    cleaned = cleaned.encode('utf-8').decode('unicode_escape')

    # 3. Final strip
    cleaned = cleaned.strip()

    # 4. Parse
    return json.loads(cleaned)
