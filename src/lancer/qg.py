import json
import re

# TODO: move the topk-truncation at the later stage, making it more flexible to different truncation
# TODO: fix the first two settings
def question_generation(
    llm, 
    queries: dict, 
    topics: dict = None,
    n_subquestions: int = 2,
    use_oracle: bool = False,
    output_file: str = None,
):
    if use_oracle:
        subquestions = load_subtopic()
        return subquestions

    # prompts
    prompts = []
    for qid in queries:
        prompts.append(
            prompt_with_example(
                problem_statement=queries[qid],
                user_background=topics[qid]['background'] if topics else "",
                title=topics[qid]['title'] if topics else "",
                n=n_subquestions
            )
        )
    outputs = llm.generate_questions(prompts)

    subquestions = {}
    for i, qid in enumerate(queries):
        subquestions[qid] = extract_subtopics(outputs[i], n_subquestions)

    if output_file:
        json.dump(subquestions, open(output_file, "w"), indent=2)

    return subquestions

def prompt_with_example(
    problem_statement,
    user_background=None,
    title=None,
    n=2
):
    template = """
    Instruction: Given the following report request, write {NUM} diverse and non-repeating sub-questions that can help guide the creation of a focused and comprehensive report. The sub-questions should help break down the topic into key areas that need to be investigated or explained. Each sub-question should be short (ideally under 20 words) and should focus on a single aspect or dimension of the report.

    Here are examples of sub-questions for a report request about the mysteries of Machu Picchu's architecture:
    - Where is Machu Picchu located?
    - How high is the mountain ridge on which Machu Picchu sits?
    - What make Machu Picchu one of the world's most visited sites?
    - What are the most remarkable aspects of the construction structure of Machu Picchu?

    Report Request:
    - Title: {title}
    - Background: {background}
    - Problem Statement: {problem_statement}

    Output format:
    - List each sub-question on a new line. Do not number the sub-questions.
    - Do not add any comment or explanation.
    - Output without adding additional questions after the specified {NUM}. Begin with "<START OF LIST>" and, when you are finished, output "<END OF LIST>". Never ever add anything else after "<END OF LIST>", my life depends on it!!!

    Now, generate the {NUM} sub-questions:
    """
    problem_statement = ("" or problem_statement)
    title = ("" or title)

    template = template.replace("{NUM}", str(n))
    template = template.replace("{background}", user_background)
    template = template.replace("{title}", title)
    template = template.replace("{problem_statement}", problem_statement)
    return template.strip()

def extract_subtopics(llm_output, n):
    pattern = r'<START OF LIST>(.*?)<END OF LIST>'
    match = re.search(pattern, llm_output, flags=re.MULTILINE | re.DOTALL)
    if match:
        extracted = match.group(1).strip()
    else:
        extracted = llm_output

    subtopics = extracted.split("\n")
    subtopics = [s.strip() for s in subtopics if (s and s not in ["START OF LIST", "END OF LIST"])]
    subtopics = [re.sub(r'^[\-\*\d\.\)\s]+', '', s) for s in subtopics]
    subtopics = [s for s in subtopics if s != ""]
    # if len(subtopics) > n:
    #     print(f" Truncating subtopics from {len(subtopics)} to {n}.")
    subtopics = subtopics[:n]
    return subtopics

