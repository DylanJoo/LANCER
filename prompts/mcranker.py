
def prompt_team_recruiting(
        user_background, problem_statement, title,
        k=10):

    template = """
    You are mrᴇᴄʀᴜɪᴛ, an expert recruiter of domain-specific evaluation teams. 
    Your task is to form a team of expert personas to evaluate and annotate content about a given topic. 

    Guidelines:
    1. The team must ALWAYS include an NLP Scientist (this member provides advanced text and language analysis expertise).
    2. You will then recruit two additional members with interdisciplinary expertise relevant to the topic. These members should be professionals, scholars, or practitioners who collectively cover the main dimensions of the subject matter.
    3. Each member should have a distinct role and perspective. Avoid redundancy.
    4. Provide a short description for each persona: name (can be descriptive like "Medical Ethicist" or "Urban Planner"), background, and what unique perspective they contribute to the team.

    Return ONLY valid JSON. Do not include explanations, formatting, or Markdown code fences. Follow this schema:
    {
        "team": [
            {
                "id": "a0",
                "role": "NLP Scientist",
                "description": "<short description>"
            },
            {
                "id": "a1",
                "role": "<role>",
                "description": "<short description>"
            }
        ]
    }

    Now, form the team for the following topic:

    {problem_statement}
    """
    #joined_subquestions = '\n'.join(prev_subquestions)
    #template = template.replace("{NUM}", str(k))
    #template = template.replace("{prev_subquestions}", joined_subquestions)
    #template = template.replace("{background}", user_background)
    #template = template.replace("{title}", title)
    template = template.replace("{problem_statement}", problem_statement)
    return template.strip()


def prompt_criteria_generation(
        user_background, problem_statement, title,
        team_data):

    template = """
    You are Mᴄʀɪᴛᴇʀɪᴀ, an expert in constructing rigorous, query-centric scoring criteria 
    for interdisciplinary evaluation teams. 

    You are given a JSON object that defines a team of evaluator personas. 
    Each persona has an id, a role, and a description of their expertise. You are also given the
    topic for which a report needs to be generated. 

    Your task:
    - For each persona, create a set of scoring criteria they will use when evaluating a report.  
    - Criteria must reflect the persona’s expertise and perspective.  
    - Each criterion should have:
    1. A short title (concise, domain-specific)  
    2. A weight (a number between 0 and 1)  
    3. A brief description explaining what is being evaluated  

    - The weights for each persona’s criteria must sum to **1.0**, ensuring a systematic distribution.  
    - Output must be in **valid JSON only** following this schema:

    {
    "criteria": [
        {
        "id": "<persona id>",
        "role": "<persona role>",
        "criteria": [
            {
            "title": "<criterion title>",
            "weight": <float>,
            "description": "<criterion description>"
            },
            ...
        ]
        }
    ]
    }

    Return only valid JSON. Do not include explanations, natural language, or markdown code fences. 
    Topic: {problem_statement}

    Team:
    {team_data}
    """
    #joined_subquestions = '\n'.join(prev_subquestions)
    #template = template.replace("{NUM}", str(k))
    #template = template.replace("{prev_subquestions}", joined_subquestions)
    #template = template.replace("{background}", user_background)
    #template = template.replace("{title}", title)
    template = template.replace("{problem_statement}", problem_statement)
    template = template.replace("{team_data}", str(team_data["team"]))
    return template.strip()


def prompt_rating_gen_mcranker(persona_criterias="", problem_statement="", context="", **kargs):
    
    template = """
    You are Mᴇᴠᴀʟᴜᴀᴛɪᴏɴ, an expert evaluator. 
    You are given:
    1. A persona (their role and expertise),
    2. A structured set of scoring criteria with weights,
    3. A query,
    4. A passage.

    Task:
    - Evaluate the passage with respect to the query using ONLY the given criteria. 
    - For each criterion:
        - Briefly assess how well the passage meets the criterion
        - Assign a provisional score from 0 to 5.
        - Multiply this score by the criterion’s weight.
    - Compute the weighted average score across all criteria.
    - Round the final score to the nearest integer in the range 0–5.
    - Return the final score only; do not write anything except the rating, do not add explanations.

    Persona and Criteria:
    {PERSONA_JSON}

    Query:
    {QUERY}

    Passage:
    {PASSAGE}

    Rating:
    """
    persona_criterias = persona_criterias.copy()
    persona_criterias.pop("id")
    template = template.replace("{PERSONA_JSON}", str(persona_criterias))
    template = template.replace("{QUERY}", problem_statement)

    if isinstance(context, dict):
        context = context.get('title', '') + ' ' + context.get('text', '')
    template = template.replace("{PASSAGE}", context)
    
    return template.strip()