import os
import json
import re
import openai

template_1 = """
Instruction: Given the following report request, write {NUM} diverse and non-repeating sub-questions that can help guide the creation of a focused and comprehensive report. The sub-questions should help break down the topic into key areas that need to be investigated or explained. Each sub-question should be short (ideally under 20 words) and should focus on a single aspect or dimension of the report.

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

template_2 = """
Instruction: Given the following report request, write {NUM} diverse and non-repeating sub-questions that can help guide the creation of a focused and comprehensive report. The sub-questions should help break down the topic into key areas that need to be investigated or explained. Each sub-question should be short (ideally under 20 words) and should focus on a single aspect or dimension of the report.

Report Request: {problem_statement}

Output format:
- List each sub-question on a new line. Do not number the sub-questions.
- Do not add any comment or explanation.
- Output without adding additional questions after the specified {NUM}. Begin with "<START OF LIST>" and, when you are finished, output "<END OF LIST>". Never ever add anything else after "<END OF LIST>", my life depends on it!!!

Now, generate the {NUM} sub-questions:
"""

client = openai.OpenAI(
    api_key="tgp_v1_iIX1koXpE7sBJLykB7yvNqabv5XGQEZqpQl3hOdqALU",
    base_url="https://api.together.xyz/v1",
)

NUM = str(2)
# NUM = str(1)

def get_result(problem_statement):
    response = client.chat.completions.create(
        model="Qwen/Qwen3-Next-80B-A3B-Instruct",
        messages=[
            {
                "role": "user",
                "content": template_2.replace("{problem_statement}", problem_statement).replace("{NUM}", NUM)
            },
        ], 
        temperature=0,
        top_p=1,
    )

    output = response.choices[0].message.content
    return output


input_request_file = "/exp/scale25/artifacts/crux/temp/ranking_3/testb_topics.jsonl"
output_file = "output.json"

llm_outputs = []
with open(input_request_file, "r") as infile:
    for line in infile:
        id = json.loads(line)["example_id"]
        request = json.loads(line)["topic"]
        output = get_result(request)
        llm_outputs.append({"id": id, "output": output})

with open(output_file, "w") as outfile:
    for llm_output in llm_outputs:
        pattern = r'<START OF LIST>(.*?)<END OF LIST>'
        match = re.search(pattern, llm_output["output"], flags=re.MULTILINE | re.DOTALL)
        if match:
            extracted = match.group(1).strip()
        else:
            extracted = llm_output

        try:
            subtopics = extracted.split("\n")
            suptopics = [s for s in subtopics if (s and s not in ["START OF LIST", "END OF LIST"])]
            llm_output["subtopics"] = subtopics
        except:
            llm_output["subtopics"] = 'empty'

    # output
    json.dump(llm_outputs, outfile)
