###################################################
# prompt for statement generation # (deprecated)  #
###################################################

###################################
# prompt for question generation  #
###################################
def prompt_subquestion_gen_from_report(report, NUM=10):
    template = """
    Instruction: Write {NUM} diverse sub-questions that can reveal the information contained in the given document. Each question should be self-contained and have necessary context. Write the question within `<q>` and `</q>` tags.

    Document: {report}

    Questions:
        <q>
    """
    p = template.replace("{report}", reprot).strip()
    return p


# [NOTE] start from here
def prompt_subquestion_gen_from_report_request(user_background, problem_stataement, NUM=10):
    system_prompt = "You are an expert academic research assistant."

    template = """
    Instruction: Given the following report request, write {NUM} diverse and insightful sub-questions that can help guide the creation of a focused and research report. The sub-questions should help break down the topic into key areas that need to be investigated or explained.

    Report Request:
    - Background: {background}
    - Problem Statement: {problem_statement}
    - Title: {title}

    Now, generate the subquestion
    """
    p = template.replace("{report}", reprot).strip()
    return p

################################
# prompt for rating generation #
################################
def prompt_rating_gen(question="", context="", lang='eng', **kargs):
    if lang == 'eng':
        language_setting = ''
    else:
        language_setting = ' The question is asked in English, but the context is provided in either English, Chinese, Persian, or Russian.'
    template = """
    Instruction: Determine whether the question can be answered based on the provided context?{language_setting} Rate the context on a scale from 0 to 5 according to the guideline below. Do not write anything except the rating.

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
    p = template.replace("{language_setting}", language_setting).strip()
    p = p.replace("{question}", question).strip()
    if isinstance(context, dict):
        context = context.get('title', '') + ' ' + context.get('text', '')
    p = p.replace("{context}", context).strip()
    return p


########################################################
# prompt for rating generation for OR question/answers #
########################################################
def prompt_rating_gen_or(question="", answers=[], context="", lang='eng'):
    if lang == 'eng':
        language_setting = ''
    else:
        language_setting = ' The question is asked in English, but the context is provided in either English, Chinese, Persian, or Russian.'
    template = """
    Instruction: Given the following context and a question, determine if ANY of the given possible answers are answered by the context.{language_setting} The context does not need to include all answers, but needs to include at least one answer. Rate the context on a scale from 0 to 5 according to the guideline below. Do not write anything except the rating.

    Guideline:
    - 5: The context is highly relevant, complete, and accurate, and answers the question with the given answers.
    - 4: The context is mostly relevant and complete but may have minor gaps or inaccuracies. The question is mostly answered with the given answers.
    - 3: The context is partially relevant and complete, with noticeable gaps or inaccuracies. Parts of the question are answered.
    - 2: The context has limited relevance and completeness, with significant gaps or inaccuracies. The answer to the question is incomplete.
    - 1: The context is minimally relevant or complete, with substantial shortcomings. The answer to the question is mostly incorrect or incomplete.
    - 0: The context is not relevant or complete at all. The question is not answered, or the context does not contain the given answer.

    Question: {question}

    Possible Answers:
    {answers}

    Context: {context}

    Rating:
    """
    p = template.replace('{language_setting}', language_setting).strip()
    p = p.replace('{question}', question).strip()
    p = p.replace('{answers}', '\n    '.join(answers))
    if isinstance(context, dict):
        context = context.get('title', '') + ' ' + context.get('text', '')
    p = p.replace('{context}', context).strip()
    return p

#########################################################
# prompt for rating generation for AND question/answers #
#########################################################
def prompt_rating_gen_and(question="", answers=[], context="", lang='eng'):
    if lang == 'eng':
        language_setting = ''
    else:
        language_setting = ' The question is asked in English, but the context is provided in either English, Chinese, Persian, or Russian.'
    template = """
    Instruction: Given the following context and a question, determine if ALL of the given possible answers are answered by the context. The context needs to address all given answers with respect to the given question. Rate the context on a scale from 0 to 5 according to the guideline below. Do not write anything except the rating.

    Guideline:
    - 5: The context is highly relevant, complete, and accurate, and answers the question with the given answers.
    - 4: The context is mostly relevant and complete but may have minor gaps or inaccuracies. The question is mostly answered with the given answers.
    - 3: The context is partially relevant and complete, with noticeable gaps or inaccuracies. Parts of the question are answered.
    - 2: The context has limited relevance and completeness, with significant gaps or inaccuracies. The answer to the question is incomplete.
    - 1: The context is minimally relevant or complete, with substantial shortcomings. The answer to the question is mostly incorrect or incomplete.
    - 0: The context is not relevant or complete at all. The question is not answered, or the context does not contain the given answer.

    Question: {question}

    Required Answers:
    {answers}

    Context: {context}

    Rating:
    """
    p = template.replace('{language_setting}', language_setting).strip()
    p = p.replace('{question}', question).strip()
    p = p.replace('{answers}', '\n    '.join(answers))
    if isinstance(context, dict):
        context = context.get('title', '') + ' ' + context.get('text', '')
    p = p.replace('{context}', context).strip()
    return p

# [TODO] Fix 
############################################
# prompt for topic generation (report gen) #
############################################
template_request = "Instruction: {INST}\n\n{DEMO}\n\nReport: {D}\n\n{PREFIX}"
# instruction_request (the one with questions) = "Create a general statement of report request that corresponds to the questions the user is about to ask. Given that a report generation system has produced the passage below, write a report request statement of approximately 30 words. Write the statement within <r> and </r> tags."
instruction_request = "Create a statement of report request that corresponds to given report. Write the report request of approximately 50 words within <r> and </r> tags."
demo_input = "Please produce a report on investigations within the United States in either the public or private sector into Unidentified Flying Objects (UFOs). The report should cover only investigative activities into still unidentified phenomena, and not the phenomena themselves. It should include information on the histories, costs, goals, and results of such investigations."
demo_output = "Whether you dismiss UFOs as a fantasy or believe that extraterrestrials are visiting the Earth and flying rings around our most sophisticated aircraft, the U.S. government has been taking them seriously for quite some time. “Project Blue Book”, commissioned by the U.S. Air Force, studied reports of “flying saucers” but closed down in 1969 with a conclusion that they did not present a threat to the country. As the years went by UFO reports continued to be made and from 2007 to 2012 the Aerospace Threat Identification Program, set up under the sponsorship of Senator Harry Reid, spent $22 million looking into the issue once again. Later, the Pentagon formed a “working group for the study of unidentified aerial phenomena”. This study, staffed with personnel from Naval Intelligence, was not aimed at finding extraterrestrials, but rather at determining whether craft were being flown by potential U.S. opponents with new technologies. In June, 2022, in a report issued by the Office of the Director for National Intelligence and based on the observations made by members of the U.S. military and intelligence  from 2004 to 2021 it was stated that at that time there was, with one exception, not enough information to explain the 144 cases of what were renamed as “Unidentified Aerial Phenomena” examined."
demo = f"Report: {demo_output}\n\nReport request: <r>{demo_input}</r>" # an in-context example

def prompt_topic_gen(INST="", DEMO=demo, D="", PREFIX="Report request: <r>"):
    p = template_request
    p = p.replace("{INST}", INST).strip()
    p = p.replace("{DEMO}", DEMO).strip()
    p = p.replace("{D}", D)
    p = p.replace("{PREFIX}", PREFIX).strip()
    return p

def apply_docs_prompt(doc_items, ndoc=None, field='text'):
    p = ""
    for idx, doc_item in enumerate(doc_items[:ndoc]):
        p_doc = doc_prompt_template
        p_doc = p_doc.replace("{ID}", str(idx+1))
        p_doc = p_doc.replace("{T}", doc_item['title'])
        p_doc = p_doc.replace("{P}", doc_item[field])
        p += p_doc
    return p


################################
# prompt for query reduction #
################################
# guideline = "- 5: The context is highly relevant, complete, and accurate.\n- 4: The context is mostly relevant and complete but may have minor gaps or inaccuracies.\n- 3: The context is partially relevant and complete, with noticeable gaps or inaccuracies.\n- 2: The context has limited relevance and completeness, with significant gaps or inaccuracies.\n- 1: The context is minimally relevant or complete, with substantial shortcomings.\n- 0: The context is not relevant or complete at all."
template_rating = "Instruction: {INST}\n\nGuideline:\n{G}\n\nQuestion: {Q}\n\nContext: {C}\n\n{PREFIX}" 
instruction_rating = "Summarize the provided query into a one line google search that would be useful to fulfill the query."
def prompt_query_reduction(INST="", Q="", C="", PREFIX="Rating:"):
    p = template_rating
    p = p.replace("{INST}", INST).strip()
    p = p.replace("{Q}", Q)
    return p

####################################################
# prompt for complementary subquestions generation #
####################################################
def prompt_complementary_subquestion_gen(
        user_background, problem_statement, title, prev_subquestions,
        k=10):

    template = """
    Instruction: Given the following report request, write {NUM} additional, diverse and non-repeating sub-questions that can help guide the creation of a focused and comprehensive report. The sub-questions should help break down the topic into key areas that need to be investigated or explained. Each sub-question should be short (ideally under 20 words) and should focus on a single aspect or dimension of the report. 
    Do not repeat or overlap with previously generated sub-questions.

    Previously generated subquestions: 
{prev_subquestions}
    
    Report Request:
    - Title: {title}
    - Background: {background}
    - Problem Statement: {problem_statement}

    Output format:
    - List each sub-question on a new line. Do not number the sub-questions.
    - Do not add any comment or explanation.
    - Do not repeat or overlap with previously generated sub-questions.
    - Output without adding additional questions after the specified {NUM}. Begin with "<START OF LIST>" and, when you are finished, output "<END OF LIST>". Never ever add anything else after "<END OF LIST>", my life depends on it!!!
    
    Now, generate the {NUM} sub-questions:
    """
    joined_subquestions = '\n'.join(prev_subquestions)
    template = template.replace("{NUM}", str(k))
    template = template.replace("{prev_subquestions}", joined_subquestions)
    template = template.replace("{background}", user_background)
    template = template.replace("{title}", title)
    template = template.replace("{problem_statement}", problem_statement)
    return template.strip()


# NOTE: add the workaround for the query that has only the problem_statement
####################################################
# prompt for subquestions generation #
####################################################
def prompt_subquestion_gen(
        user_background, problem_statement, title,
        k=10):

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

    if user_background.strip() == "":
        template = template_1
    else:
        template = template_2

    #joined_subquestions = '\n'.join(prev_subquestions)
    template = template.replace("{NUM}", str(k))
    #template = template.replace("{prev_subquestions}", joined_subquestions)
    template = template.replace("{background}", user_background)
    template = template.replace("{title}", title)
    template = template.replace("{problem_statement}", problem_statement)
    return template.strip()

####################################################
# prompt for subquestions generation (reranking) #
####################################################
def prompt_subquestion_gen_reranking(
        user_background, problem_statement, title,
        k=10):

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
    #joined_subquestions = '\n'.join(prev_subquestions)
    template = template.replace("{NUM}", str(k))
    #template = template.replace("{prev_subquestions}", joined_subquestions)
    template = template.replace("{background}", user_background)
    template = template.replace("{title}", title)
    template = template.replace("{problem_statement}", problem_statement)
    return template.strip()

"""
Here are examples of sub-questions for a report request about the mysteries of Machu Picchu's architecture:
    - Where is Machu Picchu located?
    - How high is the mountain ridge on which Machu Picchu sits?
    - What make Machu Picchu one of the world's most visited sites?
    - What are the most remarkable aspects of the construction structure of Machu Picchu?
"""

def prompt_subquestion_ranking(
        subtopics, user_background, problem_statement, title,
        k=10):

    template = """
    Instruction: You are given a report request along with {NUM} sub-questions. Your task is to rank these sub-questions in order of importance to guide the creation of a focused and comprehensive report. The ranking should reflect how essential each sub-question is to understanding or solving the core problem.
    
    Report Request:
    - Title: {title}
    - Background: {background}
    - Problem Statement: {problem_statement}

    Subtopics:
    {subtopics}

    Output format:
    - Output exactly the provided {NUM} sub-questions, reordered from most important to least important.
    - Each sub-question must be on a new line.
    - Do not number the sub-questions or add bullet points.
    - Do not add any comments, summaries, or explanations.
    - Begin the output with: <START OF LIST>
    - End the output with: <END OF LIST>
    - Do not include any text after "<END OF LIST>". This is critical.
    
    Now, return the ranked list of sub-questions:
    """
    #joined_subquestions = '\n'.join(prev_subquestions)
    subtopics = "\n".join(subtopics)
    template = template.replace("{NUM}", str(k))
    #template = template.replace("{prev_subquestions}", joined_subquestions)
    template = template.replace("{background}", user_background)
    template = template.replace("{title}", title)
    template = template.replace("{problem_statement}", problem_statement)
    template = template.replace("{subtopics}", subtopics)
    return template.strip()

################################
# prompt for rating generation for reranking #
################################

def prompt_rating_gen_reranking(question="", context="", lang='eng', topic=None, **kargs):
    if lang == 'eng':
        language_setting = ''
    else:
        language_setting = ' The question is asked in English, but the context is provided in either English, Chinese, Persian, or Russian.'
    template = """
    Instruction: Determine whether the question can be answered based on the provided context?{language_setting} The question is being asked to support the writing of a report, which is described in the provided report request. Rate the context on a scale from 0 to 5 according to the guideline below. Do not write anything except the rating.

    Guideline: 
    - 5: The context is highly relevant, complete, and accurate to the question.
    - 4: The context is mostly relevant and complete but may have minor gaps or inaccuracies to the question.
    - 3: The context is partially relevant and complete, with noticeable gaps or inaccuracies to the question.
    - 2: The context has limited relevance and completeness, with significant gaps or inaccuracies to the question.
    - 1: The context is minimally relevant or complete, with substantial shortcomings to the question.
    - 0: The context is not relevant or complete at all.
    
    Question: {question}

    Report Request: {problem_statement}

    Context: {context} 

    Rating:
    """
    p = template.replace("{language_setting}", language_setting).strip()
    p = p.replace("{question}", question).strip()
    if isinstance(context, dict):
        context = context.get('title', '') + ' ' + context.get('text', '')
    p = p.replace("{context}", context).strip()
    if topic:
        # p = p.replace("{title}", topic["title"]).strip()
        # p = p.replace("{background}", topic["background"]).strip()
        p = p.replace("{problem_statement}", topic["problem_statement"]).strip()
    return p

def prompt_subquestion_gen_reranking_detailed(
        user_background, problem_statement, title,
        k=10):

    template = """
    Instruction: Given the following report request, write {NUM} diverse and non-repeating sub-questions that can help guide the creation of a focused and comprehensive report. The sub-questions should help break down the topic into key areas that need to be investigated or explained. Each sub-question should be detailed but should focus on a single aspect or dimension of the report.
    
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
    #joined_subquestions = '\n'.join(prev_subquestions)
    template = template.replace("{NUM}", str(k))
    #template = template.replace("{prev_subquestions}", joined_subquestions)
    template = template.replace("{background}", user_background)
    template = template.replace("{title}", title)
    template = template.replace("{problem_statement}", problem_statement)
    return template.strip()
