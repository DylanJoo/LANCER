###################################################
# prompt for statement generation # (deprecated)  #
###################################################
template_statement = "Instruction: {INST}\n\nDocument: {D}\n\n{PREFIX}"
instruction_statement = "Write 10 diverse statements that are evident from the given document. Each statement should be standalone and have necessary context. Every statements should start with the format of a square bracket with number (e.g., [1])."
def prompt_statement_gen(INST="", D="", PREFIX="Statements:\n[1]: "):
    p = template_statement
    p = p.replace("{INST}", INST).strip()
    p = p.replace("{D}", D)
    p = p.replace("{PREFIX}", PREFIX).strip()
    return p

###################################
# prompt for question generation  #
###################################
# [this is for 2round generation] = "Write corrsponding questions for each of the given list of statements. The origianl context is also provided as reference. Each question should contain the necessary context if needed. Ensure the questions follow the same order of statements. Every questions should start with the format of a square bracket with number (e.g., [1]), like the statements."
template_question = "Instruction: {INST}\n\nDocument: {D}\n\n{PREFIX}"
instruction_question = "Write 10 diverse questions that can reveal the information contained in the given document. Each question should be self-contained and have necessary context. Write the question within `<q>` and `</q>` tags."
def prompt_question_gen(INST="", D="", PREFIX="Questions:\n<q>"):
    p = template_question
    p = p.replace("{INST}", INST).strip()
    p = p.replace("{D}", D)
    p = p.replace("{PREFIX}", PREFIX).strip()
    return p

####################################
# prompt for paragraphs generation #
####################################
template_passage = "Instruction: {INST}\n\nDocument: {D}\n\n{PREFIX}"
# instruction_summary = "Break down the given document into 2 to 3 standalone summaries that have similar length. Each summaries should be a self-contained passage with necessary context. Keep the wording in summaries as similar as possible to the original document. Each summaries should start with the format of a square bracket with number (e.g., [1])."
instruction_passage = "Break down the given document into 2-3 standalone passages of approximately 200 words each, providing essential context and information. Use similar wording and phrasing as the original document. Write each passages within `<p>` and `</p>` tags."
# instruction_summary = "Write two standalone summaries for the given document."
def prompt_passage_gen(INST="", D="", PREFIX="Passages:\n<p>"):
    p = template_passage
    p = p.replace("{INST}", INST).strip()
    p = p.replace("{D}", D)
    p = p.replace("{PREFIX}", PREFIX).strip()
    return p

################################
# prompt for rating generation #
################################
guideline = "- 5: The context is highly relevant, complete, and accurate.\n- 4: The context is mostly relevant and complete but may have minor gaps or inaccuracies.\n- 3: The context is partially relevant and complete, with noticeable gaps or inaccuracies.\n- 2: The context has limited relevance and completeness, with significant gaps or inaccuracies.\n- 1: The context is minimally relevant or complete, with substantial shortcomings.\n- 0: The context is not relevant or complete at all."
# - 5: The context is highly relevant, complete, and accurate.
# - 4: The context is mostly relevant and complete but may have minor gaps or inaccuracies.
# - 3: The context is partially relevant and complete, with noticeable gaps or inaccuracies.
# - 2: The context has limited relevance and completeness, with significant gaps or inaccuracies.
# - 1: The context is minimally relevant or complete, with substantial shortcomings.
# - 0: The context is not relevant or complet at all.
template_rating = "Instruction: {INST}\n\nGuideline:\n{G}\n\nQuestion: {Q}\n\nContext: {C}\n\n{PREFIX}" 
instruction_rating = "Determine whether the question can be answered based on the provided context? The question is asked in English, but the context is provided in Chinese. Rate the context with on a scale from 0 to 5 according to the guideline below. Do not write anything except the rating."
def prompt_rating_gen(INST="", Q="", C="", PREFIX="Rating:"):
    p = template_rating
    p = p.replace("{G}", guideline)
    p = p.replace("{INST}", INST).strip()
    p = p.replace("{Q}", Q)
    p = p.replace("{C}", C)
    p = p.replace("{PREFIX}", PREFIX).strip()
    return p

# Below are task-aware query for different task
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

# example_doc = "National Archives\nYes, it’s that time again, folks. It’s the first Friday of the month, when for one ever-so-brief moment the interests of Wall Street, Washington and Main Street are all aligned on one thing: Jobs. \n \n A fresh update on the U.S. employment situation for January hits the wires at 8:30 a.m. New York time offering one of the most important snapshots on how the economy fared during the previous month. Expectations are for 203,000 new jobs to be created, according to economists polled by Dow Jones Newswires, compared to 227,000 jobs added in February. The unemployment rate is expected to hold steady at 8.3%. \n \n Here at MarketBeat HQ, we’ll be offering color commentary before and after the data crosses the wires. Feel free to weigh-in yourself, via the comments section. And while you’re here, why don’t you sign up to follow us on Twitter. \n \n Enjoy the show."

# instruction_summary = "Instruction: Write an accurate, engaging, and concise answer for the given question using only the provided documents as references (some of which might be irrelevant). Always cite at least one document for every sentences in the answer. Use the citation format of square brackets to indicate the cited documents (e.g., [1] for the first reference). If multiple documents support the sentence, only cite a minimum sufficient subset of the documents. Only generate the answer, excluding any disclaimers, notes or list of references at the end of the answer."


################################
# prompt for query generator #
################################
INST = """Write ten google-like search keywords or questions for search engines based on the user request to help generate relevant information.
Each aspect of the user request should yield separate queries.
Look for possible details, such as specific numbers, causes, methodolgies, conflicting information, etc., if applicable.
They may not need to be grammatically correct sentences.
They will be used to query a semantic search engine.
Return the ten queries with one query in each line WITHOUT numbering.
"""
template_query_generator = "Instruction: {INST}\n\nQuestion: {Q}\n\n{PREFIX}" 
def prompt_query_generator(INST=INST, Q="", C="", PREFIX="Search keywords or questions:"):
    p = template_query_generator
    p = p.replace("{INST}", INST).strip()
    p = p.replace("{Q}", Q)
    p = p.replace("{PREFIX}", PREFIX).strip()
    return p
