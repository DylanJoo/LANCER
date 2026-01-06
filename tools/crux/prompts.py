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
instruction_rating = "Determine whether the question can be answered based on the provided context? Rate the context with on a scale from 0 to 5 according to the guideline below. Do not write anything except the rating."
def prompt_rating_gen(INST="", Q="", C="", PREFIX="Rating:"):
    p = template_rating
    p = p.replace("{G}", guideline)
    p = p.replace("{INST}", INST).strip()
    p = p.replace("{Q}", Q)
    p = p.replace("{C}", C)
    p = p.replace("{PREFIX}", PREFIX).strip()
    return p

