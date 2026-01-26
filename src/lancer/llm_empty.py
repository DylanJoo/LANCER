import random

class LLM:
    def __init__(self, **kwargs):
        self.empty = None

    def generate_ratings(self, prompts, **kwargs):
        return [str(random.randint(1, 5)) for _ in prompts]
        # return ['5'] * len(prompts)

    def generate_questions(self, prompts, **kwargs):
        return ['What is the meaning of life?\nWhat is the capital of France?\n What is 2 + 2?'] * len(prompts)
