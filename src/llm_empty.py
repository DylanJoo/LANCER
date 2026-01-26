class LLM:
    def __init__(self, **kwargs):
        self.empty = None

    async def _agenerate(self, prompts, **kwargs):
        return [5] * len(prompts)
