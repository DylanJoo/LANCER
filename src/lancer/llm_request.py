import os
import uuid
import asyncio
import openai
from typing import List, Union
from transformers import AutoTokenizer

class LLM:

    def __init__(
        self,
        api_key: str = 'EMPTY',
        base_url: str = 'http://localhost:8000/v1',
        model_name_or_path: str = 'meta-llama/Llama-3.2-1B-Instruct',
        temperature=0.0,
        top_p=1.0,
        logprobs=20,
        max_tokens=10,
        gpu_memory_utilization=0.9,
        **kwargs
    ):
        self.model_name_or_path = model_name_or_path
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.logprobs = logprobs
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)

        self.client = openai.OpenAI(
            api_key=os.environ.get('OPENAI_API_KEY', api_key),
            base_url=base_url,
            max_retries=10
        )

        try:
            self.loop = asyncio.get_event_loop()
        except RuntimeError:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)

    def generate_questions(self, prompts):
        return self.generate(prompts)

    def generate_ratings(self, prompts):
        return self.generate(prompts)

    ## NOTE: see if we want to repalce the judgment with autollmreranker
    def generate(
        self, 
        user_prompts: Union[List[str]] = None,
        system_prompt: str = "You are a helpful, honest, and harmless assistant."
    ):

        if isinstance(user_prompts, str):
            user_prompts = [user_prompts]

        prompts = [self.tokenizer.apply_chat_template(
            conversation=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            tokenize=False,
            add_generation_prompt=True
        ) for prompt in user_prompts]
        
        return self.loop.run_until_complete(
                self._agenerate(prompts, use_binary_probs=False, use_dist_probs=False)
        )

    async def _agenerate(self, prompts, use_binary_probs=False, use_dist_probs=False):
        request_ids = [str(uuid.uuid4()) for _ in prompts]

        def _get_output(prompt, use_binary_probs, use_dist_probs):
            response = self.client.completions.create(
                model=self.model_name_or_path,
                prompt=prompt,
                logprobs=self.logprobs,
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
            )

            # TODO: Integrate the autollmreranker here
            output = response.choices[0].text

            return output

        # Gather all the outputs
        outputs = await asyncio.gather(*[
            asyncio.to_thread(_get_output, prompt,
                use_binary_probs, 
                use_dist_probs) for prompt in prompts
        ])
        return list(outputs)
