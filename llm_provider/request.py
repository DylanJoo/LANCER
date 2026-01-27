import os
import uuid
import math
import asyncio
import openai
from typing import List
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
        print(f"Unused kwargs: {kwargs}")
        self.model_name_or_path = model_name_or_path
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.logprobs = logprobs

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

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.yes_tokens = None
        self.no_tokens = None
        if logprobs is not None:
            self.set_classification()

    # TODO: Look into the tokeization mismatch/inconsistency issues
    def set_classification(self, 
        yes_strings=[' Yes', 'Yes', ' yes', 'yes', 'YES', ' YES'],
        no_strings=[' No', 'No', ' no', 'no', 'NO', ' NO'],
        id_strings=[chr(i) for i in range(65, 91)]
    ):
        self.yes_tokens = [self.tokenizer.tokenize(item)[0] for item in yes_strings]
        self.no_tokens = [self.tokenizer.tokenize(item)[0] for item in no_strings]
        self.id_tokens = [self.tokenizer.tokenize(item)[0] for item in id_strings]

        # also include the strings
        self.yes_tokens += yes_strings
        self.no_tokens += no_strings

    def generate(self, prompts, binary_probs=False, dist_logp=False) -> List:
        if isinstance(prompts, str):
            prompts = [prompts]
        
        return self.loop.run_until_complete(
                self._agenerate(prompts, 
                                use_binary_probs=binary_probs,
                                use_dist_probs=dist_logp)
                )

    async def _agenerate(self, prompts, use_binary_probs=False, use_dist_probs=False):
        request_ids = [str(uuid.uuid4()) for _ in prompts]

        # Use normal function and add run in thread
        ## NOTE: in serving mode, it will stop util hitting criteria

        def _get_output(prompt, use_binary_probs, use_dist_probs):
            response = self.client.completions.create(
                model=self.model_name_or_path,
                prompt=prompt,
                logprobs=self.logprobs,
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
            )

            if use_binary_probs:
                tok_logps = response.choices[0].logprobs.top_logprobs[0]  # this is strings
                yes_ = math.exp(max(
                    [-1e2] + [
                        logp for tok, logp in tok_logps.items() 
                        if tok in self.yes_tokens
                    ]
                ))
                no_ = math.exp(max(
                    [-1e2] + [
                        logp for tok, logp in tok_logps.items() 
                        if tok in self.no_tokens 
                    ]
                ))
                output = yes_ / (no_ + yes_)

            elif use_dist_probs:
                tok_logps = response.choices[0].logprobs.top_logprobs[0] # this is strings
                min_logprob = min([logp for logp in tok_logps.values()])
                output = [min_logprob for _ in self.id_tokens]
                for topk, logp in tok_logps.items():
                    decoded_token = topk.replace('[', '').replace(']', '')
                    if len(decoded_token)==1 and (65 <= ord(decoded_token) <= 90):
                        output[ord(decoded_token)-65] = max(logp, output[ord(decoded_token)-65])
            else:
                output = response.choices[0].text

            ## NOTE: check vllm's contrastined decoding.
            ## NOTE: see the attention.
            return output

        # Gather all the outputs
        outputs = await asyncio.gather(*[
            asyncio.to_thread(_get_output, prompt,
                use_binary_probs, 
                use_dist_probs) for prompt in prompts
        ])
        return list(outputs)
