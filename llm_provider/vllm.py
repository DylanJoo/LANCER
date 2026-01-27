# TODO: setup different style of token `id` categories
# TODO: in addition to that, also allow use to save token's `term` or `id`.
import math
import argparse
import asyncio
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine, AsyncStream
from vllm.sampling_params import SamplingParams
from transformers import AutoTokenizer
import uuid
import re
from typing import List
import logging
logger = logging.getLogger("vllm.engine.async_llm_engine").setLevel(logging.WARNING)

class LLM:

    def __init__(
        self,
        model_name_or_path: str = 'meta-llama/Llama-3.2-1B-Instruct',
        temperature=0.0,
        top_p=1.0,
        logprobs=None,
        max_tokens=128,
        dtype='half',
        gpu_memory_utilization=0.9,
        num_gpus=1, 
        max_model_len=10240,
        **kwargs
    ):
        print(f"Unused kwargs: {kwargs}")
        """
        # AMPERE GPU: dtype='float16', enable_prefix_caching=True
        # VOLTA GPU: dtype='float32', enable_prefix_caching=True
        """
        args = AsyncEngineArgs(
            model=model_name_or_path,
            dtype=dtype,
            tensor_parallel_size=num_gpus,
            gpu_memory_utilization=gpu_memory_utilization,
            enable_prefix_caching=True if dtype == 'float32' else False,
            max_model_len=max_model_len,
        )
        self.model = AsyncLLMEngine.from_engine_args(AsyncEngineArgs.from_cli_args(args))

        self.sampling_params = SamplingParams(
            temperature=temperature, 
            top_p=top_p,
            logprobs=logprobs,
            skip_special_tokens=False,
            min_tokens=1,
            max_tokens=max_tokens,
        )
        try:
            self.loop = asyncio.get_running_loop()
        except RuntimeError:
            # there is no actively running loop
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.yes_tokens = None
        self.no_tokens = None

    # TODO: Set the id_tokens as dynamic based on window size
    def set_classification(self, 
        yes_strings=[' Yes', 'Yes', ' yes', 'yes', 'YES', ' YES'],
        no_strings=[' No', 'No', ' no', 'no', 'NO', ' NO'],
        id_strings=[chr(i) for i in range(65, 91)]
    ):
        self.yes_tokens = [self.tokenizer.encode(item, add_special_tokens=False)[0] for item in yes_strings]
        self.no_tokens = [self.tokenizer.encode(item, add_special_tokens=False)[0] for item in no_strings]
        self.id_tokens = [self.tokenizer.encode(item, add_special_tokens=False)[0] for item in id_strings]
        print(f"YES TOKENS: {self.yes_tokens}")
        print(f"NO TOKENS: {self.no_tokens}")
        print(f"ID TOKENS: {self.id_tokens}")

    def generate(self, prompts, binary_probs=False, dist_logp=False) -> List:
        if isinstance(prompts, str):
            prompts = [prompts]

        return self.loop.run_until_complete(
                self._agenerate(prompts, 
                                use_binary_probs=binary_probs,
                                use_dist_probs=dist_logp)
                )

    async def _iterate_over_output(self, output_iterator: AsyncStream, use_binary_probs=False, use_dist_probs=False) -> str:

        async for output in output_iterator:
            if use_binary_probs:
                tok_item = output.outputs[0].logprobs[0]
                yes_ = math.exp(max(
                    [-1e2] + [
                        item.logprob for tok, item in tok_item.items() 
                        if tok in self.yes_tokens
                    ]
                ))
                no_ = math.exp(max(
                    [-1e2] + [
                        item.logprob for tok, item in tok_item.items() 
                        if tok in self.no_tokens 
                    ]
                ))
                output = score = yes_ / (no_ + yes_)

            # NOTE: the transformation is a bit hacky.
            # NOTE: make sure the numeric identifiers can also work
            elif use_dist_probs:
                tok_item = output.outputs[0].logprobs[0]
                min_logprob = min([item.logprob for item in tok_item.values()])
                output = [min_logprob for _ in self.id_tokens]
                for topk, item in tok_item.items():
                    decoded_token = item.decoded_token.replace('[', '').replace(']', '')
                    if len(decoded_token)==1 and (65 <= ord(decoded_token) <= 90):
                        output[ord(decoded_token)-65] = max(item.logprob, output[ord(decoded_token)-65])
            else:
                output = last_text = output.outputs[0].text
        return output

    async def _agenerate(self, prompts, **kwargs):
        request_ids = [str(uuid.uuid4()) for _ in prompts]

        # Add requests to the engine
        output_iterators = [
            await self.model.add_request(request_id, prompt, self.sampling_params)
            for request_id, prompt in zip(request_ids, prompts)
        ]

        # Gather all the outputs
        outputs = await asyncio.gather(*[
            self._iterate_over_output(output_iterator, **kwargs)
            for output_iterator in output_iterators
        ])
        return list(outputs)
