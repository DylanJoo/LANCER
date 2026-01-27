import uuid
import math
import argparse
import asyncio
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
import logging

# optional: suppress vLLM debug logs
logging.getLogger("vllm.engine.async_llm_engine").setLevel(logging.WARNING)

async def main():
    # 1. Initialize engine arguments
    args = AsyncEngineArgs(
        model='meta-llama/Llama-3.2-1B-Instruct',
        dtype='half',
        gpu_memory_utilization=0.9,
    )

    # 2. Create AsyncLLMEngine 
    engine = AsyncLLMEngine.from_engine_args(args)

    # 3. Define prompts and sampling params
    sampling_params = SamplingParams(temperature=0.7)
    request_ids = [str(uuid.uuid4()) for _ in range(3)]
    prompts = [
        "Explain the theory of relativity in simple terms.",
        "What are the key differences between Python and Java?",
        "Describe the process of photosynthesis.",
    ]

    # 4. Add requests asynchronously
    outputs = []
    for req_id, prompt in zip(request_ids, prompts):
        output = await engine.add_request(req_id, prompt, sampling_params)
        outputs.append(output)

    # 5. Collect responses asynchronously
    results = []
    for output in outputs:
        while True:
            response = await output.get()
            print(response)
            if response.finished:
                break

    return 0

# 6. Run the async main
if __name__ == "__main__":
    asyncio.run(main())

