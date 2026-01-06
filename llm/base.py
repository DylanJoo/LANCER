from vllm import LLM as v_llm 
from vllm import SamplingParams

class LLM:

    def __init__(self, args):
        self.model = v_llm(
            args.model, 
            dtype='bfloat16',
            enforce_eager=True,
            pipeline_parallel_size=(args.num_gpus or 1)
        )
        self.sampling_params = SamplingParams(
            temperature=args.temperature, 
            top_p=args.top_p,
            skip_special_tokens=False
        )

    def generate(self, x, **kwargs):
        if isinstance(prompts, str):
            prompts = [prompts]

        self.sampling_params.max_tokens = kwargs.pop('max_tokens', 256)
        self.sampling_params.min_tokens = kwargs.pop('min_tokens', 32)

        output = self.model.generate(x, self.sampling_params)
        if len(output) == 1:
            return [output[0].outputs[0].text]
        else:
            return [o.outputs[0].text for o in output] 

