import torch
import json
import re
import os
import string
import time
import gc

def cleanup_vllm(llm):
    del llm.model
    del llm

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.distributed.destroy_process_group()

    return 'vllm engine has been cleaned up'

def is_ampere_gpu():
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        return major == 8
    return False
