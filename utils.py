import torch.backends.cudnn as cudnn
import random
import numpy as np
import torch


def set_seed(seed):
    cudnn.deterministic = True
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)


import torch
import subprocess
import re


def get_gpu_with_max_free_memory():
    # Run nvidia-smi command to get memory usage info
    result = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=memory.free,memory.total', '--format=csv,nounits,noheader'])
    # Decode byte string and split into lines
    result = result.decode('utf-8').strip().split('\n')

    max_memory = 0
    max_memory_gpu = 0

    # Iterate over each line (each GPU)
    for idx, line in enumerate(result):
        free_memory, total_memory = map(int, line.split(','))
        # Check if this GPU has more free memory
        print("GPU", idx, "Free memory:", free_memory, "Total memory:", total_memory)
        if free_memory > max_memory:
            max_memory = free_memory
            max_memory_gpu = idx

    torch.cuda.empty_cache()

    return max_memory_gpu

import seaborn as sns
def adjust_plots(font_scale=1.4, linewidth=1.3):

    # sns.set_style('dark')
    sns.set_style('white')
    # make font bold
    sns.set_context('notebook', font_scale=font_scale, rc={'lines.linewidth': linewidth})
    sns.set_palette('bright')

    sns.despine()
