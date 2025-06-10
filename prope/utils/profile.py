import time
from typing import Callable

import torch


def timeit(repeats: int, f: Callable, *args, **kwargs) -> float:
    torch.cuda.reset_peak_memory_stats()
    mem_tic = torch.cuda.max_memory_allocated() / 1024**3

    for _ in range(5):  # warmup
        f(*args, **kwargs)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(repeats):
        results = f(*args, **kwargs)
    torch.cuda.synchronize()
    end = time.time()

    mem = torch.cuda.max_memory_allocated() / 1024**3 - mem_tic
    return (end - start) / repeats, mem, results
