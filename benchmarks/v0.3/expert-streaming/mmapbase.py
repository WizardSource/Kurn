import sys; sys.path.insert(0,'/home/ubuntu/kw')
from xsexp import *
for m in ('gemma4','qwen3'):
    experiment(f'mmap_n1_{m}', [(f'mmap_{mem}_n1', dict(model=m, mode='mmap', mem_mb=mem, n_gen=1)) for mem in (2048,4096,8192)]
               + [(f'xs_{mem}_n1', dict(model=m, mode='xs', mem_mb=mem, cache_mb=cache_for(m,mem), n_gen=1)) for mem in (2048,4096,8192)], reps=2)
