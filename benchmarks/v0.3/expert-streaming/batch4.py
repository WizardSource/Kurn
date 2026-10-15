import sys; sys.path.insert(0,'/home/ubuntu/kw')
from xsexp import *
LONG=['/home/ubuntu/kw/prompts/long.txt']
PF={'GGML_KXS_PREFETCH':1,'GGML_KXS_SPEC_INFLIGHT':4,'GGML_KXS_PREFETCH_K':4}
experiment('long2g_gemma4', [
    ('xs_2g_ub128', dict(model='gemma4', mode='xs', mem_mb=2048, cache_mb=1, ubatch=128, env=PF)),
], reps=1, prompts=LONG, n_gen=32)
for m in ('gemma4','qwen3'):
    c=cache_for(m,4096)
    xs=lambda env: dict(model=m, mode='xs', mem_mb=4096, cache_mb=c, env=env, rbps=2e9)
    experiment(f'throttle2g_{m}_4096', [
        ('mmap', dict(model=m, mode='mmap', mem_mb=4096, n_gen=64, rbps=2e9)),
        ('s3_overlap', xs({})),
        ('s4_prefetch', xs(PF)),
    ], reps=3)
