import sys; sys.path.insert(0,'/home/ubuntu/kw')
from xsexp import *
LONG=['/home/ubuntu/kw/prompts/long.txt']
PF={'GGML_KXS_PREFETCH':1,'GGML_KXS_SPEC_INFLIGHT':4,'GGML_KXS_PREFETCH_K':4}
FB={'GGML_KURN_FALLBACK':0}
OLD={'GGML_KXS_FREQ_MAX_TOKENS':1000000}
m='qwen3'; c=cache_for(m,4096)
xs=lambda env: dict(model=m, mode='xs', mem_mb=4096, cache_mb=c, env=env)
experiment('fastgate_qwen3', [
    ('ram_fb0', dict(model=m, mode='ram', env=FB)),
    ('s2_fb0', xs({**FB,'GGML_KXS_OVERLAP':0})),
    ('s3_fb0', xs({**FB})),
    ('s4_fb0', xs({**FB, **PF})),
], reps=5)
for m in ('gemma4','qwen3'):
    c=cache_for(m,4096)
    xs=lambda env: dict(model=m, mode='xs', mem_mb=4096, cache_mb=c, env=env)
    experiment(f'lfufix_{m}', [
        ('s4_freq_decode_only', xs({**PF})),
        ('s4_freq_all_ops', xs({**PF, **OLD})),
    ], reps=6)
    experiment(f'lfufix_long_{m}', [
        ('s4_freq_decode_only', xs({**PF})),
        ('s4_freq_all_ops', xs({**PF, **OLD})),
    ], reps=3, prompts=LONG, n_gen=64)
