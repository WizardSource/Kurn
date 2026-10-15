import sys; sys.path.insert(0,'/home/ubuntu/kw')
from xsexp import *
LONG=['/home/ubuntu/kw/prompts/long.txt']
PF={'GGML_KXS_PREFETCH':1,'GGML_KXS_SPEC_INFLIGHT':4,'GGML_KXS_PREFETCH_K':4}
# resident footprint without slabs after a ~1.8k-token prompt, by ubatch (MiB, measured in prefill_*_4096)
BASE={'gemma4':{128:1848,512:1994,2048:2466}, 'qwen3':{128:979,512:1069,2048:1385}}
mem=4096
for m in sys.argv[1:]:
    xs=lambda ub, env=PF: dict(model=m, mode='xs', mem_mb=mem, cache_mb=mem-BASE[m][ub]-250, ubatch=ub, env=env)
    experiment(f'prefill2_{m}_{mem}', [
        ('ram_ub512', dict(model=m, mode='ram', ubatch=512)),
        ('xs_ub128', xs(128)),
        ('xs_ub512', xs(512)),
        ('xs_ub2048', xs(2048)),
        ('xs_ub512_freq_all_ops', xs(512, {**PF, 'GGML_KXS_FREQ_MAX_TOKENS':1000000})),
    ], reps=3, prompts=LONG, n_gen=64)
