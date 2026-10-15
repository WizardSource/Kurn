import sys, json; sys.path.insert(0,'/home/ubuntu/kw')
from xsexp import *
LONG='/home/ubuntu/kw/prompts/long.txt'
PF={'GGML_KXS_PREFETCH':1,'GGML_KXS_SPEC_INFLIGHT':4,'GGML_KXS_PREFETCH_K':4}
CB=json.load(open('/home/ubuntu/kw/results/compute_buf.json'))   # MiB of compute buffer per model / ubatch
mem=int(sys.argv[1])
for m in sys.argv[2:]:
    c=cache_for(m, mem)
    def xs(ub):
        delta = CB[m][str(ub)] - CB[m]['512'] + (150 if ub == 2048 else 0)  # + kurn work buffer
        return dict(model=m, mode='xs', mem_mb=mem, cache_mb=max(1, int(c - delta)), ubatch=ub, env=PF)
    experiment(f'prefill_{m}_{mem}', [
        ('ram_ub512', dict(model=m, mode='ram', ubatch=512)),
        ('mmap_ub512', dict(model=m, mode='mmap', mem_mb=mem, ubatch=512)),
        ('xs_ub128', xs(128)),
        ('xs_ub512', xs(512)),
        ('xs_ub2048', xs(2048)),
    ], reps=3, prompts=[LONG], n_gen=16)
