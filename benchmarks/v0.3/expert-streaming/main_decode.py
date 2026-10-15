import sys; sys.path.insert(0,'/home/ubuntu/kw')
from xsexp import *
PF={'GGML_KXS_PREFETCH':1,'GGML_KXS_SPEC_INFLIGHT':4,'GGML_KXS_PREFETCH_K':4}
for m in sys.argv[1:]:
    experiment(f'ram_{m}', [
        ('kurn_ram', dict(model=m, mode='ram')),
        ('stock_ram_repack', dict(model=m, mode='ram', env={'GGML_KURN':0})),
        ('stock_ram_norepack', dict(model=m, mode='mmap')),
    ], reps=5)
    for mem in (2048, 4096, 8192):
        c=cache_for(m, mem)
        xs=lambda env: dict(model=m, mode='xs', mem_mb=mem, cache_mb=c, env=env)
        experiment(f'decode_{m}_{mem}', [
            ('mmap', dict(model=m, mode='mmap', mem_mb=mem, n_gen=32 if mem==2048 else 64)),
            ('s2_cache', xs({'GGML_KXS_OVERLAP':0})),
            ('s3_overlap', xs({})),
            ('s4_prefetch', xs(PF)),
        ], reps=5)
