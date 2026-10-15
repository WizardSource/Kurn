import sys; sys.path.insert(0,'/home/ubuntu/kw')
from xsexp import *
LONG='/home/ubuntu/kw/prompts/long.txt'
PF={'GGML_KXS_PREFETCH':1,'GGML_KXS_SPEC_INFLIGHT':4,'GGML_KXS_PREFETCH_K':4}
for m in sys.argv[1:]:
    EX={'GGML_KURN_FA_MODE':'exact','GGML_KURN_FALLBACK':0}
    c=cache_for(m,4096)
    experiment(f'gates2_exact_{m}', [
        ('ram_exact_ub512_a', dict(model=m, mode='ram', env=EX, ubatch=512)),
        ('ram_exact_ub512_b', dict(model=m, mode='ram', env=EX, ubatch=512)),
        ('xs_exact_ub512', dict(model=m, mode='xs', mem_mb=4096, cache_mb=c, env=EX, ubatch=512)),
        ('xs_exact_ub512_pf', dict(model=m, mode='xs', mem_mb=4096, cache_mb=c, env={**EX, **PF}, ubatch=512)),
        ('xs_exact_ub512_ovl0', dict(model=m, mode='xs', mem_mb=4096, cache_mb=c, env={**EX,'GGML_KXS_OVERLAP':0}, ubatch=512)),
        ('xs_exact_2g_pf', dict(model=m, mode='xs', mem_mb=2048, cache_mb=cache_for(m, 2048), env={**EX, **PF}, ubatch=512)),
        ('xs_exact_ub512_gguf', dict(model=m, mode='xs', mem_mb=4096, cache_mb=c, env={**EX,'GGML_KXS_FROM_GGUF':1}, ubatch=512)),
        ('ram_exact_ub2048', dict(model=m, mode='ram', env=EX, ubatch=2048)),
        ('xs_exact_ub2048_pf', dict(model=m, mode='xs', mem_mb=None, cache_mb=c, env={**EX, **PF}, ubatch=2048)),
    ], reps=1, prompts=[LONG], n_gen=32)
