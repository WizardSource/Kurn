import sys; sys.path.insert(0,'/home/ubuntu/kw')
from xsexp import *
for m in sys.argv[1:]:
    c=cache_for(m,4096)
    xs=lambda env: dict(model=m, mode='xs', mem_mb=4096, cache_mb=c, env=env)
    experiment(f'ablate_{m}', [
        ('s3_lfu', xs({})),
        ('s3_lru', xs({'GGML_KXS_POLICY':'lru'})),
        ('s3_from_gguf', xs({'GGML_KXS_FROM_GGUF':1})),
    ], reps=5)
