import sys; sys.path.insert(0,'/home/ubuntu/kw')
from xsexp import *
m=sys.argv[1]; c=cache_for(m,4096)
cfg=lambda env: dict(model=m, mode='xs', mem_mb=4096, cache_mb=c, env=env)
P={'GGML_KXS_PREFETCH':1}
p=experiment(f'tune_pf_{m}', [
 ('nopf', cfg({})),
 ('k8_i1', cfg({**P})),
 ('k8_i4', cfg({**P,'GGML_KXS_SPEC_INFLIGHT':4})),
 ('k4_i4', cfg({**P,'GGML_KXS_SPEC_INFLIGHT':4,'GGML_KXS_PREFETCH_K':4})),
 ('k8_i4_f1', cfg({**P,'GGML_KXS_SPEC_INFLIGHT':4,'GGML_KXS_SPEC_MAX_FREQ':1})),
 ('k6_i2_f2', cfg({**P,'GGML_KXS_SPEC_INFLIGHT':2,'GGML_KXS_PREFETCH_K':6,'GGML_KXS_SPEC_MAX_FREQ':2})),
], reps=5, n_gen=64)
print_summary(p, base='nopf')
