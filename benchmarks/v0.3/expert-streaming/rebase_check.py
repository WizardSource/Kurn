import sys; sys.path.insert(0,'/home/ubuntu/kw')
from xsexp import *
LONG=['/home/ubuntu/kw/prompts/long.txt']
PF={'GGML_KXS_PREFETCH':1,'GGML_KXS_SPEC_INFLIGHT':4,'GGML_KXS_PREFETCH_K':4}
EX={'GGML_KURN_FA_MODE':'exact','GGML_KURN_FALLBACK':0}
experiment('rebase_exact_qwen3', [
    ('ram_exact', dict(model='qwen3', mode='ram', env=EX)),
    ('xs_exact_pf', dict(model='qwen3', mode='xs', mem_mb=4096, cache_mb=cache_for('qwen3',4096)-150, env={**EX, **PF})),
], reps=1, prompts=LONG, n_gen=32)
experiment('rebase_fast_gemma4', [
    ('ram', dict(model='gemma4', mode='ram')),
    ('xs_s3', dict(model='gemma4', mode='xs', mem_mb=4096, cache_mb=cache_for('gemma4',4096), env={})),
    ('xs_s4', dict(model='gemma4', mode='xs', mem_mb=4096, cache_mb=cache_for('gemma4',4096), env=PF)),
], reps=3)
