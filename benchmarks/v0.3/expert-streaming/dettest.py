import sys; sys.path.insert(0,'/home/ubuntu/kw')
from xsexp import *
m='qwen3'; c=cache_for(m,8192); P5=['/home/ubuntu/kw/prompts/p5.txt']
experiment('det2_qwen3', [
  ('xs8g_exact', dict(model=m, mode='xs', mem_mb=8192, cache_mb=c, env={'GGML_KURN_FA_MODE':'exact'})),
  ('xs8g_ggmlfa_noamx', dict(model=m, mode='xs', mem_mb=8192, cache_mb=c, env={'GGML_KURN_AMX':0,'GGML_KURN_FA':0})),
], reps=6, prompts=P5)
experiment('det2_qwen3', [
  ('ram_exact', dict(model=m, mode='ram', env={'GGML_KURN_FA_MODE':'exact'})),
  ('ram_ggmlfa_noamx', dict(model=m, mode='ram', env={'GGML_KURN_AMX':0,'GGML_KURN_FA':0})),
], reps=2, prompts=P5)
