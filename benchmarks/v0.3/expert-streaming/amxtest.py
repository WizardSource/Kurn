import sys; sys.path.insert(0,'/home/ubuntu/kw')
from xsexp import *
m='qwen3'; c=cache_for(m,8192); P5=['/home/ubuntu/kw/prompts/p5.txt']
experiment('amx_det_qwen3', [
  ('xs8g_amx', dict(model=m, mode='xs', mem_mb=8192, cache_mb=c, env={'GGML_KURN_VERBOSE':1})),
  ('xs8g_noamx', dict(model=m, mode='xs', mem_mb=8192, cache_mb=c, env={'GGML_KURN_AMX':0})),
], reps=6, prompts=P5)
experiment('amx_det_qwen3', [
  ('ram_noamx', dict(model=m, mode='ram', env={'GGML_KURN_AMX':0})),
  ('ram_amx', dict(model=m, mode='ram')),
], reps=2, prompts=P5)
