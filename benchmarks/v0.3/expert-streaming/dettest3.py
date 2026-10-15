import sys; sys.path.insert(0,'/home/ubuntu/kw')
from xsexp import *
m='qwen3'; c=cache_for(m,8192); P5=['/home/ubuntu/kw/prompts/p5.txt']
experiment('det3_qwen3', [
  ('xs8g_fb0', dict(model=m, mode='xs', mem_mb=8192, cache_mb=c, env={'GGML_KURN_FALLBACK':0})),
  ('xs8g_default', dict(model=m, mode='xs', mem_mb=8192, cache_mb=c, env={})),
], reps=8, prompts=P5)
experiment('det3_qwen3', [
  ('ram_fb0', dict(model=m, mode='ram', env={'GGML_KURN_FALLBACK':0})),
], reps=3, prompts=P5)
