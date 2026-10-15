import sys; sys.path.insert(0,'/home/ubuntu/kw')
from xsexp import *
m='gemma4'; c=cache_for(m,4096)
cfg=lambda env: dict(model=m, mode='xs', mem_mb=4096, cache_mb=c, env=env)
p=experiment('tune_io_gemma4', [
 ('t4_c1024', cfg({'GGML_KXS_IO_THREADS':4,'GGML_KXS_IO_CHUNK_KB':1024})),
 ('t8_c1024', cfg({'GGML_KXS_IO_THREADS':8,'GGML_KXS_IO_CHUNK_KB':1024})),
 ('t8_c512',  cfg({'GGML_KXS_IO_THREADS':8,'GGML_KXS_IO_CHUNK_KB':512})),
 ('t2_c4096', cfg({'GGML_KXS_IO_THREADS':2,'GGML_KXS_IO_CHUNK_KB':4096})),
 ('t16_c256', cfg({'GGML_KXS_IO_THREADS':16,'GGML_KXS_IO_CHUNK_KB':256})),
], reps=5, n_gen=64)
print_summary(p, base='t4_c1024')
