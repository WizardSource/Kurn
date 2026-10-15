"""sizes.py FILE.gguf ...: total size (model + sidecar) and bits per weight over the linear layers (main + side)"""
import os, re, sys
from gguf import GGUFReader

LIN = re.compile(r"blk\.\d+\.(attn_q|attn_k|attn_v|attn_output|ffn_gate|ffn_up|ffn_down)\.weight$")
for f in sys.argv[1:]:
    r = GGUFReader(f)
    lb = sum(int(t.n_bytes) for t in r.tensors if LIN.match(t.name))
    ln = sum(int(t.n_elements) for t in r.tensors if LIN.match(t.name))
    sb = 0
    sc = f + ".oa.gguf"
    if os.path.exists(sc):
        sb = sum(int(t.n_bytes) for t in GGUFReader(sc).tensors)
    tot = os.path.getsize(f) + (os.path.getsize(sc) if os.path.exists(sc) else 0)
    print(f"{os.path.basename(f)}\t{tot / 2**20:.1f} MiB\tlinear {8 * (lb + sb) / ln:.3f} bpw (main {8 * lb / ln:.3f}, side {8 * sb / ln:.3f})")
