"""Dump Q8_0 tensors + a global 12-bit Huffman code for entropy_bench.c, build it, run it:
    entropy_bench.py dump OUT.bin [--gguf Q8_0.gguf] [--kinds ffn_down,attn_q,...] [--layers N] [--streams 4]
    entropy_bench.py run OUT.bin [--threads 1,2,4,8] [--secs 1] [--hot]
`run` is timed: call it through benchlock.sh."""

import argparse
import os
import subprocess
import sys

import numpy as np

from kurn import entropy as en
from kurn import mixed

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("cmd", choices=["dump", "run"])
ap.add_argument("out")
ap.add_argument("--gguf", default=os.path.expanduser("~/models/compress/qwen3-1.7b-q8_0.gguf"))
ap.add_argument("--kinds", default="attn_q,attn_k,attn_v,attn_output,ffn_gate,ffn_up,ffn_down")
ap.add_argument("--layers", type=int, default=28)
ap.add_argument("--streams", type=int, default=4)
ap.add_argument("--threads", default="1,2,4,8")
ap.add_argument("--secs", type=float, default=1.0)
ap.add_argument("--hot", action="store_true")
a = ap.parse_args()

exe = os.path.join(os.environ.get("KURN_CACHE_DIR", "/tmp"), "entropy_bench")
if a.cmd == "dump":
    ts = mixed.tensors(a.gguf)
    sel = [n for n, t in ts.items() if mixed.is_matrix(n, t) and mixed.layer(n) in range(a.layers) and mixed.kind(n) in a.kinds.split(",")
           and mixed.type_name(t) == "Q8_0"]  # fmt: skip
    h = np.zeros(256, dtype=np.int64)
    for n in sel:
        h += en.histogram(en.q8_0_split(np.asarray(ts[n].data))[1].view(np.uint8))
    lens = en.huffman_lengths(h, 12)
    codes = en.canonical_codes(lens)
    with open(a.out, "wb") as fh:
        fh.write(np.array([len(sel), a.streams], dtype=np.uint32).tobytes())
        fh.write(lens.astype(np.uint8).tobytes())
        fh.write(codes.astype(np.uint32).tobytes())
        for n in sel:
            t = ts[n]
            fh.write(np.array([mixed.n_rows(t), mixed.n_cols(t)], dtype=np.uint32).tobytes())
            fh.write(np.ascontiguousarray(t.data).tobytes())
    print(f"{a.out}: {len(sel)} tensors, H={en.entropy(h):.4f} huff12={en.avg_len(h, lens):.4f} bits/code")
else:
    src = os.path.join(HERE, "entropy_bench.c")
    if not os.path.exists(exe) or os.path.getmtime(exe) < os.path.getmtime(src):
        subprocess.check_call(["gcc", "-O3", "-march=native", "-pthread", "-o", exe, src, "-lm"])
    sys.stdout.flush()
    subprocess.check_call([exe, a.out, a.threads, str(a.secs)] + (["hot"] if a.hot else []))
