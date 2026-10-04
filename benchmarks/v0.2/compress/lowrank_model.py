"""Whole-model LoRC: copy a quantized GGUF and replace the selected tensors by Q(W) + U V (int8
factors of the imatrix-weighted residual vs the BF16 reference), stored as F16 so llama-perplexity
can evaluate it. The F16 storage is only an evaluation vehicle; the reported bpw counts the base
format plus int8 factors (lowrank.extra_bpw).
    lowrank_model.py OUT.gguf --base Q.gguf --rank 16 --kinds attn_k,attn_v [--stats OUT.json]"""

import argparse
import json
import os

import numpy as np

from kurn import lowrank as lr
from kurn import mixed

M = os.path.expanduser("~/models")
ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--base", required=True)
ap.add_argument("--ref", default=f"{M}/Qwen3-1.7B-BF16.gguf")
ap.add_argument("--imatrix", default=f"{M}/compress/qwen3-1.7b.imatrix.gguf")
ap.add_argument("--rank", type=int, default=16)
ap.add_argument("--kinds", default="attn_k,attn_v")
ap.add_argument("--stats")
a = ap.parse_args()
kinds = set(a.kinds.split(","))
imx = mixed.load_imatrix(a.imatrix)
ref = mixed.tensors(a.ref)
base = mixed.tensors(a.base)
stats = {}


def corrected(name):
    def f():
        W = mixed.dequant(ref[name]).astype(np.float64)
        Wq = mixed.dequant(base[name]).astype(np.float64)
        h = imx.get(name)
        What, _ = lr.lorc(W, Wq, h, a.rank)
        e0, e1 = lr.rel_err(W, Wq, h), lr.rel_err(W, What, h)
        stats[name] = {"err_base": e0, "err_lorc": e1, "extra_bpw": lr.extra_bpw(*W.shape, a.rank)}
        print(f"{name:28s} {mixed.type_name(base[name]):6s} r={a.rank} err {e0:.5f} -> {e1:.5f}", flush=True)
        return What.astype(np.float32)

    return f


sel = [n for n, t in base.items() if mixed.is_matrix(n, t) and mixed.kind(n) in kinds]
mixed.write_gguf(a.base, a.out, {n: corrected(n) for n in sel})
bits = n_w = extra = 0.0
for n, t in base.items():
    if not mixed.is_matrix(n, t):
        continue
    w = int(t.n_elements)
    n_w += w
    bits += 8 * int(t.n_bytes)
    if n in stats:
        extra += stats[n]["extra_bpw"] * w
stats["_total"] = {"base_bpw": bits / n_w, "bpw": (bits + extra) / n_w, "rank": a.rank, "kinds": sorted(kinds)}
print(json.dumps(stats["_total"]))
if a.stats:
    with open(a.stats, "w") as fh:
        json.dump(stats, fh, indent=1)
