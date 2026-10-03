"""Per-tensor imatrix-weighted relative error of E8P variants (same metric as `kurn mix profile`):
    e8p_tensor_err.py TENSOR [TENSOR...] [--alpha A ...] [--quant TYPE=GGUF ...]
Variants: plain RHT+E8P; importance-scaled (columns x h^(alpha/2) before the rotation, so the
uniform-MSE quantizer minimises sum_j h_j^alpha dW_j^2); E8P RVQ x2 (4.1 bpw)."""

import argparse
import os
import time

import numpy as np

from kurn import codebook as cb
from kurn import mixed

ap = argparse.ArgumentParser()
ap.add_argument("tensors", nargs="+")
ap.add_argument("--ref", default=os.path.expanduser("~/models/Qwen3-1.7B-BF16.gguf"))
ap.add_argument("--imatrix", default=os.path.expanduser("~/models/compress/qwen3-1.7b.imatrix.gguf"))
ap.add_argument("--alpha", type=float, nargs="*", default=[0.5, 1.0])
ap.add_argument("--quant", action="append", default=[])
ap.add_argument("--rows", type=int, default=0, help="only the first ROWS rows (0: all)")
a = ap.parse_args()

imx = mixed.load_imatrix(a.imatrix)
ts = mixed.tensors(a.ref)
qts = {q: mixed.tensors(p) for q, p in (s.split("=", 1) for s in a.quant)}


def err(W, Wq, h):
    D = Wq - W
    return float(((D * D) @ h).sum() / ((W * W) @ h).sum())


for name in a.tensors:
    sl = slice(0, a.rows) if a.rows else None
    W = mixed.dequant(ts[name], sl).astype(np.float64)
    h = imx[name]
    res = {}
    t0 = time.time()
    d, codes, _ = cb.quantize_e8p(cb.rotate_weight(W))
    res["e8p"] = err(W, cb.unrotate_weight(cb.dequant_e8p(d, codes)), h)
    t_e8p = time.time() - t0
    for al in a.alpha:
        s = (h / h.mean()) ** (al / 2)
        d, codes, _ = cb.quantize_e8p(cb.rotate_weight(W * s))
        res[f"e8p_h^{al:g}"] = err(W, cb.unrotate_weight(cb.dequant_e8p(d, codes)) / s, h)
    _, What = cb.quantize_e8p_rvq(cb.rotate_weight(W), 2)
    res["e8p_rvq2"] = err(W, cb.unrotate_weight(What), h)
    for q, qt in qts.items():
        res[q] = err(W, mixed.dequant(qt[name], sl).astype(np.float64), h)
    print(f"{name:28s} {W.shape} e8p {t_e8p:.1f}s  " + "  ".join(f"{k}={v:.4f}" for k, v in res.items()), flush=True)
