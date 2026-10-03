"""Quality side of LoRC: per-tensor imatrix-weighted relative error of Q(W) + U V vs rank, against
the error of simply using the next format up (same metric as `kurn mix profile`):
    lowrank_curve.py OUT.csv TENSOR... --base TYPE=GGUF [--base ...] [--e8p] [--ranks 0,8,16,32,64,128]
--e8p adds the RHT+E8P (alpha 0.5) residual as a base."""

import argparse
import csv
import os

import numpy as np

from kurn import codebook as cb
from kurn import lowrank as lr
from kurn import mixed

M = os.path.expanduser("~/models")
ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("tensors", nargs="+")
ap.add_argument("--ref", default=f"{M}/Qwen3-1.7B-BF16.gguf")
ap.add_argument("--imatrix", default=f"{M}/compress/qwen3-1.7b.imatrix.gguf")
ap.add_argument("--base", action="append", default=[])
ap.add_argument("--e8p", action="store_true")
ap.add_argument("--ranks", default="0,8,16,32,64,128")
a = ap.parse_args()
ranks = [int(r) for r in a.ranks.split(",")]
imx = mixed.load_imatrix(a.imatrix)
ref = mixed.tensors(a.ref)
bases = {q: mixed.tensors(p) for q, p in (s.split("=", 1) for s in a.base)}
rows = []
for name in a.tensors:
    W = mixed.dequant(ref[name]).astype(np.float64)
    h = imx.get(name)
    cand = {f"{q}({mixed.type_name(t[name])})": mixed.dequant(t[name]).astype(np.float64) for q, t in bases.items()}
    if a.e8p:
        cand["e8p_a0.5"] = cb.quantize_tensor(W, h, 0.5)[0]
    for q, Wq in cand.items():
        spec = lr.residual_spectrum(W, Wq, h)
        for r, bpw, e in lr.rank_curve(W, Wq, h, ranks):
            rows.append({"tensor": name, "base": q, "rank": r, "extra_bpw": bpw, "err": e,
                         "energy_captured": float(spec[r - 1]) if r else 0.0})  # fmt: skip
            print(f"{name:28s} {q:18s} r={r:4d} +{bpw:.3f} bpw  err={e:.5f}  captured={rows[-1]['energy_captured']:.3f}", flush=True)
with open(a.out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
