"""Predictor side for whole GGUFs: per-tensor imatrix-weighted relative error vs the BF16 reference
(the `kurn mix` objective), summed per tensor kind, for any set of quantized GGUFs. Joined with the
measured KLD (ppl.csv) it validates the predictor and fits per-kind weights.
    mix_err.py OUT.csv GGUF...        (untimed; streams one tensor at a time)"""

import csv
import os
import sys

import numpy as np

from kurn import mixed

M = os.path.expanduser("~/models")
ref = mixed.tensors(f"{M}/Qwen3-1.7B-BF16.gguf")
imx = mixed.load_imatrix(f"{M}/compress/qwen3-1.7b.imatrix.gguf")
KINDS = ["token_embd", "attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down"]
out, paths = sys.argv[1], sys.argv[2:]
done = set()
if os.path.exists(out):
    done = {r["model"] for r in csv.DictReader(open(out))}
new = not os.path.exists(out)
with open(out, "a", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["model", "bpw", "sum_err"] + KINDS)
    if new:
        w.writeheader()
    for p in paths:
        model = os.path.basename(p)[: -len(".gguf")]
        if model in done:
            continue
        qt = mixed.tensors(p)
        acc = dict.fromkeys(KINDS, 0.0)
        for n, t in ref.items():
            if not mixed.is_matrix(n, t):
                continue
            h = imx.get(n)
            h = np.ones(mixed.n_cols(t)) if h is None else h.astype(np.float64)
            num = den = 0.0
            for sl in mixed.row_chunks(t):
                W = mixed.dequant(t, sl).astype(np.float64)
                D = mixed.dequant(qt[n], sl).astype(np.float64) - W
                num += float(((D * D) @ h).sum())
                den += float(((W * W) @ h).sum())
            acc[mixed.kind(n)] += num / den
        row = {"model": model, "bpw": round(mixed.gguf_bpw(p), 4), "sum_err": sum(acc.values()), **acc}
        w.writerow(row)
        fh.flush()
        print(row, flush=True)
