"""Validate the `kurn mix` predictor against measured KLD and fit per-kind weights.
Joins mix_err.csv (per-kind sums of relative error per GGUF) with ppl.csv (KLD vs BF16):
  1. KLD ~ a * sum_err (equal weights, what `kurn mix plan` minimizes by default): R^2, rank corr.
  2. KLD ~ sum_k w_k * S_k (non-negative least squares), leave-one-out error.
Prints the fitted weights as a `kurn mix plan --weights` argument.
    fit_predictor.py results/mix_err.csv results/ppl.csv [--exclude REGEX]"""

import argparse
import csv
import re

import numpy as np
from scipy.optimize import nnls
from scipy.stats import spearmanr

ap = argparse.ArgumentParser()
ap.add_argument("mix_err")
ap.add_argument("ppl")
ap.add_argument("--exclude", default="e8p|lorc", help="models not quantized with ggml types (F16 writebacks)")
a = ap.parse_args()
E = {r["model"]: r for r in csv.DictReader(open(a.mix_err))}
P = {r["model"]: r for r in csv.DictReader(open(a.ppl))}
kinds = [k for k in next(iter(E.values())) if k not in ("model", "bpw", "sum_err")]
names = [m for m in E if m in P and not re.search(a.exclude, m)]
X = np.array([[float(E[m][k]) for k in kinds] for m in names])
y = np.array([float(P[m]["kld"]) for m in names])
s = X.sum(1)
c = float(s @ y / (s @ s))
r2 = 1 - ((y - c * s) ** 2).sum() / ((y - y.mean()) ** 2).sum()
print(f"{len(names)} models. equal weights: KLD = {c:.4f} * sum_err, R^2 {r2:.3f}, spearman {spearmanr(s, y)[0]:.3f}")
w, _ = nnls(X, y)
fit = X @ w
r2w = 1 - ((y - fit) ** 2).sum() / ((y - y.mean()) ** 2).sum()
loo_eq, loo_w = [], []
for i in range(len(names)):
    keep = np.arange(len(names)) != i
    ci = float(s[keep] @ y[keep] / (s[keep] @ s[keep]))
    wi, _ = nnls(X[keep], y[keep])
    loo_eq.append(abs(ci * s[i] - y[i]) / y[i])
    loo_w.append(abs(X[i] @ wi - y[i]) / y[i])
print(f"per-kind NNLS: R^2 {r2w:.3f}; leave-one-out mean |rel err| equal {np.mean(loo_eq):.1%} vs per-kind {np.mean(loo_w):.1%}")
for m, yi, f1, f2 in zip(names, y, c * s, fit):
    print(f"  {m:28s} KLD {yi:.4f}  equal {f1:.4f}  per-kind {f2:.4f}")
norm = np.mean([w[i] for i, k in enumerate(kinds) if k != "token_embd"]) or 1.0
print("weights (layer-kind mean = 1):", " ".join(f"{k}={w[i] / norm:.3g}" for i, k in enumerate(kinds)))
print("--weights " + ",".join(f"{k}={max(w[i] / norm, 1e-3):.3g}" for i, k in enumerate(kinds)))
