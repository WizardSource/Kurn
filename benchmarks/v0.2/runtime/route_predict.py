#!/usr/bin/env python3
"""How predictable is MoE expert selection? Reads route_trace's dump and the GGUF's router
weights; prints temporal reuse and cross-layer prediction hit rates (fraction of the
actually selected experts that a predictor names in advance).

    route_predict.py MODEL.gguf TRACE_DIR [--n-used 8] [--json out.json]

Predictors for layer l+1 at token t, all available before layer l+1's attention:
  reuse-1     experts layer l+1 used at token t-1
  reuse-4     union of layer l+1's experts over tokens t-4..t-1 (more bytes)
  pregate     top-k of router_{l+1}(rmsnorm(l_out_l) * ffn_norm_{l+1})   (residual after layer l)
  pregate-x   same, but prefetch the top 1.5k (12 of 8) to raise the hit rate
"""

import argparse
import glob
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf")
    ap.add_argument("trace")
    ap.add_argument("--n-used", type=int, default=8)
    ap.add_argument("--json")
    a = ap.parse_args()
    import gguf

    r = gguf.GGUFReader(a.gguf)
    W, NW = {}, {}
    for t in r.tensors:
        if t.name.endswith("ffn_gate_inp.weight"):
            W[int(t.name.split(".")[1])] = np.asarray(t.data, dtype=np.float32).reshape(-1, int(t.shape[0]))
        elif t.name.endswith("ffn_norm.weight"):
            NW[int(t.name.split(".")[1])] = np.asarray(t.data, dtype=np.float32)
    L = len(glob.glob(os.path.join(a.trace, "ffn_moe_topk-*.bin")))
    k = a.n_used
    topk = [np.fromfile(os.path.join(a.trace, f"ffn_moe_topk-{ly}.bin"), dtype=np.int32).reshape(-1, k) for ly in range(L)]
    T = topk[0].shape[0]
    n_embd = NW[0].shape[0]
    norm = [np.fromfile(os.path.join(a.trace, f"ffn_norm-{ly}.bin"), dtype=np.float32).reshape(T, n_embd) for ly in range(L)]
    lout = [np.fromfile(os.path.join(a.trace, f"l_out-{ly}.bin"), dtype=np.float32).reshape(T, n_embd) for ly in range(L)]
    sets = [[set(row) for row in topk[ly]] for ly in range(L)]

    def top(logits, n):
        return [set(np.argsort(-row, kind="stable")[:n]) for row in logits]

    def rms(x):
        return x / np.sqrt((x.astype(np.float64) ** 2).mean(axis=1, keepdims=True) + 1e-6).astype(np.float32)

    # parsing sanity: the router applied to the traced router input reproduces the traced top-k
    sanity = np.mean([len(p & s) / k for ly in range(L) for p, s in zip(top(norm[ly] @ W[ly].T, k), sets[ly])])
    res = {"tokens": T, "layers": L, "n_expert": W[0].shape[0], "n_used": k, "sanity_router_reproduces_topk": round(float(sanity), 4)}

    def hit(pred, ly, t0=0):
        return float(np.mean([len(pred[t] & sets[ly][t]) / k for t in range(t0, T)]))

    out = {"reuse-1": [], "reuse-4": [], "reuse-4 prefetched experts": [], "pregate": [], "pregate-x": []}
    for ly in range(L):
        prev1 = [set()] + sets[ly][:-1]
        u4 = [set().union(*sets[ly][max(0, t - 4) : t]) for t in range(T)]
        out["reuse-1"].append(hit(prev1, ly, 1))
        out["reuse-4"].append(hit(u4, ly, 4))
        out["reuse-4 prefetched experts"].append(float(np.mean([len(s) for s in u4[4:]])))
        if ly > 0:
            x = rms(lout[ly - 1]) * NW[ly]
            logits = x @ W[ly].T
            out["pregate"].append(hit(top(logits, k), ly))
            out["pregate-x"].append(hit(top(logits, k + k // 2), ly))
    res.update({name: round(float(np.mean(v)), 4) for name, v in out.items()})
    res["per_layer_pregate"] = [round(v, 3) for v in out["pregate"]]
    res["per_layer_reuse1"] = [round(v, 3) for v in out["reuse-1"]]
    print(json.dumps(res, indent=1))
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
