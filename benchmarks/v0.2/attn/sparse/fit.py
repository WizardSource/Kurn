"""Thresholds and low-rank gate predictors for act_sparsity (calibrated on a different text).

fit.py thr STATS.csv KIND S OUT.txt         KIND: h | silu_gate; per-layer t_l with P(|.| < t_l) = S
fit.py pred MODEL.gguf X.bin R S OUT.bin OUT_thr.txt
    rank-R predictor g~ = B (A x), A = V_r^T W_gate, B = V_r, V_r = top right singular vectors
    of G = X W_gate^T on the calibration inputs X (ffn_norm dump); thresholds t_l so that a
    fraction S of |silu(g~)| on X falls below t_l.
"""

import sys

import numpy as np


def thresholds(stats, kind, s):
    rows = [ln.strip().split(",") for ln in open(stats).read().splitlines()[1:]]
    out = []
    for layer in sorted({int(r[0]) for r in rows}):
        h = sorted((float(r[2]), float(r[3])) for r in rows if int(r[0]) == layer and r[1] == kind)
        tot, acc, t = sum(c for _, c in h), 0.0, 0.0
        for lo, c in h:
            if acc + c > s * tot:
                frac = (s * tot - acc) / c
                t = 10 ** (lo + 0.01 * frac)
                break
            acc += c
        out.append(t)
    return out


def read_dump(path):
    X = {}
    with open(path, "rb") as fh:
        while hdr := fh.read(12):
            il, n, d = np.frombuffer(hdr, np.int32)
            X.setdefault(int(il), []).append(np.frombuffer(fh.read(4 * n * d), np.float32).reshape(n, d))
    return {k: np.concatenate(v) for k, v in X.items()}


def gate_weights(model):
    from gguf import GGUFReader
    from gguf.quants import dequantize

    r = GGUFReader(model)
    for t in r.tensors:
        if t.name.endswith("ffn_gate.weight"):
            yield int(t.name.split(".")[1]), dequantize(t.data, t.tensor_type).astype(np.float32)


def silu(x):
    return x / (1 + np.exp(-x))


def main(argv):
    if argv[0] == "thr":
        t = thresholds(argv[1], argv[2], float(argv[3]))
        open(argv[4], "w").write("\n".join(f"{v:.6g}" for v in t) + "\n")
        print(" ".join(f"{v:.3g}" for v in t))
        return
    if argv[0] != "pred":
        raise SystemExit(__doc__)
    model, xpath, r, s, out, out_thr = argv[1], argv[2], int(argv[3]), float(argv[4]), argv[5], argv[6]
    X = read_dump(xpath)
    W = dict(gate_weights(model))
    nl = len(W)
    dff, d = W[0].shape
    thr = []
    with open(out, "wb") as fh:
        fh.write(np.array([nl, r, d, dff], np.int32).tobytes())
        for layer in range(nl):
            x, w = X[layer].astype(np.float64), W[layer].astype(np.float64)
            g = x @ w.T
            _, sv, vt = np.linalg.svd(g, full_matrices=False)
            v = vt[:r].T  # dff x r
            a = v.T @ w  # r x d
            gp = (x @ a.T) @ v.T
            act, pact = np.abs(silu(g)), np.abs(silu(gp))
            t = float(np.quantile(pact, s))
            thr.append(t)
            # how well the predicted inactive set matches the true one at the same sparsity
            true_off = act < np.quantile(act, s)
            pred_off = pact < t
            agree = (true_off & pred_off).sum() / max(1, true_off.sum())
            energy = (sv[:r] ** 2).sum() / (sv**2).sum()
            print(f"layer {layer:2d} rank {r} energy {energy:.3f} pred-off/true-off overlap {agree:.3f} t {t:.4g}", flush=True)
            fh.write(a.astype(np.float32).tobytes())
            fh.write(v.astype(np.float32).tobytes())
    open(out_thr, "w").write("\n".join(f"{v:.6g}" for v in thr) + "\n")


if __name__ == "__main__":
    main(sys.argv[1:])
