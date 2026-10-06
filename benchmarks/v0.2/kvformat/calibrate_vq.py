"""TaSQ-style K codebooks for the engine's vqk_* KV formats (KURN_VQ).

    python calibrate_vq.py PREFIX N_LAYER N_KV D OUT.bin [--samples 16384] [--iters 10] [--procs N]

PREFIX is a KURN_DUMP_KQ dump taken with THREADS = N_KV (thread t owns kv head t): PREFIX.t holds
f16 pre-RoPE K [steps][N_LAYER][D] and PREFIX.qq.t the per-channel query second moments.

Per (layer, kv head), following TaSQ (arXiv 2610.03027) without its Fisher weighting:
  * query-guided weights: for a NEOX RoPE pair (i, i + D/2), averaging R^T q q^T R over relative
    angles gives w_i = w_{i+D/2} = (E q_i^2 + E q_{i+D/2}^2) / 2; keys are scaled by sqrt(w);
  * per-token RMS normalization per head (TaSQ shares one scale across heads; the engine's threads
    own one head each, so this costs 16/D more bits per channel);
  * covariance-aware grouping: whole RoPE pairs merged by minimum-weight perfect matching on
    det(Sigma_G + eps I)^(1/|G|) until groups have 8 channels;
  * k-means (1024 centroids) per group on the normalized, weighted keys.
Output: int32 header (magic, N_LAYER, N_KV, D, 8, 1024), then per (layer, head) float wsq[D],
int32 perm[D] (permuted position -> channel), float codebook[D/8][1024][8].
"""

import argparse
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import struct  # noqa: E402
import time  # noqa: E402
from multiprocessing import Pool  # noqa: E402

import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402

G, K = 8, 1024


def group_channels(cov, d, eps=1e-6):
    units = [(i, i + d // 2) for i in range(d // 2)]

    def cost(idx):
        sub = cov[np.ix_(idx, idx)] + eps * np.eye(len(idx))
        sign, logdet = np.linalg.slogdet(sub)
        return float(np.exp(logdet / len(idx))) if sign > 0 else 1e9

    while len(units[0]) < G:
        gr = nx.Graph()
        for a in range(len(units)):
            for b in range(a + 1, len(units)):
                gr.add_edge(a, b, weight=cost(list(units[a] + units[b])))
        m = nx.min_weight_matching(gr)
        units = [tuple(sorted(units[a] + units[b])) for a, b in m]
    return np.array([c for u in units for c in u], dtype=np.int32)


def kmeans(x, k, iters, rng):
    c = x[rng.choice(len(x), k, replace=len(x) < k)].copy()
    xn = (x * x).sum(1)
    for _ in range(iters):
        d = xn[:, None] - 2 * x @ c.T + (c * c).sum(1)[None]
        a = d.argmin(1)
        cnt = np.bincount(a, minlength=k)
        s = np.zeros_like(c)
        np.add.at(s, a, x)
        live = cnt > 0
        c[live] = s[live] / cnt[live, None]
        dead = np.flatnonzero(~live)
        if len(dead):  # re-seed empty clusters on the worst-fit points
            far = np.argsort(d[np.arange(len(x)), a])[::-1][np.arange(len(dead)) % len(x)]
            c[dead] = x[far]
    return c.astype(np.float32)


_DATA = {}


def fit_head(job):
    layer, h = job
    kd, qq, pick, D, iters = _DATA["k"][h], _DATA["qq"][h], _DATA["pick"], _DATA["D"], _DATA["iters"]
    rng = np.random.default_rng(1000 * layer + h)
    k = np.asarray(kd[pick, layer, :], dtype=np.float32)
    w = np.empty(D)
    w[: D // 2] = w[D // 2 :] = 0.5 * (qq[layer][: D // 2] + qq[layer][D // 2 :])
    wsq = np.sqrt(np.maximum(w, 1e-12 * w.mean())).astype(np.float32)
    x = k * wsq
    x /= np.sqrt((x * x).mean(1, keepdims=True)) + 1e-12
    perm = group_channels(np.cov(x, rowvar=False), D)
    cb = np.stack([kmeans(x[:, perm[g * G : (g + 1) * G]], K, iters, rng) for g in range(D // G)])
    return wsq.tobytes() + perm.tobytes() + cb.tobytes()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix")
    ap.add_argument("n_layer", type=int)
    ap.add_argument("n_kv", type=int)
    ap.add_argument("d", type=int)
    ap.add_argument("out")
    ap.add_argument("--samples", type=int, default=16384)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--procs", type=int, default=os.cpu_count())
    a = ap.parse_args()
    L, H, D = a.n_layer, a.n_kv, a.d
    ks, qqs = [], []
    for h in range(H):
        raw = np.memmap(f"{a.prefix}.{h}", dtype=np.float16, mode="r")
        steps = raw.size // (L * D)
        ks.append(raw[: steps * L * D].reshape(steps, L, D))
        qq = np.fromfile(f"{a.prefix}.qq.{h}", dtype=np.float64)
        qqs.append(qq[:-1].reshape(L, D) / qq[-1])
    steps = ks[0].shape[0]
    pick = np.sort(np.random.default_rng(0).choice(steps, min(a.samples, steps), replace=False))
    _DATA.update(k=ks, qq=qqs, pick=pick, D=D, iters=a.iters)
    print(f"{steps} calibration tokens, {len(pick)} used per codebook, {a.procs} processes", flush=True)
    t0 = time.time()
    jobs = [(layer, h) for layer in range(L) for h in range(H)]
    with Pool(a.procs) as pool, open(a.out + ".tmp", "wb") as fh:
        fh.write(struct.pack("<6i", 0x4B565131, L, H, D, G, K))
        for i, blob in enumerate(pool.imap(fit_head, jobs)):
            fh.write(blob)
            if (i + 1) % H == 0:
                print(f"layer {(i + 1) // H - 1} done, {time.time() - t0:.0f}s", flush=True)
    os.replace(a.out + ".tmp", a.out)


if __name__ == "__main__":
    main()
