"""predictor.py MODEL CORPUS : the blog's speculative KV coding on kurn caches.

mu = KV of a cheaper predictor (same model, Q4_K_M weights, F16 KV) on the same tokens; sigma^2 per (layer, kv, channel)
= residual variance on calibration text (wiki-train + code-calib, 8K tokens each), frozen. Target symbols are coded under
the bin-integrated mixture 0.95 N(mu, s^2) + 0.03 N(mu, 9 s^2) + 0.02 empirical marginal (fitted on calibration).
Reports bits per element for F16, Q8_0, Q4_0 (V of k4c_q4), k4c codes (K of k4c_*), incl. side info (scales / headers,
coded with the static models of kvent.py).
"""
import json, os, sys
import numpy as np
from scipy.special import ndtr
from kvseq import Seq
from kvent import *

model, corpus = sys.argv[1], sys.argv[2]
D = os.environ.get("KV_DUMPS", "dumps") + "/"
NT = 8192
LAYERS = None
W = (0.95, 0.03, 0.02)


def f32(T, n=NT):
    return np.array(T[:n]).view(np.float16).astype(np.float32)


# llama.cpp stores quantized K / V caches after a Walsh-Hadamard rotation (K: 128-point here, V: 64-point;
# llama_kv_cache::build_input_k_rot / _v_rot); F16 caches and k4c keys are not rotated
def hadamard(n):
    h = np.array([[1.0]])
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return (h / np.sqrt(n)).astype(np.float32)


ROT = {"K": hadamard(128), "V": hadamard(64)}


def rot(x, kv):
    H = ROT[kv]
    n = H.shape[0]
    return (x.reshape(x.shape[0], -1, n) @ H).reshape(x.shape)


# calibration: per-channel residual variance and per-(layer,kv) empirical f16 marginal
calib = [(Seq(D + f"{model}-{c}-f16.seq"), Seq(D + f"{model}-{c}-pred.seq")) for c in ("wiki-train", "code-calib")]
nl = calib[0][0].nlayer
LAYERS = range(nl)
sig = {}
sig_rot = {}
marg = {}
for kv in ("K", "V"):
    for L in LAYERS:
        num, num_r, den = 0.0, 0.0, 0
        hist = np.zeros(65536)
        for t, p in calib:
            a, b = f32(getattr(t, kv)[L]), f32(getattr(p, kv)[L])
            num = num + ((a - b) ** 2).sum(0)
            num_r = num_r + ((rot(a, kv) - rot(b, kv)) ** 2).sum(0)
            den += a.shape[0]
            hist += np.bincount(np.array(getattr(t, kv)[L][:NT]).view(np.uint16).ravel(), minlength=65536)
        sig[kv, L] = np.sqrt(num / den) + 1e-6
        sig_rot[kv, L] = np.sqrt(num_r / den) + 1e-6
        marg[kv, L] = (hist + 0.5) / (hist.sum() + 0.5 * 65536)


def mix_prob(lo, hi, mu, s, pm):
    """probability of the interval [lo, hi] under the mixture; pm = marginal mass of the symbol."""
    g1 = ndtr((hi - mu) / s) - ndtr((lo - mu) / s)
    g2 = ndtr((hi - mu) / (3 * s)) - ndtr((lo - mu) / (3 * s))
    return np.maximum(W[0] * g1 + W[1] * g2 + W[2] * pm, 1e-30)


def f16_edges(u):
    """f16 bit patterns -> (lo, hi) bin edges in value space (midpoints to the neighbours)."""
    k = ordered16(u)
    def key2val(kk):
        kk = np.clip(kk, -0x7BFF - 1, 0x7BFF)  # finite range
        uu = np.where(kk >= 0, kk, (-kk - 1) | 0x8000).astype(np.uint16)
        return uu.view(np.float16).astype(np.float64)
    x = key2val(k)
    lo = (x + key2val(k - 1)) / 2
    hi = (x + key2val(k + 1)) / 2
    return lo, hi


tgt = {f: Seq(D + f"{model}-{corpus}-{f}.seq") for f in ("f16", "q8", "k4cq4")}
prd = Seq(D + f"{model}-{corpus}-pred.seq")
res = {}
acc = {}
for kv in ("K", "V"):
    for L in LAYERS:
        mu32 = f32(getattr(prd, kv)[L])
        mu = mu32.astype(np.float64)
        s = sig[kv, L][None, :].astype(np.float64)
        mur = rot(mu32, kv).astype(np.float64)  # for the rotated (quantized) caches
        sr = sig_rot[kv, L][None, :].astype(np.float64)
        # F16 target
        u = np.array(getattr(tgt["f16"], kv)[L][:NT]).view(np.uint16)
        lo, hi = f16_edges(u)
        pm = marg[kv, L][u]
        b = -np.log2(mix_prob(lo, hi, mu, s, pm)).mean()
        b1 = -np.log2(np.maximum(ndtr((hi - mu) / s) - ndtr((lo - mu) / s), 1e-30)).mean()
        acc.setdefault(("f16", kv), []).append(b)
        acc.setdefault(("f16_gauss", kv), []).append(b1)
        # Q8_0 target (real q8 cache run)
        d, q = q8_0_split(np.array(getattr(tgt["q8"], kv)[L][:NT]))
        df = np.repeat(d.view(np.float16).astype(np.float64), 32, axis=1)
        qc = q.astype(np.float64) - 128
        with np.errstate(divide="ignore", invalid="ignore"):
            muq, sq = mur / df, sr / df
            pmq = 1.0 / 255
            p = mix_prob(qc - 0.5, qc + 0.5, np.nan_to_num(muq), np.nan_to_num(sq, nan=1.0, posinf=1e6) + 1e-9, pmq)
        sb = model_codes(d, q, 256, None)["scale_bits_per_elem"]
        acc.setdefault(("q8_0", kv), []).append(-np.log2(p).mean() + sb)
        if kv == "V":
            # Q4_0 V of the k4c_q4 run: value = (q - 8) * d
            d, q = q4_0_split(np.array(tgt["k4cq4"].V[L][:NT]))
            df = np.repeat(d.view(np.float16).astype(np.float64), 32, axis=1)
            qc = q.astype(np.float64) - 8
            with np.errstate(divide="ignore", invalid="ignore"):
                # Q4_0 scales are signed (d = max / -8): the grid is in units of d, the spread is |s / d|
                muq, sq = np.nan_to_num(mur / df), np.nan_to_num(np.abs(sr / df), nan=1.0, posinf=1e6) + 1e-9
            lo_, hi_ = np.where(qc == -8, -1e9, qc - 0.5), np.where(qc == 7, 1e9, qc + 0.5)
            p = mix_prob(lo_, hi_, muq, sq, 1.0 / 16)
            sb = model_codes(d, q, 16, None)["scale_bits_per_elem"]
            acc.setdefault(("q4_0", kv), []).append(-np.log2(p).mean() + sb)
        else:
            # k4c codes of the k4c_q4 run's exact keys: value = q * s + m
            sh, mh, q = k4c_encode(np.array(tgt["k4cq4"].K[L][:NT]).view(np.float16))
            G = q.shape[0]
            sc = np.repeat(sh.astype(np.float64), 32, axis=0).reshape(G * 32, -1)
            mn = np.repeat(mh.astype(np.float64), 32, axis=0).reshape(G * 32, -1)
            qc = q.reshape(G * 32, -1).astype(np.float64)
            with np.errstate(divide="ignore", invalid="ignore"):
                muq = np.nan_to_num((mu - mn) / sc)
                sq = np.nan_to_num(s / sc, nan=1.0, posinf=1e6) + 1e-9
            lo_, hi_ = np.where(qc == 0, -1e9, qc - 0.5), np.where(qc == 15, 1e9, qc + 0.5)
            p = mix_prob(lo_, hi_, muq, sq, 1.0 / 16)
            hb = model_k4c(sh, mh, q)["header_bits_per_elem"]
            acc.setdefault(("k4c", kv), []).append(-np.log2(p).mean() + hb)
        print(kv, L, {f"{k[0]}": round(v[-1], 3) for k, v in acc.items() if k[1] == kv}, flush=True)
res = {f"{k[0]}.{k[1]}": float(np.mean(v)) for k, v in acc.items()}
res["per_layer"] = {f"{k[0]}.{k[1]}": [float(x) for x in v] for k, v in acc.items()}
print(json.dumps({k: round(v, 3) for k, v in res.items() if k != "per_layer"}))
json.dump(res, open(os.path.join(os.environ.get("KV_RESULTS", "results"), f"pred-{model}-{corpus}.json"), "w"), indent=1)
