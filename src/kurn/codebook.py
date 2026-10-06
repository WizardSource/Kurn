"""Codebook / vector quantization: QuIP#-style E8 lattice codebook ("E8P") with randomized
Hadamard incoherence processing, plus an AQLM-style additive 2x8 codebook, their exact
Python references and LUT/decode GEMV lowerings.

E8P (2 bits/weight + a fp16 scale per 256 weights = 2.0625 bpw)
----------------------------------------------------------------
Weights are grouped in vectors of 8. Each vector is one point of the shifted E8 lattice
coset (D8 + 1/2) + t, t = +-1/4, restricted to 2^16 points:

    c = s (.) a + t,  a in A (256 "absolute value" patterns, coordinates in {1/2, 3/2, 5/2}),
                      s in {+-1}^8 with sum(c - t) even, t in {-1/4, +1/4}.

A holds the 227 half-integer patterns with |a|^2 <= 10 plus 29 of norm 12 (21 with one 5/2
and two 3/2, 8 with five 3/2), chosen so that exactly 128 patterns have an even coordinate
sum. A is sorted by that parity, so the parity constraint becomes: table index bit 7 =
parity of the sign byte. The 16-bit code is therefore

    low byte: table index bits 0-6 | t << 7 (1: +1/4);  high byte: s_1..s_8 (1: negative)

and the sign byte *is* the 8-lane negate mask. Integer form: q_i = 4 c_i is an odd integer
in [-11, 11]; value_i = d * q_i.

Block `e8p` (66 bytes / 256 weights, Q8_K activations), planar so the kernel can load sign
masks straight into mask registers: fp16 d; u8 lo[32] (low code bytes); u8 hi[32] (sign bytes).
The kernel computes sum_i q_i x_i = sum_i 4a_i (s_i x_i) + sum_groups t * (sum of the group's
8 activations): the negation goes onto the activations, the shift onto precomputed group sums.

Incoherence processing: before quantization, W' = W R^T with R = H_b D / sqrt(b) (block
Walsh-Hadamard of size b = largest power of two dividing K, capped at 4096, random signs D
from a fixed seed per K). The kernel multiplies W' with x' = R x (one fast Walsh-Hadamard
transform per input vector, shared by every tensor that reads the same activation).

vq2x8 (AQLM-style additive codebooks, 2.0625 bpw)
-------------------------------------------------
Each vector of 8 is C0[i] + C1[j] with two 256-entry int8 codebooks learned on rotated
weights and frozen into the format (`VQ2X8_C0/C1`, per-format constant). Block: fp16 d;
u8 idx[64] (pairs i, j); value = d * (C0[i] + C1[j]). Two lowerings: decode (gather the two
8-byte codewords, add, vpdpbusd) and LUT (per 8-activation chunk, precompute the 2 x 256
codeword-activation dot products, then each weight vector costs two table lookups).
"""

import functools
import os

from ._numpy import np
from .ext.compress import E8P_ABS2, E8P_BYTES, QK, e8p_abs_u64, e8p_blocks, lower_e8p_gemv, ref_e8p  # noqa: F401

# ---------------------------------------------------------------- E8P codebook


@functools.cache
def e8p_abs():
    """(256, 8) int8 table of 2*a (odd integers 1, 3, 5), sorted so that rows 0-127 have an
    even coordinate sum of a and rows 128-255 an odd one."""
    return np.array(E8P_ABS2, dtype=np.int8)


def _parity8(x):
    x = x ^ (x >> 4)
    x = x ^ (x >> 2)
    x = x ^ (x >> 1)
    return x & 1


def e8p_decode(codes):
    """u16 codes (any shape) -> int8 q (shape + (8,)), q = 4 * c (odd integers in [-11, 11])."""
    codes = np.asarray(codes, dtype=np.uint16).astype(np.int32)
    sgn = codes >> 8
    idx = (codes & 0x7F) | (_parity8(sgn) << 7)
    a4 = 2 * e8p_abs()[idx].astype(np.int32)  # 4 * a_i
    neg = (sgn[..., None] >> np.arange(8)) & 1
    t = np.where((codes & 0x80) != 0, 1, -1)[..., None]
    return (np.where(neg == 1, -a4, a4) + t).astype(np.int8)


def e8p_encode(u):
    """Nearest E8P point for each row of u (..., 8) given in units of c (= value / (4 d)).
    Returns (u16 codes, q int8 (..., 8))."""
    u = np.asarray(u, dtype=np.float64)
    shp = u.shape[:-1]
    u = u.reshape(-1, 8)
    A = e8p_abs().astype(np.float64) / 2  # (256, 8) a
    An = (A * A).sum(1)
    best_err = np.full(len(u), np.inf)
    best = np.zeros((len(u), 3), dtype=np.int64)  # idx, signmask, t
    for t, tbit in ((-0.25, 0), (0.25, 1)):
        v = u - t
        av = np.abs(v)
        negbits = (v < 0).astype(np.int64)
        npar = negbits.sum(1) & 1
        for lo in range(0, len(u), 8192):
            sl = slice(lo, lo + 8192)
            avs = av[sl]
            err = (avs * avs).sum(1)[:, None] - 2 * avs @ A.T + An[None, :]  # (m, 256)
            flip = 4 * (avs[:, None, :] * A[None, :, :])  # cost of flipping coordinate i
            fi = flip.argmin(2)
            fmin = np.take_along_axis(flip, fi[..., None], 2)[..., 0]
            mism = npar[sl, None] != (np.arange(256) >= 128)[None, :]
            err = err + np.where(mism, fmin, 0.0)
            j = err.argmin(1)
            e = err[np.arange(len(j)), j]
            better = e < best_err[sl]
            nb = negbits[sl].copy()
            fl = mism[np.arange(len(j)), j]
            rows = np.nonzero(fl)[0]
            nb[rows, fi[rows, j[rows]]] ^= 1
            mask = (nb << np.arange(8)).sum(1)
            cand = np.stack([j, mask, np.full_like(j, tbit)], 1)
            idx = np.nonzero(better)[0] + lo
            best[idx] = cand[better]
            best_err[idx] = e[better]
    codes = (best[:, 0] & 0x7F) | (best[:, 2] << 7) | (best[:, 1] << 8)
    codes = codes.astype(np.uint16)
    return codes.reshape(shp), e8p_decode(codes).reshape(shp + (8,))


_ENCODE_C = r"""
#include <math.h>
#include <stdint.h>
static float AT[8][256] __attribute__((aligned(64))), AN[256] __attribute__((aligned(64)));
void e8p_enc_init(const int8_t *tab2a) {
    for (int j = 0; j < 256; j++) {
        AN[j] = 0;
        for (int i = 0; i < 8; i++) { AT[i][j] = 0.5f * tab2a[8 * j + i]; AN[j] += AT[i][j] * AT[i][j]; }
    }
}
/* nearest E8P point to each u[8k..8k+7] (units of c); exact search over the 2 x 256 x 2^7 points */
void e8p_enc(const float *u, int64_t n, uint16_t *codes) {
    float e[256] __attribute__((aligned(64))), fm[256] __attribute__((aligned(64)));
    for (int64_t k = 0; k < n; k++) {
        const float *x = u + 8 * k;
        float best = INFINITY;
        unsigned bcode = 0;
        for (int tb = 0; tb < 2; tb++) {
            const float t = tb ? 0.25f : -0.25f;
            float av[8], vn = 0;
            unsigned nb = 0;
            for (int i = 0; i < 8; i++) {
                const float v = x[i] - t;
                av[i] = fabsf(v); vn += v * v; nb |= (unsigned)(v < 0) << i;
            }
            const int npar = __builtin_popcount(nb) & 1;
            for (int j = 0; j < 256; j++) {
                float dot = 0, m = INFINITY;
                for (int i = 0; i < 8; i++) { const float p = AT[i][j] * av[i]; dot += p; m = p < m ? p : m; }
                fm[j] = m;
                e[j] = vn - 2 * dot + AN[j];
            }
            const float add0 = npar ? 4.0f : 0.0f, add1 = npar ? 0.0f : 4.0f;
            for (int j = 0; j < 128; j++) e[j] += add0 * fm[j];
            for (int j = 128; j < 256; j++) e[j] += add1 * fm[j];
            int bj = 0;
            float be = e[0];
            for (int j = 1; j < 256; j++) if (e[j] < be) { be = e[j]; bj = j; }
            if (be < best) {
                unsigned s = nb;
                if (((unsigned)bj >> 7) != (unsigned)npar) {
                    int fi = 0;
                    float f = INFINITY;
                    for (int i = 0; i < 8; i++) { const float p = AT[i][bj] * av[i]; if (p < f) { f = p; fi = i; } }
                    s ^= 1u << fi;
                }
                best = be;
                bcode = ((unsigned)bj & 0x7f) | ((unsigned)tb << 7) | (s << 8);
            }
        }
        codes[k] = (uint16_t)bcode;
    }
}
"""


@functools.cache
def _encoder():
    """ctypes handle of the compiled exact encoder, or None when no C compiler is usable."""
    import ctypes

    from .toolchain import compile_source, host_arch

    try:
        so = compile_source(_ENCODE_C, "avx512_vnni" if host_arch() == "x86_64" else "scalar", stem="e8p_encode",
                            extra_flags=["-fno-trapping-math"])  # fmt: skip
    except Exception:
        return None
    lib = ctypes.CDLL(so)
    lib.e8p_enc.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p]
    tab = np.ascontiguousarray(e8p_abs())
    lib.e8p_enc_init(tab.ctypes.data_as(ctypes.c_void_p))
    lib._keep = tab
    return lib


def e8p_encode_fast(u, threads=None):
    """Same as e8p_encode (exact nearest point; ties may resolve differently) via the C encoder,
    multithreaded. Falls back to e8p_encode without a compiler."""
    lib = _encoder()
    if lib is None:
        return e8p_encode(u)
    from concurrent.futures import ThreadPoolExecutor

    u = np.ascontiguousarray(u, dtype=np.float32)
    shp = u.shape[:-1]
    flat = u.reshape(-1, 8)
    codes = np.empty(len(flat), dtype=np.uint16)
    step = 65536

    def run(lo):
        n = min(step, len(flat) - lo)
        lib.e8p_enc(flat[lo:].ctypes.data, n, codes[lo:].ctypes.data)

    with ThreadPoolExecutor(threads or min(4, os.cpu_count() or 1)) as ex:
        list(ex.map(run, range(0, len(flat), step)))
    return codes.reshape(shp), e8p_decode(codes).reshape(shp + (8,))


# ---------------------------------------------------------------- randomized Hadamard


def hadamard_block(k, cap=4096):
    b = k & -k
    return min(b, cap)


def rht_signs(k, seed=0x5EED):
    return np.random.default_rng(seed + k).choice(np.array([-1.0, 1.0]), k)


def fwht(x, b):
    """Unnormalized Walsh-Hadamard transform along the last axis, in independent blocks of b."""
    x = np.array(x, dtype=np.float64, copy=True)
    shp = x.shape
    x = x.reshape(-1, shp[-1] // b, b)
    h = 1
    while h < b:
        y = x.reshape(x.shape[0], x.shape[1], b // (2 * h), 2, h)
        a, c = y[:, :, :, 0, :].copy(), y[:, :, :, 1, :].copy()
        y[:, :, :, 0, :] = a + c
        y[:, :, :, 1, :] = a - c
        h *= 2
    return x.reshape(shp)


def rotate_act(x, seed=0x5EED):
    """x' = R x (last axis = K)."""
    k = x.shape[-1]
    b = hadamard_block(k)
    return fwht(np.asarray(x) * rht_signs(k, seed), b) / np.sqrt(b)


def rotate_weight(W, seed=0x5EED):
    """W' = W R^T, so that W' (R x) = W x."""
    k = W.shape[-1]
    b = hadamard_block(k)
    return fwht(np.asarray(W) * rht_signs(k, seed), b) / np.sqrt(b)


def unrotate_weight(Wr, seed=0x5EED):
    """W = W' R (inverse of rotate_weight)."""
    k = Wr.shape[-1]
    b = hadamard_block(k)
    return fwht(Wr, b) / np.sqrt(b) * rht_signs(k, seed)


# ---------------------------------------------------------------- quantizers

E8P_SCALES = (0.85, 0.95, 1.05, 1.15, 1.3)  # block-scale candidates, x the RMS-based scale
E8P_TARGET_RMS = 1.0  # RMS of the vectors in c units the base scale aims at


def _f16(x):
    return np.asarray(x, dtype=np.float16).astype(np.float64)


def quantize_e8p(W, imp=None, scales=E8P_SCALES, target=E8P_TARGET_RMS):
    """Rotated weights W' (rows, K) -> (d fp16 (rows, K/256), codes u16 (rows, K/8), q int8 (rows, K)).
    Per 256-block scale chosen from `scales` by (importance-weighted) squared error."""
    W = np.asarray(W, dtype=np.float64)
    rows, k = W.shape
    nb = k // QK
    B = W.reshape(rows, nb, QK)
    w8 = np.ones(QK) if imp is None else None
    rms = np.sqrt((B * B).mean(2)) + 1e-12
    best_err = np.full((rows, nb), np.inf)
    best_d = np.zeros((rows, nb))
    best_codes = np.zeros((rows, nb, QK // 8), dtype=np.uint16)
    for m in scales:
        d = _f16(rms * m / (4 * target))
        d = np.where(d == 0, 1e-8, d)
        codes, q = e8p_encode_fast(B.reshape(rows, nb, QK // 8, 8) / (4 * d[..., None, None]))
        R = B - d[..., None] * q.reshape(rows, nb, QK)
        if imp is None:
            err = (R * R) @ w8
        else:
            err = (R * R * imp.reshape(nb, QK)[None]).sum(2)
        better = err < best_err
        best_err = np.where(better, err, best_err)
        best_d = np.where(better, d, best_d)
        best_codes = np.where(better[..., None], codes, best_codes)
    q = e8p_decode(best_codes).reshape(rows, k)
    return best_d.astype(np.float16), best_codes.reshape(rows, k // 8), q


def dequant_e8p(d, codes):
    rows = d.shape[0]
    q = e8p_decode(codes).reshape(rows, -1, QK).astype(np.float64)
    return (q * d.astype(np.float64)[..., None]).reshape(rows, -1)


def quantize_e8p_rvq(W, stages=2, imp=None):
    """Residual VQ with E8P stages (2 bits per stage). Returns list of (d, codes) and W_hat."""
    R = np.asarray(W, dtype=np.float64)
    out, What = [], np.zeros_like(R)
    for _ in range(stages):
        d, codes, _ = quantize_e8p(R, imp)
        Wq = dequant_e8p(d, codes)
        out.append((d, codes))
        What += Wq
        R = R - Wq
    return out, What


def quantize_tensor(W, imp=None, alpha=0.5, stages=1):
    """Incoherence-processed E8P of one weight matrix (rows, K) with importance scaling.

    Columns are scaled by s = (imp / mean imp)^(alpha/2) (fp16) before the rotation, so the
    uniform-MSE lattice quantizer minimises sum_j imp_j^alpha |dW_:j|^2; the kernel applies
    x'' = R (x / s) in its activation preprocessing (one pass shared by every tensor reading x).
    Returns (W_hat in the original basis, [(d, codes)] per RVQ stage, s)."""
    W = np.asarray(W, dtype=np.float64)
    s = np.ones(W.shape[1])
    if imp is not None and alpha:
        h = np.maximum(np.asarray(imp, dtype=np.float64), 1e-6 * float(np.mean(imp)) + 1e-30)
        s = _f16((h / h.mean()) ** (alpha / 2))
    parts, what = quantize_e8p_rvq(rotate_weight(W * s), stages)
    return unrotate_weight(what) / s, parts, s


def model_bpw(stages, rows):
    """Bits per weight of an E8P tensor with `stages` RVQ stages and one fp16 scale per column."""
    return stages * 8 * E8P_BYTES / QK + 16 / rows


def quantize_model(src, out, imatrix=None, alpha=0.5, stages=1, keep=("token_embd.weight",), raw_from=None, log=print):
    """Quantize every 2-D weight of GGUF `src` except `keep` with E8P (stages: int or {tensor: int})
    and write the dequantized weights (original basis) as F16 to `out`, so stock llama.cpp measures
    the format's quality. `keep` tensors are copied as stored in `raw_from` (e.g. a llama.cpp
    mix, for a like-for-like token_embd) or in `src`. Returns per-tensor stats."""
    from . import mixed

    imx = mixed.load_imatrix(imatrix) if imatrix else {}
    ts = mixed.tensors(src)
    rawt = mixed.tensors(raw_from) if raw_from else {}
    stats = {}

    def job(name, t, st):
        def run():
            W = mixed.dequant(t).astype(np.float64)
            h = imx.get(name)
            what, _, _ = quantize_tensor(W, h, alpha, st)
            hw = np.ones(W.shape[1]) if h is None else h
            D = what - W
            err = float(((D * D) @ hw).sum() / ((W * W) @ hw).sum())
            stats[name] = {"stages": st, "params": int(W.size), "bpw": model_bpw(st, W.shape[0]), "err": err}
            log(f"{name:32s} stages={st} err={err:.4f}")
            return what.astype(np.float32)

        return run

    rep = {}
    for name, t in ts.items():
        if mixed.is_matrix(name, t) and name not in keep:
            st = stages.get(name, 1) if isinstance(stages, dict) else stages
            rep[name] = job(name, t, st)
    raw = {n: rawt[n] for n in keep if n in rawt}
    mixed.write_gguf(src, out, rep, "F16", raw=raw)
    for n in keep:
        if n in ts:
            t = raw.get(n, ts[n])
            stats[n] = {"stages": 0, "params": int(t.n_elements), "bpw": 8 * int(t.n_bytes) / int(t.n_elements),
                        "type": mixed.type_name(t)}  # fmt: skip
    tot = sum(v["params"] for v in stats.values())
    stats["_total"] = {"bpw": sum(v["bpw"] * v["params"] for v in stats.values()) / tot, "params": tot,
                       "bpw_blocks": sum(v["bpw"] * v["params"] for n, v in stats.items() if n not in keep and n[0] != "_")
                       / max(1, sum(v["params"] for n, v in stats.items() if n not in keep and n[0] != "_"))}  # fmt: skip
    return stats


def pack_e8p(d, codes):
    """(rows, nb) fp16 + (rows, K/8) u16 -> bytes in block_e8p layout, row-major."""
    rows, nb = d.shape
    out = np.zeros((rows, nb, E8P_BYTES), dtype=np.uint8)
    out[:, :, 0:2] = d.astype("<f2").view(np.uint8).reshape(rows, nb, 2)
    c = codes.astype(np.uint16).reshape(rows, nb, QK // 8)
    out[:, :, 2:34] = (c & 0xFF).astype(np.uint8)
    out[:, :, 34:66] = (c >> 8).astype(np.uint8)
    return out.tobytes()


def block_codes(w):
    """One e8p block (bytes) -> 32 u16 codes."""
    lo = np.frombuffer(w, dtype=np.uint8, count=32, offset=2).astype(np.uint16)
    hi = np.frombuffer(w, dtype=np.uint8, count=32, offset=34).astype(np.uint16)
    return lo | (hi << 8)


# ---------------------------------------------------------------- AQLM-style additive 2x8


def _kmeans(X, k, iters=20, seed=0):
    rng = np.random.default_rng(seed)
    C = X[rng.choice(len(X), k, replace=False)].copy()
    for _ in range(iters):
        a = vq_assign(X, C)
        for j in range(k):
            m = a == j
            C[j] = X[m].mean(0) if m.any() else X[rng.integers(len(X))]
    return C


def vq_assign(X, C):
    out = np.empty(len(X), dtype=np.int64)
    Cn = (C * C).sum(1)
    for lo in range(0, len(X), 16384):
        out[lo : lo + 16384] = (Cn[None] - 2 * X[lo : lo + 16384] @ C.T).argmin(1)
    return out


def vq2_assign(X, C0, C1, beam=8):
    """Beam search for additive codes: top-`beam` C0 candidates, best C1 for each residual."""
    i_out = np.empty(len(X), dtype=np.int64)
    j_out = np.empty(len(X), dtype=np.int64)
    C0n, C1n = (C0 * C0).sum(1), (C1 * C1).sum(1)
    for lo in range(0, len(X), 8192):
        x = X[lo : lo + 8192]
        d0 = C0n[None] - 2 * x @ C0.T
        top = np.argsort(d0, 1)[:, :beam]
        best = np.full(len(x), np.inf)
        bi = np.zeros(len(x), dtype=np.int64)
        bj = np.zeros(len(x), dtype=np.int64)
        for b in range(beam):
            i = top[:, b]
            r = x - C0[i]
            d1 = C1n[None] - 2 * r @ C1.T
            j = d1.argmin(1)
            e = ((r - C1[j]) ** 2).sum(1)
            m = e < best
            best[m], bi[m], bj[m] = e[m], i[m], j[m]
        i_out[lo : lo + 8192], j_out[lo : lo + 8192] = bi, bj
    return i_out, j_out


def train_vq2x8(X, iters=6, seed=0, log=None):
    """Learn two additive 256-entry codebooks on vectors X (n, 8) (float, unit-RMS-ish)."""
    C0 = _kmeans(X, 256, 15, seed)
    C1 = _kmeans(X - C0[vq_assign(X, C0)], 256, 15, seed + 1)
    for it in range(iters):
        i, j = vq2_assign(X, C0, C1)
        for Cs, other, oi, si in ((C0, C1, j, i), (C1, C0, i, j)):
            tgt = X - other[oi]
            cnt = np.bincount(si, minlength=256)
            for dim in range(8):
                Cs[:, dim] = np.where(cnt > 0, np.bincount(si, weights=tgt[:, dim], minlength=256) / np.maximum(cnt, 1),
                                      Cs[:, dim])  # fmt: skip
        if log:
            i, j = vq2_assign(X, C0, C1)
            log(f"vq2x8 iter {it}: mse {((X - C0[i] - C1[j]) ** 2).mean():.4f}")
    return C0, C1


def int8_codebooks(C0, C1, limit=63):
    """Scale two float codebooks to int8 so |C0 + C1| <= 2*limit (fits int8). Returns (C0q, C1q, unit)
    where value = unit * (C0q + C1q) in the float codebooks' units."""
    m = max(np.abs(C0).max(), np.abs(C1).max())
    unit = m / limit
    return np.round(C0 / unit).astype(np.int8), np.round(C1 / unit).astype(np.int8), unit
