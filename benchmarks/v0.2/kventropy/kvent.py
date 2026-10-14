"""Entropy of real KV caches: adaptive (KT) code lengths under simple context models, plus zstd baselines.

Code lengths are what an adaptive arithmetic / rANS coder with a KT (add-1/2) estimator per context pays, so model cost
is included and no table is transmitted. Everything is in bits per element of the stored format.
"""
import numpy as np
from scipy.special import gammaln

LN2 = np.log(2.0)


def kt_bits(ctx, sym, nctx, A):
    """Adaptive KT code length (bits) of symbols sym (0..A-1) under contexts ctx (0..nctx-1)."""
    cnt = np.bincount(ctx.astype(np.int64) * A + sym.astype(np.int64), minlength=nctx * A).reshape(nctx, A)
    a = 0.5
    N = cnt.sum(1)
    used = N > 0
    bits = (gammaln(N[used] + A * a) - gammaln(A * a)).sum() - (gammaln(cnt[used] + a) - gammaln(a)).sum()
    return bits / LN2


def h0(sym, A):
    c = np.bincount(sym.ravel().astype(np.int64), minlength=A).astype(np.float64)
    p = c[c > 0] / c.sum()
    return -(p * np.log2(p)).sum()


# ---- format views -------------------------------------------------------------------------------------------------

def f16_planes(rows):
    """rows: (n, 2C) uint8 of f16 -> hi, lo bytes (n, C)."""
    u = rows.view(np.uint16)
    return (u >> 8).astype(np.uint8), (u & 0xFF).astype(np.uint8)


def q8_0_split(rows):
    n, rb = rows.shape
    blk = rows.reshape(n, rb // 34, 34)
    d = np.ascontiguousarray(blk[:, :, :2]).view(np.uint16)[..., 0]
    q = blk[:, :, 2:].view(np.int8).reshape(n, -1)
    return d, (q.astype(np.int16) + 128).astype(np.uint8)


def q4_0_split(rows):
    n, rb = rows.shape
    blk = rows.reshape(n, rb // 18, 18)
    d = np.ascontiguousarray(blk[:, :, :2]).view(np.uint16)[..., 0]
    b = blk[:, :, 2:]
    q = np.concatenate([b & 15, b >> 4], axis=2).reshape(n, -1)
    return d, q


def k4c_encode(xf16, dk=128):
    """Exact numpy port of ggml-k4c.c k4c_encode for full groups. xf16: (n, C) f16 with n % 32 == 0.
    Returns scale (G, C) f16, min (G, C) f16, codes (G, 32, C) uint8."""
    n, C = xf16.shape
    x = xf16.astype(np.float32).reshape(n // 32, 32, C)
    lo = x.min(1)
    hi = x.max(1)
    mh = lo.astype(np.float16)
    m = mh.astype(np.float32)
    sh = ((hi - lo) / np.float32(15.0)).astype(np.float16)
    s = sh.astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.rint((x - m[:, None, :]) / s[:, None, :])
    q = np.clip(q, 0, 15)
    q = np.where(s[:, None, :] == 0, 8, q).astype(np.uint8)
    return sh, mh, q


def k4c_bytes(sh, mh, q, dk=128):
    """Pack into the in-memory K4C layout (for byte-level baselines)."""
    G, _, C = q.shape
    H = C // dk
    out = np.zeros((G, H, dk * 20), np.uint8)
    for h in range(H):
        sl = slice(h * dk, (h + 1) * dk)
        out[:, h, :2 * dk] = sh[:, sl].view(np.uint8)
        out[:, h, 2 * dk:4 * dk] = mh[:, sl].view(np.uint8)
        qq = q[:, :, sl].reshape(G, 32, dk // 32, 2, 16)
        nib = qq[:, :, :, 0, :] | (qq[:, :, :, 1, :] << 4)
        out[:, h, 4 * dk:] = nib.reshape(G, 32 * dk // 2)
    return out.reshape(-1)


def ordered16(u):
    """f16 bit pattern -> monotone int32 key."""
    u = u.astype(np.int32)
    return np.where(u & 0x8000, 0x8000 - (u & 0x7FFF) - 1, u + 0x8000) - 0x8000


# ---- models -------------------------------------------------------------------------------------------------------

def model_f16(rows):
    """bits per element for an f16 tensor (n, C)."""
    n = rows.shape[0]
    hi, lo = f16_planes(rows)
    C = hi.shape[1]
    ch = np.broadcast_to(np.arange(C, dtype=np.int64), hi.shape)
    N = hi.size
    r = {}
    r["byte_h0"] = h0(rows, 256) * 2  # bytewise order-0 entropy (the blog's "~11 bits")
    r["planes_h0"] = h0(hi, 256) + h0(lo, 256)
    r["hi|ch+lo|hi"] = (kt_bits(ch.ravel(), hi.ravel(), C, 256) + kt_bits(hi.ravel(), lo.ravel(), 256, 256)) / N
    r["hi|ch+lo|ch"] = (kt_bits(ch.ravel(), hi.ravel(), C, 256) + kt_bits(ch.ravel(), lo.ravel(), C, 256)) / N
    # previous-token predictor (residual of the monotone key, per channel): the "cheap predictor" variant
    k = ordered16(rows.view(np.uint16))
    d = np.diff(k, axis=0, prepend=0)
    dz = np.where(d >= 0, 2 * d, -2 * d - 1)  # zigzag
    dh = np.minimum(dz >> 8, 255).astype(np.int64)  # escape bucket 255 (rare)
    dl = (dz & 0xFF).astype(np.int64)
    esc = (dz >> 8) >= 255
    r["delta_t"] = (kt_bits(ch.ravel(), dh.ravel(), C, 256) + kt_bits(dh.ravel(), dl.ravel(), 256, 256)) / N + esc.mean() * 17
    return r


def model_codes(d, q, A, nch_scale):
    """Q8_0 / Q4_0: scales d (n, nb) f16 bits, codes q (n, C). bits per element incl. scales."""
    n, C = q.shape
    N = q.size
    ch = np.broadcast_to(np.arange(C, dtype=np.int64), q.shape)
    dhi, dlo = (d >> 8).astype(np.int64), (d & 0xFF).astype(np.int64)
    bc = np.broadcast_to(np.arange(d.shape[1], dtype=np.int64), d.shape)
    sbits = kt_bits(bc.ravel(), dhi.ravel(), d.shape[1], 256) + kt_bits(dhi.ravel(), dlo.ravel(), 256, 256)
    r = {"scale_bits_per_elem": sbits / N}
    r["q|ch"] = (kt_bits(ch.ravel(), q.ravel(), C, A) + sbits) / N
    r["q_h0"] = h0(q, A) + sbits / N
    qd = (np.diff(q.astype(np.int64), axis=0, prepend=0)) % A
    r["dq_t|ch"] = (kt_bits(ch.ravel(), qd.ravel(), C, A) + sbits) / N
    return r


def f16_given_k4c(xf16, sh, mh, q):
    """bits per element for the exact f16 keys given their k4c group (mu = dequantized value, free predictor)."""
    G, _, C = q.shape
    mu = (q.astype(np.float32) * sh.astype(np.float32)[:, None, :] + mh.astype(np.float32)[:, None, :]).astype(np.float16)
    kx = ordered16(xf16.reshape(G, 32, C).view(np.uint16))
    km = ordered16(mu.view(np.uint16))
    r = kx - km
    z = np.where(r >= 0, 2 * r, -2 * r - 1)
    # context: bin width in ulps of mu, log2-bucketed
    ulp = np.abs(np.spacing(mu.astype(np.float16)).astype(np.float32))
    w = np.log2(np.maximum(sh.astype(np.float32)[:, None, :] / np.maximum(ulp, 1e-12), 1.0))
    ctx = np.clip(np.rint(w * 2), 0, 47).astype(np.int64)
    zc = np.minimum(z, 4095).astype(np.int64)
    esc = (z >= 4095).mean() * 16
    return (kt_bits(ctx.ravel(), zc.ravel(), 48, 4096)) / q.size + esc


def model_k4c(sh, mh, q):
    """k4c: per (group, channel) f16 scale + min, 32 4-bit codes. bits per element (5.0 raw)."""
    G, _, C = q.shape
    N = q.size
    ch = np.broadcast_to(np.arange(C, dtype=np.int64), (G, 32, C))
    chg = np.broadcast_to(np.arange(C, dtype=np.int64), (G, C))
    qb = kt_bits(ch.ravel(), q.ravel(), C, 16)
    hb = 0.0
    for v in (sh.view(np.uint16), mh.view(np.uint16)):
        vh, vl = (v >> 8).astype(np.int64), (v & 0xFF).astype(np.int64)
        hb += kt_bits(chg.ravel(), vh.ravel(), C, 256) + kt_bits(vh.ravel(), vl.ravel(), 256, 256)
    return {"q|ch": (qb + hb) / N, "codes_only": qb / N, "header_bits_per_elem": hb / N, "q_h0": h0(q, 16) + hb / N}
