"""MXFP4 and NVFP4: FP4 (E2M1) weight formats, byte-identical to ggml's `block_mxfp4`
and `block_nvfp4`, with exact Python references and numpy quantizers.

Both store E2M1 codes as nibbles; ggml's `kvalues_mxfp4` holds 2 * E2M1 so the codes
are small integers {0, ±1, ±2, ±3, ±4, ±6, ±8, ±12} and every block scale carries the
compensating factor 1/2:

    mxfp4  32 values, 17 bytes: e (E8M0) + qs[16]; value = kv[q] * 2^(e - 128)
           (ggml_e8m0_to_fp32_half); low nibbles hold values 0..15, high 16..31.
    nvfp4  64 values, 36 bytes: d[4] (UE4M3, one per 16 values) + qs[32];
           value = kv[q] * ue4m3(d[s]) / 2 (ggml_ue4m3_to_fp32); sub-block s uses
           qs[8s .. 8s+7]: low nibble = value j, high nibble = value j + 8.
           0x7F (NaN) decodes as 0. NVFP4 normally carries a per-tensor fp32 scale
           on top; ggml keeps it outside the block (an optional `ggml_mul`), so does kurn:
           kernels return the dot product without it, the caller multiplies.

Kernels multiply these with Q8_0 activations (ggml's vec_dot_type for both). The
generated kernels are exact against the references below except for one documented
corner: an MXFP4 block with e == 0 (scale 2^-128) is treated as zero.

Quantizers (numpy, float32 arithmetic):
    quantize_mxfp4(x, method="ggml" | "mse", weights=None)
        "ggml": bit-identical to quantize_row_mxfp4_ref (e = floor(log2 amax) - 2,
        nearest code). "mse": per block, the exponent among e-1, e, e+1 with the
        least (optionally importance-weighted) squared error.
    quantize_nvfp4(x, method="ggml" | "mse", tensor_scale=None, weights=None)
        "ggml": bit-identical to quantize_row_nvfp4_ref (no per-tensor scale, so
        typical LLM weights land in the subnormal UE4M3 range and small sub-blocks
        flush to zero). "mse": NVFP4 as specified, with a per-tensor scale
        s = amax / (6 * 448) (returned) and the best of 4 neighbouring UE4M3 codes
        per sub-block.
"""

import struct

from .formats import Field, Format

KV_FP4 = (0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12)  # ggml kvalues_mxfp4 = 2 * E2M1


def e8m0_half(e):
    """ggml_e8m0_to_fp32_half."""
    return 2.0 ** (e - 128)


def ue4m3_half(u):
    """ggml_ue4m3_to_fp32 (UE4M3 value / 2)."""
    if u in (0, 0x7F):
        return 0.0
    e, m = (u >> 3) & 15, u & 7
    return m * 2.0**-10 if e == 0 else 2.0 ** (e - 8) * (1 + m / 8)


def _q8(x):
    return struct.unpack_from("<e", x, 0)[0], struct.unpack_from("<32b", x, 2)


def _ref_mxfp4(wblocks, xblocks):
    total = 0.0
    for w, xb in zip(wblocks, xblocks):
        dx, q8 = _q8(xb)
        vals = [KV_FP4[w[1 + i] & 15] for i in range(16)] + [KV_FP4[w[1 + i] >> 4] for i in range(16)]
        total += e8m0_half(w[0]) * dx * sum(a * b for a, b in zip(vals, q8))
    return total


def _nvfp4_vals(w):
    return [KV_FP4[(w[4 + (v // 16) * 8 + v % 8] >> (4 * ((v % 16) >= 8))) & 15] for v in range(64)]


def _ref_nvfp4(wblocks, xblocks):  # 64 values per block, two Q8_0 activation blocks
    xs = [_q8(x) for x in xblocks]
    total = 0.0
    for i, w in enumerate(wblocks):
        vals = _nvfp4_vals(w)
        for s in range(4):
            dx, q8 = xs[2 * i + s // 2]
            o = 16 * (s % 2)
            total += ue4m3_half(w[s]) * dx * sum(a * b for a, b in zip(vals[16 * s : 16 * s + 16], q8[o : o + 16]))
    return total


MXFP4 = Format("mxfp4", 32, 17, "q8_0", (Field("e", "u8", 0), Field("qs", "u4", 1, 32)),
               "OCP MXFP4: value = kvalues_mxfp4[q] * 2^(e - 128) (E2M1 x 2 codes, E8M0 scale per 32)",
               _ref_mxfp4)  # fmt: skip
NVFP4 = Format("nvfp4", 64, 36, "q8_0", (Field("d", "u8", 0, 4), Field("qs", "u4", 4, 64)),
               "NVFP4: value = kvalues_mxfp4[q] * ue4m3(d[v / 16]) / 2; per-tensor scale applied by the caller",
               _ref_nvfp4)  # fmt: skip


# --------------------------------------------------------------------------- numpy quantizers
def _np():
    import numpy as np

    return np


def _kv(np):
    return np.array(KV_FP4, dtype=np.float32)


def _nearest(x, d):
    """ggml best_index_mxfp4 for rows of x (n, g) with per-row scale d (n,): first index of
    the least |kv * d - x| in float32."""
    np = _np()
    err = np.abs(_kv(np)[None, None, :] * d[:, None, None] - x[:, :, None])
    return np.argmin(err, axis=2).astype(np.uint8)


def _sse(x, q, d, w):
    np = _np()
    r = _kv(np)[q] * d[:, None] - x
    return np.sum(r * r * (1.0 if w is None else w), axis=1)


def _rows(x, g):
    np = _np()
    x = np.ascontiguousarray(x, dtype=np.float32)
    if x.shape[-1] % g:
        raise ValueError(f"last dimension {x.shape[-1]} is not a multiple of {g}")
    return x.reshape(-1, g)


def _weights(weights, x, g):
    """Importance weights per column (K,) broadcast to the (n, g) group view."""
    if weights is None:
        return None
    np = _np()
    w = np.asarray(weights, dtype=np.float32)
    k = x.shape[-1]
    return np.broadcast_to(w.reshape(-1)[:k], x.shape).reshape(-1, g)


def _pack_halves(q, half):
    """q (n, 2*half) codes -> (n, half) bytes: low nibble = first half, high = second."""
    return (q[:, :half] | (q[:, half:] << 4)).astype(_np().uint8)


def _chunks(n, step=1 << 15):
    for i in range(0, n, step):
        yield slice(i, min(n, i + step))


def quantize_mxfp4(x, method="ggml", weights=None):
    """float array (..., K), K % 32 == 0 -> bytes (ggml block_mxfp4 rows)."""
    np = _np()
    xb = _rows(x, 32)
    wb = _weights(weights, np.asarray(x), 32)
    out = np.empty((xb.shape[0], 17), dtype=np.uint8)
    for sl in _chunks(xb.shape[0]):
        v = xb[sl]
        amax = np.max(np.abs(v), axis=1)
        with np.errstate(divide="ignore"):
            lg = np.floor(np.log2(amax.astype(np.float64)).astype(np.float32))
        e = np.where(amax > 0, lg - 2 + 127, 0).astype(np.int64)
        e = np.clip(e, 0, 254)
        if method == "mse":
            best_e, best_err, best_q = e, None, None
            for de in (0, -1, 1):
                ec = np.clip(e + de, 0, 254)
                d = np.ldexp(np.float32(1), (ec - 128).astype(np.int32)).astype(np.float32)
                q = _nearest(v, d)
                err = _sse(v, q, d, None if wb is None else wb[sl])
                if best_err is None:
                    best_e, best_err, best_q = ec, err, q
                else:
                    better = err < best_err
                    best_e = np.where(better, ec, best_e)
                    best_err = np.where(better, err, best_err)
                    best_q = np.where(better[:, None], q, best_q)
            e, q = best_e, best_q
        elif method == "ggml":
            d = np.ldexp(np.float32(1), (e - 128).astype(np.int32)).astype(np.float32)
            q = _nearest(v, d)
        else:
            raise ValueError(f"unknown MXFP4 method {method!r}")
        out[sl, 0] = e
        out[sl, 1:] = _pack_halves(q, 16)
    return out.tobytes()


def fp32_to_ue4m3(x):
    """ggml_fp32_to_ue4m3, vectorised (x float32 array)."""
    np = _np()
    x = np.minimum(np.asarray(x, dtype=np.float32), np.float32(448.0))
    bits = x.view(np.uint32).astype(np.int64)
    fexp = ((bits >> 23) & 0xFF) - 127
    man = (bits >> 20) & 7
    uexp = fexp + 7
    sub = np.clip((x * np.float32(512.0) + np.float32(0.5)).astype(np.int64), 0, 7)
    m2 = man + ((bits >> 19) & 1)
    e2 = uexp + (m2 > 7)
    m2 = np.where(m2 > 7, 0, m2)
    normal = np.where(e2 >= 15, 0x7E, (e2 << 3) | m2)
    r = np.where(uexp <= 0, sub, np.where(uexp >= 15, 0x7E, normal))
    return np.where(x > 0, r, 0).astype(np.uint8)


def ue4m3_to_fp32_half(u):
    """ggml_ue4m3_to_fp32, vectorised."""
    np = _np()
    u = np.asarray(u).astype(np.int64)
    e, m = (u >> 3) & 15, u & 7
    v = np.where(e == 0, m * np.float32(2.0**-10), np.ldexp(np.float32(1), (e - 8).astype(np.int32)) * (1 + m / 8))
    return np.where((u == 0) | (u == 0x7F), 0, v).astype(np.float32)


def quantize_nvfp4(x, method="ggml", tensor_scale=None, weights=None):
    """float array (..., K), K % 64 == 0 -> (bytes of ggml block_nvfp4 rows, tensor scale).
    Dequantized value = tensor scale * block value (the scale is 1.0 for method="ggml")."""
    np = _np()
    xs = _rows(x, 16)
    wb = _weights(weights, np.asarray(x), 16)
    if _rows(x, 64).shape[0] * 4 != xs.shape[0]:
        raise AssertionError
    if method == "ggml":
        ts = np.float32(1.0)
    elif method == "mse":
        amax_t = float(np.max(np.abs(xs))) if xs.size else 0.0
        ts = np.float32(tensor_scale if tensor_scale is not None else (amax_t / (6 * 448) if amax_t > 0 else 1.0))
    else:
        raise ValueError(f"unknown NVFP4 method {method!r}")
    codes = np.empty((xs.shape[0], 8), dtype=np.uint8)
    scales = np.empty(xs.shape[0], dtype=np.uint8)
    for sl in _chunks(xs.shape[0]):
        v = xs[sl] / ts if method == "mse" else xs[sl]
        amax = np.max(np.abs(v), axis=1)
        ue = fp32_to_ue4m3((amax / np.float32(6.0)).astype(np.float32))
        if method == "mse":
            cands = [ue] + [np.clip(ue.astype(np.int64) + k, 0, 0x7E).astype(np.uint8) for k in (-1, 1, 2)]
            best_u = best_err = best_q = None
            for u in cands:
                d = ue4m3_to_fp32_half(u)
                q = _nearest(v, d)
                err = _sse(v, q, d, None if wb is None else wb[sl])
                if best_err is None:
                    best_u, best_err, best_q = u, err, q
                else:
                    better = err < best_err
                    best_u = np.where(better, u, best_u)
                    best_err = np.where(better, err, best_err)
                    best_q = np.where(better[:, None], q, best_q)
            ue, q = best_u, best_q
        else:
            q = _nearest(v, ue4m3_to_fp32_half(ue))
        scales[sl] = ue
        codes[sl] = _pack_halves(q, 8)
    nb = xs.shape[0] // 4
    out = np.concatenate([scales.reshape(nb, 4), codes.reshape(nb, 32)], axis=1)
    return out.tobytes(), float(ts)


def dequantize_mxfp4(data, k):
    """bytes of block_mxfp4 rows -> float32 array (rows, k)."""
    np = _np()
    b = np.frombuffer(data, dtype=np.uint8).reshape(-1, 17)
    q = np.concatenate([b[:, 1:] & 15, b[:, 1:] >> 4], axis=1)
    d = np.ldexp(np.float32(1), b[:, 0].astype(np.int32) - 128).astype(np.float32)
    return (_kv(np)[q] * d[:, None]).reshape(-1, k)


def dequantize_nvfp4(data, k, tensor_scale=1.0):
    """bytes of block_nvfp4 rows -> float32 array (rows, k), times the per-tensor scale."""
    np = _np()
    b = np.frombuffer(data, dtype=np.uint8).reshape(-1, 36)
    d = ue4m3_to_fp32_half(b[:, :4]).reshape(-1, 1)
    qs = b[:, 4:].reshape(-1, 8)
    q = np.concatenate([qs & 15, qs >> 4], axis=1)
    return (_kv(np)[q] * d * np.float32(tensor_scale)).reshape(-1, k)
