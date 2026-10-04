"""Latent (compressed) KV attention, DeepSeek-V2 "absorbed" MLA form, on the kurn attn kernel.

MLA caches one latent row per token: c_j (d_c = 512, after the kv_a norm) and a shared RoPE key
k_pe_j (d_r = 64). With W_UK_h / W_UV_h the per-head key / value up-projections (rows of
attn_kv_b), the scores and outputs of head h are

    s_h(t, j) = q_nope_h(t) . (W_UK_h c_j) + q_pe_h(t) . k_pe_j = q~_h(t) . [c_j, k_pe_j]
    o_h(t)    = W_UV_h sum_j p_h(t, j) c_j,          q~_h = [W_UK_h^T q_nope_h, q_pe_h]

so attention runs over a single 576-wide KV head shared by all query heads (G = n_head), with V
the first 512 dims of the same row (`mla=1`: kattn_args.v aliases k). The kernel is built with
`dk=576` (-> dv=512) and any kurn attn target.
"""

import ctypes
import threading

from ._numpy import np


class KattnArgs(ctypes.Structure):
    _fields_ = [
        ("n_q", ctypes.c_int64), ("n_kv", ctypes.c_int64), ("q_pos0", ctypes.c_int64),
        ("n_head", ctypes.c_int32), ("n_head_kv", ctypes.c_int32), ("causal", ctypes.c_int32), ("scale", ctypes.c_float),
        ("q", ctypes.c_void_p), ("q_s_tok", ctypes.c_int64), ("q_s_head", ctypes.c_int64),
        ("k", ctypes.c_void_p), ("k_s_tok", ctypes.c_int64), ("k_s_head", ctypes.c_int64),
        ("v", ctypes.c_void_p), ("v_s_tok", ctypes.c_int64), ("v_s_head", ctypes.c_int64),
        ("mask", ctypes.c_void_p), ("mask_s_tok", ctypes.c_int64),
        ("out", ctypes.c_void_p), ("o_s_tok", ctypes.c_int64), ("o_s_head", ctypes.c_int64),
    ]  # fmt: skip


def load(so):
    lib = ctypes.CDLL(so)
    lib.kattn_workspace.argtypes = [ctypes.POINTER(KattnArgs), ctypes.c_int]
    lib.kattn_workspace.restype = ctypes.c_size_t
    lib.kattn.argtypes = [ctypes.POINTER(KattnArgs), ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    lib.kattn.restype = None
    return lib


def run_kattn(lib, q, k, v, dv, scale, causal=True, pos0=None, threads=1):
    """q float32 [T, H, DK]; k (and v, or None for MLA aliasing) float16 [N, HKV, D]. Returns
    float32 [T, H, dv]. Causal: query t sees keys j <= pos0 + t (pos0 defaults to N - T)."""
    q = np.ascontiguousarray(q, np.float32)
    k = np.ascontiguousarray(k, np.float16)
    v = k if v is None else np.ascontiguousarray(v, np.float16)
    T, H, _ = q.shape
    N, HKV, _ = k.shape
    out = np.empty((T, H, dv), np.float32)
    f = 4
    a = KattnArgs(T, N, N - T if pos0 is None else pos0, H, HKV, int(causal), scale,
                  q.ctypes.data, q.strides[0] // f, q.strides[1] // f, k.ctypes.data, k.strides[0], k.strides[1],
                  v.ctypes.data, v.strides[0], v.strides[1], None, 0,
                  out.ctypes.data, out.strides[0] // f, out.strides[1] // f)  # fmt: skip
    ws = np.zeros(lib.kattn_workspace(ctypes.byref(a), threads) + 64, np.uint8)
    if threads == 1:
        lib.kattn(ctypes.byref(a), ws.ctypes.data, 0, 1)
    else:
        ts = [threading.Thread(target=lib.kattn, args=(ctypes.byref(a), ws.ctypes.data, i, threads)) for i in range(threads)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
    return out


def split_kv_b(w, n_head, d_nope, d_v):
    """attn_kv_b as [n_head * (d_nope + d_v), d_c] -> (W_UK [H, d_nope, d_c], W_UV [H, d_v, d_c])."""
    w = np.asarray(w).reshape(n_head, d_nope + d_v, -1)
    return w[:, :d_nope], w[:, d_nope:]


def absorb_queries(q_nope, q_pe, w_uk):
    """[T, H, d_nope], [T, H, d_r], [H, d_nope, d_c] -> q~ [T, H, d_c + d_r] (float32)."""
    qa = np.einsum("thn,hnc->thc", q_nope.astype(np.float64), w_uk.astype(np.float64))
    return np.concatenate([qa, q_pe.astype(np.float64)], axis=-1).astype(np.float32)


def latent_cache(c, k_pe):
    """[N, d_c], [N, d_r] -> one latent KV head [N, 1, d_c + d_r] (float16, as cached)."""
    return np.concatenate([c, k_pe], axis=-1).astype(np.float16)[:, None, :]


def up_project(o, w_uv):
    """[T, H, d_c], [H, d_v, d_c] -> [T, H, d_v]."""
    return np.einsum("thc,hvc->thv", o.astype(np.float64), w_uv.astype(np.float64)).astype(np.float32)


def reference(qt, cache, d_v, scale, pos0=None):
    """float64 absorbed attention on the cached latent rows: [T, H, d_c + d_r] -> [T, H, d_v]."""
    T, H, _ = qt.shape
    kv = cache[:, 0].astype(np.float64)
    N = kv.shape[0]
    pos0 = N - T if pos0 is None else pos0
    out = np.zeros((T, H, d_v))
    for t in range(T):
        lim = min(pos0 + t + 1, N)
        s = qt[t].astype(np.float64) @ kv[:lim].T * scale  # [H, lim]
        p = np.exp(s - s.max(axis=1, keepdims=True))
        out[t] = (p @ kv[:lim, :d_v]) / p.sum(axis=1, keepdims=True)
    return out


def kv_bytes_per_token(n_layer, d_c=512, d_r=64, n_head=16, d_nope=128, d_v=128, n_head_kv=None, d_head=128, elt=2):
    """Bytes of KV cache per token (all layers): latent MLA, MLA decompressed per head (as llama.cpp
    caches PLM / non-absorbed DeepSeek), and a GQA model with n_head_kv heads of d_head."""
    r = {"mla_latent": n_layer * (d_c + d_r) * elt, "mla_decompressed": n_layer * n_head * (d_nope + d_r + d_v) * elt}
    if n_head_kv:
        r["gqa"] = n_layer * 2 * n_head_kv * d_head * elt
    return r
