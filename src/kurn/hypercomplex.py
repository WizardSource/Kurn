"""Hypercomplex (quaternion / complex) linear layers: exact references for kurn's compact kernels
(integration/hypercomplex/kurn-hc.cpp).

A layer maps Kq hypercomplex inputs to Nq outputs, y_n = sum_k W_nk * x_k (Hamilton or complex product, weight on
the left). Compact weights are component-major arrays W[p, n, k] (p = 0 real, 1 i, 2 j, 3 k). Vectors use the
block-planar layout: groups of 32 numbers, each group stored component by component.

The real expansion is the left-multiplication matrix of each weight. For output component o and input component
q, the weight component is p = o ^ q and the sign is SIGN[o][q]; with block-planar rows and columns, every 32-value
block of the expansion is one weight component of one row (possibly negated), so a Q8_0-quantized compact layer
expands to an ordinary Q8_0 matrix with the same scales."""

import numpy as np

QUAT_SIGN = np.array([[1, -1, -1, -1], [1, 1, -1, 1], [1, 1, 1, -1], [1, -1, 1, 1]], dtype=np.int8)
CPLX_SIGN = np.array([[1, -1], [1, 1]], dtype=np.int8)


def sign_table(D):
    if D == 4:
        return QUAT_SIGN
    if D == 2:
        return CPLX_SIGN
    raise ValueError("D must be 2 (complex) or 4 (quaternion)")


def component(o, q):
    """Weight component feeding output component o from input component q (both algebras: o XOR q)."""
    return o ^ q


def hamilton(a, b):
    """Quaternion product a * b for arrays [..., 4] (real, i, j, k)."""
    a0, a1, a2, a3 = np.moveaxis(a, -1, 0)
    b0, b1, b2, b3 = np.moveaxis(b, -1, 0)
    return np.stack(
        [
            a0 * b0 - a1 * b1 - a2 * b2 - a3 * b3,
            a0 * b1 + a1 * b0 + a2 * b3 - a3 * b2,
            a0 * b2 - a1 * b3 + a2 * b0 + a3 * b1,
            a0 * b3 + a1 * b2 - a2 * b1 + a3 * b0,
        ],
        axis=-1,
    )


def left_matrix(w):
    """D x D real matrix L with L @ x == w * x for one hypercomplex number w (shape [D])."""
    D = w.shape[-1]
    S = sign_table(D)
    return np.array([[S[o, q] * w[component(o, q)] for q in range(D)] for o in range(D)], dtype=np.float64)


def bp_index(D, n, c):
    """Block-planar index of component c of element n."""
    return (n // 32) * 32 * D + c * 32 + n % 32


def to_block_planar(v):
    """[n, D] hypercomplex vector -> block-planar flat array (n a multiple of 32)."""
    n, D = v.shape
    return v.reshape(n // 32, 32, D).transpose(0, 2, 1).reshape(-1)


def from_block_planar(x, D):
    return x.reshape(-1, D, 32).transpose(0, 2, 1).reshape(-1, D)


def expand(W):
    """Compact W[p, n, k] -> real matrix (D Nq, D Kq) with block-planar rows and columns."""
    D, Nq, Kq = W.shape
    S = sign_table(D)
    E = np.zeros((D * Nq, D * Kq), dtype=np.float64)
    rows = np.array([[bp_index(D, n, o) for n in range(Nq)] for o in range(D)])
    cols = np.array([[bp_index(D, k, q) for k in range(Kq)] for q in range(D)])
    for o in range(D):
        for q in range(D):
            E[np.ix_(rows[o], cols[q])] = S[o, q] * W[component(o, q)]
    return E


def layer(W, x):
    """Hypercomplex layer on [Kq, D] inputs -> [Nq, D] outputs, by the algebra's product (float64)."""
    D, Nq, Kq = W.shape
    Wn = np.moveaxis(W, 0, -1)  # [Nq, Kq, D]
    if D == 4:
        return hamilton(Wn, x[None, :, :]).sum(axis=1)
    a, b = Wn[..., 0], Wn[..., 1]
    return np.stack([a @ x[:, 0] - b @ x[:, 1], a @ x[:, 1] + b @ x[:, 0]], axis=-1)


def quantize_q8_0(x):
    """Compact Q8_0 per component (ggml's quantize_row_q8_0_ref): blocks of 32 along the last axis.
    Returns (int8 codes, fp16 scales as float32)."""
    xb = x.reshape(*x.shape[:-1], -1, 32).astype(np.float32)
    amax = np.abs(xb).max(axis=-1, keepdims=True)
    d = amax / np.float32(127)
    with np.errstate(divide="ignore"):
        idv = np.where(d != 0, np.float32(1) / d, np.float32(0)).astype(np.float32)
    q = np.trunc(xb * idv + np.copysign(np.float32(0.5), xb * idv)).astype(np.int8)  # roundf: half away from zero
    return q.reshape(x.shape), d[..., 0].astype(np.float16).astype(np.float32)


def dequantize_q8_0(q, d):
    return (q.reshape(*q.shape[:-1], -1, 32).astype(np.float32) * d[..., None]).reshape(q.shape)


def param_bytes(D, Nq, Kq, fmt):
    """Weight bytes of the compact layer and of its real expansion."""
    if fmt == "q8":
        return D * Nq * Kq * 34 // 32, D * D * Nq * Kq * 34 // 32
    return 4 * D * Nq * Kq, 4 * D * D * Nq * Kq
