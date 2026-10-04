"""Low-rank kernels: W ~ Q(W) + U V (LoRC-style residual correction) and the fused GEMV
y = Q(W) x + U (V x).

Factorization: the importance-weighted best rank-r approximation of the quantization residual
E = W - Q(W) under sum_j h_j |E_:j - (UV)_:j|^2 (h = llama.cpp imatrix, diagonal input Hessian)
is the truncated SVD of E diag(sqrt h), un-scaled: U = P_r S_r, V = Q_r^T diag(1/sqrt h).
U and V are stored as int8 with one f32 scale per row (Q8-style), i.e. r (N + K) bytes plus
4 r + 4 N bytes of scales -- about 8 r (N + K) / (N K) extra bits per weight.

Cost side (decode, bandwidth-bound): the factors are extra bytes read once per token, so the
correction costs ~ (bpw_extra / bpw_base) of the base GEMV time; the r-vector V x is shared by
all rows. Quality side: measured as the imatrix-weighted relative error (same metric as
`kurn mix`) and, for whole models, perplexity of the F16 writeback.
"""

from ._numpy import np


def weighted_svd(E, h=None):
    """SVD of E diag(sqrt h). Returns (P S, Q^T, sqrt h) with E diag(sqrt h) = (P S) Q^T."""
    E = np.asarray(E, dtype=np.float64)
    sh = np.ones(E.shape[1]) if h is None else np.sqrt(np.maximum(np.asarray(h, dtype=np.float64), 1e-12 * float(np.mean(h))))
    P, s, Qt = np.linalg.svd(E * sh, full_matrices=False)
    return P * s, Qt, sh


def factors(svd, r):
    """Rank-r factors (U (N, r), V (r, K)) from weighted_svd output."""
    PS, Qt, sh = svd
    return PS[:, :r], Qt[:r] / sh


def q8_rows(M):
    """Per-row absmax int8 quantization -> (int8 (rows, cols), f32 scales (rows,))."""
    M = np.asarray(M, dtype=np.float64)
    s = np.abs(M).max(1) / 127.0
    s = np.where(s == 0, 1.0, s)
    return np.clip(np.round(M / s[:, None]), -127, 127).astype(np.int8), s.astype(np.float32)


def deq_rows(q, s):
    return q.astype(np.float64) * s.astype(np.float64)[:, None]


def lorc(W, Wq, h=None, rank=16, int8=True):
    """Q(W) + U V correction of rank `rank`. Returns (W_hat, (U, V) as stored)."""
    U, V = factors(weighted_svd(np.asarray(W, np.float64) - Wq, h), rank)
    if int8:
        uq, us = q8_rows(U)
        vq, vs = q8_rows(V)
        U, V = deq_rows(uq, us), deq_rows(vq, vs)
        return Wq + U @ V, ((uq, us), (vq, vs))
    return Wq + U @ V, (U, V)


def extra_bpw(n, k, r, int8=True):
    """Extra bits per weight of rank-r factors for an (n, k) matrix."""
    if r == 0:
        return 0.0
    return (8 * r * (n + k) + 32 * (n + r)) / (n * k) if int8 else 16 * r * (n + k) / (n * k)


def rel_err(W, What, h=None):
    W = np.asarray(W, np.float64)
    hw = np.ones(W.shape[1]) if h is None else np.asarray(h, np.float64)
    D = What - W
    return float(((D * D) @ hw).sum() / ((W * W) @ hw).sum())


def rank_curve(W, Wq, h=None, ranks=(0, 8, 16, 32, 64, 128)):
    """[(rank, extra bpw, rel err)] for int8 factors of the residual of Wq (one SVD)."""
    W = np.asarray(W, np.float64)
    svd = weighted_svd(W - Wq, h)
    out = []
    for r in ranks:
        if r == 0:
            out.append((0, 0.0, rel_err(W, Wq, h)))
            continue
        U, V = factors(svd, r)
        uq, us = q8_rows(U)
        vq, vs = q8_rows(V)
        out.append((r, extra_bpw(*W.shape, r), rel_err(W, Wq + deq_rows(uq, us) @ deq_rows(vq, vs), h)))
    return out


def residual_spectrum(W, Wq, h=None):
    """Fraction of the weighted residual energy captured by the top-r singular directions, per r."""
    PS, _, _ = weighted_svd(np.asarray(W, np.float64) - Wq, h)
    e = (PS * PS).sum(0)
    return np.cumsum(e) / e.sum()
