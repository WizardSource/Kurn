"""kattn called in-process (ctypes) against an independent numpy float64 reference: arbitrary
strides (ggml-style permuted q / output), workspace reuse across calls of different shapes
(the workspace resets itself), and real concurrent threads (ctypes releases the GIL)."""

import ctypes
import threading

import pytest

from kurn import attention as A

np = pytest.importorskip("numpy")


class Args(ctypes.Structure):
    _fields_ = [
        ("n_q", ctypes.c_int64), ("n_kv", ctypes.c_int64), ("q_pos0", ctypes.c_int64),
        ("n_head", ctypes.c_int32), ("n_head_kv", ctypes.c_int32), ("causal", ctypes.c_int32), ("scale", ctypes.c_float),
        ("q", ctypes.c_void_p), ("q_s_tok", ctypes.c_int64), ("q_s_head", ctypes.c_int64),
        ("k", ctypes.c_void_p), ("k_s_tok", ctypes.c_int64), ("k_s_head", ctypes.c_int64),
        ("v", ctypes.c_void_p), ("v_s_tok", ctypes.c_int64), ("v_s_head", ctypes.c_int64),
        ("mask", ctypes.c_void_p), ("mask_s_tok", ctypes.c_int64),
        ("out", ctypes.c_void_p), ("o_s_tok", ctypes.c_int64), ("o_s_head", ctypes.c_int64),
    ]  # fmt: skip


def _lib(target, kv="f16", **kw):
    c = A.resolve({"target": target, "kv": kv, **kw})
    ok, missing = A.runnable(target)
    if not ok:
        pytest.skip(f"host CPU lacks {missing}")
    lib = ctypes.CDLL(A.build(c))
    lib.kattn_workspace.argtypes, lib.kattn_workspace.restype = [ctypes.POINTER(Args), ctypes.c_int], ctypes.c_size_t
    lib.kattn.argtypes, lib.kattn.restype = [ctypes.POINTER(Args), ctypes.c_void_p, ctypes.c_int, ctypes.c_int], None
    return lib


def _problem(rng, nq, nkv, nh, nhkv, d, head_major_q):
    q = (3 * rng.standard_normal((nq, nh, d))).astype(np.float32)
    if head_major_q:  # q stored [head][token][d]
        q = np.ascontiguousarray(q.transpose(1, 0, 2)).transpose(1, 0, 2)
    k = rng.standard_normal((nkv, nhkv, d)).astype(np.float16)
    v = rng.standard_normal((nkv, nhkv, d)).astype(np.float16)
    return q, k, v


def _reference(q, k, v, pos0, causal):
    nq, nh, d = q.shape
    g = nh // k.shape[1]
    out = np.zeros((nq, nh, d))
    kd, vd = k.astype(np.float64), v.astype(np.float64)
    for t in range(nq):
        lim = min(pos0 + t + 1, k.shape[0]) if causal else k.shape[0]
        for h in range(nh):
            s = kd[:lim, h // g] @ q[t, h].astype(np.float64) / np.sqrt(d)
            p = np.exp(s - s.max())
            out[t, h] = p @ vd[:lim, h // g] / p.sum()
    return out


def _args(q, k, v, out, pos0, causal):
    f = 4
    return Args(q.shape[0], k.shape[0], pos0, q.shape[1], k.shape[1], causal, 1 / np.sqrt(q.shape[2]),
                q.ctypes.data, q.strides[0] // f, q.strides[1] // f,
                k.ctypes.data, k.strides[0], k.strides[1], v.ctypes.data, v.strides[0], v.strides[1],
                None, 0, out.ctypes.data, out.strides[0] // f, out.strides[1] // f)  # fmt: skip


def _call(lib, a, ws, nth):
    if nth == 1:
        lib.kattn(ctypes.byref(a), ws.ctypes.data, 0, 1)
        return
    ts = [threading.Thread(target=lib.kattn, args=(ctypes.byref(a), ws.ctypes.data, i, nth)) for i in range(nth)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()


@pytest.mark.parametrize("target,tol", [("avx512", 1e-4), ("amx_bf16", 1e-2)])
def test_strides_workspace_reuse_and_threads(target, tol):
    lib = _lib(target)
    rng = np.random.default_rng(7)
    cases = [  # (nq, nkv, heads, kv_heads, head-major q, permuted out, causal, threads)
        (1, 3000, 16, 8, False, False, 1, 4),
        (200, 700, 8, 2, True, True, 1, 3),
        (2, 1500, 4, 1, False, True, 1, 2),
        (65, 65, 4, 4, True, False, 0, 4),
    ]
    big = max(lib.kattn_workspace(ctypes.byref(_args(*_problem(rng, nq, nkv, nh, nhkv, 128, hm),
                                                         np.zeros((nq, nh, 128), np.float32), nkv - nq, c)), nth)
              for nq, nkv, nh, nhkv, hm, _, c, nth in cases)  # fmt: skip
    ws = np.zeros(big + 64, np.uint8)
    for _ in range(2):  # the second round reuses a workspace that every call must have reset
        for nq, nkv, nh, nhkv, hm, perm, causal, nth in cases:
            q, k, v = _problem(rng, nq, nkv, nh, nhkv, 128, hm)
            out = np.full((nh, nq, 128), np.nan, np.float32).transpose(1, 0, 2) if perm else np.full((nq, nh, 128), np.nan, np.float32)
            a = _args(q, k, v, out, nkv - nq, causal)
            assert lib.kattn_workspace(ctypes.byref(a), nth) <= big
            _call(lib, a, ws, nth)
            ref = _reference(q, k, v, nkv - nq, causal)
            err = np.abs(out - ref).max() / np.abs(ref).max()
            assert err < tol, (nq, nkv, nh, nhkv, err)


@pytest.mark.parametrize("nth", [1, 4])
def test_amx_repeat_runs_bit_identical(nth):
    """AMX tile data is not preserved across preemption on some KVM guests; the kernel retries
    blocks that spanned a long gap, so repeated calls must agree bit for bit (split-KV merges in
    split order, so this holds for any fixed thread count)."""
    lib = _lib("amx_bf16", dk=256)
    rng = np.random.default_rng(3)
    nq, nkv, nh, nhkv = 129, 1500, 4, 1
    q, k, v = _problem(rng, nq, nkv, nh, nhkv, 256, False)
    ref = _reference(q, k, v, nkv - nq, 1)
    first = None
    for _ in range(20):
        out = np.full((nq, nh, 256), np.nan, np.float32)
        a = _args(q, k, v, out, nkv - nq, 1)
        ws = np.zeros(lib.kattn_workspace(ctypes.byref(a), nth) + 64, np.uint8)
        _call(lib, a, ws, nth)
        if first is None:
            first = out
            assert np.abs(out - ref).max() / np.abs(ref).max() < 1e-2
        assert np.array_equal(out, first)
