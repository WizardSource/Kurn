"""Numerical check of generated kernels against the independent pure-Python
reference in kurn.formats, called in-process through ctypes. Runs every legal
configuration the host CPU can execute; others are skipped (see test_harness.py
for NEON under qemu)."""

import ctypes
import random
import zlib

import pytest

from kurn.formats import FORMATS, reference_gemm, reference_gemv
from kurn.kernels import kernel
from kurn.spec import legal_configs
from kurn.toolchain import build, run_mode

from conftest import BLOCKS

c_i64, c_vp = ctypes.c_int64, ctypes.c_void_p
F32P = ctypes.POINTER(ctypes.c_float)
SIZES = {"gemv": {"K": 512, "N": 37}, "gemm": {"K": 256, "N": 37, "M": 5}}


def _fn(lib, name, args, res=None):
    f = getattr(lib, name, None)
    if f is not None:
        f.argtypes, f.restype = args, res
    return f


def _assert_close(y, ref):
    mag = max(abs(v) for v in ref) or 1.0
    err = max(abs(a - b) for a, b in zip(y, ref))
    assert err / mag < 1e-5, f"max rel err {err / mag:.2e}"


def run_gemv(lib, c, W, x, K, N, splits):
    sym = kernel(c).entry
    plain = _fn(lib, sym, [c_vp, c_vp, F32P, c_i64, c_i64, c_i64])
    prep = _fn(lib, f"{sym}_prepare", [c_vp, c_i64, c_i64], c_vp)
    y = (ctypes.c_float * N)(*([float("nan")] * N))
    if prep:
        handle = prep(W, K, N)
        packed = _fn(lib, f"{sym}_packed", [c_vp, c_vp, F32P, c_i64, c_i64, c_i64])
        for r0, r1 in splits:
            packed(handle, x, y, K, r0, r1)
    else:
        for r0, r1 in splits:
            plain(W, x, y, K, r0, r1)
    return list(y)


def run_gemm(lib, W, X, K, N, M, splits):
    if init := _fn(lib, "kern_thread_init", []):
        init()
    Y = (ctypes.c_float * (N * M))(*([float("nan")] * (N * M)))
    if prep := _fn(lib, "kq8_gemm_prepare", [c_vp, c_i64, c_i64], c_vp):
        handle = prep(W, K, N)
        packed = _fn(lib, "kq8_gemm_packed", [c_vp, c_vp, F32P, c_i64, c_i64, c_i64, c_i64, c_i64])
        for n0, n1 in splits:
            packed(handle, X, Y, K, N, M, n0, n1)
    else:
        gemm = _fn(lib, "kq8_gemm", [c_vp, c_vp, F32P, c_i64, c_i64, c_i64, c_i64, c_i64])
        for n0, n1 in splits:
            gemm(W, X, Y, K, N, M, n0, n1)
    return list(Y)


def run_verify(lib, c, W, X, K, N, M, splits):
    """Multi-token verify kernels export only the _prepare / _packed pair (M <= cols)."""
    sym = kernel(c).entry
    prep = _fn(lib, f"{sym}_prepare", [c_vp, c_i64, c_i64], c_vp)
    packed = _fn(lib, f"{sym}_packed", [c_vp, c_vp, F32P, c_i64, c_i64, c_i64, c_i64, c_i64])
    Y = (ctypes.c_float * (N * M))(*([float("nan")] * (N * M)))
    handle = prep(W, K, N)
    for n0, n1 in splits:
        packed(handle, X, Y, K, N, M, n0, n1)
    return list(Y)


CONFIGS = list(legal_configs())


def _id(c):
    return "-".join(str(c[k]) for k in ("weights", "op", "target", "layout", "align", "rows", "cols", "act", "prefetch"))


@pytest.mark.parametrize("data", ["random", "extreme"])
@pytest.mark.parametrize("c", CONFIGS, ids=_id)
def test_kernel_matches_reference(c, data):
    mode, why = run_mode(c["target"])
    if mode != "native":
        pytest.skip(why or "needs qemu (covered in test_harness.py)")
    rng = random.Random(zlib.crc32(f"{_id(c)}/{data}".encode()))
    fmt = FORMATS[c["weights"]]
    act = fmt.act
    sz = SIZES["gemv" if c["op"] == "verify" else c["op"]]
    K, N = sz["K"], sz["N"]
    extreme = data == "extreme"
    W = BLOCKS[c["weights"]](rng, N * K // fmt.block, extreme)
    lib = ctypes.CDLL(build(c))
    # one call over all rows, then the same rows split at a 16-row boundary (thread slices)
    for splits in ([(0, N)], [(0, 16), (16, 32), (32, N)]):
        if c["op"] == "gemv":
            x = BLOCKS[act](rng, K // FORMATS[act].block, extreme)
            _assert_close(run_gemv(lib, c, W, x, K, N, splits), reference_gemv(fmt, W, x, K, N))
        elif c["op"] == "verify":
            M = max(2, c["cols"] - 1)  # one column fewer than compiled exercises the padding
            X = BLOCKS[act](rng, M * K // FORMATS[act].block, extreme)
            _assert_close(run_verify(lib, c, W, X, K, N, M, splits), reference_gemm(fmt, W, X, K, N, M))
        else:
            M = sz["M"]
            X = BLOCKS[act](rng, M * K // FORMATS[act].block, extreme)
            _assert_close(run_gemm(lib, W, X, K, N, M, splits), reference_gemm(fmt, W, X, K, N, M))
