"""kurn.epilogue Q4_0 engine GEMVs (kq4e_*): Q4_0 weights in the vnni16 layout (two k-steps per
byte), Q8_0 activations, the same epilogues as the Q8_0 kernels. Checked against a float64
reference built from the native block_q4_0 bytes (gguf's quantizer), and fused vs unfused SwiGLU."""

import atexit
import ctypes
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from kurn import spec, toolchain
from kurn.epilogue import engine_kernels

gguf = pytest.importorskip("gguf")

MODEL_DIR = Path(__file__).resolve().parent.parent / "src" / "kurn" / "model"
F, I64 = ctypes.c_void_p, ctypes.c_int64
Q4 = np.dtype([("d", "<f2"), ("qs", "u1", 16)])
Q8 = np.dtype([("d", "<f2"), ("qs", "i1", 32)])
ACT_BYTES = (8 * 1024 + 1024 + 1024) * 4

pytestmark = pytest.mark.skipif("avx512_vnni" not in toolchain.cpu_flags(), reason="host CPU lacks AVX-512 VNNI")
_OUT = tempfile.mkdtemp(prefix="kq4e-")
atexit.register(shutil.rmtree, _OUT, True)


@pytest.fixture(scope="module", params=[(8, "packed", 0), (4, 64, 8), (1, "packed", 8)], ids=lambda p: f"rows{p[0]}_align{p[1]}_pf{p[2]}")
def lib(request):
    rows, align, pf = request.param
    c = spec.resolve({"op": "gemv", "weights": "q8_0", "target": "avx512_vnni", "layout": "vnni16", "rows": rows, "align": align,
                      "prefetch": pf})
    src = engine_kernels(c, fmts=("q8_0", "q4_0"))
    so = toolchain.compile_source(src, "avx512_vnni", out_dir=_OUT, stem="kq4e", extra_flags=["-I", str(MODEL_DIR)])
    L = ctypes.CDLL(so)
    L.kq4e_blk_bytes.restype = ctypes.c_size_t
    L.kq4e_store_sumsq.restype = ctypes.c_float
    for name in ("kq4e_store", "kq4e_store_sumsq", "kq8e_store"):
        getattr(L, name).argtypes = [F, I64, F, I64, I64, I64, I64, F]
    L.kq4e_axpy.argtypes = [F, I64, F, I64, I64, I64, I64, F, ctypes.c_float]
    L.kq4e_swiglu_q8.argtypes = [F, I64, F, I64, I64, I64, I64, F, F]
    L.kq4e_pack.argtypes = [F, F, I64, I64]
    L.kq8e_prep.argtypes = [F, I64, F]
    L.kq8e_quantize.argtypes = [F, F, I64]
    L.kq8e_swiglu.argtypes = [F, F, F, I64]
    return L


def _q4_weights(rng, n, nb):
    w = (rng.standard_normal((n, 32 * nb)) * rng.uniform(0.01, 1.0, (n, 1))).astype(np.float32)
    raw = gguf.quants.quantize(w, gguf.GGMLQuantizationType.Q4_0)
    return np.frombuffer(raw.tobytes(), Q4).reshape(n, nb).copy()


def _q4_values(w):
    lo, hi = (w["qs"] & 0x0F).astype(np.int64), (w["qs"] >> 4).astype(np.int64)
    return np.concatenate([lo, hi], axis=-1) - 8  # [n, nb, 32]


def _q8(x):
    xb = x.astype(np.float32).reshape(-1, 32)
    amax = np.abs(xb).max(axis=1)
    inv = np.where(amax != 0, np.float32(127) / np.where(amax != 0, amax, 1), 0).astype(np.float32)
    out = np.zeros(len(xb), Q8)
    out["d"] = (amax / np.float32(127)).astype(np.float16)
    out["qs"] = np.rint(xb * inv[:, None]).astype(np.int8)
    return out


def _ref(w, xq, b0=0, b1=None):
    b1 = w.shape[1] if b1 is None else b1
    dots = np.einsum("nbk,bk->nb", _q4_values(w)[:, b0:b1], xq["qs"].astype(np.int64))
    return (dots * w["d"][:, b0:b1].astype(np.float64) * xq["d"].astype(np.float64)[None, :]).sum(axis=1)


class K4:
    def __init__(self, lib, w):
        self.lib, self.n, self.nb = lib, w.shape[0], w.shape[1]
        self.ng = (self.n + 15) // 16
        self.buf = np.zeros(self.ng * self.nb * lib.kq4e_blk_bytes() + 64, np.uint8)
        self.pk = self.buf.ctypes.data + (-self.buf.ctypes.data) % 64
        src = np.ascontiguousarray(w)
        lib.kq4e_pack(self.pk, src.ctypes.data, self.nb, self.n)

    def act(self, xq):
        a = ctypes.create_string_buffer(ACT_BYTES)
        xq = np.ascontiguousarray(xq)
        self.lib.kq8e_prep(xq.ctypes.data, len(xq), a)
        return a


@pytest.mark.parametrize("n,nb", [(16, 1), (40, 3), (256, 64), (2048, 8), (100, 24)])
def test_q4_store_axpy_sumsq_match_reference(lib, n, nb):
    rng = np.random.default_rng(n + 7 * nb)
    w = _q4_weights(rng, n, nb)
    xq = _q8(rng.standard_normal(32 * nb))
    k = K4(lib, w)
    a = k.act(xq)
    ref = _ref(w, xq)
    tol = 1e-5 * np.abs(ref).max()
    y = np.zeros(k.ng * 16, np.float32)
    lib.kq4e_store(k.pk, nb, a, 0, nb, 0, k.ng, y.ctypes.data)
    assert np.abs(y[:n] - ref).max() <= tol
    assert (y[n:] == 0).all()  # padding rows of the last group
    y2 = np.zeros_like(y)
    ss = lib.kq4e_store_sumsq(k.pk, nb, a, 0, nb, 0, k.ng, y2.ctypes.data)
    assert np.array_equal(y, y2) and abs(ss - float((ref**2).sum())) <= 1e-5 * float((ref**2).sum())
    y0 = rng.standard_normal(k.ng * 16).astype(np.float32)
    y3 = y0.copy()
    lib.kq4e_axpy(k.pk, nb, a, 0, nb, 0, k.ng, y3.ctypes.data, ctypes.c_float(-0.6))
    assert np.abs(y3[:n] - (y0[:n] - 0.6 * ref)).max() <= tol + 1e-6 * np.abs(y0).max()


def test_q4_group_and_block_ranges(lib):
    rng = np.random.default_rng(1)
    n, nb = 192, 16
    w = _q4_weights(rng, n, nb)
    k = K4(lib, w)
    for b0, b1, g0, g1 in [(0, 16, 2, 9), (3, 11, 0, 12), (15, 16, 5, 6)]:
        xq = _q8(rng.standard_normal(32 * (b1 - b0)))
        y = np.full(n, 99.0, np.float32)
        lib.kq4e_store(k.pk, nb, k.act(xq), b0, b1, g0, g1, y.ctypes.data)
        ref = _ref(w, xq, b0, b1)
        sl = slice(16 * g0, 16 * g1)
        assert np.abs(y[sl] - ref[sl]).max() <= 1e-5 * np.abs(ref).max()
        assert (y[: 16 * g0] == 99.0).all() and (y[16 * g1 :] == 99.0).all()


@pytest.mark.parametrize("nf,nb", [(32, 2), (768, 64)])
def test_q4_swiglu_q8_bit_identical_to_unfused(lib, nf, nb):
    rng = np.random.default_rng(nf)
    gate, up = _q4_weights(rng, nf, nb), _q4_weights(rng, nf, nb)
    inter = np.zeros((2 * nf, nb), Q4)
    for j in range(nf // 16):
        inter[32 * j : 32 * j + 16], inter[32 * j + 16 : 32 * j + 32] = gate[16 * j : 16 * j + 16], up[16 * j : 16 * j + 16]
    xq = _q8(rng.standard_normal(32 * nb) * 2)
    k = K4(lib, inter)
    a = k.act(xq)
    y = np.zeros(2 * nf, np.float32)
    lib.kq4e_store(k.pk, nb, a, 0, nb, 0, k.ng, y.ctypes.data)
    g = np.concatenate([y[32 * j : 32 * j + 16] for j in range(nf // 16)])
    u = np.concatenate([y[32 * j + 16 : 32 * j + 32] for j in range(nf // 16)])
    act = np.zeros(nf, np.float32)
    lib.kq8e_swiglu(g.ctypes.data, u.ctypes.data, act.ctypes.data, nf)
    q_unfused = np.zeros(nf // 32, Q8)
    lib.kq8e_quantize(act.ctypes.data, q_unfused.ctypes.data, nf)
    q = np.zeros(nf // 32, Q8)
    lib.kq4e_swiglu_q8(k.pk, nb, a, 0, nb, 0, k.ng, q.ctypes.data, None)
    assert q.tobytes() == q_unfused.tobytes()
    gr, ur = _ref(gate, xq), _ref(up, xq)
    assert np.abs(act - gr / (1 + np.exp(-gr)) * ur).max() <= 2e-5 * np.abs(gr / (1 + np.exp(-gr)) * ur).max()
