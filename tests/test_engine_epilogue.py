"""kurn.epilogue: fused vnni16 GEMV epilogues, checked against an independent float64 reference
and, for the SwiGLU + Q8_0 output, bit-for-bit against ggml's own functions when libggml-cpu is
available (KURN_GGML_CPU_LIB or ~/src/llama.cpp/build/bin/libggml-cpu.so)."""

import atexit
import ctypes
import os
import random
import shutil
import tempfile
from pathlib import Path

import pytest

import kurn.model
from kurn import spec, toolchain
from kurn.epilogue import engine_kernels
from kurn.kernels import generate

np = pytest.importorskip("numpy")

MODEL_DIR = Path(kurn.model.__file__).resolve().parent  # installed package or src/, whichever is imported
F = ctypes.c_void_p
I64 = ctypes.c_int64

pytestmark = pytest.mark.skipif("avx512_vnni" not in toolchain.cpu_flags(), reason="host CPU lacks AVX-512 VNNI")


def _cfg(**kw):
    base = {"op": "gemv", "weights": "q8_0", "target": "avx512_vnni", "layout": "vnni16"}
    return spec.resolve({**base, **kw})


_OUT = tempfile.mkdtemp(prefix="kq8e-")  # per process: toolchain's shared cache writes <hash>.c in place
atexit.register(shutil.rmtree, _OUT, True)


def _lib(c):
    so = toolchain.compile_source(engine_kernels(c), "avx512_vnni", out_dir=_OUT, stem="kq8e", extra_flags=["-I", str(MODEL_DIR)])
    lib = ctypes.CDLL(so)
    lib.kq8e_blk_bytes.restype = ctypes.c_size_t
    lib.kq8e_store_sumsq.restype = ctypes.c_float
    for name in ("kq8e_store", "kq8e_store_sumsq"):
        getattr(lib, name).argtypes = [F, I64, F, I64, I64, I64, I64, F]
    lib.kq8e_axpy.argtypes = [F, I64, F, I64, I64, I64, I64, F, ctypes.c_float]
    lib.kq8e_swiglu_q8.argtypes = [F, I64, F, I64, I64, I64, I64, F, F]
    lib.kq8e_pack.argtypes = [F, F, I64, I64]
    lib.kq8e_prep.argtypes = [F, I64, F]
    lib.kq8e_quantize.argtypes = [F, F, I64]
    lib.kq8e_swiglu.argtypes = [F, F, F, I64]
    return lib


def _ggml():
    p = os.environ.get("KURN_GGML_CPU_LIB", os.path.expanduser("~/src/llama.cpp/build/bin/libggml-cpu.so"))
    if not os.path.exists(p):
        return None
    lib = ctypes.CDLL(p)
    lib.quantize_row_q8_0.argtypes = [F, F, I64]
    lib.ggml_vec_swiglu_f32.argtypes = [ctypes.c_int, F, F, F]
    return lib


BLOCK = np.dtype([("d", "<f2"), ("qs", "i1", 32)])
ACT_BYTES = (8 * 1024 + 1024 + 1024) * 4  # kq8e_act with KQ8E_MAXNB = 1024


def _weights(rng, n, nb):
    w = np.zeros((n, nb), BLOCK)
    w["d"] = rng.uniform(2**-9, 2**-5, (n, nb)).astype(np.float16)
    w["qs"] = rng.integers(-127, 128, (n, nb, 32))
    return w


def _quant_ref(x):
    """ggml quantize_row_q8_0 semantics in numpy (x86 path: round half to even)."""
    xb = x.astype(np.float32).reshape(-1, 32)
    amax = np.abs(xb).max(axis=1)
    d = (amax / np.float32(127)).astype(np.float32)
    idv = np.where(amax != 0, np.float32(127) / np.where(amax != 0, amax, 1), 0).astype(np.float32)
    out = np.zeros(len(xb), BLOCK)
    out["d"] = d.astype(np.float16)
    out["qs"] = np.rint(xb * idv[:, None]).astype(np.int8)
    return out


def _gemv_ref(w, xq, b0=0, b1=None):
    b1 = w.shape[1] if b1 is None else b1
    wd = w["d"][:, b0:b1].astype(np.float64)
    xd = xq["d"].astype(np.float64)
    dots = np.einsum("nbk,bk->nb", w["qs"][:, b0:b1].astype(np.int64), xq["qs"].astype(np.int64))
    return (dots * wd * xd[None, :]).sum(axis=1)


class Kern:
    def __init__(self, lib, w):
        self.lib, self.n, self.nb = lib, w.shape[0], w.shape[1]
        ng = (self.n + 15) // 16
        self.buf = np.zeros(ng * self.nb * lib.kq8e_blk_bytes() + 64, np.uint8)
        self.pk = self.buf.ctypes.data + (-self.buf.ctypes.data) % 64  # align=64 layouts use aligned loads
        src = np.ascontiguousarray(w)
        lib.kq8e_pack(self.pk, src.ctypes.data, self.nb, self.n)
        self.ng = ng

    def act(self, xq):
        a = ctypes.create_string_buffer(ACT_BYTES)
        xq = np.ascontiguousarray(xq)
        self.lib.kq8e_prep(xq.ctypes.data, len(xq), a)
        return a


@pytest.fixture(
    scope="module", params=[(8, "packed", 0), (4, 64, 8), (2, "packed", 8), (1, 64, 0)], ids=lambda p: f"rows{p[0]}_align{p[1]}_pf{p[2]}"
)
def lib(request):
    rows, align, pf = request.param
    return _lib(_cfg(rows=rows, align=align, prefetch=pf))


@pytest.mark.parametrize("n,nb", [(16, 1), (48, 3), (256, 64), (2048, 8), (512, 192)])
def test_store_axpy_sumsq_match_reference(lib, n, nb):
    rng = np.random.default_rng(n * 1000 + nb)
    w = _weights(rng, n, nb)
    xq = _quant_ref(rng.standard_normal(32 * nb).astype(np.float32))
    k = Kern(lib, w)
    a = k.act(xq)
    ref = _gemv_ref(w, xq)
    tol = 1e-5 * np.abs(ref).max()
    y = np.zeros(k.ng * 16, np.float32)
    lib.kq8e_store(k.pk, nb, a, 0, nb, 0, k.ng, y.ctypes.data)
    assert np.abs(y[:n] - ref).max() <= tol
    y2 = np.zeros(k.ng * 16, np.float32)
    ss = lib.kq8e_store_sumsq(k.pk, nb, a, 0, nb, 0, k.ng, y2.ctypes.data)
    assert np.array_equal(y, y2)
    assert abs(ss - float((ref**2).sum())) <= 1e-5 * float((ref**2).sum())
    y0 = rng.standard_normal(k.ng * 16).astype(np.float32)
    y3 = y0.copy()
    lib.kq8e_axpy(k.pk, nb, a, 0, nb, 0, k.ng, y3.ctypes.data, ctypes.c_float(0.37))
    assert np.abs(y3[:n] - (y0[:n] + 0.37 * ref)).max() <= tol + 1e-6 * np.abs(y0).max()


def test_group_and_block_ranges(lib):
    """Row groups [g0, g1) and a K slice [b0, b1) (column-split GEMV: activation covers only the slice)."""
    rng = np.random.default_rng(7)
    n, nb = 256, 24
    w = _weights(rng, n, nb)
    k = Kern(lib, w)
    for b0, b1, g0, g1 in [(0, 24, 3, 11), (5, 13, 0, 16), (23, 24, 8, 9), (0, 1, 0, 16)]:
        xq = _quant_ref(rng.standard_normal(32 * (b1 - b0)).astype(np.float32))
        a = k.act(xq)
        y = np.full(n, 99.0, np.float32)
        lib.kq8e_store(k.pk, nb, a, b0, b1, g0, g1, y.ctypes.data)
        ref = _gemv_ref(w, xq, b0, b1)
        sl = slice(16 * g0, 16 * g1)
        assert np.abs(y[sl] - ref[sl]).max() <= 1e-5 * np.abs(ref).max()
        assert (y[: 16 * g0] == 99.0).all() and (y[16 * g1 :] == 99.0).all()


def _gate_up(rng, nf, nb):
    gate, up = _weights(rng, nf, nb), _weights(rng, nf, nb)
    inter = np.zeros((2 * nf, nb), BLOCK)
    for j in range(nf // 16):
        inter[32 * j : 32 * j + 16] = gate[16 * j : 16 * j + 16]
        inter[32 * j + 16 : 32 * j + 32] = up[16 * j : 16 * j + 16]
    return gate, up, inter


@pytest.mark.parametrize("nf,nb", [(32, 2), (96, 8), (768, 64), (1024, 64)])
def test_swiglu_q8_is_bit_identical_to_unfused(lib, nf, nb):
    rng = np.random.default_rng(nf + nb)
    gate, up, inter = _gate_up(rng, nf, nb)
    xq = _quant_ref(rng.standard_normal(32 * nb).astype(np.float32) * 3)
    k = Kern(lib, inter)
    a = k.act(xq)
    # unfused: plain GEMV to floats, then SwiGLU, then quantize (the engine's KURN_EPILOGUE=0 path)
    y = np.zeros(2 * nf, np.float32)
    lib.kq8e_store(k.pk, nb, a, 0, nb, 0, k.ng, y.ctypes.data)
    g = np.concatenate([y[32 * j : 32 * j + 16] for j in range(nf // 16)])
    u = np.concatenate([y[32 * j + 16 : 32 * j + 32] for j in range(nf // 16)])
    act = np.zeros(nf, np.float32)
    lib.kq8e_swiglu(g.ctypes.data, u.ctypes.data, act.ctypes.data, nf)
    q_unfused = np.zeros(nf // 32, BLOCK)
    lib.kq8e_quantize(act.ctypes.data, q_unfused.ctypes.data, nf)
    # fused
    q = np.zeros(nf // 32, BLOCK)
    act_f = np.zeros(nf, np.float32)
    lib.kq8e_swiglu_q8(k.pk, nb, a, 0, nb, 0, k.ng, q.ctypes.data, act_f.ctypes.data)
    assert q.tobytes() == q_unfused.tobytes()
    assert act_f.tobytes() == act.tobytes()
    # and the float activation against float64 math
    gr, ur = _gemv_ref(gate, xq), _gemv_ref(up, xq)
    ref = gr / (1 + np.exp(-gr)) * ur
    assert np.abs(act - ref).max() <= 2e-5 * np.abs(ref).max()
    gg = _ggml()
    if gg is not None:  # ggml's own SwiGLU + quantize_row_q8_0 on the same GEMV outputs
        act_g = np.zeros(nf, np.float32)
        gg.ggml_vec_swiglu_f32(nf, act_g.ctypes.data, g.ctypes.data, u.ctypes.data)
        assert act_g.tobytes() == act.tobytes()
        q_g = np.zeros(nf // 32, BLOCK)
        gg.quantize_row_q8_0(act_g.ctypes.data, q_g.ctypes.data, nf)
        assert q.tobytes() == q_g.tobytes()


def test_swiglu_q8_group_subrange(lib):
    """MoE units: only FF blocks [fb0, fb1) of an expert; Q8_0 blocks land at their absolute index."""
    rng = np.random.default_rng(3)
    nf, nb = 256, 16
    _, _, inter = _gate_up(rng, nf, nb)
    k = Kern(lib, inter)
    a = k.act(_quant_ref(rng.standard_normal(32 * nb).astype(np.float32)))
    full = np.zeros(nf // 32, BLOCK)
    lib.kq8e_swiglu_q8(k.pk, nb, a, 0, nb, 0, k.ng, full.ctypes.data, None)
    part = np.zeros(nf // 32, BLOCK)
    lib.kq8e_swiglu_q8(k.pk, nb, a, 0, nb, 4 * 3, 4 * 6, part.ctypes.data, None)
    assert part[3:6].tobytes() == full[3:6].tobytes()
    assert not part[:3].tobytes().strip(b"\0") and not part[6:].tobytes().strip(b"\0")


def test_quantize_and_swiglu_match_ggml_exactly():
    gg = _ggml()
    if gg is None:
        pytest.skip("libggml-cpu not found (set KURN_GGML_CPU_LIB)")
    lib = _lib(_cfg(rows=8))
    rng = np.random.default_rng(11)
    x = rng.standard_normal(32 * 512).astype(np.float32) * rng.uniform(0.01, 100, 32 * 512).astype(np.float32)
    x[:32] = 0  # all-zero block (d = 0, id = 0)
    x[32:64] = np.arange(32, dtype=np.float32) - 15.5  # exact .5 ties after scaling for some values
    x[64:96] = 127.0 * (np.arange(32) % 3 - 1) / 2  # values that land exactly on k + 0.5
    ours, ref = np.zeros(512, BLOCK), np.zeros(512, BLOCK)
    lib.kq8e_quantize(x.ctypes.data, ours.ctypes.data, len(x))
    gg.quantize_row_q8_0(x.ctypes.data, ref.ctypes.data, len(x))
    assert ours.tobytes() == ref.tobytes()
    assert ours.tobytes() == _quant_ref(x).tobytes()
    gv = (rng.standard_normal(4096) * 20).astype(np.float32)
    gv[:4] = [-200.0, 200.0, 0.0, -0.0]
    uv = rng.standard_normal(4096).astype(np.float32)
    a, b = np.zeros(4096, np.float32), np.zeros(4096, np.float32)
    lib.kq8e_swiglu(gv.ctypes.data, uv.ctypes.data, a.ctypes.data, 4096)
    gg.ggml_vec_swiglu_f32(4096, b.ctypes.data, gv.ctypes.data, uv.ctypes.data)
    assert a.tobytes() == b.tobytes()


def test_epilogue_schedule_key():
    c = _cfg(rows=8)
    assert c["epilogue"] == "none"
    fused = generate(dict(c, epilogue="fused"))
    base = generate(c)
    assert fused.startswith(base) and "kq8e_swiglu_q8" in fused and "kq8e_swiglu_q8" not in base
    with pytest.raises(spec.SpecError):
        spec.resolve({"op": "gemv", "weights": "q8_0", "target": "avx512_vnni", "layout": "native", "epilogue": "fused"})
    with pytest.raises(spec.SpecError):
        spec.resolve({"op": "gemv", "weights": "q4_K", "target": "avx512_vnni", "epilogue": "fused"})
    so = toolchain.compile_source(fused, "avx512_vnni", out_dir=_OUT, stem="kq8e_fused")
    lib = ctypes.CDLL(so)
    assert lib.kq8e_swiglu_q8 and lib.kq8_gemv_packed


def test_fused_build_matches_base_kernel():
    """The appended epilogue entry points use the base kernel's packed layout: kq8_gemv_packed on
    kq8_gemv_prepare's buffer and kq8e_store on kq8e_pack's buffer give the same floats."""
    c = _cfg(rows=4, prefetch=8)
    lib = ctypes.CDLL(toolchain.compile_source(generate(dict(c, epilogue="fused")), "avx512_vnni", out_dir=_OUT, stem="kq8e_fused"))
    rng = np.random.default_rng(5)
    n, nb = 64, 8
    w = np.ascontiguousarray(_weights(rng, n, nb))
    xq = np.ascontiguousarray(_quant_ref(rng.standard_normal(32 * nb).astype(np.float32)))
    lib.kq8_gemv_prepare.restype = F
    lib.kq8_gemv_prepare.argtypes = [F, I64, I64]
    lib.kq8_gemv_packed.argtypes = [F, F, F, I64, I64, I64]
    pk = lib.kq8_gemv_prepare(w.ctypes.data, 32 * nb, n)
    y1 = np.zeros(n, np.float32)
    lib.kq8_gemv_packed(pk, xq.ctypes.data, y1.ctypes.data, 32 * nb, 0, n)
    lib.kq8e_blk_bytes.restype = ctypes.c_size_t
    lib.kq8e_pack.argtypes = [F, F, I64, I64]
    lib.kq8e_prep.argtypes = [F, I64, F]
    lib.kq8e_store.argtypes = [F, I64, F, I64, I64, I64, I64, F]
    buf = ctypes.create_string_buffer(4 * nb * lib.kq8e_blk_bytes())
    lib.kq8e_pack(buf, w.ctypes.data, nb, n)
    a = ctypes.create_string_buffer(ACT_BYTES)
    lib.kq8e_prep(xq.ctypes.data, nb, a)
    y2 = np.zeros(n, np.float32)
    lib.kq8e_store(buf, nb, a, 0, nb, 0, 4, y2.ctypes.data)
    assert y1.tobytes() == y2.tobytes()


def test_random_shapes_fuzz(lib):
    r = random.Random(99)
    for _ in range(6):
        nb = r.choice([1, 2, 5, 32, 96])
        n = 16 * r.randint(1, 40)
        rng = np.random.default_rng(r.randrange(1 << 30))
        w = _weights(rng, n, nb)
        xq = _quant_ref(rng.standard_normal(32 * nb).astype(np.float32))
        k = Kern(lib, w)
        y = np.zeros(k.ng * 16, np.float32)
        lib.kq8e_store(k.pk, nb, k.act(xq), 0, nb, 0, k.ng, y.ctypes.data)
        ref = _gemv_ref(w, xq)
        assert np.abs(y[:n] - ref).max() <= 1e-5 * np.abs(ref).max()
