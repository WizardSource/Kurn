"""E8P lattice codebook: table structure, encode/decode, rotation, and the GEMV kernels
against the Python reference (exact: the decoded math is integer)."""

import ctypes
import random
import struct

import pytest

from kurn import codebook as cb
from kurn.formats import FORMATS, reference_gemv
from kurn.spec import resolve
from kurn.toolchain import build, data_path, run_mode

from conftest import BLOCKS

np = pytest.importorskip("numpy")


def test_e8p_table_is_e8_coset_and_parity_sorted():
    A = cb.e8p_abs().astype(int)  # 2a: odd
    assert A.shape == (256, 8) and len({tuple(r) for r in A}) == 256
    assert (A % 2 == 1).all()
    n2 = (A * A).sum(1) / 4
    assert n2.max() == 12 and (n2 <= 10).sum() == 227
    par = (A.sum(1) // 2) & 1  # parity of sum(a)
    assert (par[:128] == 0).all() and (par[128:] == 1).all()


def test_every_code_decodes_to_a_shifted_e8_point():
    codes = np.arange(65536, dtype=np.uint16)
    q = cb.e8p_decode(codes).astype(int)  # 4c
    assert len({tuple(r) for r in q}) == 65536
    t = np.where(codes & 0x80, 1, -1)
    half = (q - t[:, None]) // 2  # 2 (c - t): odd integers
    assert (half % 2 == 1).all()
    assert ((half.sum(1) // 2) % 2 == 0).all()  # sum(c - t) even: point of D8 + 1/2 in E8


def test_encode_is_nearest_point():
    rng = np.random.default_rng(0)
    u = rng.normal(size=(300, 8)) * 0.9
    codes, q = cb.e8p_encode(u)
    assert (cb.e8p_decode(codes) == q).all()
    allq = cb.e8p_decode(np.arange(65536, dtype=np.uint16)).astype(np.float64) / 4
    d_all = ((u[:, None, :] - allq[None]) ** 2).sum(2).min(1)
    d_enc = ((u - q / 4) ** 2).sum(1)
    assert np.allclose(d_enc, d_all)


def test_rotation_roundtrip_and_invariance():
    rng = np.random.default_rng(1)
    W = rng.normal(size=(5, 768))
    x = rng.normal(size=768)
    Wr = cb.rotate_weight(W)
    assert np.allclose(Wr @ cb.rotate_act(x), W @ x)
    assert np.allclose(cb.unrotate_weight(Wr), W)


def test_quantize_e8p_error_is_two_bit_class():
    rng = np.random.default_rng(2)
    W = rng.normal(size=(16, 512))
    d, codes, q = cb.quantize_e8p(W)
    rel = ((cb.dequant_e8p(d, codes) - W) ** 2).sum() / (W * W).sum()
    assert rel < 0.10  # 2-bit scalar Lloyd-Max on a Gaussian is 0.118; E8 gains ~0.2 bits
    blob = cb.pack_e8p(d, codes)
    assert len(blob) == 16 * 2 * cb.E8P_BYTES


def test_reference_matches_dequantized_math():
    rng = np.random.default_rng(3)
    W = rng.normal(size=(3, 512))
    d, codes, _ = cb.quantize_e8p(W)
    wb = cb.pack_e8p(d, codes)
    xr = random.Random(5)
    xb = BLOCKS["q8_K"](xr, 2)
    xq = np.concatenate([np.frombuffer(xb, np.int8, 256, 292 * i + 4) for i in range(2)]).astype(np.float64)
    xd = np.repeat([struct.unpack_from("<f", xb, 292 * i)[0] for i in range(2)], 256)
    y = reference_gemv("e8p", wb, xb, 512, 3)
    assert np.allclose(y, cb.dequant_e8p(d, codes) @ (xq * xd), rtol=1e-9)


C_I64, C_VP = ctypes.c_int64, ctypes.c_void_p


@pytest.mark.parametrize("target,rows,pf", [("scalar", 4, 0), ("avx512_vnni", 1, 0), ("avx512_vnni", 4, 4),
                                            ("avx512_vnni", 8, 8)])  # fmt: skip
@pytest.mark.parametrize("extreme", [False, True])
def test_e8p_kernel_exact(target, rows, pf, extreme):
    if run_mode(target)[0] != "native":
        pytest.skip("target not runnable here")
    c = resolve({"op": "gemv", "weights": "e8p", "target": target, "rows": rows, "prefetch": pf})
    lib = ctypes.CDLL(build(c))
    f = lib.ke8p_gemv
    f.argtypes = [C_VP, C_VP, ctypes.POINTER(ctypes.c_float), C_I64, C_I64, C_I64]
    rng = random.Random(7 + rows)
    K, N = 768, 37
    W = BLOCKS["e8p"](rng, N * K // 256, extreme)
    x = BLOCKS["q8_K"](rng, K // 256, extreme)
    y = (ctypes.c_float * N)()
    for r0, r1 in ((0, 16), (16, 32), (32, N)):
        f(W, x, y, K, r0, r1)
    ref = reference_gemv(FORMATS["e8p"], W, x, K, N)
    mag = max(abs(v) for v in ref)
    assert max(abs(a - b) for a, b in zip(y, ref)) / mag < 1e-5


def test_bench_table_matches_python():
    src = open(data_path("bench.c")).read()
    body = src.split("E8P_ABS4[256] = {", 1)[1].split("};", 1)[0]
    vals = [int(v.strip().removesuffix("ULL"), 16) for v in body.replace("\n", " ").split(",") if v.strip()]
    assert vals == cb.e8p_abs_u64()


def test_c_encoder_is_exact_nearest():
    rng = np.random.default_rng(7)
    u = np.concatenate([rng.standard_normal((3000, 8)) * s for s in (0.3, 1.0, 2.5)])
    c1, q1 = cb.e8p_encode(u)
    c2, q2 = cb.e8p_encode_fast(u)
    e1 = ((q1 / 4 - u) ** 2).sum(1)
    e2 = ((q2 / 4 - u) ** 2).sum(1)
    np.testing.assert_allclose(e2, e1, rtol=1e-5, atol=1e-6)
    np.testing.assert_array_equal(cb.e8p_decode(c2), q2)


def test_importance_scaled_quantize_tensor():
    rng = np.random.default_rng(3)
    W = rng.standard_normal((64, 512))
    imp = rng.lognormal(0, 1.5, 512)
    what0, parts, s = cb.quantize_tensor(W, imp, alpha=0)
    what, parts, s = cb.quantize_tensor(W, imp, alpha=0.5)
    assert len(parts) == 1 and s.shape == (512,)

    def err(Wq):
        return float((((Wq - W) ** 2) @ imp).sum() / ((W * W) @ imp).sum())

    assert err(what) < err(what0)
    # stored form reproduces W_hat: dequantized codes, un-rotated, divided by the column scales
    d, codes = parts[0]
    np.testing.assert_allclose(cb.unrotate_weight(cb.dequant_e8p(d, codes)) / s, what, rtol=1e-9, atol=1e-12)
    _, parts2, _ = cb.quantize_tensor(W, imp, stages=2)
    assert len(parts2) == 2
    assert cb.model_bpw(1, 2048) == pytest.approx(2.0625 + 16 / 2048)
