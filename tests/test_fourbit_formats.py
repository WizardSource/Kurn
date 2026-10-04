"""MXFP4 / NVFP4 (kurn.mx): references, decoders and numpy quantizers."""

import ctypes
import os
import random
import struct

import pytest

from kurn import mx
from kurn.formats import FORMATS, reference_dot, split_blocks

from conftest import BLOCKS

np = pytest.importorskip("numpy")

LIBGGML = os.path.expanduser(os.environ.get("KURN_LIBGGML", "~/src/llama.cpp/build/bin/libggml-base.so"))


def _q8_values(xb):
    d = struct.unpack_from("<e", xb, 0)[0]
    return np.array(struct.unpack_from("<32b", xb, 2), dtype=np.float64) * d


@pytest.mark.parametrize("fmt", ["mxfp4", "nvfp4"])
def test_reference_equals_dequantized_dot(fmt, rng):
    f = FORMATS[fmt]
    k = 512
    w = BLOCKS[fmt](rng, k // f.block)
    x = BLOCKS["q8_0"](rng, k // 32)
    deq = (mx.dequantize_mxfp4 if fmt == "mxfp4" else mx.dequantize_nvfp4)(w, k).astype(np.float64)[0]
    xv = np.concatenate([_q8_values(b) for b in split_blocks("q8_0", x)])
    ref = reference_dot(fmt, split_blocks(fmt, w), split_blocks("q8_0", x))
    assert ref == pytest.approx(float(deq @ xv), rel=1e-12, abs=1e-12)


def test_scalar_decoders_match_ggml_definitions():
    assert mx.e8m0_half(0) == 2.0**-128 and mx.e8m0_half(1) == 2.0**-127 and mx.e8m0_half(127) == 0.5
    assert mx.ue4m3_half(0x7F) == 0 and mx.ue4m3_half(0) == 0
    assert mx.ue4m3_half(0x38) == 0.5  # 1.0 / 2
    assert mx.ue4m3_half(0x7E) == 224.0  # 448 / 2
    assert mx.ue4m3_half(3) == 3 * 2.0**-10  # subnormal: 3 * 2^-9 / 2
    u = np.arange(256, dtype=np.uint8)
    assert list(mx.ue4m3_to_fp32_half(u)) == [np.float32(mx.ue4m3_half(int(v))) for v in u]


def _ggml_fp32_to_ue4m3(x):  # line-by-line port of ggml_fp32_to_ue4m3
    x = np.float32(x)
    if not x > 0:
        return 0
    x = min(x, np.float32(448.0))
    bits = int(np.array([x], dtype=np.float32).view(np.uint32)[0])
    fexp, fman = ((bits >> 23) & 0xFF) - 127, (bits >> 20) & 7
    e = fexp + 7
    if e <= 0:
        man = int(x * np.float32(512.0) + np.float32(0.5))
        return 0 if man < 1 else min(man, 7)
    if e >= 15:
        return 0x7E
    m = fman + ((bits >> 19) & 1)
    if m > 7:
        m, e = 0, e + 1
        if e >= 15:
            return 0x7E
    return (e << 3) | m


def test_ue4m3_encoder_matches_ggml():
    r = np.random.default_rng(7)
    xs = np.concatenate([
        np.exp2(r.uniform(-14, 10, 20000)).astype(np.float32),
        np.array([0, -1, 2.0**-10, 2.0**-9, 1.5 * 2.0**-9, 2.0**-6, 448, 480, 1e9, 0.0625 * 1.0625], dtype=np.float32),
    ])  # fmt: skip
    assert list(mx.fp32_to_ue4m3(xs)) == [_ggml_fp32_to_ue4m3(v) for v in xs]


def _ggml():
    if not os.path.exists(LIBGGML):
        pytest.skip(f"{LIBGGML} not found (set KURN_LIBGGML)")
    lib = ctypes.CDLL(LIBGGML)
    for sym in ("quantize_row_mxfp4_ref", "quantize_row_nvfp4_ref"):
        if not hasattr(lib, sym):
            pytest.skip(f"{sym} not exported by {LIBGGML}")
    return lib


@pytest.mark.parametrize("fmt", ["mxfp4", "nvfp4"])
def test_ggml_method_is_bit_identical_to_ggml(fmt):
    lib = _ggml()
    r = np.random.default_rng(3)
    k, n = 4096, 16
    x = (r.standard_normal((n, k)) * np.exp2(r.uniform(-9, 2, (n, 1)))).astype(np.float32)
    x[0, :64] = 0  # an all-zero block / sub-block
    x[1, :32] = 2.0**-3  # exact powers of two
    f = FORMATS[fmt]
    buf = ctypes.create_string_buffer(n * f.row_bytes(k))
    getattr(lib, f"quantize_row_{fmt}_ref")(x.ctypes.data_as(ctypes.c_void_p), buf, ctypes.c_int64(n * k))
    ours = mx.quantize_mxfp4(x) if fmt == "mxfp4" else mx.quantize_nvfp4(x)[0]
    assert ours == buf.raw


def test_mxfp4_mse_method_reduces_error():
    r = np.random.default_rng(5)
    x = (r.standard_normal((64, 2048)) * 0.02).astype(np.float32)
    err = {m: float(np.sum((mx.dequantize_mxfp4(mx.quantize_mxfp4(x, m), 2048) - x) ** 2)) for m in ("ggml", "mse")}
    assert err["mse"] < 0.97 * err["ggml"]
    w = r.uniform(0.1, 10, 2048).astype(np.float32)
    werr = {
        m: float(np.sum(w * (mx.dequantize_mxfp4(mx.quantize_mxfp4(x, "mse", weights=wt), 2048) - x) ** 2))
        for m, wt in (("plain", None), ("weighted", w))
    }
    assert werr["weighted"] <= werr["plain"]


@pytest.mark.parametrize("std,gain", [(0.02, 0.85), (0.004, 0.2)])
def test_nvfp4_tensor_scale_fixes_small_weights(std, gain):
    """ggml's NVFP4 quantizer has no per-tensor scale: typical weights (|w| ~ 0.01) get
    subnormal UE4M3 scales or flush to zero. The `mse` method scales the tensor first."""
    r = np.random.default_rng(6)
    x = (r.standard_normal((64, 2048)) * std).astype(np.float32)
    qg, sg = mx.quantize_nvfp4(x, "ggml")
    qm, sm = mx.quantize_nvfp4(x, "mse")
    assert sg == 1.0 and sm == pytest.approx(float(np.max(np.abs(x))) / (6 * 448), rel=1e-6)
    eg = np.mean((mx.dequantize_nvfp4(qg, 2048, sg) - x) ** 2)
    em = np.mean((mx.dequantize_nvfp4(qm, 2048, sm) - x) ** 2)
    assert em < gain * eg
    assert em / np.mean(x**2) < 0.01  # E2M1 with a fine scale: < 1% relative MSE


def test_quantizers_reject_bad_shapes():
    with pytest.raises(ValueError):
        mx.quantize_mxfp4(np.zeros(48, dtype=np.float32))
    with pytest.raises(ValueError):
        mx.quantize_nvfp4(np.zeros(48, dtype=np.float32))
    with pytest.raises(ValueError):
        mx.quantize_mxfp4(np.zeros(32, dtype=np.float32), method="best")


def test_quantized_bytes_are_valid_blocks():
    r = random.Random(1)
    x = np.array([r.gauss(0, 1) for _ in range(256)], dtype=np.float32)
    q = mx.quantize_nvfp4(x, "mse")[0]
    assert len(q) == 4 * 36 and all(b != 0x7F for i, b in enumerate(q) if i % 36 < 4)
    assert len(mx.quantize_mxfp4(x, "mse")) == 8 * 17
