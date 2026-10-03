"""lowbit kernels beyond test_numerics: worst-case magnitudes at large K (int16
accumulation limits of the lut / addsub layouts), and the split table build API."""

import ctypes
import random
import struct

import pytest
from test_lowbit_formats import ggml_quantize_tq1_0_trits

from kurn import lowbit
from kurn.formats import FORMATS, reference_gemv
from kurn.kernels import kernel
from kurn.spec import legal_configs
from kurn.toolchain import build, run_mode

from conftest import BLOCKS, f16_bits

c_i64, c_vp = ctypes.c_int64, ctypes.c_void_p
F32P = ctypes.POINTER(ctypes.c_float)
LOWBIT = [c for c in legal_configs(op="gemv", prefetch=(0,))
          if c["layout"] in ("lut", "addsub", "k16") or c["weights"] in ("tq1_0", "q2_K")]


def _id(c):
    extra = c["lut"] if c["layout"] == "lut" else c["addsub"] if c["layout"] == "addsub" else ""
    return "-".join(str(v) for v in (c["weights"], c["target"], c["layout"], extra, c["rows"]))


def max_blocks(fmt, nblocks, rng):
    """Every weight at its largest code (value 2 for the 2-bit recipes, +1 for q1_0, trit 2 for tq1_0)."""
    if fmt == "tq1_0":
        return b"".join(ggml_quantize_tq1_0_trits([2] * 256, f16_bits(rng)) for _ in range(nblocks))
    d = lambda: struct.pack("<H", f16_bits(rng))  # noqa: E731
    if fmt == "q2_K":
        return b"".join(bytes([0xFF] * 80) + d() + d() for _ in range(nblocks))
    if fmt == "tq2_0":
        return b"".join(bytes([0xFF] * 64) + d() for _ in range(nblocks))
    return b"".join(d() + bytes([0xFF] * 16) for _ in range(nblocks))  # q1_0, q2_0


def const_act(act, nblocks, v):
    out = bytearray()
    for _ in range(nblocks):
        if act == "q8_0":
            out += struct.pack("<H32b", 0x3000, *([v] * 32))
        else:
            out += struct.pack("<f256b16h", 0.01, *([v] * 256), *([16 * v] * 16))
    return bytes(out)


def _close(y, ref):
    mag = max(abs(v) for v in ref) or 1.0
    err = max(abs(a - b) for a, b in zip(y, ref))
    assert err / mag < 1e-5, f"max rel err {err / mag:.2e}"


def _gemv(lib, c, W, x, K, N, splits):
    sym = kernel(c).entry
    y = (ctypes.c_float * N)(*([float("nan")] * N))
    prep = getattr(lib, f"{sym}_prepare", None)
    if prep is None:
        f = getattr(lib, sym)
        f.argtypes = [c_vp, c_vp, F32P, c_i64, c_i64, c_i64]
        for r0, r1 in splits:
            f(W, x, y, K, r0, r1)
        return list(y)
    prep.argtypes, prep.restype = [c_vp, c_i64, c_i64], c_vp
    packed = getattr(lib, f"{sym}_packed")
    packed.argtypes = [c_vp, c_vp, F32P, c_i64, c_i64, c_i64]
    h = prep(W, K, N)
    for r0, r1 in splits:
        packed(h, x, y, K, r0, r1)
    return list(y)


@pytest.mark.parametrize("c", LOWBIT, ids=_id)
def test_worst_case_magnitudes_large_k(c):
    mode, why = run_mode(c["target"])
    if mode != "native":
        pytest.skip(why or "not runnable here")
    rng = random.Random(1)
    fmt = FORMATS[c["weights"]]
    K, N = 4096, 40
    W = max_blocks(c["weights"], N * K // fmt.block, rng)
    lib = ctypes.CDLL(build(c))
    for v in (127, -127):
        x = const_act(fmt.act, K // FORMATS[fmt.act].block, v)
        _close(_gemv(lib, c, W, x, K, N, [(0, 16), (16, N)]), reference_gemv(fmt, W, x, K, N))


LUT = [c for c in LOWBIT if c["layout"] == "lut" and c["rows"] == 1]


@pytest.mark.parametrize("c", LUT, ids=_id)
def test_split_table_build_matches_fused(c):
    """Tables built in pieces (as by several threads before a barrier) give bit-identical rows."""
    rng = random.Random(2)
    fmt = FORMATS[c["weights"]]
    K, N = 2048, 70
    W = BLOCKS[c["weights"]](rng, N * K // fmt.block)
    x = BLOCKS[fmt.act](rng, K // FORMATS[fmt.act].block)
    lib = ctypes.CDLL(build(c))
    sym = kernel(c).entry
    info, bld, rows = (getattr(lib, f"{sym}_lut_{n}") for n in ("info", "build", "rows"))
    info.argtypes = [c_i64, ctypes.POINTER(c_i64), ctypes.POINTER(c_i64)]
    bld.argtypes = [c_vp, c_i64, c_vp, c_i64, c_i64]
    rows.argtypes = [c_vp, c_vp, F32P, c_i64, c_i64, c_i64]
    nbytes, units = c_i64(), c_i64()
    info(K, ctypes.byref(nbytes), ctypes.byref(units))
    assert units.value == K // lowbit.LutPlan(c).unit
    tabs = ctypes.create_string_buffer(nbytes.value + 64)
    base = (ctypes.addressof(tabs) + 63) & ~63
    cuts = sorted({0, units.value, *rng.sample(range(1, units.value), min(2, units.value - 1))})
    for u0, u1 in reversed(list(zip(cuts, cuts[1:]))):
        bld(x, K, base, u0, u1)
    prep = getattr(lib, f"{sym}_prepare")
    prep.argtypes, prep.restype = [c_vp, c_i64, c_i64], c_vp
    h = prep(W, K, N)
    y = (ctypes.c_float * N)()
    rows(h, base, y, K, 0, N)
    fused = _gemv(lib, c, W, x, K, N, [(0, N)])
    assert list(y) == fused
    _close(fused, reference_gemv(fmt, W, x, K, N))
