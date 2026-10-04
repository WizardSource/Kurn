"""4-bit nibble-path kernels (unpack=mask16/perm/pair, correction=dpmin) and MXFP4 /
NVFP4 corners not covered by the random blocks of test_numerics.py."""

import ctypes
import random
import struct

import pytest

from kurn import generic, spec
from kurn.formats import FORMATS, reference_gemm, reference_gemv
from kurn.kernels import kernel
from kurn.toolchain import build, run_mode

from conftest import BLOCKS

c_i64, c_vp = ctypes.c_int64, ctypes.c_void_p
F32P = ctypes.POINTER(ctypes.c_float)
K, N = 512, 40


def _native(c):
    mode, why = run_mode(c["target"])
    if mode != "native":
        pytest.skip(why or "target not runnable on this host")


def _run(c, W, X, M=1, flags=()):
    lib = ctypes.CDLL(build(c, extra_flags=flags))
    sym = kernel(c).entry
    prep = getattr(lib, f"{sym}_prepare")
    prep.argtypes, prep.restype = [c_vp, c_i64, c_i64], c_vp
    packed = getattr(lib, f"{sym}_packed")
    handle = prep(W, K, N)
    Y = (ctypes.c_float * (N * M))(*([float("nan")] * (N * M)))
    for r0, r1 in ((0, 16), (16, N)):
        if c["op"] == "verify":
            packed.argtypes = [c_vp, c_vp, F32P, c_i64, c_i64, c_i64, c_i64, c_i64]
            packed(handle, X, Y, K, N, M, r0, r1)
        else:
            packed.argtypes = [c_vp, c_vp, F32P, c_i64, c_i64, c_i64]
            packed(handle, X, Y, K, r0, r1)
    return list(Y)


def _close(y, ref, tol=1e-5):
    mag = max(abs(v) for v in ref) or 1.0
    err = max(abs(a - b) for a, b in zip(y, ref))
    assert err / mag < tol, f"max rel err {err / mag:.2e}"


PERM = [
    {"op": "gemv", "weights": w, "target": "avx512_vnni", "layout": "i16", "unpack": "perm", "rows": r}
    for w, r in (("iq4_nl", 1), ("mxfp4", 2), ("nvfp4", 1))
] + [{"op": "verify", "weights": "nvfp4", "target": "avx512_vnni", "layout": "i16", "unpack": "perm", "cols": 4}]


@pytest.mark.parametrize("cfg", PERM, ids=lambda c: f"{c['weights']}-{c['op']}")
def test_perm_fallback_without_vbmi(cfg):
    """unpack=perm dispatches at run time; -DKURN_NO_VBMI forces the vpshufb path."""
    c = spec.resolve(cfg)
    _native(c)
    rng = random.Random(11)
    f = FORMATS[c["weights"]]
    W = BLOCKS[c["weights"]](rng, N * K // f.block)
    M = 3 if c["op"] == "verify" else 1
    X = BLOCKS["q8_0"](rng, M * K // 32)
    ref = reference_gemm(f, W, X, K, N, M)
    _close(_run(c, W, X, M, flags=("-DKURN_NO_VBMI",)), ref)
    _close(_run(c, W, X, M), ref)


def _nvfp4_rows(rng, scales):
    out = bytearray()
    for _ in range(N * K // 64):
        out += bytes(rng.choice(scales) for _ in range(4)) + bytes(rng.randrange(256) for _ in range(32))
    return bytes(out)


@pytest.mark.parametrize("scales", ["packed", "unpacked"])
@pytest.mark.parametrize("target", ["avx512_vnni", "avx2_vnni"])
def test_nvfp4_subnormal_scales_exact(target, scales):
    """Rows whose UE4M3 scales are all subnormal (or 0 / 0x7F) are still exact, relative to
    their own (tiny) magnitude."""
    layout = "i16" if target == "avx512_vnni" else "i8"
    try:
        c = spec.resolve({"op": "gemv", "weights": "nvfp4", "target": target, "layout": layout, "scales": scales})
    except spec.SpecError:
        pytest.skip(f"scales={scales} not legal on {target}")
    _native(c)
    rng = random.Random(5)
    W = _nvfp4_rows(rng, [1, 2, 3, 4, 5, 6, 7, 0, 0x7F])
    x = BLOCKS["q8_0"](rng, K // 32)
    _close(_run(c, W, x), reference_gemv("nvfp4", W, x, K, N))


@pytest.mark.parametrize("target", ["avx512_vnni", "avx2_vnni"])
def test_mxfp4_low_exponents(target):
    """e in 1..3 is exact; e == 0 (scale 2^-128, a block of ~zeros) is read as 0."""
    layout = "i16" if target == "avx512_vnni" else "i8"
    c = spec.resolve({"op": "gemv", "weights": "mxfp4", "target": target, "layout": layout})
    _native(c)
    rng = random.Random(9)
    blocks = [bytes([1 + rng.randrange(3)]) + bytes(rng.randrange(256) for _ in range(16)) for _ in range(N * K // 32)]
    W = b"".join(blocks)
    x = BLOCKS["q8_0"](rng, K // 32)
    _close(_run(c, W, x), reference_gemv("mxfp4", W, x, K, N))
    zero = b"".join(bytes([0]) + b[1:] for b in blocks)
    assert _run(c, zero, x) == [0.0] * N


EXTREME = [
    {"weights": "q4_K", "unpack": "mask16", "correction": "dpmin", "rows": 2},
    {"weights": "q4_K", "unpack": "pair", "correction": "dpmin", "rows": 1},
    {"weights": "q4_K", "unpack": "pair", "correction": "act", "rows": 2},
    {"weights": "q4_0", "unpack": "pair", "rows": 1},
]


@pytest.mark.parametrize("cfg", EXTREME, ids=lambda c: "-".join(str(v) for v in c.values()))
@pytest.mark.parametrize("target", ["avx512_vnni", "avx2_vnni"])
def test_int16_and_bsum_limits(cfg, target):
    """Largest codes, scales, mins and activations: the int16 pair sums (mask16 Q4_K) and
    the split bsums (dpmin) must not overflow."""
    layout = "i16" if target == "avx512_vnni" else "i8"
    c = spec.resolve({"op": "gemv", "target": target, "layout": layout, **cfg})
    _native(c)
    f = FORMATS[c["weights"]]
    if c["weights"] == "q4_K":
        W = (struct.pack("<HH", 0x3C00, 0x3C00) + b"\xff" * 140) * (N * K // 256)
    else:
        W = (struct.pack("<H", 0x3C00) + b"\xff" * 16) * (N * K // 32)
    for sign in (127, -127):
        if f.act == "q8_K":
            x = struct.pack("<f256b16h", 0.01, *([sign] * 256), *([16 * sign] * 16)) * (K // 256)
        else:
            x = struct.pack("<H32b", 0x2000, *([sign] * 32)) * (K // 32)
        _close(_run(c, W, x), reference_gemv(f, W, x, K, N))


@pytest.mark.parametrize("w,rows,cols,M", [("q4_0", 1, 8, 8), ("q4_0", 2, 4, 4), ("q4_0", 1, 8, 5), ("q4_K", 1, 8, 8)])
def test_pair_single_chain_verify(w, rows, cols, M):
    """pair with more than PAIR_SPLIT_MAX (row group, column) pairs uses one accumulator chain;
    every column must still match the reference, and column 0 must equal the GEMV bit for bit."""
    c = spec.resolve(
        {
            "op": "verify",
            "weights": w,
            "target": "avx512_vnni",
            "layout": "i16",
            "unpack": "pair",
            "rows": rows,
            "cols": cols,
            **({"correction": "dpmin"} if w == "q4_K" else {}),
        }
    )
    _native(c)
    rng = random.Random(5)
    f = FORMATS[w]
    W = BLOCKS[w](rng, N * K // f.block)
    X = BLOCKS[f.act](rng, M * K // 256 if f.act == "q8_K" else M * K // 32)
    y = _run(c, W, X, M)
    _close(y, reference_gemm(f, W, X, K, N, M))
    g = spec.resolve(
        {
            "op": "gemv",
            "weights": w,
            "target": "avx512_vnni",
            "layout": "i16",
            "unpack": "pair",
            "rows": 1,
            **({"correction": "dpmin"} if w == "q4_K" else {}),
        }
    )
    assert _run(g, W, X) == y[:N]


ILV = [("q4_0", "pair", 4, 4), ("q4_0", "pair", 2, 4), ("q4_0", "mask16", 4, 4), ("q4_K", "mask16", 2, 2),
       ("iq4_nl", "perm", 4, 2), ("q4_0", "pair", 1, 4)]  # fmt: skip


@pytest.mark.parametrize("w,unpack,rows,ilv", ILV, ids=lambda v: str(v))
@pytest.mark.parametrize("target", ["avx512_vnni", "avx2_vnni"])
def test_ilv_is_a_pure_reordering(w, unpack, rows, ilv, target):
    """ilv interleaves the records of consecutive row groups: GEMV and verify results must be
    bit-identical to ilv=1 for any row range, including ranges not aligned to ilv groups."""
    if target == "avx2_vnni" and unpack == "perm":
        pytest.skip("perm is AVX-512 only")
    layout = "i16" if target == "avx512_vnni" else "i8"
    extra = {"correction": "dpmin"} if w == "q4_K" else {}
    base = {"weights": w, "target": target, "layout": layout, "unpack": unpack, **extra}
    rng = random.Random(3)
    f = FORMATS[w]
    W = BLOCKS[w](rng, N * K // f.block)
    X = BLOCKS[f.act](rng, 4 * K // (256 if f.act == "q8_K" else 32))
    for op, M in (("gemv", 1), ("verify", 3)):
        kw = {"rows": rows} if op == "gemv" else {"cols": 4}
        a = spec.resolve({"op": op, **base, **kw, "ilv": ilv})
        b = spec.resolve({"op": op, **base, **kw})
        _native(a)
        ya = _run(a, W, X, M)
        assert ya == _run(b, W, X, M)
        _close(ya, reference_gemm(f, W, X, K, N, M))


XPREP = [("q4_0", "pair", "gemv", 4), ("q4_0", "pair", "verify", 8), ("q4_0", "mask16", "verify", 4),
         ("q4_K", "pair", "gemv", 2), ("q4_K", "pair", "verify", 8), ("q4_K", "mask16", "gemv", 1),
         ("iq4_nl", "perm", "gemv", 4), ("iq4_nl", "perm", "verify", 2), ("q8_0", "none", "gemv", 4),
         ("q8_0", "none", "verify", 8), ("q4_0", "mask", "gemv", 2), ("q2_0", "mask", "gemv", 2),
         ("q1_0", "mask", "verify", 4), ("tq2_0", "mask", "gemv", 2), ("iq4_nl", "lut", "verify", 8),
         ("q8_0", "none", "gemv", 8)]  # fmt: skip


def run_xprep(c, W, X, C, cols, splits):
    """E_packed_x over a workspace built by E_xprep in pieces, columns [c0, c0 + cols) at a time."""
    lib = ctypes.CDLL(build(c))
    e = kernel(c).entry
    f = FORMATS[c["weights"]]
    prep = getattr(lib, f"{e}_prepare")
    prep.argtypes, prep.restype = [c_vp, c_i64, c_i64], c_vp
    nb = getattr(lib, f"{e}_xprep_bytes")
    nb.argtypes, nb.restype = [c_i64, c_i64], ctypes.c_size_t
    xp = getattr(lib, f"{e}_xprep")
    xp.argtypes = [c_vp, c_i64, c_i64, c_i64, c_i64, c_i64, c_i64, c_vp]
    px = getattr(lib, f"{e}_packed_x")
    handle = prep(W, K, N)
    ws = ctypes.create_string_buffer(nb(K, C) + 64)
    nk, step = K // 32, 8 if f.act == "q8_K" else 1
    xrow = len(X) // C
    for m0, m1 in ((0, C // 2), (C // 2, C)):  # columns and K blocks in separate calls, like threads do
        for k0, k1 in ((0, nk // 2 // step * step), (nk // 2 // step * step, nk)):
            xp(X, K, C, m0, m1, k0, k1, ws)
    Y = (ctypes.c_float * (N * C))(*([float("nan")] * (N * C)))
    for c0 in range(0, C, cols):
        m = min(cols, C - c0)
        xc = ctypes.cast(ctypes.c_char_p(X), c_vp).value + c0 * xrow
        yc = ctypes.cast(Y, c_vp).value + 4 * c0 * N
        for r0, r1 in splits:
            if c["op"] == "verify":
                px.argtypes = [c_vp, c_vp, c_vp, c_i64, c_i64, c_vp, c_i64, c_i64, c_i64, c_i64, c_i64]
                px(handle, xc, ws, C, c0, yc, K, N, m, r0, r1)
            else:
                px.argtypes = [c_vp, c_vp, c_vp, c_i64, c_i64, c_vp, c_i64, c_i64, c_i64]
                px(handle, xc, ws, C, c0, yc, K, r0, r1)
    return list(Y)


@pytest.mark.parametrize("w,unpack,op,n", XPREP, ids=lambda v: str(v))
def test_xprep_matches_packed(w, unpack, op, n):
    """xprep (shared activation prep, llama.cpp integration): E_packed_x on a workspace built in
    pieces must be bit-identical to E_packed, for column slices and any row split."""
    kw = {"rows": n} if op == "gemv" else {"cols": n}
    extra = {"correction": "dpmin"} if w == "q4_K" else {}
    c = spec.resolve({"op": op, "weights": w, "target": "avx512_vnni", "layout": "i16", "unpack": unpack, **kw, **extra})
    _native(c)
    rng = random.Random(21)
    f = FORMATS[w]
    W = BLOCKS[w](rng, N * K // f.block)
    C = 5 if op == "verify" else 3
    X = BLOCKS[f.act](rng, C * K // (256 if f.act == "q8_K" else 32))
    cols = n if op == "verify" else 1
    got = run_xprep(dict(c, xprep=1), W, X, C, cols, ((0, 16), (16, N)))
    xrow = len(X) // C
    want = []
    for c0 in range(0, C, cols):
        m = min(cols, C - c0)
        y = _run(c, W, X[c0 * xrow : (c0 + m) * xrow], m)
        want += y[: N * m]
    assert got == want
    _close(got, reference_gemm(f, W, X, K, N, C))


def test_q8_0_i16_rows8():
    """The Q8_0 GEMV on i16 with vnni16's rows=8: exact against the reference, and the verify
    kernels on the same records give column 0 bit-identical to the GEMV."""
    g = spec.resolve({"op": "gemv", "weights": "q8_0", "target": "avx512_vnni", "layout": "i16", "rows": 8})
    v = spec.resolve({"op": "verify", "weights": "q8_0", "target": "avx512_vnni", "layout": "i16", "cols": 8})
    _native(g)
    rng = random.Random(8)
    f = FORMATS["q8_0"]
    W = BLOCKS["q8_0"](rng, N * K // 32)
    X = BLOCKS["q8_0"](rng, 6 * K // 32)
    yv = _run(v, W, X, 6)
    _close(yv, reference_gemm(f, W, X, K, N, 6))
    assert _run(g, W, X) == yv[:N]


def test_q8_0_rows8_rejected_for_verify_and_other_formats():
    with pytest.raises(spec.SpecError):
        spec.resolve({"op": "gemv", "weights": "q4_0", "target": "avx512_vnni", "layout": "i16", "rows": 8})
    with pytest.raises(spec.SpecError):
        spec.resolve({"op": "verify", "weights": "q8_0", "target": "avx512_vnni", "layout": "i16", "rows": 8, "cols": 1})
    with pytest.raises(spec.SpecError):
        spec.resolve({"op": "gemv", "weights": "q8_0", "target": "avx512_vnni", "layout": "i16", "align": 64})


def test_ilv_rejected_outside_nibble_path():
    with pytest.raises(spec.SpecError, match="ilv"):
        spec.resolve({"op": "gemv", "weights": "q4_0", "target": "avx512_vnni", "layout": "i16", "unpack": "mask",
                      "ilv": 2})  # fmt: skip


@pytest.mark.parametrize(
    "cfg,msg",
    [
        ({"weights": "q4_0", "unpack": "mask16", "correction": "weight"}, "needs correction=act"),
        ({"weights": "q4_K", "unpack": "mask", "correction": "dpmin"}, "dpmin needs unpack=mask16 or pair"),
        ({"weights": "q4_K", "unpack": "mask16", "scales": "packed"}, "needs scales=unpacked"),
    ],
)
def test_invalid_nibble_combinations(cfg, msg):
    with pytest.raises(spec.SpecError, match=msg):
        spec.resolve({"op": "gemv", "target": "avx512_vnni", "layout": "i16", "rows": 1, **cfg})
    assert generic.combo_problem({"layout": "i16", "cols": 1, "rows": 1, "correction": "act", "scales": "unpacked",
                                  **cfg}) is not None  # fmt: skip


def test_nibble_values_are_legal_for_4bit_formats():
    for w in ("q4_0", "q4_K", "iq4_nl", "mxfp4", "nvfp4"):
        legal = generic.legal_keys(w, "avx512_vnni")
        assert "perm" in legal["unpack"] or "pair" in legal["unpack"], w
    assert "dpmin" in generic.legal_keys("q4_K", "avx2_vnni")["correction"]
    assert "perm" not in generic.legal_keys("iq4_nl", "avx2_vnni")["unpack"]
