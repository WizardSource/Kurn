"""lowbit formats (tq1_0, q2_K) against transcriptions of ggml's own quantize/dequantize
loops (ggml-quants.c), and the schedule keys of the lowbit layouts."""

import random
import struct

import pytest

from kurn import lowbit
from kurn.formats import FORMATS, reference_gemv
from kurn.spec import SpecError, legal_configs, resolve

from conftest import BLOCKS

POW3 = (1, 3, 9, 27, 81)


def ggml_quantize_tq1_0_trits(trits, d_bits):
    """quantize_row_tq1_0_ref with x = trit - 1 and amax = 1: 5 trits per byte (most
    significant first), ceil(q * 256 / 243)."""
    out = bytearray(54)
    t = [v for v in trits]

    def pack(vals, shift_last=False):
        q = 0
        for v in vals:
            q = q * 3 + v
        if shift_last:
            q *= 3
        return (q * 256 + 242) // 243

    for m in range(32):
        out[m] = pack([t[m + 32 * n] for n in range(5)])
    for m in range(16):
        out[32 + m] = pack([t[160 + m + 16 * n] for n in range(5)])
    for j in range(4):
        out[48 + j] = pack([t[240 + j + 4 * m] for m in range(4)], shift_last=True)
    struct.pack_into("<H", out, 52, d_bits)
    return bytes(out)


def ggml_dequantize_tq1_0(b):
    d = struct.unpack_from("<e", b, 52)[0]
    y = []
    for n in range(5):
        for m in range(32):
            y.append(((((b[m] * POW3[n]) & 0xFF) * 3) >> 8) - 1)
    for n in range(5):
        for m in range(16):
            y.append(((((b[32 + m] * POW3[n]) & 0xFF) * 3) >> 8) - 1)
    for n in range(4):
        for j in range(4):
            y.append(((((b[48 + j] * POW3[n]) & 0xFF) * 3) >> 8) - 1)
    return [d * v for v in y]


def ggml_dequantize_q2_K(b):
    d, dmin = struct.unpack_from("<ee", b, 80)
    q, y, s = 16, [], 0
    for _ in range(2):
        for j in range(4):
            for half in range(2):
                sc = b[s]
                s += 1
                y += [d * (sc & 15) * ((b[q + 16 * half + i] >> (2 * j)) & 3) - dmin * (sc >> 4) for i in range(16)]
        q += 32
    return y


def q8_K_values(x):
    d = struct.unpack_from("<f", x, 0)[0]
    return [d * v for v in struct.unpack_from("<256b", x, 4)]


@pytest.mark.parametrize("seed", range(4))
def test_tq1_0_matches_ggml_encoder(seed):
    rng = random.Random(seed)
    trits = [rng.randrange(3) for _ in range(256)]
    blk = ggml_quantize_tq1_0_trits(trits, 0x3C00)
    assert [lowbit.tq1_0_trit(blk, v) for v in range(256)] == trits
    assert ggml_dequantize_tq1_0(blk) == [float(t - 1) for t in trits]


def test_tq1_0_every_byte_decodes_like_ggml():
    """Every byte value (not only canonical encodings, e.g. 0xFF in the extreme tests)."""
    for q in range(256):
        blk = bytes([q] * 52) + b"\x00\x3c"
        assert [lowbit.tq1_0_trit(blk, v) + 0.0 - 1 for v in range(256)] == ggml_dequantize_tq1_0(blk)


@pytest.mark.parametrize("fmt", ["tq1_0", "q2_K"])
def test_reference_matches_ggml_dequantize(fmt):
    rng = random.Random(7)
    K, N = 512, 3
    W = BLOCKS[fmt](rng, N * K // 256)
    x = BLOCKS["q8_K"](rng, K // 256)
    deq = {"tq1_0": ggml_dequantize_tq1_0, "q2_K": ggml_dequantize_q2_K}[fmt]
    nb = FORMATS[fmt].nbytes
    xv = [v for i in range(K // 256) for v in q8_K_values(x[i * 292:(i + 1) * 292])]
    ref = reference_gemv(fmt, W, x, K, N)
    for r in range(N):
        wv = [v for i in range(K // 256) for v in deq(W[(r * K // 256 + i) * nb:(r * K // 256 + i + 1) * nb])]
        exp = sum(a * b for a, b in zip(wv, xv))
        assert abs(ref[r] - exp) <= 1e-4 * max(1.0, abs(exp))


def test_base3_word_extraction_exhaustive():
    """10 trits per 16-bit word: W' = ceil(W * 2^16 / 3^10); the lut kernel's
    mulhi/mullo-by-27 chain yields (9t0+3t1+t2, 9t3+3t4+t5, 9t6+3t7+t8, t9) for every W."""
    for w in range(3**10):
        t = [(w // 3 ** (9 - k)) % 3 for k in range(10)]
        wp = (w * 65536 + 59048) // 59049
        assert wp < 65536
        i0, r = (wp * 27) >> 16, (wp * 27) & 0xFFFF
        i1, r = (r * 27) >> 16, (r * 27) & 0xFFFF
        i2, r = (r * 27) >> 16, (r * 27) & 0xFFFF
        i3 = (r * 3) >> 16
        assert (i0, i1, i2, i3) == (9 * t[0] + 3 * t[1] + t[2], 9 * t[3] + 3 * t[4] + t[5], 9 * t[6] + 3 * t[7] + t[8], t[9]), w


def test_lut_variants_parse_and_fit():
    for f, variants in lowbit.LUT_VARIANTS.items():
        assert lowbit.LUT_DEFAULT[f] in variants
        for v in variants:
            bits, g, mirror = lowbit.lut_variant(v)
            assert bits in ("direct", "serial", "tern") and 2 <= g <= 6 and mirror in (0, 1)
            p = lowbit.LutPlan({"weights": f, "lut": v})
            assert p.se in (16, 32, 64) and p.op == ("perm" if p.se <= 32 else "perm2")
            assert p.groups * p.unit == lowbit.generic.RECIPES[f].period


def test_lut_storage():
    """Index storage per weight: dense (1 or 2 bits) for g=4 1-bit, g=2 direct and g=4 serial 2-bit,
    1.625 bits for tern (26 words per 256)."""
    bpw = {(f, v): 16 * lowbit.LutPlan({"weights": f, "lut": v}).wpu / lowbit.LutPlan({"weights": f, "lut": v}).unit
           for f, vs in lowbit.LUT_VARIANTS.items() for v in vs}
    assert bpw[("q1_0", "direct4")] == 1.0
    assert bpw[("tq2_0", "direct2")] == 2.0 and bpw[("tq2_0", "serial4")] == 2.0
    assert bpw[("tq1_0", "tern3")] == 1.625


@pytest.mark.parametrize("bad", [
    {"layout": "i16", "lut": "direct4"},
    {"layout": "lut", "lut": "serial4"},  # not a 1-bit variant
    {"layout": "lut", "addsub": "mask"},
    {"layout": "addsub", "rows": 4},
])
def test_lowbit_keys_rejected(bad):
    with pytest.raises(SpecError):
        resolve({"op": "gemv", "weights": "q1_0", "target": "avx512_vnni", **bad})


def test_lowbit_defaults():
    c = resolve({"op": "gemv", "weights": "tq1_0", "target": "avx512_vnni", "layout": "lut", "rows": 1})
    assert c["lut"] == "tern3"
    c = resolve({"op": "gemv", "weights": "q2_0", "target": "avx512_vnni", "layout": "addsub", "rows": 1})
    assert c["addsub"] == "mask"
    c = resolve({"op": "gemv", "weights": "q2_K", "target": "avx512_vnni"})
    assert c["layout"] == "k16" and c["rows"] == 4


def test_every_lut_variant_is_a_legal_config():
    seen = {(c["weights"], c["lut"]) for c in legal_configs(op="gemv", target="avx512_vnni") if c["layout"] == "lut"}
    assert seen == {(f, v) for f, vs in lowbit.LUT_VARIANTS.items() for v in vs}
    modes = {(c["weights"], c["addsub"]) for c in legal_configs(op="gemv", target="avx512_vnni") if c["layout"] == "addsub"}
    assert modes == {(f, m) for f in lowbit.ADDSUB_FORMATS for m in ("mask", "sad")}
