"""Test data and references for the GPU kernels (stdlib only).

- random ggml weight blocks per format (random and extreme data),
- f32 activations with the awkward cases (all-zero blocks, +-max ties),
- ggml's reference activation quantizers (quantize_row_q8_0_ref / quantize_row_q8_K_ref), emulated
  in exact float32 arithmetic so the GPU quantizers can be checked byte for byte,
- the exact GEMM reference from kurn.formats.
"""

import math
import random
import struct

from .. import kernels  # noqa: F401  (loads kurn.ext, which registers e8p and friends)
from ..formats import reference_gemm
from .spec import ACT, FORMATS


def f32(v):
    return struct.unpack("<f", struct.pack("<f", v))[0]


def f16_bits(rng):
    """Positive fp16 in [2^-6, 2^-1) with a random mantissa."""
    return ((9 + rng.randrange(5)) << 10) | rng.randrange(1024)


def _i8s(rng, n, extreme):
    if extreme:
        return [127 if rng.random() < 0.5 else -127 for _ in range(n)]
    return [rng.randint(-127, 127) for _ in range(n)]


def weight_blocks(fmt, rng, nblocks, extreme=False):
    """Random valid blocks of a weight format (ggml byte layout)."""
    out = bytearray()
    for _ in range(nblocks):
        if fmt == "q8_0":
            out += struct.pack("<H32b", f16_bits(rng), *_i8s(rng, 32, extreme))
        elif fmt in ("q4_0", "iq4_nl", "q2_0", "q1_0"):
            out += struct.pack("<H", f16_bits(rng)) + bytes(0xFF if extreme else rng.randrange(256) for _ in range(16))
        elif fmt == "q4_K":
            out += struct.pack("<HH", f16_bits(rng), f16_bits(rng)) + bytes(0xFF if extreme else rng.randrange(256) for _ in range(140))
        elif fmt == "tq2_0":
            out += bytes(0xAA if extreme else rng.randrange(256) for _ in range(64)) + struct.pack("<H", f16_bits(rng))
        elif fmt == "e8p":
            from ..ext.compress import e8p_blocks

            out += e8p_blocks(rng, 1, extreme)
        elif fmt == "mxfp4":  # E8M0 scale 2^-8 .. 2^-2 (times the 1/2 of the doubled codes); 0xFF = all codes -12
            out += bytes([120 + rng.randrange(7)]) + bytes(0xFF if extreme else rng.randrange(256) for _ in range(16))
        elif fmt == "nvfp4":  # UE4M3 scales 2^-5 .. 2^1 (codes 0x20-0x4F), sometimes 0 or subnormal (codes 1-7); no NaN
            d = [(lambda r: 0 if r < 0.05 else 1 + rng.randrange(7) if r < 0.1 else 0x20 + rng.randrange(0x30))(rng.random())
                 for _ in range(4)]  # fmt: skip
            out += bytes(d) + bytes(0xFF if extreme else rng.randrange(256) for _ in range(32))
        else:
            raise KeyError(fmt)
    return bytes(out)


def weights(fmt, rng, n, k, extreme=False):
    f = FORMATS[fmt]
    return weight_blocks(fmt, rng, n * (k // f["block"]), extreme)


def activations(rng, k, m, block=32):
    """f32 activations [m][k] with an all-zero block, a +-max tie and a lone extreme in each row."""
    x = []
    for _ in range(m):
        row = [f32(rng.gauss(0.0, 1.0)) for _ in range(k)]
        if k >= 4 * block:
            for j in range(block):
                row[block + j] = 0.0
            row[2 * block + 3] = 2.5
            row[2 * block + 9] = -2.5
            row[3 * block + 1] = f32(rng.choice((-1, 1)) * 40.0)
        x.append(row)
    return x


def pack_f32(x):
    return b"".join(struct.pack(f"<{len(r)}f", *r) for r in x)


def _roundf(v):
    return math.copysign(math.floor(abs(v) + 0.5), v)


def quant_q8_0(row):
    """ggml quantize_row_q8_0_ref, in float32."""
    out = bytearray()
    for b in range(0, len(row), 32):
        xs = row[b : b + 32]
        amax = max(abs(v) for v in xs)
        d = f32(amax / 127.0)
        iv = f32(1.0 / d) if d else 0.0
        out += struct.pack("<e", d) + struct.pack("<32b", *(int(_roundf(f32(v * iv))) for v in xs))
    return bytes(out)


def quant_q8_K(row):
    """ggml quantize_row_q8_K_ref, in float32 (round-to-nearest-even, first max on ties)."""
    out = bytearray()
    for b in range(0, len(row), 256):
        xs = row[b : b + 256]
        amax, mx = 0.0, 0.0
        for v in xs:
            if abs(v) > amax:
                amax, mx = abs(v), v
        if amax == 0:
            out += struct.pack("<f", 0.0) + bytes(256) + bytes(32)
            continue
        iscale = f32(-127.0 / mx)
        qs = [min(127, round(f32(iscale * v))) for v in xs]
        out += struct.pack("<f", f32(1.0 / iscale)) + struct.pack("<256b", *qs)
        out += struct.pack("<16h", *(sum(qs[16 * t : 16 * t + 16]) for t in range(16)))
    return bytes(out)


def act_blocks(fmt, x):
    q = quant_q8_0 if FORMATS[fmt]["act"] == "q8_0" else quant_q8_K
    return b"".join(q(r) for r in x)


def act_row_bytes(fmt, k):
    a = ACT[FORMATS[fmt]["act"]]
    return k // a["block"] * a["nbytes"]


def reference(fmt, W, xb, n, k, m):
    """Exact Y[i * n + r] from the weights and the quantized activation blocks."""
    return reference_gemm(fmt, W, xb, k, n, m)


def relerr(y, ref):
    scale = max(1e-30, max(abs(v) for v in ref))
    return max(abs(a - b) for a, b in zip(y, ref)) / scale


def problem(fmt, n, k, m, seed=0, extreme=False):
    rng = random.Random(seed)
    W = weights(fmt, rng, n, k, extreme)
    x = activations(rng, k, m)
    return W, x
