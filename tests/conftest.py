import os
import random
import struct
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / "examples"

# Keep test builds out of the user's cache.
os.environ.setdefault("KURN_CACHE_DIR", tempfile.mkdtemp(prefix="kurn-test-cache-"))
sys.path.insert(0, str(ROOT / "src"))


def f16_bits(rng):
    """Positive fp16 in [2^-6, 2^-1) with a random mantissa."""
    return ((9 + rng.randrange(5)) << 10) | rng.randrange(1024)


def i8s(rng, n, extreme=False):
    if extreme:
        return [127 if rng.random() < 0.5 else -127 for _ in range(n)]
    return [rng.randint(-127, 127) for _ in range(n)]


def q8_0_blocks(rng, nblocks, extreme=False):
    out = bytearray()
    for _ in range(nblocks):
        out += struct.pack("<H32b", f16_bits(rng), *i8s(rng, 32, extreme))
    return bytes(out)


def q4_K_blocks(rng, nblocks, extreme=False):
    out = bytearray()
    for _ in range(nblocks):
        out += struct.pack("<HH", f16_bits(rng), f16_bits(rng))
        out += bytes(0xFF if extreme else rng.randrange(256) for _ in range(12 + 128))
    return bytes(out)


def q8_K_blocks(rng, nblocks, extreme=False):
    out = bytearray()
    for _ in range(nblocks):
        qs = i8s(rng, 256, extreme)
        bsums = [sum(qs[16 * t : 16 * t + 16]) for t in range(16)]
        out += struct.pack("<f256b16h", 0.001 + rng.random() * 0.02, *qs, *bsums)
    return bytes(out)


def d16_blocks(rng, nblocks, extreme=False):
    """fp16 d + 16 code bytes: q4_0, iq4_nl (32 values), q2_0 (64), q1_0 (128)."""
    out = bytearray()
    for _ in range(nblocks):
        out += struct.pack("<H", f16_bits(rng))
        out += bytes(0xFF if extreme else rng.randrange(256) for _ in range(16))
    return bytes(out)


def tq2_0_blocks(rng, nblocks, extreme=False):
    out = bytearray()
    for _ in range(nblocks):
        out += bytes(0xAA if extreme else rng.randrange(256) for _ in range(64))
        out += struct.pack("<H", f16_bits(rng))
    return bytes(out)


BLOCKS = {
    "q8_0": q8_0_blocks,
    "q4_K": q4_K_blocks,
    "q8_K": q8_K_blocks,
    "q4_0": d16_blocks,
    "iq4_nl": d16_blocks,
    "q2_0": d16_blocks,
    "q1_0": d16_blocks,
    "tq2_0": tq2_0_blocks,
}


def _extension_blocks():
    from kurn import hooks, kernels  # noqa: F401  (importing kernels loads kurn.ext)

    return hooks.TEST_BLOCKS


BLOCKS.update(_extension_blocks())


@pytest.fixture
def rng():
    return random.Random(1234)
