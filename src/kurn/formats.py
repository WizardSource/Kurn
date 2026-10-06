"""Format registry: block quantization formats as first-class types.

A format fixes the block size, byte layout, the activation format it pairs with
(ggml's `vec_dot_type`), and its exact semantics. Weight formats carry an
independent pure-Python `reference(wblocks, xblocks)` used by the tests to check
generated kernels. Layouts are byte-identical to ggml's `block_q8_0`,
`block_q4_K` and `block_q8_K`.

Adding a format = a `_ref_<name>` function and one `Format(...)` entry below
(see kernels.py for the rest of the checklist).
"""

import struct
from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass(frozen=True)
class Field:
    name: str
    ctype: str  # f16, f32, i8, u8, i16, u4 (packed nibbles), u2 (packed crumbs)
    offset: int
    count: int = 1


@dataclass(frozen=True)
class Format:
    name: str
    block: int  # values per block
    nbytes: int  # bytes per block
    act: str  # activation format it is multiplied with
    fields: tuple = field(default_factory=tuple)
    doc: str = ""
    reference: Optional[Callable] = None  # (weight blocks, activation blocks) -> float; None for activation-only formats

    @property
    def bits_per_weight(self):
        return 8 * self.nbytes / self.block

    def field(self, name):
        return next(f for f in self.fields if f.name == name)

    def row_bytes(self, k):
        if k % self.block:
            raise ValueError(f"K={k} is not a multiple of the {self.name} block size {self.block}")
        return k // self.block * self.nbytes


def _f16(b, o):
    return struct.unpack_from("<e", b, o)[0]


def _i8(b, o, n):
    return struct.unpack_from(f"<{n}b", b, o)


def _ref_q8_0(wblocks, xblocks):
    total = 0.0
    for w, x in zip(wblocks, xblocks):
        s = sum(a * b for a, b in zip(_i8(w, 2, 32), _i8(x, 2, 32)))
        total += _f16(w, 0) * _f16(x, 0) * s
    return total


def _ref_q4_K(wblocks, xblocks):
    total = 0.0
    for w, x in zip(wblocks, xblocks):
        q = w[4:16]
        sc = [q[j] & 63 if j < 4 else (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4) for j in range(8)]
        mn = [q[j + 4] & 63 if j < 4 else (q[j + 4] >> 4) | ((q[j] >> 6) << 4) for j in range(8)]
        dx = struct.unpack_from("<f", x, 0)[0]
        q8 = _i8(x, 4, 256)
        bsums = struct.unpack_from("<16h", x, 260)
        isum = msum = 0
        for s in range(8):
            chunk = w[16 + 32 * (s // 2) : 16 + 32 * (s // 2) + 32]
            nib = [(c >> (4 * (s & 1))) & 0xF for c in chunk]
            isum += sc[s] * sum(a * b for a, b in zip(nib, q8[32 * s : 32 * s + 32]))
            msum += mn[s] * (bsums[2 * s] + bsums[2 * s + 1])
        total += dx * (_f16(w, 0) * isum - _f16(w, 2) * msum)
    return total


KV_IQ4NL = (-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113)


def _q8_blocks(xblocks):
    """Q8_0 activation blocks -> (list of (d, qs))."""
    return [(_f16(x, 0), _i8(x, 2, 32)) for x in xblocks]


def _ref_nibble32(values_of):
    def ref(wblocks, xblocks):
        total = 0.0
        for w, (dx, q8) in zip(wblocks, _q8_blocks(xblocks)):
            vals = values_of(w)
            total += _f16(w, 0) * dx * sum(a * b for a, b in zip(vals, q8))
        return total

    return ref


def _vals_q4_0(w):
    return [(w[2 + i] & 15) - 8 for i in range(16)] + [(w[2 + i] >> 4) - 8 for i in range(16)]


def _vals_iq4_nl(w):
    return [KV_IQ4NL[w[2 + i] & 15] for i in range(16)] + [KV_IQ4NL[w[2 + i] >> 4] for i in range(16)]


def _ref_q2_0(wblocks, xblocks):  # 64 values per block, two Q8_0 activation blocks
    xs = _q8_blocks(xblocks)
    total = 0.0
    for i, w in enumerate(wblocks):
        vals = [((w[2 + v // 4] >> (2 * (v % 4))) & 3) - 1 for v in range(64)]
        for h in range(2):
            dx, q8 = xs[2 * i + h]
            total += _f16(w, 0) * dx * sum(a * b for a, b in zip(vals[32 * h : 32 * h + 32], q8))
    return total


def _ref_q1_0(wblocks, xblocks):  # 128 values per block, four Q8_0 activation blocks
    xs = _q8_blocks(xblocks)
    total = 0.0
    for i, w in enumerate(wblocks):
        vals = [1 if (w[2 + v // 8] >> (v % 8)) & 1 else -1 for v in range(128)]
        for h in range(4):
            dx, q8 = xs[4 * i + h]
            total += _f16(w, 0) * dx * sum(a * b for a, b in zip(vals[32 * h : 32 * h + 32], q8))
    return total


def _ref_tq2_0(wblocks, xblocks):  # 256 values, qs[64] then d; Q8_K activations
    total = 0.0
    for w, x in zip(wblocks, xblocks):
        dx = struct.unpack_from("<f", x, 0)[0]
        q8 = _i8(x, 4, 256)
        s = 0
        for v in range(256):
            q = (w[(v // 128) * 32 + v % 32] >> (2 * ((v % 128) // 32))) & 3
            s += (q - 1) * q8[v]
        total += dx * _f16(w, 64) * s
    return total


FORMATS = {
    f.name: f
    for f in (
        Format(
            "q8_0",
            32,
            34,
            "q8_0",
            (Field("d", "f16", 0), Field("qs", "i8", 2, 32)),
            "value[i] = d * qs[i]; qs in [-127, 127] (ggml's quantizer never emits -128)",
            _ref_q8_0,
        ),  # fmt: skip
        Format(
            "q4_K",
            256,
            144,
            "q8_K",
            (Field("d", "f16", 0), Field("dmin", "f16", 2), Field("scales", "u8", 4, 12), Field("qs", "u4", 16, 256)),
            "8 sub-blocks of 32; 6-bit scale sc[s] and min mn[s] packed in `scales`; value = d*sc[s]*q - dmin*mn[s]; "
            "qs chunk j (32 B) holds sub-block 2j in low and 2j+1 in high nibbles",
            _ref_q4_K,
        ),  # fmt: skip
        Format(
            "q4_0",
            32,
            18,
            "q8_0",
            (Field("d", "f16", 0), Field("qs", "u4", 2, 32)),
            "value = d * (q - 8); low nibbles hold values 0..15, high nibbles 16..31",
            _ref_nibble32(_vals_q4_0),
        ),  # fmt: skip
        Format(
            "iq4_nl",
            32,
            18,
            "q8_0",
            (Field("d", "f16", 0), Field("qs", "u4", 2, 32)),
            "value = d * kvalues_iq4nl[q] (non-linear codebook)",
            _ref_nibble32(_vals_iq4_nl),
        ),  # fmt: skip
        Format(
            "q2_0",
            64,
            18,
            "q8_0",
            (Field("d", "f16", 0), Field("qs", "u2", 2, 64)),
            "value = d * (q - 1), 4 values per byte, LSB first (ternary Bonsai: q in 0..2)",
            _ref_q2_0,
        ),  # fmt: skip
        Format(
            "tq2_0",
            256,
            66,
            "q8_K",
            (Field("qs", "u2", 0, 256), Field("d", "f16", 64)),
            "ternary: value = d * (q - 1); value v in byte (v/128)*32 + v%32, bits 2*((v%128)/32)",
            _ref_tq2_0,
        ),  # fmt: skip
        Format(
            "q1_0",
            128,
            18,
            "q8_0",
            (Field("d", "f16", 0), Field("qs", "u1", 2, 128)),
            "1-bit (Bonsai): value = d * (bit ? +1 : -1), LSB first",
            _ref_q1_0,
        ),  # fmt: skip
        Format(
            "q8_K",
            256,
            292,
            "q8_K",
            (Field("d", "f32", 0), Field("qs", "i8", 4, 256), Field("bsums", "i16", 260, 16)),
            "activation format: value[i] = d * qs[i]; bsums[t] = sum of qs[16t:16t+16] (lets q4_K apply mins without touching qs)",
        ),  # fmt: skip
    )
}


def reference_dot(fmt, wblocks, xblocks):
    """Exact dot(weight row, activation row): integer sums per (sub-)block, then
    scaling, accumulated in Python floats (double precision)."""
    fmt = FORMATS[fmt] if isinstance(fmt, str) else fmt
    if fmt.reference is None:
        raise ValueError(f"{fmt.name} is an activation format; it has no weight reference")
    return fmt.reference(wblocks, xblocks)


def split_blocks(fmt, row):
    fmt = FORMATS[fmt] if isinstance(fmt, str) else fmt
    return [row[i : i + fmt.nbytes] for i in range(0, len(row), fmt.nbytes)]


def reference_gemv(fmt, W, x, k, n):
    """y[r] = dot(W[r, :], x) for r < n. W and x are bytes in ggml block layout."""
    fmt = FORMATS[fmt] if isinstance(fmt, str) else fmt
    rb = fmt.row_bytes(k)
    xb = split_blocks(fmt.act, x)
    return [reference_dot(fmt, split_blocks(fmt, W[r * rb : (r + 1) * rb]), xb) for r in range(n)]


def reference_gemm(fmt, W, X, k, n, m):
    """Y[i * n + r] = dot(W[r, :], X[i, :]) for i < m, r < n."""
    fmt = FORMATS[fmt] if isinstance(fmt, str) else fmt
    xr = FORMATS[fmt.act].row_bytes(k)
    out = []
    for i in range(m):
        out += reference_gemv(fmt, W, X[i * xr : (i + 1) * xr], k, n)
    return out
