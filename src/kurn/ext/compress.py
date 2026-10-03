"""Compressed weight formats (workstream D): E8P lattice codebook GEMV.

Registers the `e8p` format (QuIP#-style 2-bit E8 lattice codebook; quantizer and incoherence
processing in kurn.codebook, which needs numpy) and its decode GEMV kernel (`ke8p_gemv`, scalar +
AVX-512 VNNI). This module is stdlib-only: the table, the exact reference and the lowerings.
"""

import itertools
import struct

from .. import hooks
from ..formats import FORMATS, Field, Format
from ..kernels import KERNELS, Kernel

QK = 256  # weights per block
E8P_BYTES = 2 + 2 * (QK // 8)  # 66


def _e8p_abs2():
    pats = []
    for k in itertools.product(range(3), repeat=8):  # a_i = k_i + 1/2
        n2 = sum((x + 0.5) ** 2 for x in k)
        if n2 <= 10:
            pats.append((n2, k))
    assert len(pats) == 227, len(pats)
    norm12_odd = sorted(k for k in itertools.product(range(3), repeat=8) if sorted(k) == [0, 0, 0, 1, 1, 1, 1, 1])
    norm12_even = sorted(k for k in itertools.product(range(3), repeat=8) if sorted(k) == [0, 0, 0, 0, 0, 1, 1, 2])
    extra = [(12.0, k) for k in norm12_even[:21]] + [(12.0, k) for k in norm12_odd[:8]]
    rows = [k for _, k in sorted(pats + extra)]
    par = lambda k: int(round(sum(x + 0.5 for x in k))) & 1  # noqa: E731  (parity of sum(a))
    even = [k for k in rows if par(k) == 0]
    odd = [k for k in rows if par(k) == 1]
    assert len(even) == 128 and len(odd) == 128, (len(even), len(odd))
    return tuple(tuple(2 * x + 1 for x in k) for k in even + odd)


# 256 x 8 values of 2*a (1, 3, 5): the 227 half-integer patterns with |a|^2 <= 10 plus 29 of
# norm 12, sorted so rows 0-127 have an even coordinate sum of a, rows 128-255 an odd one.
E8P_ABS2 = _e8p_abs2()


def e8p_abs_u64():
    """The table as 256 little-endian u64 of the bytes 4*a_i (2, 6, 10) -- the kernel's gather table."""
    return [int.from_bytes(bytes(2 * v for v in row), "little") for row in E8P_ABS2]


# ---------------------------------------------------------------- Python reference (format)


def _par8(x):
    x ^= x >> 4
    x ^= x >> 2
    x ^= x >> 1
    return x & 1


def e8p_block_q(w):
    """One e8p block (bytes) -> 256 ints q = 4c (odd, in [-11, 11]) in weight order."""
    out = []
    for g in range(QK // 8):
        lo, sg = w[2 + g], w[34 + g]
        a = E8P_ABS2[(lo & 0x7F) | (_par8(sg) << 7)]
        t = 1 if lo & 0x80 else -1
        out += [(-2 * a[i] if (sg >> i) & 1 else 2 * a[i]) + t for i in range(8)]
    return out


def ref_e8p(wblocks, xblocks):
    total = 0.0
    for w, x in zip(wblocks, xblocks):
        q = e8p_block_q(w)
        q8 = struct.unpack_from(f"<{QK}b", x, 4)
        total += struct.unpack_from("<e", w, 0)[0] * struct.unpack_from("<f", x, 0)[0] * sum(a * b for a, b in zip(q, q8))
    return total


def e8p_blocks(rng, nblocks, extreme=False):
    """Random valid e8p blocks (every 16-bit code is a valid codeword)."""
    out = bytearray()
    for _ in range(nblocks):
        d = ((9 + rng.randrange(5)) << 10) | rng.randrange(1024)
        if extreme:  # largest |q|: the |a| = 5/2 rows, all signs negative, t = -1/4
            big = [i for i, r in enumerate(E8P_ABS2) if 5 in r]
            codes = []
            for _ in range(QK // 8):
                i = big[rng.randrange(len(big))]
                sgn = 0xFF if (i >> 7) == 0 else 0x7F  # parity(sign byte) must equal index bit 7
                codes.append((i & 0x7F) | (sgn << 8))
        else:
            codes = [rng.randrange(65536) for _ in range(QK // 8)]
        out += struct.pack("<H", d) + bytes(c & 0xFF for c in codes) + bytes(c >> 8 for c in codes)
    return bytes(out)


# ---------------------------------------------------------------- lowerings (C)


def _u64_table(name, vals, per_line=4):
    lines = [", ".join(f"0x{v:016x}ULL" for v in vals[i : i + per_line]) for i in range(0, len(vals), per_line)]
    return f"static const uint64_t {name}[{len(vals)}] __attribute__((aligned(64))) = {{\n    " + ",\n    ".join(lines) + "\n};\n"


_COMMON = r"""
#include <stdint.h>
#include <string.h>
#include "kurn.h"
typedef struct {{ uint16_t d; uint8_t lo[32]; uint8_t hi[32]; }} block_e8p;
_Static_assert(sizeof(block_e8p) == 66, "e8p size");
static inline float f16f(uint16_t h) {{
    uint32_t s = (uint32_t)(h & 0x8000) << 16, e = (h >> 10) & 0x1f, m = h & 0x3ff, b;
    if (e == 0) {{
        if (!m) b = s;
        else {{ e = 113; while (!(m & 0x400)) {{ m <<= 1; e--; }} b = s | (e << 23) | ((m & 0x3ff) << 13); }}
    }} else if (e == 31) b = s | 0x7f800000 | (m << 13);
    else b = s | ((e + 112) << 23) | (m << 13);
    float f; memcpy(&f, &b, 4); return f;
}}
{table}
"""


def lower_e8p_gemv(target, c):
    table = _u64_table("E8P_ABS4", e8p_abs_u64())
    head = _COMMON.format(table=table)
    entry = c["entry"]
    if target == "scalar":
        return head + _E8P_SCALAR.format(entry=entry)
    return "#include <immintrin.h>\n" + head + _E8P_AVX512.format(entry=entry, R=c["rows"], PF=c["prefetch"])


_E8P_SCALAR = r"""
static inline int par8(unsigned x) {{ x ^= x >> 4; x ^= x >> 2; x ^= x >> 1; return x & 1; }}
void {entry}(const void *W, const void *x, float *y, int64_t K, int64_t r0, int64_t r1) {{
    const int64_t nb = K / QK_K;
    const block_q8_K *xb = (const block_q8_K *)x;
    for (int64_t r = r0; r < r1; r++) {{
        const block_e8p *w = (const block_e8p *)W + r * nb;
        float acc = 0.0f;
        for (int64_t b = 0; b < nb; b++) {{
            int32_t isum = 0;
            for (int g = 0; g < 32; g++) {{
                const unsigned lo = w[b].lo[g], sg = w[b].hi[g];
                const uint64_t a = E8P_ABS4[(lo & 0x7F) | (par8(sg) << 7)];
                const int t = (lo & 0x80) ? 1 : -1;
                for (int i = 0; i < 8; i++) {{
                    int q = (int)((a >> (8 * i)) & 0xFF);
                    q = ((sg >> i) & 1 ? -q : q) + t;
                    isum += q * xb[b].qs[8 * g + i];
                }}
            }}
            acc += f16f(w[b].d) * xb[b].d * (float)isum;
        }}
        y[r] = acc;
    }}
}}
"""

# R rows share the activation loads. Per block and row: parity of the 32 sign bytes (nibble
# LUT) -> 8-bit table indices; t bits (MSB of the low bytes) -> +-1 words dotted with the
# precomputed 8-activation group sums (vpdpwssd); per group of 8 codes: one 8-qword gather of
# the 4|a| bytes, the 8 sign bytes loaded straight into a mask register negate the
# activations, then vpdpbusd (|a| is unsigned, so no bias correction).
_E8P_AVX512 = r"""
#define R {R}
#define PF {PF}
static void e8p_rows(const block_e8p *const *w, const block_q8_K *xb, const int16_t *xg, int64_t nb, float *out, int nr) {{
    const __m256i PLUT = _mm256_setr_epi8(0, -128, -128, 0, -128, 0, 0, -128, -128, 0, 0, -128, 0, -128, -128, 0,
                                          0, -128, -128, 0, -128, 0, 0, -128, -128, 0, 0, -128, 0, -128, -128, 0);
    const __m256i NIB = _mm256_set1_epi8(0x0F), LOW7 = _mm256_set1_epi8(0x7F);
    const __m512i ONE = _mm512_set1_epi16(1), MONE = _mm512_set1_epi16(-1), Z = _mm512_setzero_si512();
    __m512 acc[R];
    for (int i = 0; i < R; i++) acc[i] = _mm512_setzero_ps();
    for (int64_t b = 0; b < nb; b++) {{
        const __m512i xs[4] = {{_mm512_loadu_si512(xb[b].qs), _mm512_loadu_si512(xb[b].qs + 64),
                                _mm512_loadu_si512(xb[b].qs + 128), _mm512_loadu_si512(xb[b].qs + 192)}};
        const __m512i xgs = _mm512_loadu_si512(xg + 32 * b);
        const float dx = xb[b].d;
        for (int i = 0; i < nr; i++) {{
            const block_e8p *blk = w[i] + b;
#if PF
            _mm_prefetch((const char *)(blk + PF), _MM_HINT_T0);
#endif
            const __m256i lo = _mm256_loadu_si256((const __m256i *)blk->lo);
            const __m256i hi = _mm256_loadu_si256((const __m256i *)blk->hi);
            const __m256i par = _mm256_xor_si256(_mm256_shuffle_epi8(PLUT, _mm256_and_si256(hi, NIB)),
                                                 _mm256_shuffle_epi8(PLUT, _mm256_and_si256(_mm256_srli_epi16(hi, 4), NIB)));
            const __m256i idx = _mm256_or_si256(_mm256_and_si256(lo, LOW7), par);
            const __mmask32 tm = _mm256_movepi8_mask(lo);
            __m512i ai = _mm512_dpwssd_epi32(Z, _mm512_mask_blend_epi16(tm, MONE, ONE), xgs);
            const __m512i i0 = _mm512_cvtepu8_epi32(_mm256_castsi256_si128(idx));
            const __m512i i1 = _mm512_cvtepu8_epi32(_mm256_extracti128_si256(idx, 1));
            const __m256i gi[4] = {{_mm512_castsi512_si256(i0), _mm512_extracti64x4_epi64(i0, 1),
                                    _mm512_castsi512_si256(i1), _mm512_extracti64x4_epi64(i1, 1)}};
            for (int g = 0; g < 4; g++) {{
                const __m512i a = _mm512_i32gather_epi64(gi[g], (const void *)E8P_ABS4, 8);
                const __mmask64 neg = _load_mask64((__mmask64 *)(blk->hi + 8 * g));
                ai = _mm512_dpbusd_epi32(ai, a, _mm512_mask_sub_epi8(xs[g], neg, Z, xs[g]));
            }}
            acc[i] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(ai), _mm512_set1_ps(_cvtsh_ss(blk->d) * dx), acc[i]);
        }}
    }}
    for (int i = 0; i < nr; i++) out[i] = _mm512_reduce_add_ps(acc[i]);
}}
void {entry}(const void *W, const void *x, float *y, int64_t K, int64_t r0, int64_t r1) {{
    const int64_t nb = K / QK_K;
    const block_q8_K *xb = (const block_q8_K *)x;
    int16_t xg[32768 / 8] __attribute__((aligned(64)));  // sums of 8 consecutive activations
    const __m512i B = _mm512_set1_epi8(-128);
    for (int64_t b = 0; b < nb; b++)
        for (int g = 0; g < 4; g++) {{
            const __m512i s = _mm512_sad_epu8(_mm512_xor_si512(_mm512_loadu_si512(xb[b].qs + 64 * g), B), _mm512_setzero_si512());
            _mm_storeu_si128((__m128i *)(xg + 32 * b + 8 * g), _mm_sub_epi16(_mm512_cvtepi64_epi16(s), _mm_set1_epi16(1024)));
        }}
    const block_e8p *wb = (const block_e8p *)W;
    for (int64_t r = r0; r < r1; r += R) {{
        const int nr = r1 - r < R ? (int)(r1 - r) : R;
        const block_e8p *rows[R];
        for (int i = 0; i < R; i++) rows[i] = wb + (r + (i < nr ? i : 0)) * nb;
        float out[R];
        e8p_rows(rows, xb, xg, nb, out, nr);
        for (int i = 0; i < nr; i++) y[r + i] = out[i];
    }}
}}
"""


# ---------------------------------------------------------------- registration

FORMATS["e8p"] = Format(
    "e8p", QK, E8P_BYTES, "q8_K", (Field("d", "f16", 0), Field("lo", "u8", 2, 32), Field("hi", "u8", 34, 32)),
    "QuIP#-style E8 lattice codebook: 8 weights per u16 code; value = d * (s (.) 4a + t), "
    "a from a 256-entry table (index low 7 bits | parity of sign byte), signs = code >> 8, t = +-1 (bit 7)",
    ref_e8p,
)  # fmt: skip

KERNELS[("gemv", "e8p")] = Kernel(
    "gemv", "e8p", ("scalar", "avx512_vnni"), lower_e8p_gemv, "ke8p_gemv", "e8pgemv",
    "E8P (2.06 bpw lattice codebook, RHT-rotated weights) x Q8_K activation vector (decode)",
)  # fmt: skip

hooks.TEST_BLOCKS["e8p"] = e8p_blocks
hooks.GOLDEN["e8p_gemv_scalar"] = {"op": "gemv", "weights": "e8p", "target": "scalar"}
hooks.GOLDEN["e8p_gemv_avx512_rows4_pf4"] = {"op": "gemv", "weights": "e8p", "target": "avx512_vnni", "rows": 4,
                                             "prefetch": 4}  # fmt: skip
