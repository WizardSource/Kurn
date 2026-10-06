"""Formats and lowerings for 2 bits per weight and below (KURN v0.2 workstream `lowbit`).

Formats (byte layouts identical to ggml):
  q2_K   256 values, 84 B: 16 sub-blocks of 16, 4-bit scale sc and min mn per sub-block,
         value = d * sc[s] * q - dmin * mn[s], q in 0..3 (Q8_K activations)
  tq1_0  256 values, 54 B (1.6875 bpw): ternary, 5 trits per byte (qs[48]) + 4 per byte (qh[4]),
         base-3 fixed point: trit l of byte q = ((uint8_t)(q * 3^l) * 3) >> 8; value = d * (trit - 1)

Lowerings (AVX-512, decode GEMV; all exact):
  layout lut     T-MAC / bitnet.cpp-style table lookup, no multiplies in the inner loop.
                 Per activation chunk of g values a table of exact int16 partial sums is built once
                 per activation vector (vpdpbusd against constant coefficient vectors); weights are
                 stored as table indices, 32 rows per vector (int16 lanes). Key lut=<bits><g>[m]:
                   bits  direct  index = the g codes of a chunk (2^g or 4^g entries)
                         serial  bit-serial: one 1-bit plane per weight bit, shared +-1 tables
                         tern    tq1_0 only: 10 trits per 16-bit word in base-3 fixed point
                                 (1.6875 bpw, the size of TQ1_0); indices via mulhi/mullo by 27
                   g     values per table index (chunk size); tables of <= 32 entries are looked
                         up with vpermw, 64-entry tables with vpermi2w
                   m     mirror: +-1 tables stored half size, sign bit applied with a masked negate
                 Tables can be built cooperatively: `<entry>_lut_build` (a range of activation
                 units, run by every thread before a barrier) + `<entry>_lut_rows`.
  layout addsub  multiplication-free ternary/binary math. Weights are 1-bit planes (b0, b1 of the
                 code; 1-bit formats: one plane); sum(w * x) = 2 * sum_b1(x) + sum_b0(x) - sum(x).
                   addsub=mask  int16 lanes (32 rows), one masked add of the broadcast activation
                                per plane and value (vpaddw with a k-mask)
                   addsub=sad   64-bit lanes (8 rows), select activations with a byte mask then
                                vpsadbw horizontal sums (activations biased to u8)
  layout k16     q2_K: 16 rows per vector, vpdpbusd on the 2-bit codes, sub-block scales and mins
                 applied in pairs with vpdpwssd (Q8_K bsums give the min term for free).
"""

import struct

from . import generic
from .formats import Field, Format

# --------------------------------------------------------------------------- formats
TQ1_0_CODE = (
    "((((unsigned)(uint8_t)((unsigned)(v < 160 ? b->qs[v % 32] : v < 240 ? b->qs[32 + (v - 160) % 16] : b->qh[(v - 240) % 4])"
    ' * (unsigned)"\\001\\003\\011\\033\\121"[v < 160 ? v / 32 : v < 240 ? (v - 160) / 16 : (v - 240) / 4])) * 3u) >> 8)'
)

TQ1_0_RECIPE = generic.Recipe(
    "tq1_0", 256, 54, "q8_K", 2, "uint8_t qs[48]; uint8_t qh[4]; uint16_t d;", TQ1_0_CODE, 256,
    maps={"mask": ("q", 1, -1), "lut": ("(uint8_t)(q - 1 + 128)", 1, -128)}, lut=(-1, 0, 1, 2), signed_c="(q - 1)",
    doc="ternary base-3: value = d * (trit - 1); generic layouts repack to 2-bit codes (2.06 bpw)")  # fmt: skip


def _f16(b, o):
    return struct.unpack_from("<e", b, o)[0]


def tq1_0_trit(w, v):
    """Trit (0..2) of value v in a native tq1_0 block (ggml's base-3 fixed point)."""
    if v < 160:
        q, e = w[v % 32], v // 32
    elif v < 240:
        q, e = w[32 + (v - 160) % 16], (v - 160) // 16
    else:
        q, e = w[48 + (v - 240) % 4], (v - 240) // 4
    return (((q * 3**e) & 0xFF) * 3) >> 8


def _q8_K(x):
    return struct.unpack_from("<f", x, 0)[0], struct.unpack_from("<256b", x, 4), struct.unpack_from("<16h", x, 260)


def _ref_tq1_0(wblocks, xblocks):
    total = 0.0
    for w, x in zip(wblocks, xblocks):
        dx, q8, _ = _q8_K(x)
        s = sum((tq1_0_trit(w, v) - 1) * q8[v] for v in range(256))
        total += dx * _f16(w, 52) * s
    return total


def q2_K_code(w, v):
    return (w[16 + (v // 128) * 32 + v % 32] >> (2 * ((v % 128) // 32))) & 3


def _ref_q2_K(wblocks, xblocks):
    total = 0.0
    for w, x in zip(wblocks, xblocks):
        dx, q8, bsums = _q8_K(x)
        isum = msum = 0
        for s in range(16):
            sc, mn = w[s] & 15, w[s] >> 4
            isum += sc * sum(q2_K_code(w, v) * q8[v] for v in range(16 * s, 16 * s + 16))
            msum += mn * bsums[s]
        total += dx * (_f16(w, 80) * isum - _f16(w, 82) * msum)
    return total


FORMATS = {
    "tq1_0": Format("tq1_0", 256, 54, "q8_K", (Field("qs", "u8", 0, 48), Field("qh", "u8", 48, 4), Field("d", "f16", 52)),
                    "ternary, 1.6875 bpw: 5 trits per byte in qs, 4 per byte in qh (base-3 fixed point); value = d * (trit - 1)",
                    _ref_tq1_0),
    "q2_K": Format("q2_K", 256, 84, "q8_K",
                   (Field("scales", "u8", 0, 16), Field("qs", "u2", 16, 256), Field("d", "f16", 80), Field("dmin", "f16", 82)),
                   "16 sub-blocks of 16; scales[s] = sc | mn << 4; value = d * sc[s] * q - dmin * mn[s]; "
                   "q of value v in byte (v/128)*32 + v%32, bits 2*((v%128)/32)", _ref_q2_K),
}  # fmt: skip


def _f16_bits(rng):
    return ((9 + rng.randrange(5)) << 10) | rng.randrange(1024)


def tq1_0_blocks(rng, nblocks, extreme=False):
    out = bytearray()
    for _ in range(nblocks):
        out += bytes(0xFF if extreme else rng.randrange(256) for _ in range(52))
        out += struct.pack("<H", _f16_bits(rng))
    return bytes(out)


def q2_K_blocks(rng, nblocks, extreme=False):
    out = bytearray()
    for _ in range(nblocks):
        out += bytes(0xFF if extreme else rng.randrange(256) for _ in range(80))
        out += struct.pack("<HH", _f16_bits(rng), _f16_bits(rng))
    return bytes(out)


TEST_BLOCKS = {"tq1_0": tq1_0_blocks, "q2_K": q2_K_blocks}

# --------------------------------------------------------------------------- schedule keys
LUT_FORMATS = ("q1_0", "q2_0", "tq2_0", "tq1_0")
ADDSUB_FORMATS = ("q1_0", "q2_0", "tq2_0", "tq1_0")
# Schedule key `lut` = <bits><g>[m]: bits direct | serial | tern, g values per table index, m = mirror
# (half-size +-1 tables + masked negate). Tables of <= 32 int16 entries use vpermw, 64 use vpermi2w.
_LUT_2BIT = ("direct2", "direct3", "serial4", "serial5", "serial6", "serial6m")
LUT_VARIANTS = {
    "q1_0": ("direct4", "direct5", "direct6", "direct6m"),
    "q2_0": _LUT_2BIT,
    "tq2_0": _LUT_2BIT,
    "tq1_0": ("tern3", "direct2", "serial4"),
}
LUT_DEFAULT = {"q1_0": "direct4", "q2_0": "serial4", "tq2_0": "serial4", "tq1_0": "tern3"}


def lut_variant(name):
    """`serial6m` -> ("serial", 6, 1)."""
    mirror = int(name.endswith("m"))
    core = name[:-1] if mirror else name
    return core.rstrip("0123456789"), int(core[len(core.rstrip("0123456789")) :]), mirror


def _lut_possible(op, f, t):
    return op == "gemv" and t == "avx512_vnni" and f in LUT_FORMATS


def layouts(op, f, t):
    out = ()
    if _lut_possible(op, f, t):
        out += ("lut",)
    if op == "gemv" and t == "avx512_vnni" and f in ADDSUB_FORMATS:
        out += ("addsub",)
    if (op, f, t) == ("gemv", "q2_K", "avx512_vnni"):
        out += ("k16",)
    return out


KEY_LEGAL = {
    "lut": lambda op, f, t: ("auto",) + (LUT_VARIANTS[f] if _lut_possible(op, f, t) else ()),
    "addsub": lambda op, f, t: ("auto", "mask", "sad") if "addsub" in layouts(op, f, t) else ("auto",),
}


INVALID = [
    (lambda c: c["layout"] != "lut" and c["lut"] != "auto", "lut applies to layout=lut only"),
    (lambda c: c["layout"] != "addsub" and c["addsub"] != "auto", "addsub applies to layout=addsub only"),
    (lambda c: c["layout"] in ("lut", "addsub") and c["rows"] not in (1, 2), "rows must be 1 or 2 for lut/addsub (32-row groups)"),
    (lambda c: c["weights"] == "q2_K" and c["target"] != "scalar" and c["rows"] not in (1, 2, 4), "rows must be 1, 2 or 4 for q2_K"),
]


def resolve(c):
    if c["layout"] == "lut" and c["lut"] == "auto":
        c["lut"] = LUT_DEFAULT[c["weights"]]
    if c["layout"] == "addsub" and c["addsub"] == "auto":
        c["addsub"] = "mask"
    if c["weights"] == "q2_K" and c["target"] != "scalar" and c["layout"] == "native":
        c["layout"] = "k16"  # the only vector layout for q2_K


# --------------------------------------------------------------------------- shared C pieces
def _prelude(what):
    return f"""// Generated by kurn (lowbit: {what}). Do not edit; edit the .kurn spec instead.
#include "kurn.h"
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>
static inline float f16f(uint16_t h) {{ return _cvtsh_ss(h); }}
static inline __m512i lo16(__m512i v) {{ return _mm512_cvtepi16_epi32(_mm512_castsi512_si256(v)); }}
static inline __m512i hi16(__m512i v) {{ return _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(v, 1)); }}
static inline __m512 fma16(__m512i s, __m512 d, __m512 dx, __m512 acc) {{
    return _mm512_fmadd_ps(_mm512_cvtepi32_ps(s), _mm512_mul_ps(d, dx), acc);
}}
static inline __m256 fma8(__m512i s, __m256 d, __m256 dx, __m256 acc) {{
    return _mm256_fmadd_ps(_mm512_cvtepi64_ps(s), _mm256_mul_ps(d, dx), acc);
}}
"""


STOREV = """
static inline void storev(float *y, int64_t row0, int64_t r0, int64_t r1, __m512 v) {
    if (row0 >= r0 && row0 + 16 <= r1) { _mm512_storeu_ps(y + row0, v); return; }
    uint32_t m = 0xFFFF;
    if (row0 < r0) m &= r0 - row0 >= 16 ? 0 : 0xFFFFu << (r0 - row0);
    if (row0 + 16 > r1) m &= (1u << (r1 > row0 ? (r1 - row0 > 16 ? 16 : r1 - row0) : 0)) - 1;
    _mm512_mask_storeu_ps(y + row0, (__mmask16)m, v);
}"""

WVEC = "#define WVEC(rec, kg, w) _mm512_loadu_si512((const void *)((rec) + HDR_BYTES + (kg) * UNIT_BYTES + 64 * (w)))"

STOREV8 = """
static inline void storev8(float *y, int64_t row0, int64_t r0, int64_t r1, __m256 v) {
    if (row0 >= r0 && row0 + 8 <= r1) { _mm256_storeu_ps(y + row0, v); return; }
    float t[8];
    _mm256_storeu_ps(t, v);
    for (int i = 0; i < 8; i++) if (row0 + i >= r0 && row0 + i < r1) y[row0 + i] = t[i];
}"""


def _act_c(act):
    if act == "q8_0":
        return "typedef struct { uint16_t d; int8_t qs[32]; } xblock;"
    return "typedef struct { float d; int8_t qs[256]; int16_t bsums[16]; } xblock;"


def _act_unit_c(act, fac):
    """C statements: copy the int8 activations of 32-value unit `k` to xq + 32*(k - k0), set dxs[k], sx[k]."""
    if act == "q8_0":
        return f"""{{ const xblock *b = xb + k;
              memcpy(dst, b->qs, 32);
              int32_t s = 0;
              for (int l = 0; l < 32; l++) s += b->qs[l];
              dxs[k] = f16f(b->d) * {fac}f; sx[k] = s; }}"""
    return f"""{{ const xblock *b = xb + k / 8;
              memcpy(dst, b->qs + (k % 8) * 32, 32);
              dxs[k] = b->d * {fac}f; sx[k] = b->bsums[2 * (k % 8)] + b->bsums[2 * (k % 8) + 1]; }}"""


def _prepare_head(r, rec, hdr, ngroups):
    return f"""
typedef struct {{ {r.struct} }} nblock;
typedef struct {{ int64_t nrec_k, ngroups; uint8_t *buf; }} packed_t;
#define REC_BYTES {rec}
#define HDR_BYTES {hdr}
"""


# --------------------------------------------------------------------------- layout lut
class LutPlan:
    """Static structure of one lut configuration (shared by repack, table builder and kernel)."""

    def __init__(self, c):
        r = generic.RECIPES[c["weights"]]
        self.r = r
        self.bits, self.g, self.mirror = lut_variant(c["lut"])
        self.act = r.act
        self.planes = 2 if self.bits == "serial" else 1
        if self.bits == "tern":
            self.unit = 256
            self.fw = 16
            self.chunks = []
            for w in range(26):
                o = 10 * w
                self.chunks += [(o, 3), (o + 3, 3)] + ([(o + 6, 3), (o + 9, 1)] if w < 25 else [])
            self.ib = 5
            self.wpu = 26
        else:
            self.unit = 32
            self.fw = 2 * self.g if (self.bits == "direct" and r.bits == 2) else self.g
            self.chunks = [(o, min(self.g, 32 - o)) for o in range(0, 32, self.g)]
            self.ib = self.g - 1 if self.mirror else self.fw
            self.fpw = 16 // self.fw
            nf = len(self.chunks) * self.planes
            self.wpu = -(-nf // self.fpw)
        self.se = max(16, 2**self.ib)
        if self.se > 64:
            raise ValueError(f"lut={c['lut']}: {self.se}-entry tables do not fit one vpermi2w")
        self.op = "perm" if self.se <= 32 else "perm2"
        self.tb = 2 * self.se  # table bytes
        self.nch = len(self.chunks)
        self.groups = r.period // self.unit  # units per scale period
        self.unit_bytes = 64 * self.wpu
        self.rec = 64 + self.groups * self.unit_bytes
        # int16 accumulation limit (values) before widening, for Q8_K formats (one scale per period)
        vmax = 2 if (self.bits == "direct" and r.bits == 2) else 1
        self.flush = max(1, min(self.groups, (32767 // (127 * vmax)) // self.unit))
        self.fac = 0.5 if self.bits == "serial" else 1.0

    def coef(self, c, e, j):
        """Coefficient of activation j (< len) of chunk c in table entry e."""
        o, ln = self.chunks[c]
        if j >= ln:
            return 0
        if self.bits == "tern":
            if ln == 3:
                return (e // 9, (e // 3) % 3, e % 3)[j] - 1 if e < 27 else 0
            return e - 1 if e < 3 else 0
        e %= 2**self.ib
        if self.bits == "direct" and self.r.bits == 2:
            return ((e >> (2 * j)) & 3) - 1
        return 2 * ((e >> j) & 1) - 1


def _lut_builder(p, entry):
    """C: tables for units [u0, u1) (the per-activation-vector precompute)."""
    cst, cst_idx, corr, corr_idx = [], {}, [], {}

    def const(table, idx, vals):
        key = tuple(vals)
        if key not in idx:
            idx[key] = len(table)
            table.append(key)
        return idx[key]

    body = []
    for c, (o, ln) in enumerate(p.chunks):
        for blk in range(p.se // 16):
            entries = range(16 * blk, 16 * blk + 16)
            if all(p.coef(c, e, j) == 0 for e in entries for j in range(ln)):
                continue
            ci = const(corr, corr_idx, [-128 * sum(p.coef(c, e, j) for j in range(ln)) for e in entries])
            dps = []
            for dw in range(o // 4, (o + ln - 1) // 4 + 1):
                vals = []
                for e in entries:
                    for b in range(4):
                        j = 4 * dw + b - o
                        vals.append(p.coef(c, e, j) if 0 <= j < ln else 0)
                si = const(cst, cst_idx, vals)
                dps.append(f"_mm512_dpbusd_epi32(A, _mm512_set1_epi32(ld32(xu + {4 * dw})), _mm512_load_si512((const void *)LB_CST[{si}]))")
            line = f"_mm512_load_si512((const void *)LB_CORR[{ci}])"
            for d in dps:
                line = d.replace("A,", line + ",", 1)
            body.append(f"            _mm256_store_si256((__m256i *)(t + {c * p.tb + 32 * blk}), _mm512_cvtepi32_epi16({line}));")
    cst_c = ",\n".join("    {" + ", ".join(str(v) for v in row) + "}" for row in cst)
    corr_c = ",\n".join("    {" + ", ".join(str(v) for v in row) + "}" for row in corr)
    per_unit = p.unit // 32
    return f"""
static const int8_t LB_CST[{len(cst)}][64] __attribute__((aligned(64))) = {{
{cst_c}
}};
static const int32_t LB_CORR[{len(corr)}][16] __attribute__((aligned(64))) = {{
{corr_c}
}};
static inline int32_t ld32(const uint8_t *p) {{ int32_t v; memcpy(&v, p, 4); return v; }}
static inline size_t lb_toff(int64_t K) {{ return ((size_t)8 * (K / 32) + 63) & ~(size_t)63; }}

/* table buffer: float dxs[K/32] | int32 sx[K/32] | pad to 64 | {p.nch} tables of {p.se} int16 per {p.unit}-value unit */
void {entry}_lut_info(int64_t K, int64_t *bytes, int64_t *units) {{
    *bytes = (int64_t)(lb_toff(K) + (size_t)(K / {p.unit}) * {p.nch * p.tb});
    *units = K / {p.unit};
}}

void {entry}_lut_build(const void *X, int64_t K, void *tabs, int64_t u0, int64_t u1) {{
    const xblock *xb = (const xblock *)X;
    float *dxs = (float *)tabs;
    int32_t *sx = (int32_t *)((uint8_t *)tabs + 4 * (K / 32));
    uint8_t *tb = (uint8_t *)tabs + lb_toff(K);
    uint8_t xu[{p.unit}] __attribute__((aligned(64)));
    for (int64_t u = u0; u < u1; u++) {{
        for (int64_t k = u * {per_unit}; k < (u + 1) * {per_unit}; k++) {{
            int8_t *dst = (int8_t *)xu + 32 * (k - u * {per_unit});
            {_act_unit_c(p.act, p.fac)}
        }}
        for (int i = 0; i < {p.unit}; i++) xu[i] ^= 0x80;  /* x + 128 as u8 (vpdpbusd's unsigned operand) */
        uint8_t *t = tb + (size_t)u * {p.nch * p.tb};
{chr(10).join(body)}
    }}
}}
"""


def _lut_field_c(p):
    if p.bits == "direct" and p.r.bits == 2:
        return "e |= q << (2 * j);"
    if p.bits == "serial":
        return "e |= ((q >> pl) & 1) << j;"
    return "e |= q << j;"


def _lut_prepare(p, entry, G):
    r = p.r
    if p.bits == "tern":
        fill = f"""
                    for (int wd = 0; wd < 26; wd++) {{
                        uint32_t t10 = 0;
                        for (int t = 0; t < 10; t++) {{
                            const int vv = 10 * wd + t;
                            int q = 0;
                            if (vv < 256) {{ const int v = (int)((v0 + vv) % {r.block}); q = {r.code_c}; }}
                            t10 = t10 * 3 + (uint32_t)q;
                        }}
                        wp[wd * 32] = (uint16_t)((t10 * 65536u + 59048u) / 59049u);  /* base-3 fixed point: digits by * 27 */
                    }}"""
    else:
        cho = ", ".join(str(o) for o, _ in p.chunks)
        chl = ", ".join(str(n) for _, n in p.chunks)
        mirror = ""
        if p.mirror:
            mirror = f"if (CH_L[c] == {p.g} && ((e >> {p.g - 1}) & 1)) e = (~e & {2 ** (p.g - 1) - 1}) | {2 ** (p.g - 1)};"
        fill = f"""
                    static const int CH_O[{p.nch}] = {{{cho}}}, CH_L[{p.nch}] = {{{chl}}};
                    for (int c = 0; c < {p.nch}; c++)
                        for (int pl = 0; pl < {p.planes}; pl++) {{
                            int e = 0;
                            for (int j = 0; j < CH_L[c]; j++) {{
                                const int v = (int)((v0 + kg * 32 + CH_O[c] + j) % {r.block});
                                const int q = {r.code_c};
                                {_lut_field_c(p)}
                            }}
                            {mirror}
                            const int slot = c * {p.planes} + pl;
                            wp[(slot / {p.fpw}) * 32] |= (uint16_t)(e << ((slot % {p.fpw}) * {p.fw}));
                            (void)pl;
                        }}"""
    return f"""
void *{entry}_prepare(const void *W, int64_t K, int64_t N) {{
    const nblock *w = (const nblock *)W;
    const int64_t nb = K / {r.block};
    packed_t *pk = malloc(sizeof *pk);
    pk->nrec_k = K / {r.period}; pk->ngroups = (N + 31) / 32 + {G - 1};
    const size_t bytes = (size_t)REC_BYTES * pk->nrec_k * pk->ngroups;
    pk->buf = aligned_alloc(64, (bytes + 63) & ~(size_t)63);
    memset(pk->buf, 0, bytes);
    for (int64_t gr = 0; gr < pk->ngroups; gr++)
        for (int64_t p = 0; p < pk->nrec_k; p++) {{
            uint8_t *hp = pk->buf + ((size_t)gr * pk->nrec_k + p) * REC_BYTES;
            for (int row = 0; row < 32; row++) {{
                const int64_t n = gr * 32 + row;
                if (n >= N) continue;  /* padding rows stay zero: d = 0 */
                const int64_t v0 = p * {r.period};
                const nblock *b = w + n * nb + v0 / {r.block};
                memcpy(hp + 2 * row, &b->d, 2);
                for (int kg = 0; kg < {p.groups}; kg++) {{
                    uint16_t *wp = (uint16_t *)(hp + HDR_BYTES + kg * UNIT_BYTES) + row;{fill}
                }}
            }}
        }}
    return pk;
}}
"""


def _lut_kernel(p, entry, G, PF):
    ind = "                "
    L = []
    q8k = p.act == "q8_K"

    def tload(c):
        base = f"T + {c * p.tb}"
        if p.op == "perm2":
            if p.se == 64:
                return (f"_mm512_load_si512((const void *)({base}))", f"_mm512_load_si512((const void *)({base} + 64))")
            raise ValueError("perm2 needs 64-entry tables")
        if p.se == 16:
            return (f"_mm512_broadcast_i64x4(_mm256_load_si256((const __m256i *)({base})))",)
        return (f"_mm512_load_si512((const void *)({base}))",)

    def lookup(idx, c):
        t = tload(c)
        if p.op == "perm2":
            return f"_mm512_permutex2var_epi16({t[0]}, {idx}, {t[1]})"
        return f"_mm512_permutexvar_epi16({idx}, {t[0]})"

    for gg in range(G):
        L.append(f"{ind}const uint8_t *rec{gg} = pk->buf + ((size_t)(g + {gg}) * pk->nrec_k + p) * REC_BYTES;")
        if PF:
            L.append(f"{ind}_mm_prefetch((const char *)(rec{gg} + {PF} * REC_BYTES), _MM_HINT_T0);")
        L.append(
            f"{ind}const __m512 dlo{gg} = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)rec{gg})), "
            f"dhi{gg} = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(rec{gg} + 32)));"
        )
        if q8k:
            L.append(f"{ind}__m512i ilo{gg} = _mm512_setzero_si512(), ihi{gg} = _mm512_setzero_si512();")
            for pl in range(p.planes):
                L.append(f"{ind}__m512i s{gg}_{pl} = _mm512_setzero_si512();")
    L.append(f"{ind}for (int kg = 0; kg < {p.groups}; kg++) {{")
    L.append(f"{ind}    const int64_t k = p * {p.groups} + kg;  /* unit index */")
    L.append(f"{ind}    const uint8_t *T = tb + (size_t)k * {p.nch * p.tb};")
    if not q8k:
        for gg in range(G):
            for pl in range(p.planes):
                L.append(f"{ind}    __m512i s{gg}_{pl} = _mm512_setzero_si512();")
    if p.bits == "tern":
        for w in range(26):
            nf = 4 if w < 25 else 2
            for gg in range(G):
                L.append(f"{ind}    {{ const __m512i W = WVEC(rec{gg}, kg, {w});")
                L.append(f"{ind}      __m512i R = _mm512_mullo_epi16(W, k27);")
                L.append(f"{ind}      s{gg}_0 = _mm512_add_epi16(s{gg}_0, {lookup('_mm512_mulhi_epu16(W, k27)', 4 * w)});")
                L.append(f"{ind}      s{gg}_0 = _mm512_add_epi16(s{gg}_0, {lookup('_mm512_mulhi_epu16(R, k27)', 4 * w + 1)});")
                if nf == 4:
                    L.append(f"{ind}      const __m512i R2 = _mm512_mullo_epi16(R, k27);")
                    L.append(f"{ind}      s{gg}_0 = _mm512_add_epi16(s{gg}_0, {lookup('_mm512_mulhi_epu16(R2, k27)', 4 * w + 2)});")
                    i3 = lookup("_mm512_mulhi_epu16(_mm512_mullo_epi16(R2, k27), k3)", 4 * w + 3)
                    L.append(f"{ind}      s{gg}_0 = _mm512_add_epi16(s{gg}_0, {i3});")
                L.append(f"{ind}      (void)R; }}")
    else:
        slots = [(c, pl) for c in range(p.nch) for pl in range(p.planes)]
        for w in range(p.wpu):
            for gg in range(G):
                L.append(f"{ind}    {{ const __m512i W = WVEC(rec{gg}, kg, {w});")
                for f in range(p.fpw):
                    s = w * p.fpw + f
                    if s >= len(slots):
                        break
                    c, pl = slots[s]
                    sh = f * p.fw
                    idx = f"_mm512_srli_epi16(W, {sh})" if sh else "W"
                    if p.mirror and p.chunks[c][1] == p.g:
                        L.append(f"{ind}      {{ const __m512i v = {lookup(idx, c)};")
                        L.append(f"{ind}        const __mmask32 neg = _mm512_test_epi16_mask(W, _mm512_set1_epi16({1 << (sh + p.g - 1)}));")
                        L.append(f"{ind}        s{gg}_{pl} = _mm512_add_epi16(s{gg}_{pl}, _mm512_mask_sub_epi16(v, neg, zero, v)); }}")
                    else:
                        L.append(f"{ind}      s{gg}_{pl} = _mm512_add_epi16(s{gg}_{pl}, {lookup(idx, c)});")
                L.append(f"{ind}    }}")

    def widen(gg, dst_lo, dst_hi):
        out = []
        for dst, h in ((dst_lo, "lo16"), (dst_hi, "hi16")):
            v = f"_mm512_add_epi32(_mm512_slli_epi32({h}(s{gg}_1), 1), {h}(s{gg}_0))" if p.planes == 2 else f"{h}(s{gg}_0)"
            out.append(f"{dst} = _mm512_add_epi32({dst}, {v});")
        return out

    if q8k:
        if p.groups == 1:
            cond = ""
        elif p.flush >= p.groups:
            cond = f"if (kg == {p.groups - 1}) "
        else:
            cond = f"if ((kg + 1) % {p.flush} == 0) "
        for gg in range(G):
            L.append(f"{ind}    {cond}{{")
            for line in widen(gg, f"ilo{gg}", f"ihi{gg}"):
                L.append(f"{ind}      {line}")
            for pl in range(p.planes):
                L.append(f"{ind}      s{gg}_{pl} = _mm512_setzero_si512();")
            L.append(f"{ind}    }}")
        L.append(f"{ind}}}")
        for gg in range(G):
            if p.planes == 2:  # 2 * sum(w x) = 2 L1 + L0 + sum(x)
                L.append(f"{ind}{{ int32_t st = 0; for (int kk = 0; kk < {p.groups}; kk++) st += sx[p * {p.groups} + kk];")
                L.append(f"{ind}  const __m512i sv = _mm512_set1_epi32(st);")
                L.append(f"{ind}  ilo{gg} = _mm512_add_epi32(ilo{gg}, sv); ihi{gg} = _mm512_add_epi32(ihi{gg}, sv); }}")
            L.append(f"{ind}{{ const __m512 dx = _mm512_set1_ps(dxs[p * {p.groups * p.unit // 32}]);")
            L.append(f"{ind}  alo{gg} = fma16(ilo{gg}, dlo{gg}, dx, alo{gg}); ahi{gg} = fma16(ihi{gg}, dhi{gg}, dx, ahi{gg}); }}")
    else:  # q8_0: one activation scale per 32-value unit
        for gg in range(G):
            if p.planes == 2:
                sv = f"_mm512_add_epi16(_mm512_add_epi16(_mm512_slli_epi16(s{gg}_1, 1), s{gg}_0), _mm512_set1_epi16((short)sx[k]))"
            else:
                sv = f"s{gg}_0"
            L.append(f"{ind}    {{ const __m512i sv = {sv};")
            L.append(f"{ind}      const __m512 dx = _mm512_set1_ps(dxs[k]);")
            L.append(f"{ind}      alo{gg} = fma16(lo16(sv), dlo{gg}, dx, alo{gg});")
            L.append(f"{ind}      ahi{gg} = fma16(hi16(sv), dhi{gg}, dx, ahi{gg}); }}")
        L.append(f"{ind}}}")
    decl = " ".join(f"__m512 alo{gg} = _mm512_setzero_ps(), ahi{gg} = _mm512_setzero_ps();" for gg in range(G))
    stores = "\n".join(
        f"        storev(Y, (g + {gg}) * 32, r0, r1, alo{gg}); storev(Y, (g + {gg}) * 32 + 16, r0, r1, ahi{gg});" for gg in range(G)
    )
    return f"""
void {entry}_lut_rows(const void *pv, const void *tabs, float *Y, int64_t K, int64_t r0, int64_t r1) {{
    const packed_t *pk = (const packed_t *)pv;
    const float *dxs = (const float *)tabs;
    const int32_t *sx = (const int32_t *)((const uint8_t *)tabs + 4 * (K / 32));
    const uint8_t *tb = (const uint8_t *)tabs + lb_toff(K);
    const __m512i zero = _mm512_setzero_si512(), k27 = _mm512_set1_epi16(27), k3 = _mm512_set1_epi16(3);
    (void)sx; (void)zero; (void)k27; (void)k3;
    const int64_t g0 = r0 / 32, g1 = (r1 + 31) / 32;
    for (int64_t g = g0; g < g1; g += {G}) {{
        {decl}
        for (int64_t p = 0; p < pk->nrec_k; p++) {{
{chr(10).join(L)}
        }}
{stores}
    }}
}}

void {entry}_packed(const void *pv, const void *X, float *Y, int64_t K, int64_t r0, int64_t r1) {{
    static __thread uint8_t *tabs;
    static __thread int64_t cap;
    int64_t bytes, units;
    {entry}_lut_info(K, &bytes, &units);
    if (cap < bytes) {{ free(tabs); tabs = aligned_alloc(64, (size_t)(bytes + 63) & ~(size_t)63); cap = bytes; }}
    {entry}_lut_build(X, K, tabs, 0, units);
    {entry}_lut_rows(pv, tabs, Y, K, r0, r1);
}}
"""


def lower_lut(target, c):
    if target != "avx512_vnni" or c["op"] != "gemv":
        raise ValueError("layout lut: gemv on avx512_vnni only")
    p = LutPlan(c)
    entry, G, PF = c["entry"], c["rows"], c["prefetch"]
    head = _prepare_head(p.r, p.rec, 64, 0) + f"#define UNIT_BYTES {p.unit_bytes}\n{WVEC}\n{_act_c(p.act)}\n"
    doc = (
        f"/* lut={c['lut']} ({p.op}): {p.nch} chunks per {p.unit}-value unit, "
        f"{p.se}-entry int16 tables, {p.wpu} index words per row per unit "
        f"({16 * p.wpu / p.unit + 16 / p.r.period:.4f} bpw incl. scales) */\n"
    )
    return (
        _prelude(f"lut {p.bits} g{p.g} {p.op}")
        + doc
        + head
        + STOREV
        + _lut_prepare(p, entry, G)
        + _lut_builder(p, entry)
        + _lut_kernel(p, entry, G, PF)
    )


# --------------------------------------------------------------------------- layout addsub (no multiplies)
def _planes_of(r):
    return 1 if r.bits == 1 else 2


def _addsub_prepare(r, entry, G, mode, planes, groups, rec):
    if mode == "mask":
        # per unit (32 values) and plane: 32 uint32 masks (value kk), bit = row
        fill = """
                    for (int kk = 0; kk < 32; kk++) {
                        const int v = (int)((v0 + kg * 32 + kk) % BLOCK);
                        const int q = CODE;
                        for (int pl = 0; pl < PLANES; pl++)
                            if ((q >> pl) & 1) ((uint32_t *)(up + pl * 128))[kk] |= 1u << row;
                    }"""
    else:
        # per unit and plane: [octet o][row vector rv] 8-byte masks: byte = row % 8, bit = value % 8; then the
        # bias correction 2*cnt1 + cnt0 per row (int16) after the planes
        fill = """
                    int32_t cnt = 0;
                    for (int kk = 0; kk < 32; kk++) {
                        const int v = (int)((v0 + kg * 32 + kk) % BLOCK);
                        const int q = CODE;
                        for (int pl = 0; pl < PLANES; pl++)
                            if ((q >> pl) & 1) {
                                up[pl * 128 + ((kk / 8) * 4 + row / 8) * 8 + row % 8] |= (uint8_t)(1u << (kk % 8));
                                cnt += PLANES == 2 && pl == 1 ? 2 : (PLANES == 1 ? 2 : 1);
                            }
                    }
                    ((int16_t *)(up + PLANES * 128))[row] = (int16_t)cnt;"""
    fill = fill.replace("BLOCK", str(r.block)).replace("CODE", r.code_c).replace("PLANES", str(planes))
    return f"""
void *{entry}_prepare(const void *W, int64_t K, int64_t N) {{
    const nblock *w = (const nblock *)W;
    const int64_t nb = K / {r.block};
    packed_t *pk = malloc(sizeof *pk);
    pk->nrec_k = K / {r.period}; pk->ngroups = (N + 31) / 32 + {G - 1};
    const size_t bytes = (size_t)REC_BYTES * pk->nrec_k * pk->ngroups;
    pk->buf = aligned_alloc(64, (bytes + 63) & ~(size_t)63);
    memset(pk->buf, 0, bytes);
    for (int64_t gr = 0; gr < pk->ngroups; gr++)
        for (int64_t p = 0; p < pk->nrec_k; p++) {{
            uint8_t *hp = pk->buf + ((size_t)gr * pk->nrec_k + p) * REC_BYTES;
            for (int row = 0; row < 32; row++) {{
                const int64_t n = gr * 32 + row;
                if (n >= N) continue;
                const int64_t v0 = p * {r.period};
                const nblock *b = w + n * nb + v0 / {r.block};
                memcpy(hp + 2 * row, &b->d, 2);
                for (int kg = 0; kg < {groups}; kg++) {{
                    uint8_t *up = hp + HDR_BYTES + kg * UNIT_BYTES;{fill}
                }}
            }}
        }}
    return pk;
}}
"""


def lower_addsub(target, c):
    if target != "avx512_vnni" or c["op"] != "gemv":
        raise ValueError("layout addsub: gemv on avx512_vnni only")
    r = generic.RECIPES[c["weights"]]
    entry, G, PF, mode = c["entry"], c["rows"], c["prefetch"], c["addsub"]
    planes = _planes_of(r)
    groups = r.period // 32
    q8k = r.act == "q8_K"
    unit_bytes = planes * 128 + (64 if mode == "sad" else 0)
    rec = 64 + groups * unit_bytes
    head = _prepare_head(r, rec, 64, 0) + f"#define UNIT_BYTES {unit_bytes}\n{_act_c(r.act)}\n"
    ind = "                "
    L = []
    if mode == "mask":
        # sum(w x) = 2 A1 + A0 - S (2-bit codes) or 2 A - S (1-bit), A_b = sum of x where bit b is set
        for gg in range(G):
            L.append(f"{ind}const uint8_t *rec{gg} = pk->buf + ((size_t)(g + {gg}) * pk->nrec_k + p) * REC_BYTES;")
            if PF:
                L.append(f"{ind}_mm_prefetch((const char *)(rec{gg} + {PF} * REC_BYTES), _MM_HINT_T0);")
            L.append(
                f"{ind}const __m512 dlo{gg} = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)rec{gg})), "
                f"dhi{gg} = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(rec{gg} + 32)));"
            )
            for pl in range(planes):
                L.append(f"{ind}__m512i a{gg}_{pl} = _mm512_setzero_si512();")
        L.append(f"{ind}for (int kg = 0; kg < {groups}; kg++) {{")
        L.append(f"{ind}    const int64_t k = p * {groups} + kg;")
        L.append(f"{ind}    const int16_t *xk = xw + k * 32;")
        L.append(f"{ind}    for (int kk = 0; kk < 32; kk++) {{")
        L.append(f"{ind}        const __m512i xv = _mm512_set1_epi16(xk[kk]);")
        for gg in range(G):
            for pl in range(planes):
                m = f"_cvtu32_mask32(((const uint32_t *)(rec{gg} + HDR_BYTES + kg * UNIT_BYTES + {pl * 128}))[kk])"
                L.append(f"{ind}        a{gg}_{pl} = _mm512_mask_add_epi16(a{gg}_{pl}, {m}, a{gg}_{pl}, xv);")
        L.append(f"{ind}    }}")
        if not q8k:
            for gg in range(G):
                comb = f"_mm512_add_epi16(_mm512_slli_epi16(a{gg}_1, 1), a{gg}_0)" if planes == 2 else f"_mm512_slli_epi16(a{gg}_0, 1)"
                L.append(f"{ind}    {{ const __m512i sv = _mm512_sub_epi16({comb}, _mm512_set1_epi16((short)sx[k]));")
                L.append(f"{ind}      const __m512 dx = _mm512_set1_ps(dxs[k]);")
                L.append(f"{ind}      alo{gg} = fma16(lo16(sv), dlo{gg}, dx, alo{gg});")
                L.append(f"{ind}      ahi{gg} = fma16(hi16(sv), dhi{gg}, dx, ahi{gg}); }}")
                for pl in range(planes):
                    L.append(f"{ind}    a{gg}_{pl} = _mm512_setzero_si512();")
            L.append(f"{ind}}}")
        else:  # one scale per 256: |A_b| <= 256 * 127 fits int16; widen once per period
            L.append(f"{ind}}}")
            for gg in range(G):
                L.append(f"{ind}{{ int32_t st = 0; for (int kk = 0; kk < {groups}; kk++) st += sx[p * {groups} + kk];")
                if planes == 2:
                    clo = f"_mm512_add_epi32(_mm512_slli_epi32(lo16(a{gg}_1), 1), lo16(a{gg}_0))"
                    chi = f"_mm512_add_epi32(_mm512_slli_epi32(hi16(a{gg}_1), 1), hi16(a{gg}_0))"
                else:
                    clo, chi = f"_mm512_slli_epi32(lo16(a{gg}_0), 1)", f"_mm512_slli_epi32(hi16(a{gg}_0), 1)"
                L.append(f"{ind}  const __m512i st16 = _mm512_set1_epi32(st);")
                L.append(f"{ind}  const __m512i ilo = _mm512_sub_epi32({clo}, st16), ihi = _mm512_sub_epi32({chi}, st16);")
                L.append(f"{ind}  const __m512 dx = _mm512_set1_ps(dxs[p * {groups}]);")
                L.append(f"{ind}  alo{gg} = fma16(ilo, dlo{gg}, dx, alo{gg}); ahi{gg} = fma16(ihi, dhi{gg}, dx, ahi{gg}); }}")
        decl = " ".join(f"__m512 alo{gg} = _mm512_setzero_ps(), ahi{gg} = _mm512_setzero_ps();" for gg in range(G))
        stores = "\n".join(
            f"        storev(Y, (g + {gg}) * 32, r0, r1, alo{gg}); storev(Y, (g + {gg}) * 32 + 16, r0, r1, ahi{gg});" for gg in range(G)
        )
        xprep = "int16_t xw[32768];\n    for (int64_t i = 0; i < K; i++) xw[i] = xq[i];"
    else:
        # sad: 4 row vectors of 8 rows (u64 lanes) per 32-row group; s = sum over selected (x + 128)
        for gg in range(G):
            L.append(f"{ind}const uint8_t *rec{gg} = pk->buf + ((size_t)(g + {gg}) * pk->nrec_k + p) * REC_BYTES;")
            if PF:
                L.append(f"{ind}_mm_prefetch((const char *)(rec{gg} + {PF} * REC_BYTES), _MM_HINT_T0);")
            for rv in range(4):
                L.append(f"{ind}const __m256 d{gg}_{rv} = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(rec{gg} + {16 * rv})));")
                if q8k:
                    L.append(f"{ind}__m512i t{gg}_{rv} = _mm512_setzero_si512();")
        L.append(f"{ind}for (int kg = 0; kg < {groups}; kg++) {{")
        L.append(f"{ind}    const int64_t k = p * {groups} + kg;")
        for o in range(4):
            L.append(f"{ind}    const __m512i x{o} = _mm512_set1_epi64(ldq(xu + k * 32 + {8 * o}));")
        for gg in range(G):
            L.append(f"{ind}    {{ const uint8_t *up = rec{gg} + HDR_BYTES + kg * UNIT_BYTES;")
            for rv in range(4):
                terms = []
                for pl in range(planes):
                    for o in range(4):
                        m = f"_cvtu64_mask64(ldq(up + {pl * 128 + (o * 4 + rv) * 8}))"
                        terms.append(f"_mm512_sad_epu8(_mm512_maskz_mov_epi8({m}, x{o}), zero)")
                pl_sums = []
                for pl in range(planes):
                    ts = terms[4 * pl : 4 * pl + 4]
                    pl_sums.append(f"_mm512_add_epi64(_mm512_add_epi64({ts[0]}, {ts[1]}), _mm512_add_epi64({ts[2]}, {ts[3]}))")
                if planes == 2:
                    comb = f"_mm512_add_epi64(_mm512_slli_epi64({pl_sums[1]}, 1), {pl_sums[0]})"
                else:
                    comb = f"_mm512_slli_epi64({pl_sums[0]}, 1)"
                corr = f"_mm512_slli_epi64(_mm512_cvtepi16_epi64(_mm_loadu_si128((const __m128i *)(up + {planes * 128 + 16 * rv}))), 7)"
                L.append(f"{ind}      const __m512i s{rv} = _mm512_sub_epi64({comb}, {corr});")
            if not q8k:
                L.append(f"{ind}      const __m256 dx = _mm256_set1_ps(dxs[k]); const __m512i sxk = _mm512_set1_epi64(sx[k]);")
                for rv in range(4):
                    L.append(f"{ind}      a{gg}_{rv} = fma8(_mm512_sub_epi64(s{rv}, sxk), d{gg}_{rv}, dx, a{gg}_{rv});")
            else:
                for rv in range(4):
                    L.append(f"{ind}      t{gg}_{rv} = _mm512_add_epi64(t{gg}_{rv}, s{rv});")
            L.append(f"{ind}    }}")
        L.append(f"{ind}}}")
        if q8k:
            for gg in range(G):
                L.append(f"{ind}{{ int64_t st = 0; for (int kk = 0; kk < {groups}; kk++) st += sx[p * {groups} + kk];")
                L.append(f"{ind}  const __m256 dx = _mm256_set1_ps(dxs[p * {groups}]); const __m512i sxk = _mm512_set1_epi64(st);")
                for rv in range(4):
                    L.append(f"{ind}  a{gg}_{rv} = fma8(_mm512_sub_epi64(t{gg}_{rv}, sxk), d{gg}_{rv}, dx, a{gg}_{rv});")
                L.append(f"{ind}}}")
        decl = " ".join(f"__m256 a{gg}_{rv} = _mm256_setzero_ps();" for gg in range(G) for rv in range(4))
        stores = "\n".join(f"        storev8(Y, (g + {gg}) * 32 + {8 * rv}, r0, r1, a{gg}_{rv});" for gg in range(G) for rv in range(4))
        xprep = "uint8_t xu[32768] __attribute__((aligned(64)));\n    for (int64_t i = 0; i < K; i++) xu[i] = (uint8_t)(xq[i] ^ 0x80);"
    per_unit = _act_unit_c(r.act, 1.0)
    body = f"""
static inline int64_t ldq(const uint8_t *p) {{ int64_t v; memcpy(&v, p, 8); return v; }}
void {entry}_packed(const void *pv, const void *X, float *Y, int64_t K, int64_t r0, int64_t r1) {{
    const packed_t *pk = (const packed_t *)pv;
    const xblock *xb = (const xblock *)X;
    int8_t xq[32768] __attribute__((aligned(64)));
    float dxs[1024]; int32_t sx[1024];
    for (int64_t k = 0; k < K / 32; k++) {{
        int8_t *dst = xq + 32 * k;
        {per_unit}
    }}
    {xprep}
    const __m512i zero = _mm512_setzero_si512();
    (void)zero;
    const int64_t g0 = r0 / 32, g1 = (r1 + 31) / 32;
    for (int64_t g = g0; g < g1; g += {G}) {{
        {decl}
        for (int64_t p = 0; p < pk->nrec_k; p++) {{
{chr(10).join(L)}
        }}
{stores}
    }}
}}
"""
    doc = f"/* addsub={mode}: multiplication-free; {planes} bit plane(s) per weight, {unit_bytes} B per 32 rows x 32 values */\n"
    return (
        _prelude(f"addsub {mode}")
        + doc
        + head
        + (STOREV if mode == "mask" else STOREV8)
        + _addsub_prepare(r, entry, G, mode, planes, groups, rec)
        + body
    )


# --------------------------------------------------------------------------- q2_K
Q2K_STRUCT = "uint8_t scales[16]; uint8_t qs[64]; uint16_t d; uint16_t dmin;"
Q2K_CODE = "((b->qs[(v / 128) * 32 + v % 32] >> (2 * ((v % 128) / 32))) & 3)"


def _q2k_scalar(c):
    entry = c["entry"]
    return f"""// Generated by kurn (lowbit: q2_K scalar reference). Do not edit.
#include "kurn.h"
#include <stdint.h>
#include <string.h>
static inline float f16f(uint16_t h) {{
    uint32_t s = (uint32_t)(h & 0x8000) << 16, e = (h >> 10) & 0x1f, m = h & 0x3ff, bb;
    if (e == 0) {{ if (!m) bb = s; else {{ e = 113; while (!(m & 0x400)) {{ m <<= 1; e--; }} bb = s | (e << 23) | ((m & 0x3ff) << 13); }} }}
    else if (e == 31) bb = s | 0x7f800000 | (m << 13);
    else bb = s | ((e + 112) << 23) | (m << 13);
    float f; memcpy(&f, &bb, 4); return f;
}}
typedef struct {{ {Q2K_STRUCT} }} nblock;
typedef struct {{ float d; int8_t qs[256]; int16_t bsums[16]; }} xblock;
void {entry}(const void *W, const void *X, float *Y, int64_t K, int64_t r0, int64_t r1) {{
    const xblock *x = (const xblock *)X;
    for (int64_t n = r0; n < r1; n++) {{
        const nblock *wr = (const nblock *)W + n * (K / 256);
        float acc = 0;
        for (int64_t sb = 0; sb < K / 256; sb++) {{
            const nblock *b = wr + sb;
            int32_t isum = 0, msum = 0;
            for (int s = 0; s < 16; s++) {{
                int32_t t = 0;
                for (int l = 0; l < 16; l++) {{ const int v = 16 * s + l; t += {Q2K_CODE} * x[sb].qs[v]; }}
                isum += (b->scales[s] & 15) * t;
                msum += (b->scales[s] >> 4) * x[sb].bsums[s];
            }}
            acc += x[sb].d * (f16f(b->d) * (float)isum - f16f(b->dmin) * (float)msum);
        }}
        Y[n] = acc;
    }}
}}
"""


def lower_q2k(target, c):
    """q2_K x Q8_K, layout k16: 16 rows per vector (int32 lanes), vpdpbusd on the unsigned 2-bit codes.
    Record per (16 rows, 256 values): d[16], dmin[16] (fp16), raw scale bytes as [pair][row][2],
    then 8 units of 32 values x 16 rows of 2-bit codes (same packing as the generic i16 layout)."""
    if target == "scalar":
        return _q2k_scalar(c)
    if target != "avx512_vnni":
        raise ValueError("q2_K: scalar and avx512_vnni only")
    entry, G, PF = c["entry"], c["rows"], c["prefetch"]
    hdr = 64 + 256
    rec = hdr + 8 * 128
    ind = "                "
    L = []
    for g in range(G):
        L.append(f"{ind}const uint8_t *rec{g} = pk->buf + ((size_t)(g + {g}) * pk->nrec_k + p) * REC_BYTES;")
        if PF:
            L.append(f"{ind}_mm_prefetch((const char *)(rec{g} + {PF} * REC_BYTES), _MM_HINT_T0);")
        L.append(f"{ind}__m512i ia{g} = _mm512_setzero_si512(), ma{g} = _mm512_setzero_si512();")
    L.append(f"{ind}for (int kg = 0; kg < 8; kg++) {{")
    L.append(f"{ind}    const int64_t k = p * 8 + kg;")
    L.append(f"{ind}    const __m512i bs = _mm512_set1_epi32(bsp[k]);  /* (bsums[2kg], bsums[2kg+1]) */")
    for kk in range(8):
        L.append(f"{ind}    const __m512i x{kk} = _mm512_set1_epi32(xw[k * 8 + {kk}]);")
    for g in range(G):
        L.append(f"{ind}    {{ const uint8_t *cp = rec{g} + {hdr} + kg * 128;")
        L.append(f"{ind}      const __m512i c0 = _mm512_loadu_si512((const void *)cp), c1 = _mm512_loadu_si512((const void *)(cp + 64));")
        L.append(f"{ind}      __m512i e = _mm512_setzero_si512(), o = _mm512_setzero_si512();")
        for kk in range(8):
            src = "c0" if kk < 4 else "c1"
            sh = 2 * (kk % 4)
            u = f"_mm512_and_si512(_mm512_srli_epi16({src}, {sh}), m3)" if sh else f"_mm512_and_si512({src}, m3)"
            acc = "e" if kk < 4 else "o"
            L.append(f"{ind}      {acc} = _mm512_dpbusd_epi32({acc}, {u}, x{kk});")
        L.append(f"{ind}      const __m512i sp = _mm512_cvtepu8_epi16(_mm256_loadu_si256((const __m256i *)(rec{g} + 64 + kg * 32)));")
        L.append(f"{ind}      const __m512i pair = _mm512_mask_blend_epi16(0xAAAAAAAAu, e, _mm512_slli_epi32(o, 16));")
        L.append(f"{ind}      ia{g} = _mm512_dpwssd_epi32(ia{g}, pair, _mm512_and_si512(sp, m15w));")
        L.append(f"{ind}      ma{g} = _mm512_dpwssd_epi32(ma{g}, _mm512_srli_epi16(sp, 4), bs); }}")
    L.append(f"{ind}}}")
    for g in range(G):
        L.append(f"{ind}{{ const __m512 dx = _mm512_set1_ps(dxs[p]);")
        L.append(
            f"{ind}  const __m512 d = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)rec{g})), "
            f"dm = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(rec{g} + 32)));"
        )
        L.append(f"{ind}  a{g} = _mm512_fmadd_ps(_mm512_cvtepi32_ps(ia{g}), _mm512_mul_ps(d, dx), a{g});")
        L.append(f"{ind}  a{g} = _mm512_fnmadd_ps(_mm512_cvtepi32_ps(ma{g}), _mm512_mul_ps(dm, dx), a{g}); }}")
    decl = " ".join(f"__m512 a{g} = _mm512_setzero_ps();" for g in range(G))
    stores = "\n".join(f"        storev(Y, (g + {g}) * 16, r0, r1, a{g});" for g in range(G))
    return (
        _prelude("q2_K k16")
        + f"""
typedef struct {{ {Q2K_STRUCT} }} nblock;
typedef struct {{ float d; int8_t qs[256]; int16_t bsums[16]; }} xblock;
typedef struct {{ int64_t nrec_k, ngroups; uint8_t *buf; }} packed_t;
#define REC_BYTES {rec}
{STOREV}

void *{entry}_prepare(const void *W, int64_t K, int64_t N) {{
    const nblock *w = (const nblock *)W;
    const int64_t nb = K / 256;
    packed_t *pk = malloc(sizeof *pk);
    pk->nrec_k = nb; pk->ngroups = (N + 15) / 16 + {G - 1};
    const size_t bytes = (size_t)REC_BYTES * pk->nrec_k * pk->ngroups;
    pk->buf = aligned_alloc(64, (bytes + 63) & ~(size_t)63);
    memset(pk->buf, 0, bytes);
    for (int64_t gr = 0; gr < pk->ngroups; gr++)
        for (int64_t p = 0; p < nb; p++) {{
            uint8_t *hp = pk->buf + ((size_t)gr * nb + p) * REC_BYTES;
            for (int row = 0; row < 16; row++) {{
                const int64_t n = gr * 16 + row;
                if (n >= N) continue;
                const nblock *b = w + n * nb + p;
                memcpy(hp + 2 * row, &b->d, 2); memcpy(hp + 32 + 2 * row, &b->dmin, 2);
                for (int s = 0; s < 16; s++) hp[64 + (s / 2) * 32 + row * 2 + (s & 1)] = b->scales[s];
                for (int kg = 0; kg < 8; kg++)
                    for (int kk = 0; kk < 8; kk++)
                        for (int j = 0; j < 4; j++) {{
                            const int v = kg * 32 + kk * 4 + j;
                            const int q = {Q2K_CODE};
                            hp[{hdr} + kg * 128 + (kk / 4) * 64 + row * 4 + j] |= (uint8_t)(q << (2 * (kk & 3)));
                        }}
            }}
        }}
    return pk;
}}

void {entry}_packed(const void *pv, const void *X, float *Y, int64_t K, int64_t r0, int64_t r1) {{
    const packed_t *pk = (const packed_t *)pv;
    const xblock *xb = (const xblock *)X;
    const int64_t nk = K / 32;
    int32_t xw[32768 / 4], bsp[1024];
    float dxs[128];
    for (int64_t k = 0; k < nk; k++) {{
        const xblock *b = xb + k / 8;
        memcpy(xw + k * 8, b->qs + (k % 8) * 32, 32);
        memcpy(bsp + k, b->bsums + 2 * (k % 8), 4);
        dxs[k / 8] = b->d;
    }}
    const __m512i m3 = _mm512_set1_epi8(3), m15w = _mm512_set1_epi16(15);
    const int64_t g0 = r0 / 16, g1 = (r1 + 15) / 16;
    for (int64_t g = g0; g < g1; g += {G}) {{
        {decl}
        for (int64_t p = 0; p < pk->nrec_k; p++) {{
{chr(10).join(L)}
        }}
{stores}
    }}
}}
"""
    )


def lower_q2k_kernel(target, c):
    from . import hooks

    if c.get("layout") in hooks.LOWERINGS and c["layout"] != "k16":
        return hooks.LOWERINGS[c["layout"]](target, c)
    return lower_q2k(target, c)


GOLDEN = {
    "q1_0_gemv_avx512_lut_direct4": {"op": "gemv", "weights": "q1_0", "target": "avx512_vnni", "layout": "lut",
                                     "lut": "direct4", "rows": 2},
    "tq2_0_gemv_avx512_lut_serial6m": {"op": "gemv", "weights": "tq2_0", "target": "avx512_vnni", "layout": "lut",
                                       "lut": "serial6m", "rows": 1},
    "q2_0_gemv_avx512_lut_direct3": {"op": "gemv", "weights": "q2_0", "target": "avx512_vnni", "layout": "lut",
                                     "lut": "direct3", "rows": 1},
    "tq1_0_gemv_avx512_lut_tern": {"op": "gemv", "weights": "tq1_0", "target": "avx512_vnni", "layout": "lut", "rows": 1},
    "tq1_0_gemv_scalar": {"op": "gemv", "weights": "tq1_0", "target": "scalar"},
    "q2_0_gemv_avx512_addsub_mask": {"op": "gemv", "weights": "q2_0", "target": "avx512_vnni", "layout": "addsub",
                                     "addsub": "mask", "rows": 1},
    "q1_0_gemv_avx512_addsub_sad": {"op": "gemv", "weights": "q1_0", "target": "avx512_vnni", "layout": "addsub",
                                    "addsub": "sad", "rows": 1},
    "q2_K_gemv_scalar": {"op": "gemv", "weights": "q2_K", "target": "scalar"},
    "q2_K_gemv_avx512_k16_rows2": {"op": "gemv", "weights": "q2_K", "target": "avx512_vnni", "layout": "k16", "rows": 2},
}  # fmt: skip
