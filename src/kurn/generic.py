"""Recipes ("what") and the generic lane-per-row lowering ("how").

A weight format is described once as a *recipe*:

  * `bits`: width of the stored code (8, 4, 2, 1),
  * how to read code v of a native ggml block (`code_c`, plain C),
  * the code -> integer map, written as  w = alpha * u + beta  where u is the
    unsigned operand fed to vpdpbusd (u = code, or LUT[code], or code + 128),
  * the scale structure: one fp16 `d` per `period` values (32, 64, 128, 256),
    or Q4_K's d * sc[s] - dmin * mn[s] two-level scales,
  * the activation format (q8_0 or q8_K).

Every recipe lowers through the same kernel template. Algorithm choices are
schedule keys, not new code:

  layout      native | i16 (AVX-512) / i8 (AVX2-VNNI): rows interleaved so one
              vector load feeds vpdpbusd for 16 (8) rows, lane = row
  unpack      mask (and/shift) | lut (vpshufb codebook; folds the offset into u)
              4-bit only (lower_nibble):
              mask16 (low nibble AND 0x0F, high nibble AND 0xF0 without a shift;
                      the x16 is removed once per K-group with an exact shift)
              perm   (vpermb 64-entry codebook: the low nibble needs no AND; VBMI)
              pair   (rows r and r + 16 (8) share a byte at the same k, so
                      dp(byte) - dp(byte & 0xF0) and dp(byte & 0xF0) >> 4 give both
                      rows: one ALU op per 64-byte load; linear codes only)
  correction  act    (beta * sum(x) per activation block, seeded into the
                      accumulator; free, shared by all rows)
              weight (weights stay signed, activations are biased by +128 and
                      128 * sum(w) per row-block is stored at repack time)
              dpmin  (Q4_K, 4-bit lowering: the dmin * mn * bsum term once per
                      super-block by two vpdpbusd of the min bytes against the
                      activation block sums split into 7-bit halves)
  scales      unpacked (pre-decoded per row) | packed (Q4_K: 6-bit, decoded in-kernel;
              NVFP4: UE4M3 bytes decoded in-kernel instead of fp16 at repack)
  accum       float (convert + FMA per 32 values) | int (integer accumulation
              across a whole scale period; one convert per period)
  rows        row groups per pass,  cols  activation columns (multi-token verify)
  prefetch    software prefetch distance in records; pfgran rec (one line per record) | line
              (every line of the record; K-group lines inside the K-group loop), pfhint t0|t1|t2|nta

The generated C reads only the repacked buffer, so it needs no ggml headers.
"""

from dataclasses import dataclass, field
from typing import Optional

PF_HINTS = {"t0": "_MM_HINT_T0", "t1": "_MM_HINT_T1", "t2": "_MM_HINT_T2", "nta": "_MM_HINT_NTA"}
KV_IQ4NL = (-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113)
KV_FP4 = (0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12)  # E2M1 x 2 (ggml kvalues_fp4)


@dataclass(frozen=True)
class Recipe:
    name: str
    block: int  # values per native block
    nbytes: int  # bytes per native block
    act: str  # q8_0 | q8_K
    bits: int
    struct: str  # C struct body of the native block
    code_c: str  # C expression: raw code of value v in native block `b` (unsigned, or int8 for q8_0)
    period: int  # values per fp16 scale d (q4_K: 256 = super-block)
    d_c: str = "b->d"  # C expression for the fp16 bits of d for the native block holding the period
    # value = alpha * u + beta, for each unpack method: (u expression from raw code `q`, alpha, beta)
    maps: dict = field(default_factory=dict)
    two_level: bool = False  # q4_K: d * sc[s] * q - dmin * mn[s]
    lut: Optional[tuple] = None  # value table for unpack=lut (indexed by raw code)
    signed_c: str = ""  # C expression: signed value of raw code `q` (correction=weight)
    doc: str = ""
    scale: str = "f16"  # f16 | e8m0 (MXFP4: 2^(e-128)) | ue4m3 (NVFP4: one scale per `sub` values)
    sub: int = 32  # values per scale (16 for NVFP4)
    dsc_c: str = "f16f(b->d)"  # scalar C: float scale of value v in native block b

    @property
    def unpacks(self):
        return tuple(self.maps)


RECIPES = {
    r.name: r
    for r in (
        Recipe(
            "q8_0",
            32,
            34,
            "q8_0",
            8,
            "uint16_t d; int8_t qs[32];",
            "b->qs[v]",
            32,
            maps={"none": ("(uint8_t)(q + 128)", 1, -128)},
            signed_c="q",
            doc="value = d * qs",
        ),  # fmt: skip
        Recipe(
            "q4_0",
            32,
            18,
            "q8_0",
            4,
            "uint16_t d; uint8_t qs[16];",
            "(v < 16 ? b->qs[v] & 15 : b->qs[v - 16] >> 4)",
            32,
            maps={"mask": ("q", 1, -8), "lut": ("(uint8_t)(q - 8 + 128)", 1, -128)},
            lut=tuple(q - 8 for q in range(16)),
            signed_c="(q - 8)",
            doc="value = d * (q - 8); low nibbles hold values 0..15, high nibbles 16..31",
        ),  # fmt: skip
        Recipe(
            "iq4_nl",
            32,
            18,
            "q8_0",
            4,
            "uint16_t d; uint8_t qs[16];",
            "(v < 16 ? b->qs[v] & 15 : b->qs[v - 16] >> 4)",
            32,
            maps={"lut": ("(uint8_t)(KV[q] + 128)", 1, -128)},
            lut=KV_IQ4NL,
            signed_c="KV[q]",
            doc="value = d * kvalues_iq4nl[q] (non-linear 4-bit codebook)",
        ),  # fmt: skip
        Recipe(
            "q2_0",
            64,
            18,
            "q8_0",
            2,
            "uint16_t d; uint8_t qs[16];",
            "((b->qs[v / 4] >> (2 * (v % 4))) & 3)",
            64,
            maps={"mask": ("q", 1, -1), "lut": ("(uint8_t)(q - 1 + 128)", 1, -128)},
            lut=(-1, 0, 1, 2),
            signed_c="(q - 1)",
            doc="value = d * (q - 1), q in 0..3 (ternary models use -1, 0, 1)",
        ),  # fmt: skip
        Recipe(
            "tq2_0",
            256,
            66,
            "q8_K",
            2,
            "uint8_t qs[64]; uint16_t d;",
            "((b->qs[(v / 128) * 32 + v % 32] >> (2 * ((v % 128) / 32))) & 3)",
            256,
            maps={"mask": ("q", 1, -1), "lut": ("(uint8_t)(q - 1 + 128)", 1, -128)},
            lut=(-1, 0, 1, 2),
            signed_c="(q - 1)",
            doc="ternary: value = d * (q - 1), q in {0, 1, 2}; one fp16 d per 256 values; Q8_K activations",
        ),  # fmt: skip
        Recipe(
            "q1_0",
            128,
            18,
            "q8_0",
            1,
            "uint16_t d; uint8_t qs[16];",
            "((b->qs[v / 8] >> (v % 8)) & 1)",
            128,
            maps={"mask": ("q", 2, -1)},
            signed_c="(2 * q - 1)",
            doc="1-bit (Bonsai): value = d * (bit ? +1 : -1); one fp16 d per 128 values",
        ),  # fmt: skip
        Recipe(
            "q4_K",
            256,
            144,
            "q8_K",
            4,
            "uint16_t d; uint16_t dmin; uint8_t scales[12]; uint8_t qs[128];",
            "((b->qs[(v / 64) * 32 + v % 32] >> (4 * ((v / 32) & 1))) & 15)",
            256,
            maps={"mask": ("q", 1, 0)},
            two_level=True,
            doc="value = d * sc[s] * q - dmin * mn[s] (s = v / 32, 6-bit sc/mn)",
        ),  # fmt: skip
        Recipe(
            "mxfp4",
            32,
            17,
            "q8_0",
            4,
            "uint8_t e; uint8_t qs[16];",
            "(v < 16 ? b->qs[v] & 15 : b->qs[v - 16] >> 4)",
            32,
            d_c="b->e",
            maps={"lut": ("(uint8_t)(KV[q] + 128)", 1, -128)},
            lut=KV_FP4,
            signed_c="KV[q]",
            scale="e8m0",
            dsc_c="e8m0h(b->e)",
            doc="MXFP4 (OCP MX): value = kvalues_fp4[q] * 2^(e - 128); E2M1 x 2 codes, E8M0 scale per 32",
        ),  # fmt: skip
        Recipe(
            "nvfp4",
            64,
            36,
            "q8_0",
            4,
            "uint8_t d[4]; uint8_t qs[32];",
            "((v % 16) < 8 ? b->qs[(v / 16) * 8 + v % 16] & 15 : b->qs[(v / 16) * 8 + v % 16 - 8] >> 4)",
            64,
            maps={"lut": ("(uint8_t)(KV[q] + 128)", 1, -128)},
            lut=KV_FP4,
            signed_c="KV[q]",
            scale="ue4m3",
            sub=16,
            dsc_c="ue4m3h(b->d[v / 16])",
            doc="NVFP4 (ggml block_nvfp4): value = kvalues_fp4[q] * ue4m3(d[s]) / 2, one UE4M3 scale per 16 values; "
            "a per-tensor fp32 scale, if any, is applied by the caller",
        ),  # fmt: skip
    )
}
NIBBLE_UNPACKS = ("mask16", "perm", "pair")

TARGET_VEC = {
    # lanes = rows per vector (interleave width); `w` = vector width in bits
    "avx512_vnni": {"lanes": 16, "w": 512, "layout": "i16"},
    "avx2_vnni": {"lanes": 8, "w": 256, "layout": "i8"},
}


def legal_keys(fmt, target):
    """Legal values of the algorithm keys for the generic lowering."""
    r = RECIPES[fmt]
    unpack = r.unpacks
    if r.bits == 4:
        if "mask" in r.maps:
            unpack += ("mask16", "pair")
        elif target == "avx512_vnni":
            unpack += ("perm",)
    if r.two_level:
        corr = ("act", "dpmin")
    elif r.signed_c and r.period == 32 and r.bits >= 4 and r.scale == "f16":
        corr = ("act", "weight")
    else:
        corr = ("act",)
    keys = {
        "unpack": unpack,
        "correction": corr,
        "scales": ("unpacked", "packed") if (r.two_level or r.scale == "ue4m3") else ("unpacked",),
        "accum": ("float", "int") if (r.two_level or r.act == "q8_K") else ("float",),
    }
    return keys


def nibble_path(r, c):
    """4-bit configs lowered by lower_nibble (the new unpack schemes and the MX scale formats)."""
    return r.bits == 4 and (c.get("unpack") in NIBBLE_UNPACKS or r.scale != "f16")


def _interleaved(c):
    return c.get("layout") in ("i16", "i8") and c["weights"] in RECIPES


NIBBLE_INVALID = [
    (lambda c: _interleaved(c) and c["unpack"] in NIBBLE_UNPACKS and c["correction"] == "weight",
     "unpack=mask16/perm/pair needs correction=act (or dpmin)"),
    (lambda c: _interleaved(c) and c["correction"] == "dpmin" and c["unpack"] not in ("mask16", "pair"),
     "correction=dpmin needs unpack=mask16 or pair"),
    (lambda c: _interleaved(c) and RECIPES[c["weights"]].two_level and c["unpack"] in NIBBLE_UNPACKS
     and c["scales"] == "packed", "unpack=mask16/pair needs scales=unpacked for two-level formats"),
]  # fmt: skip
# pair keeps two accumulator chains per (row group, column) up to this many independent
# (row group, column) pairs, one chain beyond (the register file holds 32 vectors)
PAIR_SPLIT_MAX = 4


def combo_problem(c):
    """Combinations of algorithm keys that are individually legal but not together (i16 / i8 layouts)."""
    return next((msg for bad, msg in NIBBLE_INVALID if bad(c)), None)


# --------------------------------------------------------------------------- C building blocks
def _prims(target):
    if target == "avx512_vnni":
        return dict(
            V="__m512i",
            F="__m512",
            L=16,
            zero="_mm512_setzero_si512()",
            fzero="_mm512_setzero_ps()",
            loadu=lambda p: f"_mm512_loadu_si512((const void *)({p}))",
            and_=lambda a, b: f"_mm512_and_si512({a}, {b})",
            srli16=lambda a, n: f"_mm512_srli_epi16({a}, {n})",
            set1_8=lambda v: f"_mm512_set1_epi8((char)({v}))",
            set1_32=lambda v: f"_mm512_set1_epi32({v})",
            shuf=lambda t, i: f"_mm512_shuffle_epi8({t}, {i})",
            dp=lambda acc, u, x: f"_mm512_dpbusd_epi32({acc}, {u}, {x})",
            add=lambda a, b: f"_mm512_add_epi32({a}, {b})",
            mullo=lambda a, b: f"_mm512_mullo_epi32({a}, {b})",
            cvt=lambda a: f"_mm512_cvtepi32_ps({a})",
            fma=lambda a, b, c: f"_mm512_fmadd_ps({a}, {b}, {c})",
            mul=lambda a, b: f"_mm512_mul_ps({a}, {b})",
            fset1=lambda v: f"_mm512_set1_ps({v})",
            h2f=lambda p: f"_mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)({p})))",
            u8x=lambda p: f"_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *)({p})))",
            i16x=lambda p: f"_mm512_cvtepi16_epi32(_mm256_loadu_si256((const __m256i *)({p})))",
            bits=lambda p: f"_mm512_maskz_mov_epi8(_cvtu64_mask64(*(const uint64_t *)({p})), _mm512_set1_epi8(1))",
            tbl=lambda name: f"_mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i *){name}))",
        )
    if target == "avx2_vnni":
        return dict(
            V="__m256i",
            F="__m256",
            L=8,
            zero="_mm256_setzero_si256()",
            fzero="_mm256_setzero_ps()",
            loadu=lambda p: f"_mm256_loadu_si256((const __m256i *)({p}))",
            and_=lambda a, b: f"_mm256_and_si256({a}, {b})",
            srli16=lambda a, n: f"_mm256_srli_epi16({a}, {n})",
            set1_8=lambda v: f"_mm256_set1_epi8((char)({v}))",
            set1_32=lambda v: f"_mm256_set1_epi32({v})",
            shuf=lambda t, i: f"_mm256_shuffle_epi8({t}, {i})",
            dp=lambda acc, u, x: f"_mm256_dpbusd_avx_epi32({acc}, {u}, {x})",
            add=lambda a, b: f"_mm256_add_epi32({a}, {b})",
            mullo=lambda a, b: f"_mm256_mullo_epi32({a}, {b})",
            cvt=lambda a: f"_mm256_cvtepi32_ps({a})",
            fma=lambda a, b, c: f"_mm256_fmadd_ps({a}, {b}, {c})",
            mul=lambda a, b: f"_mm256_mul_ps({a}, {b})",
            fset1=lambda v: f"_mm256_set1_ps({v})",
            h2f=lambda p: f"_mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)({p})))",
            u8x=lambda p: f"_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)({p})))",
            i16x=lambda p: f"_mm256_cvtepi16_epi32(_mm_loadu_si128((const __m128i *)({p})))",
            bits=lambda p: f"bits8x4(*(const uint32_t *)({p}))",
            tbl=lambda name: f"_mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *){name}))",
        )
    raise ValueError(target)


def _record_layout(r, c, L):
    """Bytes of one record = (L rows) x (one scale period): header then codes for period/32 K-groups."""
    groups = r.period // 32
    code_bytes = 32 * L * r.bits // 8  # per K-group
    if r.two_level:
        hdr = 4 * L + (16 * L if c["scales"] == "unpacked" else 12 * L)  # d, dmin f16 + sc/mn u8 (or raw 12 B per row)
    else:
        hdr = 2 * L
    corr = 2 * L if c["correction"] == "weight" else 0  # int16 sum(w) per row per K-group
    return hdr, code_bytes, corr, groups, hdr + groups * (code_bytes + corr)


def lower(target, c):
    """Generic GEMV (cols == 1) / multi-column verify (cols > 1) for recipe c['weights']."""
    if target == "scalar":
        return _scalar(c)
    if c.get("layout") == "l32":
        return lower_lut(target, c)
    r = RECIPES[c["weights"]]
    if nibble_path(r, c):
        return lower_nibble(target, c)
    P = _prims(target)
    L = P["L"]
    G, M, PF = c["rows"], c["cols"], c["prefetch"]
    unpack, corr_mode, acc_mode = c["unpack"], c["correction"], c["accum"]
    u_expr, alpha, beta = r.maps[unpack] if unpack in r.maps else r.maps[next(iter(r.maps))]
    hdr, cbytes, corr, groups, rec = _record_layout(r, c, L)
    V, F = P["V"], P["F"]
    two = r.two_level
    entry = c["entry"]
    gemm = M > 1

    # ---------------- repack (prepare) -------------------------------------------------------
    if corr_mode == "weight" and r.bits == 8:
        ustore = f"(uint8_t)(int8_t)({r.signed_c})"
    elif corr_mode == "weight":
        ustore = "q"
    elif unpack == "lut" or r.bits == 8:
        ustore = "q" if unpack == "lut" else u_expr  # lut: store raw code, map at runtime; q8_0: biased byte
    else:
        ustore = "q"
    lut_tbl = ""
    if unpack == "lut" or corr_mode == "weight" and r.lut:
        vals = r.lut
        if corr_mode == "weight":
            entries = [v & 0xFF for v in vals]  # signed values as bytes
        else:
            entries = [(v + 128) & 0xFF for v in vals]
        entries = (entries * (16 // len(entries) + 1))[:16]
        lut_tbl = "static const uint8_t LUT[16] = {" + ", ".join(str(e) for e in entries) + "};\n"
    kv = ""
    if "KV[" in (r.signed_c + u_expr):
        kv = "static const int8_t KV[16] __attribute__((unused)) = {" + ", ".join(str(v) for v in KV_IQ4NL) + "};\n"

    pack_code = {
        8: "rec_codes[(kk) * 64 + (row) * 4 + j] = u;",
        4: "rec_codes[(kk / 2) * 64 + (row) * 4 + j] |= (uint8_t)(u << (4 * (kk & 1)));",
        2: "rec_codes[(kk / 4) * 64 + (row) * 4 + j] |= (uint8_t)(u << (2 * (kk & 3)));",
        1: "rec_codes[(kk) * 8 + ((row) * 4 + j) / 8] |= (uint8_t)(u << (((row) * 4 + j) % 8));",
    }[r.bits]
    if L == 8:  # AVX2: 8 rows x 4 values = 32 codes per kk -> 32-byte vectors, 4-byte bit words
        pack_code = {
            8: "rec_codes[(kk) * 32 + (row) * 4 + j] = u;",
            4: "rec_codes[(kk / 2) * 32 + (row) * 4 + j] |= (uint8_t)(u << (4 * (kk & 1)));",
            2: "rec_codes[(kk / 4) * 32 + (row) * 4 + j] |= (uint8_t)(u << (2 * (kk & 3)));",
            1: "rec_codes[(kk) * 4 + ((row) * 4 + j) / 8] |= (uint8_t)(u << (((row) * 4 + j) % 8));",
        }[r.bits]
    vec_bytes = L * 4  # bytes per code vector (64 or 32)

    if two:
        if c["scales"] == "unpacked":
            hdr_fill = f"""
                {{ const uint8_t *q = b->scales;
                  for (int s = 0; s < 8; s++) {{
                      hp[4 * {L} + s * {L} + row] = s < 4 ? q[s] & 63 : (q[s + 4] & 0xF) | ((q[s - 4] >> 6) << 4);
                      hp[12 * {L} + s * {L} + row] = s < 4 ? q[s + 4] & 63 : (q[s + 4] >> 4) | ((q[s] >> 6) << 4);
                  }} }}"""
        else:
            hdr_fill = f"""
                for (int t = 0; t < 12; t++) hp[4 * {L} + t * {L} + row] = b->scales[t];"""
        hdr_fill = (
            f"""
                memcpy(hp + 2 * row, &b->d, 2); memcpy(hp + 2 * {L} + 2 * row, &b->dmin, 2);"""
            + hdr_fill
        )
    else:
        hdr_fill = """
                memcpy(hp + 2 * row, &b->d, 2);"""

    wsum_line = ("wsum += " + r.signed_c + ";") if corr_mode == "weight" else ""
    wsum_store = "{ const int16_t ws = (int16_t)wsum; memcpy(rec_codes + CODE_BYTES + 2 * row, &ws, 2); }" if corr_mode == "weight" else ""
    prepare = f"""
typedef struct {{ {r.struct} }} nblock;
{kv}{lut_tbl}typedef struct {{ int64_t nrec_k, ngroups; uint8_t *buf; }} packed_t;
#define REC_BYTES {rec}
#define HDR_BYTES {hdr}
#define KG_BYTES {cbytes + corr}
#define CODE_BYTES {cbytes}

void *{entry}_prepare(const void *W, int64_t K, int64_t N) {{
    const nblock *w = (const nblock *)W;
    const int64_t nb = K / {r.block};
    packed_t *pk = malloc(sizeof *pk);
    pk->nrec_k = K / {r.period}; pk->ngroups = (N + {L - 1}) / {L};
    const size_t bytes = (size_t)REC_BYTES * pk->nrec_k * pk->ngroups;
    pk->buf = aligned_alloc(64, (bytes + 63) & ~(size_t)63);
    memset(pk->buf, 0, bytes);
    for (int64_t g = 0; g < pk->ngroups; g++)
        for (int64_t p = 0; p < pk->nrec_k; p++) {{
            uint8_t *hp = pk->buf + ((size_t)g * pk->nrec_k + p) * REC_BYTES;
            for (int row = 0; row < {L}; row++) {{
                const int64_t n = g * {L} + row;
                if (n >= N) continue;  /* padding rows stay zero: d = 0 */
                const int64_t v0 = p * {r.period};
                const nblock *b = w + n * nb + v0 / {r.block};
                {hdr_fill}
                for (int kg = 0; kg < {groups}; kg++) {{
                    uint8_t *rec_codes = hp + HDR_BYTES + kg * KG_BYTES;
                    int32_t wsum = 0;
                    for (int kk = 0; kk < 8; kk++)
                        for (int j = 0; j < 4; j++) {{
                            const int v = (int)((v0 + kg * 32 + kk * 4 + j) % {r.block});
                            const int q = {r.code_c};
                            (void)wsum;
                            {wsum_line}
                            const uint8_t u = {ustore};
                            {pack_code}
                        }}
                    {wsum_store}
                }}
            }}
        }}
    return pk;
}}
"""

    # ---------------- activation prep ------------------------------------------------------
    if r.act == "q8_0":
        act_struct = "typedef struct { uint16_t d; int8_t qs[32]; } xblock;"
        act_prep = """
    const xblock *xb = (const xblock *)X + (m < M ? m : 0) * nbx;  /* columns beyond M repeat column 0 */
    for (int64_t k = 0; k < nk; k++) {
        int32_t s = 0;
        for (int l = 0; l < 32; l++) s += xb[k].qs[l];
        memcpy(xw + (m * nk + k) * 8, xb[k].qs, 32);
        sx[m * nk + k] = s;
        dx[m * nk + k] = f16f(xb[k].d);
    }"""
        nbx = "K / 32"
    else:
        act_struct = "typedef struct { float d; int8_t qs[256]; int16_t bsums[16]; } xblock;"
        act_prep = """
    const xblock *xb = (const xblock *)X + (m < M ? m : 0) * nbx;  /* columns beyond M repeat column 0 */
    for (int64_t k = 0; k < nk; k++) {
        const xblock *b = xb + k / 8;
        memcpy(xw + (m * nk + k) * 8, b->qs + (k % 8) * 32, 32);
        sx[m * nk + k] = b->bsums[2 * (k % 8)] + b->bsums[2 * (k % 8) + 1];
        dx[m * nk + k] = b->d;
    }"""
        nbx = "K / 256"
    act_bias = ""
    if corr_mode == "weight":
        act_bias = f"\n    for (int64_t i = 0; i < {M} * nk * 8; i++) xw[i] ^= (int32_t)0x80808080;  /* x + 128 as u8 */"
    # the same prep per (column m, block k) for the shared-prep workspace (xprep)
    if r.act == "q8_0":
        xblk, xrow = "xp[m] + 34 * k", "nk * 34"
        xprep_lines = [
            "        { const xblock *b = (const xblock *)blk; int32_t s = 0;",
            "          for (int l = 0; l < 32; l++) s += b->qs[l];",
            "          memcpy(xw + (m * nk + k) * 8, b->qs, 32); sx[m * nk + k] = s; dx[m * nk + k] = f16f(b->d); }",
        ]
    else:
        xblk, xrow = "xp[m] + 292 * (k / 8)", "(K / 256) * 292"
        xprep_lines = [
            "        { const xblock *b = (const xblock *)blk;",
            "          memcpy(xw + (m * nk + k) * 8, b->qs + (k % 8) * 32, 32);",
            "          sx[m * nk + k] = b->bsums[2 * (k % 8)] + b->bsums[2 * (k % 8) + 1]; dx[m * nk + k] = b->d; }",
        ]
    if corr_mode == "weight":
        xprep_lines.append("        for (int i = 0; i < 8; i++) xw[(m * nk + k) * 8 + i] ^= (int32_t)0x80808080;")

    # ---------------- inner kernel ---------------------------------------------------------
    lines = []
    ind = "                "

    def code_vec(g, kk):
        base = f"rec{g} + HDR_BYTES + kg * KG_BYTES"
        if r.bits == 8:
            v = P["loadu"](f"{base} + {kk * vec_bytes}")
        elif r.bits == 4:
            raw = f"c{g}_{kk // 2}"
            v = P["and_"](raw, "m4") if kk % 2 == 0 else P["and_"](P["srli16"](raw, 4), "m4")
        elif r.bits == 2:
            raw = f"c{g}_{kk // 4}"
            sh = 2 * (kk % 4)
            v = P["and_"](P["srli16"](raw, sh) if sh else raw, "m3")
        else:
            v = P["bits"](f"{base} + {kk * vec_bytes // 8}")
        if unpack == "lut" or (corr_mode == "weight" and r.lut and r.bits < 8):
            v = P["shuf"]("lut", v)
        elif corr_mode == "weight" and r.bits < 8 and not r.lut:
            pass
        return v

    for g in range(G):
        lines.append(f"{ind}const uint8_t *rec{g} = pk->buf + ((size_t)(g + {g}) * pk->nrec_k + p) * REC_BYTES;")
    if PF:
        hint = PF_HINTS[c.get("pfhint", "t0")]
        offs = [f" + {o}" for o in range(0, rec, 64)] if c.get("pfgran", "rec") == "line" else [""]
        for g in range(G):
            for off in offs:
                lines.append(f"{ind}_mm_prefetch((const char *)(rec{g} + {PF} * REC_BYTES{off}), {hint});")
    # per-record scale vectors
    for g in range(G):
        if two:
            lines.append(f"{ind}const {F} d{g} = {P['h2f'](f'rec{g}')}, dm{g} = {P['h2f'](f'rec{g} + {2 * L}')};")
        else:
            lines.append(f"{ind}const {F} d{g} = {P['h2f'](f'rec{g}')};")
    if two and c["scales"] == "packed":
        # vectorized 6-bit decode across the L rows: raw byte t of every row is a u8 vector
        for g in range(G):
            lines.append(f"{ind}{V} q{g}[12]; for (int t = 0; t < 12; t++) q{g}[t] = {P['u8x'](f'rec{g} + {4 * L} + t * {L}')};")
    int_period = acc_mode == "int"
    if int_period or two:
        for g in range(G):
            for m in range(M):
                lines.append(f"{ind}{V} ia{g}_{m} = {P['zero']};" + (f" {V} ma{g}_{m} = {P['zero']};" if two else ""))
    lines.append(f"{ind}for (int kg = 0; kg < {groups}; kg++) {{")
    lines.append(f"{ind}    const int64_t k = p * {groups} + kg;")
    for g in range(G):
        if r.bits == 4:
            for t in range(4):
                lines.append(f"{ind}    const {V} c{g}_{t} = {P['loadu'](f'rec{g} + HDR_BYTES + kg * KG_BYTES + {t * vec_bytes}')};")
        elif r.bits == 2:
            for t in range(2):
                lines.append(f"{ind}    const {V} c{g}_{t} = {P['loadu'](f'rec{g} + HDR_BYTES + kg * KG_BYTES + {t * vec_bytes}')};")
    # accumulator seeds
    for g in range(G):
        for m in range(M):
            if corr_mode == "weight":
                seed = f"{P['mullo'](P['i16x'](f'rec{g} + HDR_BYTES + kg * KG_BYTES + CODE_BYTES'), P['set1_32']('-128'))}"
            elif beta and alpha == 1:
                seed = P["set1_32"](f"{beta} * sx[{m} * nk + k]")
            else:
                seed = P["zero"]
            lines.append(f"{ind}    {V} e{g}_{m} = {seed}, o{g}_{m} = {P['zero']};")
    for kk in range(8):
        acc = "e" if kk % 2 == 0 else "o"
        for m in range(M):
            lines.append(f"{ind}    const {V} x{m}_{kk} = {P['set1_32'](f'xw[({m} * nk + k) * 8 + {kk}]')};")
        for g in range(G):
            lines.append(f"{ind}    {{ const {V} u = {code_vec(g, kk)};")
            for m in range(M):
                ops = (f"x{m}_{kk}", "u") if corr_mode == "weight" else ("u", f"x{m}_{kk}")
                lines.append(f"{ind}      {acc}{g}_{m} = {P['dp'](f'{acc}{g}_{m}', *ops)};")
            lines.append(f"{ind}    }}")
    # apply scales
    for g in range(G):
        for m in range(M):
            dsum = P["add"](f"e{g}_{m}", f"o{g}_{m}")
            if alpha != 1:  # w = alpha * u + beta with beta / alpha fractional (q1_0): fix up in integers
                dsum = P["add"](P["mullo"](dsum, P["set1_32"](str(alpha))), P["set1_32"](f"{beta} * sx[{m} * nk + k]"))
            if two:
                if c["scales"] == "unpacked":
                    scv = P["u8x"](f"rec{g} + {4 * L} + kg * {L}")
                    mnv = P["u8x"](f"rec{g} + {12 * L} + kg * {L}")
                else:
                    scv = f"q4k_sc(q{g}, kg)"
                    mnv = f"q4k_mn(q{g}, kg)"
                lines.append(f"{ind}    ia{g}_{m} = {P['add'](f'ia{g}_{m}', P['mullo'](dsum, scv))};")
                lines.append(f"{ind}    ma{g}_{m} = {P['add'](f'ma{g}_{m}', P['mullo'](mnv, P['set1_32'](f'sx[{m} * nk + k]')))};")
            elif int_period:
                lines.append(f"{ind}    ia{g}_{m} = {P['add'](f'ia{g}_{m}', dsum)};")
            else:
                sc = P["mul"](f"d{g}", P["fset1"](f"dx[{m} * nk + k]"))
                lines.append(f"{ind}    a{g}_{m} = {P['fma'](P['cvt'](dsum), sc, f'a{g}_{m}')};")
    lines.append(f"{ind}}}")
    if two or int_period:
        for g in range(G):
            for m in range(M):
                kx = f"{m} * nk + p * {groups}"
                dsc = P["mul"](f"d{g}", P["fset1"](f"dx[{kx}]"))
                lines.append(f"{ind}a{g}_{m} = {P['fma'](P['cvt'](f'ia{g}_{m}'), dsc, f'a{g}_{m}')};")
                if two:
                    msc = P["mul"](f"dm{g}", P["fset1"](f"-dx[{kx}]"))
                    lines.append(f"{ind}a{g}_{m} = {P['fma'](P['cvt'](f'ma{g}_{m}'), msc, f'a{g}_{m}')};")
    inner = "\n".join(lines)
    decl = " ".join(f"{F} a{g}_{m} = {P['fzero']};" for g in range(G) for m in range(M))
    stores = "\n".join(
        (f"        if ({m} < M) " if m else "        ") + f"storev(Y + {m} * N, (g + {g}) * {L}, r0, r1, a{g}_{m});"
        for g in range(G)
        for m in range(M)
    )

    if target == "avx512_vnni":
        storev = """
static inline void storev(float *y, int64_t row0, int64_t r0, int64_t r1, __m512 v) {
    if (row0 >= r0 && row0 + 16 <= r1) { _mm512_storeu_ps(y + row0, v); return; }
    uint32_t m = 0xFFFF;
    if (row0 < r0) m &= 0xFFFFu << (r0 - row0);
    if (row0 + 16 > r1) m &= (1u << (r1 > row0 ? r1 - row0 : 0)) - 1;
    _mm512_mask_storeu_ps(y + row0, (__mmask16)m, v);
}"""
    else:
        storev = """
static inline void storev(float *y, int64_t row0, int64_t r0, int64_t r1, __m256 v) {
    float t[8];
    _mm256_storeu_ps(t, v);
    for (int i = 0; i < 8; i++) if (row0 + i >= r0 && row0 + i < r1) y[row0 + i] = t[i];
}
static inline __m256i bits8x4(uint32_t w) {  /* 32 bits -> 32 bytes of 0/1 */
    const __m256i sh = _mm256_setr_epi8(0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1, 2,2,2,2,2,2,2,2, 3,3,3,3,3,3,3,3);
    const __m256i bm = _mm256_set1_epi64x((long long)0x8040201008040201ull);
    __m256i v = _mm256_shuffle_epi8(_mm256_set1_epi32((int)w), sh);
    return _mm256_and_si256(_mm256_cmpeq_epi8(_mm256_and_si256(v, bm), bm), _mm256_set1_epi8(1));
}"""
    q4k_packed = ""
    if two and c["scales"] == "packed":
        q4k_packed = f"""
/* 6-bit scale/min decode for {L} rows at once; q[t] holds raw scale byte t of every row */
static inline {V} q4k_sc(const {V} *q, int s) {{
    const {V} m63 = {P["set1_32"](63)}, m15 = {P["set1_32"](15)};
    if (s < 4) return {P["and_"]("q[s]", "m63")};
    return {P["add"](P["and_"]("q[s + 4]", "m15"), P["mullo"](P["srli16"]("q[s - 4]", 6), P["set1_32"](16)))};
}}
static inline {V} q4k_mn(const {V} *q, int s) {{
    const {V} m63 = {P["set1_32"](63)};
    if (s < 4) return {P["and_"]("q[s + 4]", "m63")};
    return {P["add"](P["srli16"]("q[s + 4]", 4), P["mullo"](P["srli16"]("q[s]", 6), P["set1_32"](16)))};
}}"""

    lut_decl = ("const " + V + " lut = " + P["tbl"]("LUT") + ";") if lut_tbl else ""
    sig = (
        f"void {entry}_packed(const void *pv, const void *X, float *Y, int64_t K, int64_t N, int64_t M, int64_t r0, int64_t r1)"
        if gemm
        else f"void {entry}_packed(const void *pv, const void *X, float *Y, int64_t K, int64_t r0, int64_t r1)"
    )
    mdecl = "" if gemm else "    const int64_t M = 1, N = 0;\n"
    fallback = (
        (
            f"void {entry}(const void *W, const void *X, float *Y, int64_t K, int64_t N, int64_t M, int64_t n0, int64_t n1) {{\n"
            "    (void)W; (void)X; (void)Y; (void)K; (void)N; (void)M; (void)n0; (void)n1; abort();\n}\n"
        )
        if gemm
        else ""
    )
    tail = f"""    const {V} m4 = {P["set1_8"]("0x0F")}, m3 = {P["set1_8"]("0x03")};
    (void)m4; (void)m3; (void)sx;
    {lut_decl}
    const int64_t g0 = r0 / {L}, g1 = (r1 + {L - 1}) / {L};
    int64_t g = g0;
    for (; g < g1; g += {G}) {{
        {decl}
        for (int64_t p = 0; p < pk->nrec_k; p++) {{
{inner}
        }}
{stores}
    }}
}}
"""
    xarr = [("int32_t", "xw", "nk * 8"), ("int32_t", "sx", "nk"), ("float", "dx", "nk")]
    xfuncs = (
        _xprep_funcs(
            c, entry, xarr, xprep_lines, xblk, xrow, M, gemm, mdecl, tail, "    const packed_t *pk = (const packed_t *)pv;\n", need_xp=False
        )
        if c.get("xprep")
        else ""
    )
    body = f"""
{act_struct}
{storev}
{q4k_packed}
{sig} {{
    const packed_t *pk = (const packed_t *)pv;
{mdecl}    const int64_t nk = K / 32, nbx = {nbx};
    if (M > {M}) abort();
    int32_t xw[{M} * 1024 * 8], sx[{M} * 1024];
    float dx[{M} * 1024];
    (void)sx; (void)nbx;
    for (int64_t m = 0; m < {M}; m++) {{{act_prep}
    }}{act_bias}
    const {V} m4 = {P["set1_8"]("0x0F")}, m3 = {P["set1_8"]("0x03")};
    (void)m4; (void)m3;
    {lut_decl}
    const int64_t g0 = r0 / {L}, g1 = (r1 + {L - 1}) / {L};
    int64_t g = g0;
    for (; g < g1; g += {G}) {{
        {decl}
        for (int64_t p = 0; p < pk->nrec_k; p++) {{
{inner}
        }}
{stores}
    }}
}}
{fallback}{xfuncs}"""
    # tail groups beyond g1 are guarded by storev masks; records past ngroups are never read
    # because the loop over g advances by G only while groups exist: pad ngroups in prepare.
    prepare = prepare.replace("pk->ngroups = (N + " + str(L - 1) + ") / " + str(L) + ";", f"pk->ngroups = (N + {L - 1}) / {L} + {G - 1};")
    return _prelude(target) + prepare + body


def _prelude(target):
    arch = "#include <immintrin.h>"
    f16 = "static inline float f16f(uint16_t h) { return _cvtsh_ss(h); }"
    return f"""// Generated by kurn (generic recipe lowering). Do not edit; edit the .kurn spec instead.
#include "kurn.h"
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
{arch}
{f16}
"""


def _scalar(c):
    """Native-layout scalar kernel straight from the recipe (the semantic reference in C)."""
    r = RECIPES[c["weights"]]
    entry = c["entry"]
    kv = "static const int8_t KV[16] = {" + ", ".join(str(v) for v in (r.lut or KV_IQ4NL)) + "};" if "KV[" in r.signed_c else ""
    if r.act == "q8_0":
        act = "typedef struct { uint16_t d; int8_t qs[32]; } xblock;"
        xv = "x[v / 32].qs[v % 32]"
        dxe = "f16f(x[k].d)"
    else:
        act = "typedef struct { float d; int8_t qs[256]; int16_t bsums[16]; } xblock;"
        xv = "x[v / 256].qs[v % 256]"
        dxe = "x[k / 8].d"
    if r.two_level:
        inner = f"""
        for (int64_t sb = 0; sb < K / 256; sb++) {{
            const nblock *b = wr + sb;
            const uint8_t *q6 = b->scales;
            int64_t isum = 0, msum = 0;
            for (int s = 0; s < 8; s++) {{
                const int sc = s < 4 ? q6[s] & 63 : (q6[s + 4] & 0xF) | ((q6[s - 4] >> 6) << 4);
                const int mn = s < 4 ? q6[s + 4] & 63 : (q6[s + 4] >> 4) | ((q6[s] >> 6) << 4);
                int32_t t = 0, xs = 0;
                for (int l = 0; l < 32; l++) {{
                    const int v = s * 32 + l;
                    const int q = {r.code_c};
                    const int64_t vv = sb * 256 + v;
                    t += q * x[vv / 256].qs[vv % 256];
                    xs += x[vv / 256].qs[vv % 256];
                }}
                isum += (int64_t)sc * t;
                msum += (int64_t)mn * xs;
            }}
            acc += x[sb].d * (f16f(b->d) * (float)isum - f16f(b->dmin) * (float)msum);
        }}"""
    elif r.scale != "f16":
        sub = r.sub
        inner = f"""
        for (int64_t k = 0; k < K / 32; k++)
            for (int h = 0; h < {32 // sub}; h++) {{
                const nblock *b = wr + (k * 32 + h * {sub}) / {r.block};
                const int v0 = (int)((k * 32 + h * {sub}) % {r.block});
                int32_t t = 0;
                for (int l = 0; l < {sub}; l++) {{
                    const int64_t vv = k * 32 + h * {sub} + l;
                    const int v = v0 + l;
                    const int q = {r.code_c};
                    t += ({r.signed_c}) * {xv.replace("v / ", "vv / ").replace("v % ", "vv % ")};
                }}
                {{ const int v __attribute__((unused)) = v0; acc += {r.dsc_c} * {dxe} * (float)t; }}
            }}"""
    else:
        inner = f"""
        for (int64_t k = 0; k < K / 32; k++) {{
            const nblock *b = wr + (k * 32) / {r.block};
            int32_t t = 0;
            for (int l = 0; l < 32; l++) {{
                const int64_t vv = k * 32 + l;
                const int v = (int)(vv % {r.block});
                const int q = {r.code_c};
                t += ({r.signed_c}) * {xv.replace("v / ", "vv / ").replace("v % ", "vv % ")};
            }}
            acc += f16f({r.d_c}) * {dxe} * (float)t;
        }}"""
    return f"""// Generated by kurn (generic recipe lowering, scalar reference). Do not edit.
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
{_MX_SCALAR_C if r.scale != "f16" else ""}typedef struct {{ {r.struct} }} nblock;
{act}
{kv}
void {entry}(const void *W, const void *X, float *Y, int64_t K, int64_t r0, int64_t r1) {{
    const xblock *x = (const xblock *)X;
    for (int64_t n = r0; n < r1; n++) {{
        const nblock *wr = (const nblock *)W + n * (K / {r.block});
        float acc = 0;{inner}
        Y[n] = acc;
    }}
}}
"""


# --------------------------------------------------------------------------- T-MAC-style LUT lowering (<= 2-bit)
LUT_GROUP = {1: 4, 2: 2}  # values per 4-bit table index


def lower_lut(target, c):
    """Lookup-table GEMV for 1- and 2-bit recipes (layout l32, AVX-512 only).

    Per call, every group of g activations (g = 4 for 1-bit, 2 for 2-bit) gets a
    16-entry int16 table of exact partial sums  T[i] = sum_j value(i, j) * x_j.
    Weights are stored as 4-bit table indices, 32 rows per vector (int16 lanes),
    so one vpermw returns the partial dot products of 32 rows with no multiplies.
    Tables are int16 and sums stay exact (unlike T-MAC's int8 tables).
    """
    r = RECIPES[c["weights"]]
    if target != "avx512_vnni" or r.bits > 2:
        raise ValueError("lut lowering: AVX-512 and <= 2-bit recipes only")
    g = LUT_GROUP[r.bits]
    G, PF, entry = c["rows"], c["prefetch"], c["entry"]
    int_period = c["accum"] == "int"
    groups = r.period // 32
    chunks = 32 // g  # table lookups per K-group
    idx_bytes = 32 * chunks // 2  # 32 rows x chunks nibbles
    hdr = 64  # d f16 x 32 rows
    rec = hdr + groups * idx_bytes
    # signed value of raw code q (see recipe): used to build tables
    if r.bits == 1:
        tval = "((i >> j) & 1 ? 1 : -1)"
    else:
        tval = "(((i >> (2 * j)) & 3) - 1)"
    if r.act == "q8_0":
        act_struct = "typedef struct { uint16_t d; int8_t qs[32]; } xblock;"
        xq, dxe, nbx = "xb[(v) / 32].qs[(v) % 32]", "f16f(xb[k].d)", "K / 32"
    else:
        act_struct = "typedef struct { float d; int8_t qs[256]; int16_t bsums[16]; } xblock;"
        xq, dxe, nbx = "xb[(v) / 256].qs[(v) % 256]", "xb[k / 8].d", "K / 256"
    lines = []
    ind = "                "
    for gg in range(G):
        lines.append(f"{ind}const uint8_t *rec{gg} = pk->buf + ((size_t)(g + {gg}) * pk->nrec_k + p) * REC_BYTES;")
        if PF:
            lines.append(f"{ind}_mm_prefetch((const char *)(rec{gg} + {PF} * REC_BYTES), _MM_HINT_T0);")
        lines.append(
            f"{ind}const __m512 dlo{gg} = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)rec{gg})), "
            f"dhi{gg} = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(rec{gg} + 32)));"
        )
        if int_period:
            lines.append(f"{ind}__m512i ilo{gg} = _mm512_setzero_si512(), ihi{gg} = _mm512_setzero_si512();")
    lines.append(f"{ind}for (int kg = 0; kg < {groups}; kg++) {{")
    lines.append(f"{ind}    const int64_t k = p * {groups} + kg;")
    lines.append(f"{ind}    const __m512i *T = tabs + k * {chunks};")
    for gg in range(G):
        lines.append(f"{ind}    __m512i s{gg} = _mm512_setzero_si512();")
    for pp in range(chunks // 2):
        for gg in range(G):
            src = f"(const __m256i *)(rec{gg} + HDR_BYTES + kg * IDX_BYTES + {32 * pp})"
            lo = f"_mm512_permutexvar_epi16(_mm512_and_si512(w, m15), T[{2 * pp}])"
            hi = f"_mm512_permutexvar_epi16(_mm512_srli_epi16(w, 4), T[{2 * pp + 1}])"
            lines.append(f"{ind}    {{ const __m512i w = _mm512_cvtepu8_epi16(_mm256_loadu_si256({src}));")
            lines.append(f"{ind}      s{gg} = _mm512_add_epi16(s{gg}, _mm512_add_epi16({lo},")
            lines.append(f"{ind}                                                  {hi})); }}")
    for gg in range(G):
        lo = f"_mm512_cvtepi16_epi32(_mm512_castsi512_si256(s{gg}))"
        hi = f"_mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(s{gg}, 1))"
        if int_period:
            lines.append(f"{ind}    ilo{gg} = _mm512_add_epi32(ilo{gg}, {lo}); ihi{gg} = _mm512_add_epi32(ihi{gg}, {hi});")
        else:
            lines.append(f"{ind}    {{ const __m512 dx = _mm512_set1_ps(dxs[k]);")
            lines.append(f"{ind}      alo{gg} = _mm512_fmadd_ps(_mm512_cvtepi32_ps({lo}), _mm512_mul_ps(dlo{gg}, dx), alo{gg});")
            lines.append(f"{ind}      ahi{gg} = _mm512_fmadd_ps(_mm512_cvtepi32_ps({hi}), _mm512_mul_ps(dhi{gg}, dx), ahi{gg}); }}")
    lines.append(f"{ind}}}")
    if int_period:
        for gg in range(G):
            lines.append(f"{ind}{{ const __m512 dx = _mm512_set1_ps(dxs[p * {groups}]);")
            lines.append(f"{ind}  alo{gg} = _mm512_fmadd_ps(_mm512_cvtepi32_ps(ilo{gg}), _mm512_mul_ps(dlo{gg}, dx), alo{gg});")
            lines.append(f"{ind}  ahi{gg} = _mm512_fmadd_ps(_mm512_cvtepi32_ps(ihi{gg}), _mm512_mul_ps(dhi{gg}, dx), ahi{gg}); }}")
    decl = " ".join(f"__m512 alo{gg} = _mm512_setzero_ps(), ahi{gg} = _mm512_setzero_ps();" for gg in range(G))
    stores = "\n".join(
        f"        storev(Y, (g + {gg}) * 32, r0, r1, alo{gg}); storev(Y, (g + {gg}) * 32 + 16, r0, r1, ahi{gg});" for gg in range(G)
    )
    return (
        _prelude(target)
        + f"""
typedef struct {{ {r.struct} }} nblock;
{act_struct}
typedef struct {{ int64_t nrec_k, ngroups; uint8_t *buf; }} packed_t;
#define REC_BYTES {rec}
#define HDR_BYTES {hdr}
#define IDX_BYTES {idx_bytes}

/* repack: per (32-row group, scale period): d[32] (fp16), then per K-group
   {chunks} 4-bit table indices per row; byte (pair * 32 + row) holds chunk 2*pair
   in the low nibble and chunk 2*pair + 1 in the high nibble */
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
                for (int kg = 0; kg < {groups}; kg++)
                    for (int ch = 0; ch < {chunks}; ch++) {{
                        int idx = 0;
                        for (int j = 0; j < {g}; j++) {{
                            const int v = (int)((v0 + kg * 32 + ch * {g} + j) % {r.block});
                            const int q = {r.code_c};
                            idx |= q << ({r.bits} * j);
                        }}
                        hp[HDR_BYTES + kg * IDX_BYTES + (ch / 2) * 32 + row] |= (uint8_t)(idx << (4 * (ch & 1)));
                    }}
            }}
        }}
    return pk;
}}

static inline void storev(float *y, int64_t row0, int64_t r0, int64_t r1, __m512 v) {{
    if (row0 >= r0 && row0 + 16 <= r1) {{ _mm512_storeu_ps(y + row0, v); return; }}
    uint32_t m = 0xFFFF;
    if (row0 < r0) m &= r0 - row0 >= 16 ? 0 : 0xFFFFu << (r0 - row0);
    if (row0 + 16 > r1) m &= (1u << (r1 > row0 ? (r1 - row0 > 16 ? 16 : r1 - row0) : 0)) - 1;
    _mm512_mask_storeu_ps(y + row0, (__mmask16)m, v);
}}

void {entry}_packed(const void *pv, const void *X, float *Y, int64_t K, int64_t r0, int64_t r1) {{
    const packed_t *pk = (const packed_t *)pv;
    const xblock *xb = (const xblock *)X;
    const int64_t nk = K / 32, nbx = {nbx};
    (void)nbx;
    /* per-call tables: one 16-entry int16 table per {g} activations (exact partial sums) */
    static __thread __m512i *tabs;
    static __thread int64_t tcap;
    if (tcap < nk * {chunks}) {{ free(tabs); tabs = aligned_alloc(64, sizeof(__m512i) * nk * {chunks}); tcap = nk * {chunks}; }}
    float dxs[1024];
    for (int64_t k = 0; k < nk; k++) {{
        dxs[k] = {dxe};
        for (int ch = 0; ch < {chunks}; ch++) {{
            int16_t t[32] = {{0}};
            for (int i = 0; i < 16; i++) {{
                int s = 0;
                for (int j = 0; j < {g}; j++) {{ const int64_t v = k * 32 + ch * {g} + j; s += {tval} * {xq}; }}
                t[i] = (int16_t)s;
            }}
            tabs[k * {chunks} + ch] = _mm512_loadu_si512((const void *)t);
        }}
    }}
    const __m512i m15 = _mm512_set1_epi16(15);
    const int64_t g0 = r0 / 32, g1 = (r1 + 31) / 32;
    for (int64_t g = g0; g < g1; g += {G}) {{
        {decl}
        for (int64_t p = 0; p < pk->nrec_k; p++) {{
{chr(10).join(lines)}
        }}
{stores}
    }}
}}
"""
    )


# --------------------------------------------------------------------------- 4-bit lowering (q4_vnni16 family)
_MX_SCALAR_C = """static inline float bitsf(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }
static inline float e8m0h(uint8_t x) {  /* ggml_e8m0_to_fp32_half */
    return bitsf(x < 2 ? 0x00200000u << x : (uint32_t)(x - 1) << 23);
}
static inline float ue4m3h(uint8_t x) {  /* ggml_ue4m3_to_fp32: UE4M3 scale x 0.5 (the codes are E2M1 x 2) */
    if (x == 0 || x == 0x7F) return 0.0f;
    const int e = (x >> 3) & 15, m = x & 7;
    return e ? bitsf((uint32_t)(e - 8 + 127) << 23) * (1.0f + (float)m / 8.0f) : (float)m * (1.0f / 1024.0f);
}
"""


def _nibble_prims(target):
    P = dict(_prims(target))
    if target == "avx512_vnni":
        P.update(
            srai=lambda a, n: f"_mm512_srai_epi32({a}, {n})", slli=lambda a, n: f"_mm512_slli_epi32({a}, {n})",
            sub=lambda a, b: f"_mm512_sub_epi32({a}, {b})", castps=lambda a: f"_mm512_castsi512_ps({a})",
            blend16=lambda a, b: f"_mm512_mask_blend_epi16(0xAAAAAAAAu, {a}, {b})",
            dpw=lambda acc, a, b: f"_mm512_dpwssd_epi32({acc}, {a}, {b})",
            permb=lambda i, t: f"PERMB({i}, {t})",
            scdup=lambda p: f"_mm512_shuffle_epi8(_mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i *)({p}))), scidx)",
            vload=lambda name: f"_mm512_loadu_si512((const void *){name})",
        )  # fmt: skip
    else:
        P.update(
            srai=lambda a, n: f"_mm256_srai_epi32({a}, {n})", slli=lambda a, n: f"_mm256_slli_epi32({a}, {n})",
            sub=lambda a, b: f"_mm256_sub_epi32({a}, {b})", castps=lambda a: f"_mm256_castsi256_ps({a})",
            blend16=lambda a, b: f"_mm256_blend_epi16({a}, {b}, 0xAA)",
            dpw=lambda acc, a, b: f"_mm256_dpwssd_avx_epi32({acc}, {a}, {b})",
            scdup=lambda p: f"_mm256_shuffle_epi8(_mm256_set1_epi64x((long long)ld64({p})), scidx)",
            vload=lambda name: f"_mm256_loadu_si256((const __m256i *){name})",
        )  # fmt: skip
    return P


def nibble_record(r, c, L):
    """(rows per record, header bytes, code bytes per K-group, K-groups per record, record bytes)."""
    pair = c["unpack"] == "pair"
    R = (2 if pair else 1) * L
    groups = r.period // 32 if r.two_level else 1
    if r.two_level:
        hdr = 20 * R  # d, dmin (f16) + sc[8], mn[8] (u8) per row
    elif r.scale == "f16":
        hdr = 2 * R
    elif r.scale == "e8m0":
        hdr = R
    else:
        hdr = (32 // r.sub) * R * (1 if c["scales"] == "packed" else 2)
    kgb = (8 if pair else 4) * 4 * L
    return R, hdr, kgb, groups, hdr + groups * kgb


def lower_nibble(target, c):
    """4-bit GEMV / verify on row-interleaved records (layout i16 / i8), CPU-integer-shaped.

    Record = R rows x (32 values, or a 256-value Q4_K super-block): per-row scales, then per
    K-group the codes. Non-pair: vector t holds, per lane (row), values 4*(2t)+j in the low
    and 4*(2t+1)+j in the high nibble. Pair: vector t holds values 4t+j of row `lane` in the
    low and of row `lane + L` in the high nibble, so the raw byte is lo + 16 * hi.
    The activation correction beta * sum(x) is the accumulator seed, so the codes are used
    as stored (pre-biased); Q4_K scales multiply integer sums (vpdpwssd or vpmulld) and the
    super-block is accumulated in integers.
    """
    r = RECIPES[c["weights"]]
    P = _nibble_prims(target)
    L, V, F = P["L"], P["V"], P["F"]
    G, M, PF, entry = c["rows"], c["cols"], c["prefetch"], c["entry"]
    unpack, corr = c["unpack"], c["correction"]
    IL = c.get("ilv", 1)  # records of IL consecutive row groups interleaved per K step (one stream per pass)
    gemm = M > 1
    pair = unpack == "pair"
    two = r.two_level
    lutmode = unpack in ("lut", "perm")
    if pair and lutmode:
        raise ValueError("unpack=pair needs linear codes")
    H = 2 if pair else 1
    S = 32 // r.sub  # scales per row per K-group (NVFP4: 2)
    R, hdr, kgb, groups, rec = nibble_record(r, c, L)
    rec_vals = 32 * groups
    vb = 4 * L
    nvec = 8 if pair else 4
    beta = -128 if lutmode else r.maps["mask"][2]
    pk_scales = c["scales"] == "packed"
    sc_off, mn_off = 4 * R, 12 * R
    dxmul = {"f16": "1.0f", "e8m0": "0.5f", "ue4m3": "1.0f"}[r.scale]
    xrow = "nk * 34" if r.act == "q8_0" else "(K / 256) * 292"

    def rec_addr(gg, p):
        if IL == 1:
            return f"((size_t){gg if gg.isidentifier() else f'({gg})'} * pk->nrec_k + {p})"
        return f"((((size_t)({gg}) / {IL}) * pk->nrec_k + {p}) * {IL} + ({gg}) % {IL})"

    ngroups = f"(N + {R - 1}) / {R} + {G - 1}" if IL == 1 else f"((N + {R - 1}) / {R} + {G - 1} + {IL - 1}) / {IL} * {IL}"

    # ---------------- repack ------------------------------------------------------------
    if pair:
        pack = f"rec_codes[kk * {vb} + (row % {L}) * 4 + j] |= (uint8_t)(q << (4 * (row / {L})));"
    else:
        pack = f"rec_codes[(kk / 2) * {vb} + row * 4 + j] |= (uint8_t)(q << (4 * (kk & 1)));"
    helpers = ""
    if two:
        mn_idx = f"{mn_off} + (s / 4) * {4 * R} + row * 4 + s % 4" if corr == "dpmin" else f"{mn_off} + s * {R} + row"
        hdr_fill = f"""memcpy(hp + 2 * row, &b->d, 2); memcpy(hp + {2 * R} + 2 * row, &b->dmin, 2);
                for (int s = 0; s < 8; s++) {{
                    const uint8_t *q6 = b->scales;
                    hp[{sc_off} + s * {R} + row] = s < 4 ? q6[s] & 63 : (q6[s + 4] & 0xF) | ((q6[s - 4] >> 6) << 4);
                    hp[{mn_idx}] = s < 4 ? q6[s + 4] & 63 : (q6[s + 4] >> 4) | ((q6[s] >> 6) << 4);
                }}"""
    elif r.scale == "f16":
        hdr_fill = "memcpy(hp + 2 * row, &b->d, 2);"
    elif r.scale == "e8m0":
        hdr_fill = "hp[row] = b->e;"
    else:
        if pk_scales:
            put = f"hp[s * {R} + row] = u;"
            # (u << 20) + (119 << 23) is 2^(e-8) * (1 + m/8), always normal; e = 0 lanes become 2v - 2^-7 = m * 2^-10.
            # A plain (u << 20) * 2^119 is also exact but makes subnormal UE4M3 codes denormal floats (microcode assists).
            if target == "avx512_vnni":
                helpers = """static inline __m512 ue4m3_ps(__m512i u) {  /* ggml_ue4m3_to_fp32 (incl. x0.5), exact */
    const __m512 v = _mm512_castsi512_ps(_mm512_add_epi32(_mm512_slli_epi32(u, 20), _mm512_set1_epi32(119 << 23)));
    return _mm512_mask_fmsub_ps(v, _mm512_testn_epi32_mask(u, _mm512_set1_epi32(0x78)), _mm512_set1_ps(2.0f),
                                _mm512_set1_ps(0x1p-7f));
}
"""
            else:
                helpers = """static inline __m256 ue4m3_ps(__m256i u) {  /* ggml_ue4m3_to_fp32 (incl. x0.5), exact */
    const __m256 v = _mm256_castsi256_ps(_mm256_add_epi32(_mm256_slli_epi32(u, 20), _mm256_set1_epi32(119 << 23)));
    const __m256 sub = _mm256_fmsub_ps(v, _mm256_set1_ps(2.0f), _mm256_set1_ps(0x1p-7f));
    const __m256i e0 = _mm256_cmpeq_epi32(_mm256_and_si256(u, _mm256_set1_epi32(0x78)), _mm256_setzero_si256());
    return _mm256_blendv_ps(v, sub, _mm256_castsi256_ps(e0));
}
"""
        else:
            put = f"{{ const uint16_t h = ue4m3_f16(u); memcpy(hp + 2 * (s * {R} + row), &h, 2); }}"
            helpers = """static uint16_t ue4m3_f16(uint8_t x) {  /* ggml_ue4m3_to_fp32 (incl. x0.5) as an exact fp16 */
    if (x == 0) return 0;
    const int e = (x >> 3) & 15, m = x & 7;
    if (e) return (uint16_t)(((e + 7) << 10) | (m << 7));
    return _cvtss_sh((float)m * (1.0f / 1024.0f), 0);
}
"""
        hdr_fill = f"""for (int s = 0; s < {S}; s++) {{
                    uint8_t u = b->d[(v0 % {r.block}) / {r.sub} + s];
                    if (u == 0x7F) u = 0;  /* NaN encoding: ggml decodes it as 0 */
                    {put}
                }}"""
    prepare = f"""
typedef struct {{ {r.struct} }} nblock;
typedef struct {{ int64_t nrec_k, ngroups; uint8_t *buf; }} packed_t;
#define REC_BYTES {rec}
#define HDR_BYTES {hdr}
#define KG_BYTES {kgb}
{helpers}
/* repack: records of {R} rows x {rec_vals} values; see lower_nibble in kurn/generic.py for the layout */
void *{entry}_prepare(const void *W, int64_t K, int64_t N) {{
    const nblock *w = (const nblock *)W;
    const int64_t nb = K / {r.block};
    packed_t *pk = malloc(sizeof *pk);
    pk->nrec_k = K / {rec_vals}; pk->ngroups = {ngroups};
    const size_t bytes = (size_t)REC_BYTES * pk->nrec_k * pk->ngroups;
    pk->buf = aligned_alloc(64, (bytes + 63) & ~(size_t)63);
    memset(pk->buf, 0, bytes);
    for (int64_t g = 0; g < pk->ngroups; g++)
        for (int64_t p = 0; p < pk->nrec_k; p++) {{
            uint8_t *hp = pk->buf + {rec_addr("g", "p")} * REC_BYTES;
            for (int row = 0; row < {R}; row++) {{
                const int64_t n = g * {R} + row;
                if (n >= N) continue;  /* padding rows stay zero: scale 0 */
                const int64_t v0 = p * {rec_vals};
                const nblock *b = w + n * nb + v0 / {r.block};
                {hdr_fill}
                for (int kg = 0; kg < {groups}; kg++) {{
                    uint8_t *rec_codes = hp + HDR_BYTES + kg * KG_BYTES;
                    for (int kk = 0; kk < 8; kk++)
                        for (int j = 0; j < 4; j++) {{
                            const int v = (int)((v0 + kg * 32 + kk * 4 + j) % {r.block});
                            const int q = {r.code_c};
                            {pack}
                        }}
                }}
            }}
        }}
    return pk;
}}
"""

    # ---------------- kernel --------------------------------------------------------------
    ind = "                "
    lines = []
    A = lines.append

    def xb(m, kk):
        return P["set1_32"](f"ld32(xq{m} + {4 * kk})")

    pf_line = PF and c.get("pfgran", "rec") == "line"
    pf_hint = PF_HINTS[c.get("pfhint", "t0")]
    for g in range(G):
        A(f"{ind}const uint8_t *rec{g} = pk->buf + {rec_addr(f'g + {g}', 'p')} * REC_BYTES;")
        if PF:
            offs = [f" + {o}" for o in range(0, hdr if groups > 1 else rec, 64)] if pf_line else [""]
            for off in offs:
                A(f"{ind}_mm_prefetch((const char *)(rec{g} + {PF * IL} * REC_BYTES{off}), {pf_hint});")
    # per-record scale vectors (non-two-level formats: one record = one K-group)
    if not two:
        for g in range(G):
            for h in range(H):
                for s in range(S):
                    if r.scale == "f16" or (r.scale == "ue4m3" and not pk_scales):
                        off = 2 * h * L if r.scale == "f16" else 2 * (s * R + h * L)
                        A(f"{ind}const {F} ds{g}_{h}_{s} = {P['h2f'](f'rec{g} + {off}')};")
                    elif r.scale == "e8m0":
                        A(f"{ind}const {F} ds{g}_{h}_{s} = {P['castps'](P['slli'](P['u8x'](f'rec{g} + {h * L}'), 23))};")
                    else:
                        A(f"{ind}const {F} ds{g}_{h}_{s} = ue4m3_ps({P['u8x'](f'rec{g} + {s * R + h * L}')});")
    if two:
        for g in range(G):
            for m in range(M):
                for h in range(H):
                    A(f"{ind}{V} ia{g}_{m}_{h} = {P['zero']}, ma{g}_{m}_{h} = {P['zero']};")
    if groups > 1:
        A(f"{ind}for (int kg = 0; kg < {groups}; kg++) {{")
        ind2 = ind + "    "
        A(f"{ind2}const int64_t k = p * {groups} + kg;")
    else:
        ind2 = ind
        A(f"{ind2}const int64_t k = p;")
    if pf_line and groups > 1:
        for g in range(G):
            for off in range(0, kgb, 64):
                A(f"{ind2}_mm_prefetch((const char *)(rec{g} + {PF * IL} * REC_BYTES + HDR_BYTES + kg * KG_BYTES + {off}), {pf_hint});")
    for m in range(M):
        xq = f"xp[{m}] + 34 * k + 2" if r.act == "q8_0" else f"xp[{m}] + 292 * (k / 8) + 4 + 32 * (k % 8)"
        A(f"{ind2}const uint8_t *xq{m} = {xq};")
    for g in range(G):
        for t in range(nvec):
            A(
                f"{ind2}const {V} c{g}_{t} = {P['loadu'](f'rec{g} + HDR_BYTES + kg * KG_BYTES + {t * vb}')};"
                if groups > 1
                else f"{ind2}const {V} c{g}_{t} = {P['loadu'](f'rec{g} + HDR_BYTES + {t * vb}')};"
            )
    # accumulator seeds
    for g in range(G):
        for m in range(M):
            if pair:
                sa = P["set1_32"](f"sdA[{m} * nk + k]") if beta else P["zero"]
                sb = P["set1_32"](f"sdB[{m} * nk + k]") if beta else P["zero"]
                A(f"{ind2}{V} A{g}_{m}_0 = {sa}, A{g}_{m}_1 = {P['zero']}, B{g}_{m}_0 = {sb}, B{g}_{m}_1 = {P['zero']};")
            else:
                for s in range(S):
                    se = P["set1_32"](f"sd0[({m} * nk + k) * {S} + {s}]") if beta else P["zero"]
                    A(f"{ind2}{V} e{g}_{m}_{s} = {se}, o{g}_{m}_{s} = {P['zero']};")
    # dot products
    split = G * M <= PAIR_SPLIT_MAX
    for t in range(nvec):
        for g in range(G):
            cv = f"c{g}_{t}"
            if pair:
                ch = t & 1 if split else 0
                A(f"{ind2}{{ const {V} hv = {P['and_'](cv, 'mF0')};")
                for m in range(M):
                    A(f"{ind2}  A{g}_{m}_{ch} = {P['dp'](f'A{g}_{m}_{ch}', cv, xb(m, t))};")
                    A(
                        f"{ind2}  B{g}_{m}_{ch} = {P['dp'](f'B{g}_{m}_{ch}', 'hv', xb(m, t))}; }}"
                        if m == M - 1
                        else f"{ind2}  B{g}_{m}_{ch} = {P['dp'](f'B{g}_{m}_{ch}', 'hv', xb(m, t))};"
                    )
                continue
            s = t // 2 if S == 2 else 0
            if unpack == "mask16":
                lo, hi = P["and_"](cv, "m4"), P["and_"](cv, "mF0")
            elif unpack == "lut":
                lo = P["shuf"]("lut", P["and_"](cv, "m4"))
                hi = P["shuf"]("lut", P["and_"](P["srli16"](cv, 4), "m4"))
            else:  # perm
                lo, hi = P["permb"](cv, "lut"), P["permb"](P["srli16"](cv, 4), "lut")
            A(f"{ind2}{{ const {V} lo = {lo}, hi = {hi};")
            for m in range(M):
                A(f"{ind2}  e{g}_{m}_{s} = {P['dp'](f'e{g}_{m}_{s}', 'lo', xb(m, 2 * t))};")
                A(f"{ind2}  o{g}_{m}_{s} = {P['dp'](f'o{g}_{m}_{s}', 'hi', xb(m, 2 * t + 1))};")
            A(f"{ind2}}}")
    # combine, scale
    for g in range(G):
        for m in range(M):
            dxv = P["fset1"](f"dx[{m} * nk + k]")
            if pair:
                A(f"{ind2}{{ const {V} sa = {P['add'](f'A{g}_{m}_0', f'A{g}_{m}_1')}, sb = {P['add'](f'B{g}_{m}_0', f'B{g}_{m}_1')};")
                isums = [P["sub"]("sa", "sb"), P["srai"]("sb", 4)]
                if two:
                    for h in range(H):
                        scv = P["u8x"](f"rec{g} + {sc_off} + kg * {R} + {h * L}")
                        A(f"{ind2}  ia{g}_{m}_{h} = {P['add'](f'ia{g}_{m}_{h}', P['mullo'](isums[h], scv))};")
                else:
                    for h in range(H):
                        A(f"{ind2}  a{g}_{m}_{h} = {P['fma'](P['cvt'](isums[h]), P['mul'](f'ds{g}_{h}_0', dxv), f'a{g}_{m}_{h}')};")
                A(f"{ind2}}}")
            elif two:  # mask16: e and o >> 4 are 16-value sums (|.| <= 30480) -> int16 pairs for vpdpwssd
                comb = P["blend16"](f"e{g}_{m}_0", P["slli"](f"o{g}_{m}_0", 12))
                A(f"{ind2}ia{g}_{m}_0 = {P['dpw'](f'ia{g}_{m}_0', comb, P['scdup'](f'rec{g} + {sc_off} + kg * {R}'))};")
            else:
                for s in range(S):
                    hi = P["srai"](f"o{g}_{m}_{s}", 4) if unpack == "mask16" else f"o{g}_{m}_{s}"
                    tot = P["cvt"](P["add"](f"e{g}_{m}_{s}", hi))
                    A(f"{ind2}a{g}_{m}_0 = {P['fma'](tot, P['mul'](f'ds{g}_0_{s}', dxv), f'a{g}_{m}_0')};")
            if two and corr == "act":
                for h in range(H):
                    mnv = P["u8x"](f"rec{g} + {mn_off} + kg * {R} + {h * L}")
                    A(f"{ind2}ma{g}_{m}_{h} = {P['add'](f'ma{g}_{m}_{h}', P['mullo'](mnv, P['set1_32'](f'sxk[{m} * nk + k]')))};")
    if groups > 1:
        A(f"{ind}}}")
    if two:
        for g in range(G):
            for m in range(M):
                for h in range(H):
                    if corr == "dpmin":
                        m0 = P["loadu"](f"rec{g} + {mn_off} + {h * L * 4}")
                        m1 = P["loadu"](f"rec{g} + {mn_off + 4 * R} + {h * L * 4}")
                        w = [P["set1_32"](f"mw[({m} * pk->nrec_k + p) * 4 + {i}]") for i in range(4)]
                        A(f"{ind}{{ const {V} m0 = {m0}, m1 = {m1};")
                        A(f"{ind}  const {V} lo7 = {P['dp'](P['dp'](P['zero'], 'm0', w[0]), 'm1', w[1])};")
                        A(f"{ind}  const {V} hi7 = {P['dp'](P['dp'](P['zero'], 'm0', w[2]), 'm1', w[3])};")
                        A(f"{ind}  ma{g}_{m}_{h} = {P['add']('lo7', P['slli']('hi7', 7))}; }}")
                    dxv = P["fset1"](f"dx[{m} * nk + p * 8]")
                    ndxv = P["fset1"](f"-dx[{m} * nk + p * 8]")
                    dv = P["mul"](P["h2f"](f"rec{g} + {2 * h * L}"), dxv)
                    mv = P["mul"](P["h2f"](f"rec{g} + {2 * R + 2 * h * L}"), ndxv)
                    A(f"{ind}a{g}_{m}_{h} = {P['fma'](P['cvt'](f'ia{g}_{m}_{h}'), dv, f'a{g}_{m}_{h}')};")
                    A(f"{ind}a{g}_{m}_{h} = {P['fma'](P['cvt'](f'ma{g}_{m}_{h}'), mv, f'a{g}_{m}_{h}')};")
    inner = "\n".join(lines)
    decl = " ".join(f"{F} a{g}_{m}_{h} = {P['fzero']};" for g in range(G) for m in range(M) for h in range(H))
    stores = "\n".join((f"        if ({m} < M) " if m else "        ")
                       + f"storev(Y + {m} * N, (g + {g}) * {R} + {h * L}, r0, r1, a{g}_{m}_{h});"
                       for g in range(G) for m in range(M) for h in range(H))  # fmt: skip

    # ---------------- activation prep ------------------------------------------------------
    prep = [f"        dx[m * nk + k] = f16f(ld16(blk)) * {dxmul};" if r.act == "q8_0" else
            f"        dx[m * nk + k] = ldf(blk) * {dxmul};"]  # fmt: skip
    if r.act == "q8_0":
        blk = "xp[m] + 34 * k"
        if beta and pair:
            prep.append(
                "        { const int32_t s = sum_i8(blk + 2, 32);"
                f" sdA[m * nk + k] = {17 * beta} * s; sdB[m * nk + k] = {16 * beta} * s; }}"
            )
        elif beta:
            for s in range(S):
                prep.append(f"        sd0[(m * nk + k) * {S} + {s}] = {beta} * sum_i8(blk + 2 + {s * r.sub}, {r.sub});")
    else:
        blk = "xp[m] + 292 * (k / 8)"
        prep.append("        const int16_t *bs = (const int16_t *)(blk + 260);")
        prep.append("        const int32_t sxv = bs[2 * (k % 8)] + bs[2 * (k % 8) + 1];")
        prep.append("        (void)sxv;")
        if two and corr == "act":
            prep.append("        sxk[m * nk + k] = sxv;")
        if two and corr == "dpmin":
            prep.append(
                "        { uint8_t *mb = (uint8_t *)(mw + (m * (nk / 8) + k / 8) * 4); const int j = (int)(k % 8);"
                " mb[(j / 4) * 4 + j % 4] = (uint8_t)(sxv & 127); mb[8 + (j / 4) * 4 + j % 4] = (uint8_t)(sxv >> 7); }"
            )
    arrays = f"    float dx[{M} * 1024];\n"
    xarr = [("float", "dx", "nk")]  # (C type, name, entries per activation column) for the xprep workspace
    if beta and pair:
        arrays += f"    int32_t sdA[{M} * 1024], sdB[{M} * 1024];\n"
        xarr += [("int32_t", "sdA", "nk"), ("int32_t", "sdB", "nk")]
    elif beta:
        arrays += f"    int32_t sd0[{M} * 1024 * {S}];\n"
        xarr += [("int32_t", "sd0", f"nk * {S}")]
    if two and corr == "act":
        arrays += f"    int32_t sxk[{M} * 1024];\n"
        xarr += [("int32_t", "sxk", "nk")]
    if two and corr == "dpmin":
        arrays += f"    int32_t mw[{M} * 128 * 4];\n"
        xarr += [("int32_t", "mw", "(nk / 8) * 4")]
    act_prep = f"""{arrays}    const uint8_t *xp[{M}];
    for (int64_t m = 0; m < {M}; m++) {{
        xp[m] = (const uint8_t *)X + (m < M ? m : 0) * {xrow};  /* columns beyond M repeat column 0 */
        for (int64_t k = 0; k < nk; k++) {{
        const uint8_t *blk = {blk};
{chr(10).join(prep)}
        }}
    }}"""

    # ---------------- assembly -----------------------------------------------------------------
    consts = [f"const {V} m4 = {P['set1_8']('0x0F')}, mF0 = {P['set1_8']('0xF0')};", "(void)m4; (void)mF0;"]
    tables = ""
    if lutmode:
        ent = [(v + 128) & 0xFF for v in r.lut]
        n = 64 if unpack == "perm" else 16
        tables += f"static const uint8_t LUT[{n}] __attribute__((aligned(64))) = {{" + ", ".join(str(e) for e in (ent * 4)[:n]) + "};\n"
        consts.append(f"const {V} lut = " + (P["vload"]("LUT") if unpack == "perm" else P["tbl"]("LUT")) + ";")
    if two and unpack == "mask16":
        idx = []
        for q in range(L // 4):
            for i in range(4):
                idx += [4 * q + i, 0x80, 4 * q + i, 0x80]
        tables += f"static const uint8_t SCIDX[{len(idx)}] __attribute__((aligned(64))) = {{" + ", ".join(str(e) for e in idx) + "};\n"
        consts.append(f"const {V} scidx = {P['vload']('SCIDX')};")
    if target == "avx512_vnni":
        storev = """
static inline void storev(float *y, int64_t row0, int64_t r0, int64_t r1, __m512 v) {
    if (row0 >= r0 && row0 + 16 <= r1) { _mm512_storeu_ps(y + row0, v); return; }
    uint32_t m = 0xFFFF;
    if (row0 < r0) m &= r0 - row0 >= 16 ? 0 : 0xFFFFu << (r0 - row0);
    if (row0 + 16 > r1) m &= r1 <= row0 ? 0 : (r1 - row0 >= 16 ? 0xFFFFu : (1u << (r1 - row0)) - 1);
    _mm512_mask_storeu_ps(y + row0, (__mmask16)m, v);
}"""
    else:
        storev = """
static inline void storev(float *y, int64_t row0, int64_t r0, int64_t r1, __m256 v) {
    float t[8];
    _mm256_storeu_ps(t, v);
    for (int i = 0; i < 8; i++) if (row0 + i >= r0 && row0 + i < r1) y[row0 + i] = t[i];
}"""
    helpers_k = """
static inline int32_t ld32(const uint8_t *p) { int32_t v; memcpy(&v, p, 4); return v; }
static inline int64_t ld64(const uint8_t *p) { int64_t v; memcpy(&v, p, 8); return v; }
static inline uint16_t ld16(const uint8_t *p) { uint16_t v; memcpy(&v, p, 2); return v; }
static inline float ldf(const uint8_t *p) { float v; memcpy(&v, p, 4); return v; }
static inline int32_t sum_i8(const uint8_t *q, int n) {  /* sum of n = 16 or 32 int8 */
    __m128i s;
    if (n == 32) {
        const __m256i t = _mm256_sad_epu8(_mm256_xor_si256(_mm256_loadu_si256((const __m256i *)q), _mm256_set1_epi8((char)0x80)),
                                          _mm256_setzero_si256());
        s = _mm_add_epi64(_mm256_castsi256_si128(t), _mm256_extracti128_si256(t, 1));
    } else {
        s = _mm_sad_epu8(_mm_xor_si128(_mm_loadu_si128((const __m128i *)q), _mm_set1_epi8((char)0x80)), _mm_setzero_si128());
    }
    s = _mm_add_epi64(s, _mm_unpackhi_epi64(s, s));
    return _mm_cvtsi128_si32(s) - 128 * n;
}"""
    params = (
        "const void *pv, const void *X, float *Y, int64_t K, int64_t N, int64_t M, int64_t r0, int64_t r1"
        if gemm
        else "const void *pv, const void *X, float *Y, int64_t K, int64_t r0, int64_t r1"
    )
    args = "pv, X, Y, K, N, M, r0, r1" if gemm else "pv, X, Y, K, r0, r1"
    mdecl = "" if gemm else "    const int64_t M = 1, N = 0;\n"
    fallback = (
        (
            f"void {entry}(const void *W, const void *X, float *Y, int64_t K, int64_t N, int64_t M, int64_t n0, int64_t n1) {{\n"
            "    (void)W; (void)X; (void)Y; (void)K; (void)N; (void)M; (void)n0; (void)n1; abort();\n}\n"
        )
        if gemm
        else ""
    )
    nl = "\n    "
    tail = f"""    {nl.join(consts)}
    const int64_t g0 = r0 / {R}, g1 = (r1 + {R - 1}) / {R};
    for (int64_t g = g0; g < g1; g += {G}) {{
        {decl}
        for (int64_t p = 0; p < pk->nrec_k; p++) {{
{inner}
        }}
{stores}
    }}
}}
"""
    xfuncs = (
        _xprep_funcs(
            c, entry, xarr, prep, blk, xrow, M, gemm, mdecl, tail, "    const packed_t *pk = (const packed_t *)pv;\n", unpack == "perm"
        )
        if c.get("xprep")
        else ""
    )
    kern = f"""{{
    const packed_t *pk = (const packed_t *)pv;
{mdecl}    const int64_t nk = K / 32;
    if (M > {M}) abort();
    (void)N;
{act_prep}
    {nl.join(consts)}
    const int64_t g0 = r0 / {R}, g1 = (r1 + {R - 1}) / {R};
    for (int64_t g = g0; g < g1; g += {G}) {{
        {decl}
        for (int64_t p = 0; p < pk->nrec_k; p++) {{
{inner}
        }}
{stores}
    }}
}}
"""
    if unpack == "perm":  # vpermb needs AVX512_VBMI (absent on Cascade Lake): vpshufb fallback, picked once at run time
        kern = f"""
#define PERMB(i, t) _mm512_permutexvar_epi8((i), (t))
__attribute__((target("avx512vbmi"))) static void {entry}_vbmi({params}) {kern}#undef PERMB
#define PERMB(i, t) _mm512_shuffle_epi8((t), _mm512_and_si512((i), m4))  /* LUT is 4 copies of the 16 entries */
static void {entry}_novbmi({params}) {kern}#undef PERMB

void {entry}_packed({params}) {{
#ifdef KURN_NO_VBMI
    const int vbmi = 0;
#else
    static int vbmi = -1;
    if (vbmi < 0) vbmi = __builtin_cpu_supports("avx512vbmi") ? 1 : 0;
#endif
    if (vbmi) {entry}_vbmi({args});
    else {entry}_novbmi({args});
}}
"""
    else:
        kern = f"\nvoid {entry}_packed({params}) {kern}"
    body = f"""{tables}{helpers_k}
{storev}
{kern}{fallback}{xfuncs}"""
    return _prelude(target) + prepare + body


def _xprep_funcs(c, entry, xarr, prep, blk, xrow, M, gemm, mdecl, tail, head, perm=False, need_xp=True):
    """Shared activation prep (c["xprep"], set by the llama.cpp integration): the per-block
    activation tables the kernel otherwise rebuilds on every call are built once per op.

      size_t E_xprep_bytes(K, C)            workspace bytes for C activation columns
      void   E_xprep(X, K, C, m0, m1, k0, k1, ws)   fill columns [m0, m1), 32-value blocks [k0, k1)
                                            (X = column 0; Q8_K activations: k0, k1 multiples of 8)
      void   E_packed_x(pv, X, ws, C, c0, Y, K, [N, M,] r0, r1)   the kernel on columns c0.. of ws
                                            (X = column c0), same arithmetic as E_packed
    Layout: array after array, each (C + 8) columns x per-column entries; the 8 spare columns
    keep reads of a verify kernel's unused columns (m >= M) inside the workspace."""
    ct = "(C + 8)"
    offs, run = [], "0"
    for typ, name, per in xarr:
        offs.append((typ, name, per, run))
        run = f"{run} + 4 * {ct} * ({per})"

    def ptrs(const, col):
        q = "const " if const else ""
        return "\n".join(
            f"    {q}{typ} *{name} = ({q}{typ} *)((" + q + f"uint8_t *)ws + {off}) + {col} * ({per});" for typ, name, per, off in offs
        )

    fill = "\n".join(prep)
    params_x = (
        "const void *pv, const void *X, const void *ws, int64_t C, int64_t c0, float *Y, int64_t K, int64_t N, int64_t M, "
        "int64_t r0, int64_t r1"
        if gemm
        else "const void *pv, const void *X, const void *ws, int64_t C, int64_t c0, float *Y, int64_t K, int64_t r0, int64_t r1"
    )
    args_x = "pv, X, ws, C, c0, Y, K, N, M, r0, r1" if gemm else "pv, X, ws, C, c0, Y, K, r0, r1"
    xps = (
        f"""    const uint8_t *xp[{M}];
    for (int64_t m = 0; m < {M}; m++) xp[m] = (const uint8_t *)X + (m < M ? m : 0) * {xrow};
"""
        if need_xp
        else "    (void)X;\n"
    )
    kx = f"""{{
{head}{mdecl}    const int64_t nk = K / 32;
    if (M > {M}) abort();
    (void)N; (void)C;
{ptrs(True, "c0")}
{xps}{tail}"""
    if perm:
        kx = f"""
#define PERMB(i, t) _mm512_permutexvar_epi8((i), (t))
__attribute__((target("avx512vbmi"))) static void {entry}_x_vbmi({params_x}) {kx}#undef PERMB
#define PERMB(i, t) _mm512_shuffle_epi8((t), _mm512_and_si512((i), m4))
static void {entry}_x_novbmi({params_x}) {kx}#undef PERMB

void {entry}_packed_x({params_x}) {{
#ifdef KURN_NO_VBMI
    const int vbmi = 0;
#else
    static int vbmi = -1;
    if (vbmi < 0) vbmi = __builtin_cpu_supports("avx512vbmi") ? 1 : 0;
#endif
    if (vbmi) {entry}_x_vbmi({args_x});
    else {entry}_x_novbmi({args_x});
}}
"""
    else:
        kx = f"\nvoid {entry}_packed_x({params_x}) {kx}"
    return f"""
/* --- shared activation prep (xprep) --- */
size_t {entry}_xprep_bytes(int64_t K, int64_t C) {{
    const int64_t nk = K / 32;
    return (size_t)({run});
}}
void {entry}_xprep(const void *X, int64_t K, int64_t C, int64_t m0, int64_t m1, int64_t k0, int64_t k1, void *ws) {{
    const int64_t nk = K / 32;
{ptrs(False, "0")}
    for (int64_t m = m0; m < m1; m++) {{
        const uint8_t *xpm = (const uint8_t *)X + m * {xrow};
        for (int64_t k = k0; k < k1; k++) {{
        const uint8_t *blk = {blk.replace("xp[m]", "xpm")};
{fill}
        }}
    }}
}}
{kx}"""
