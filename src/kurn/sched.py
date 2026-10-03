"""Schedule-driven lowering: (recipe, record layout, schedule) -> plain C.

"What" is a generic.Recipe (code width, code -> integer map, scale structure,
activation format). "Where" is a layout.RecordLayout composed from primitives
(lanes, bit-plane binding, K-group depth, row-group interleave, metadata
placement, swizzle, padding). "How" is the schedule: the algorithm keys below
plus register blocking and prefetch/pipelining. Nothing here is specific to one
format: Q4_K, Q8_0, Q4_0, IQ4_NL, Q2_0, TQ2_0 and Q1_0 all lower through it.

  unpack      mask (and/shift) | lut (vpshufb codebook) | perm (vpermb, 64-entry
              replicated table: no mask needed; AVX-512 VBMI)
  correction  one-level: act (beta * sum(x) seeded into the accumulator) | weight
              (signed codes, x biased by +128, 128 * sum(w) stored per row)
              two-level (Q4_K mins): act (mullo per K-group) | pair (vpdpwssd over
              int16 (mn[s], mn[s+1]) x (bsum[s], bsum[s+1]); mins stored pair-
              interleaved) | float (dmin * mn precomputed in f32; FMA per K-group)
  scales      unpacked (u8 per row) | packed (raw 12 B, decoded in-kernel) | fold
              (d * sc precomputed in f32 at repack time)
  accum       int (integer accumulation across the scale period, one convert) |
              float (convert + FMA per 32 values)
  chains      independent int32 accumulator chains per lane group (1, 2, 4)
  rows        vector groups per pass (register blocking), cols  activation columns
  prefetch    distance in records; pfhint t0|t1|t2|nta; pfgran rec (one line per
              record) | line (every line of the record, spread over the K-groups)
  stages      prefetch cascade: stage i prefetches at distance prefetch * 2^i, the farthest
              with pfhint, nearer ones with T1 / T0 (1 = a single prefetch)
  kpanel      K values per K panel, rpanel  row-group passes per row panel: loop nest
              row panel > K panel > row groups > records, partial sums parked between
              K panels (activation reuse in L1 across the row panel; 0 = untiled)

The repack loops evaluate the record layout's emitted C index expression; the
kernel's load offsets, bit-plane shifts and metadata addresses come from
evaluating the same layout object, so a composed layout needs no new code.
"""

from . import generic
from .layout import recipe_meta, record_layout

LANE_TARGETS = ("avx512_vnni", "avx2_vnni")
HINTS = {"t0": "_MM_HINT_T0", "t1": "_MM_HINT_T1", "t2": "_MM_HINT_T2", "nta": "_MM_HINT_NTA"}


class LoweringError(ValueError):
    pass


# --------------------------------------------------------------------------- vector primitives
def prims(target, lanes):
    if target == "avx512_vnni" and lanes == 16:
        return dict(
            V="__m512i", F="__m512", L=16, zero="_mm512_setzero_si512()", fzero="_mm512_setzero_ps()",
            loadu=lambda p: f"_mm512_loadu_si512((const void *)({p}))",
            and_=lambda a, b: f"_mm512_and_si512({a}, {b})", srli16=lambda a, n: f"_mm512_srli_epi16({a}, {n})",
            set1_8=lambda v: f"_mm512_set1_epi8((char)({v}))", set1_32=lambda v: f"_mm512_set1_epi32({v})",
            shuf=lambda t, i: f"_mm512_shuffle_epi8({t}, {i})", perm8=lambda i, t: f"_mm512_permutexvar_epi8({i}, {t})",
            dp=lambda acc, u, x: f"_mm512_dpbusd_epi32({acc}, {u}, {x})",
            dpw=lambda acc, a, b: f"_mm512_dpwssd_epi32({acc}, {a}, {b})",
            add=lambda a, b: f"_mm512_add_epi32({a}, {b})", mullo=lambda a, b: f"_mm512_mullo_epi32({a}, {b})",
            cvt=lambda a: f"_mm512_cvtepi32_ps({a})", fma=lambda a, b, c: f"_mm512_fmadd_ps({a}, {b}, {c})",
            fnma=lambda a, b, c: f"_mm512_fnmadd_ps({a}, {b}, {c})",
            mul=lambda a, b: f"_mm512_mul_ps({a}, {b})", fset1=lambda v: f"_mm512_set1_ps({v})",
            h2f=lambda p: f"_mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)({p})))",
            f32=lambda p: f"_mm512_loadu_ps((const float *)({p}))",
            u8x=lambda p: f"_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *)({p})))",
            u8w=lambda p: f"_mm512_cvtepu8_epi16(_mm256_loadu_si256((const __m256i *)({p})))",
            i16x=lambda p: f"_mm512_cvtepi16_epi32(_mm256_loadu_si256((const __m256i *)({p})))",
            bits=lambda p: f"_mm512_maskz_mov_epi8(_cvtu64_mask64(*(const uint64_t *)({p})), _mm512_set1_epi8(1))",
            tbl16=lambda name: f"_mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i *){name}))",
            tbl64=lambda name: f"_mm512_loadu_si512((const void *){name})",
            store=lambda p, v: f"_mm512_storeu_ps({p}, {v})",
        )
    if lanes == 8 and target in ("avx512_vnni", "avx2_vnni"):
        evex = target == "avx512_vnni"
        return dict(
            V="__m256i", F="__m256", L=8, zero="_mm256_setzero_si256()", fzero="_mm256_setzero_ps()",
            loadu=lambda p: f"_mm256_loadu_si256((const __m256i *)({p}))",
            and_=lambda a, b: f"_mm256_and_si256({a}, {b})", srli16=lambda a, n: f"_mm256_srli_epi16({a}, {n})",
            set1_8=lambda v: f"_mm256_set1_epi8((char)({v}))", set1_32=lambda v: f"_mm256_set1_epi32({v})",
            shuf=lambda t, i: f"_mm256_shuffle_epi8({t}, {i})", perm8=lambda i, t: f"_mm256_permutexvar_epi8({i}, {t})",
            dp=(lambda acc, u, x: f"_mm256_dpbusd_epi32({acc}, {u}, {x})") if evex else
               (lambda acc, u, x: f"_mm256_dpbusd_avx_epi32({acc}, {u}, {x})"),
            dpw=(lambda acc, a, b: f"_mm256_dpwssd_epi32({acc}, {a}, {b})") if evex else
                (lambda acc, a, b: f"_mm256_dpwssd_avx_epi32({acc}, {a}, {b})"),
            add=lambda a, b: f"_mm256_add_epi32({a}, {b})", mullo=lambda a, b: f"_mm256_mullo_epi32({a}, {b})",
            cvt=lambda a: f"_mm256_cvtepi32_ps({a})", fma=lambda a, b, c: f"_mm256_fmadd_ps({a}, {b}, {c})",
            fnma=lambda a, b, c: f"_mm256_fnmadd_ps({a}, {b}, {c})",
            mul=lambda a, b: f"_mm256_mul_ps({a}, {b})", fset1=lambda v: f"_mm256_set1_ps({v})",
            h2f=lambda p: f"_mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)({p})))",
            f32=lambda p: f"_mm256_loadu_ps((const float *)({p}))",
            u8x=lambda p: f"_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)({p})))",
            u8w=lambda p: f"_mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)({p})))",
            i16x=lambda p: f"_mm256_cvtepi16_epi32(_mm_loadu_si128((const __m128i *)({p})))",
            bits=(lambda p: f"_mm256_maskz_mov_epi8(_cvtu32_mask32(*(const uint32_t *)({p})), _mm256_set1_epi8(1))") if evex
                 else (lambda p: f"bits8x4(*(const uint32_t *)({p}))"),
            tbl16=lambda name: f"_mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *){name}))",
            tbl64=lambda name: f"_mm256_loadu_si256((const __m256i *){name})",
            store=lambda p, v: f"_mm256_storeu_ps({p}, {v})",
        )
    raise LoweringError(f"no {lanes}-lane primitives for {target}")


# --------------------------------------------------------------------------- legality
def planes_for(bits):
    return {8: ("none",), 4: ("kstep", "khalf", "rows"), 2: ("kstep", "khalf", "rows"), 1: ("atom", "kstep")}[bits]


def unpacks_for(recipe, target, plane="auto"):
    out = tuple(u for u in recipe.unpacks if u != "none") or ("none",)
    if recipe.bits < 8 and target == "avx512_vnni" and plane != "atom":
        out += ("perm",)
    return out


# Recipe features the composed lowering implements. Recipes with other scale encodings (MXFP4's
# e8m0, NVFP4's ue4m3 per 16 values) and generic-only algorithm values (e.g. q4_K
# correction=dpmin) stay with the generic i16 / i8 lowering.
CORRECTIONS = ("act", "weight")
SCALES = ("unpacked", "packed")


def supports(recipe):
    return recipe.scale == "f16" and recipe.sub == 32


def algo_values(recipe, target):
    """Legal algorithm-key values of the composed lowering (unions over layouts)."""
    legal = generic.legal_keys(recipe.name, target)
    corr = tuple(v for v in legal["correction"] if v in CORRECTIONS) + (("pair", "float") if recipe.two_level else ())
    scales = tuple(v for v in legal["scales"] if v in SCALES) + (("fold",) if recipe.two_level else ())
    return {"unpack": unpacks_for(recipe, target), "correction": corr, "scales": scales, "accum": legal["accum"]}


def defaults(recipe, target):
    """`auto` -> the values that reproduce the i16 / i8 layout exactly."""
    legal = generic.legal_keys(recipe.name, target)
    return {
        "lanes": 16 if target == "avx512_vnni" else 8,
        "plane": planes_for(recipe.bits)[0],
        "kblock": recipe.period,
        "meta": "head",
        "chains": 2,
        "unpack": legal["unpack"][0],
        "correction": legal["correction"][0],
        "scales": legal["scales"][0],
        "accum": legal["accum"][0],
    }


def check(c):
    """None if the composed config is lowerable, else the reason."""
    r = generic.RECIPES[c["weights"]]
    t = c["target"]
    for k, vals in algo_values(r, t).items():
        if c[k] not in vals:
            return f"{k}={c[k]} is not a composed-layout value for {r.name}/{t} (legal: {vals})"
    if c["lanes"] == 16 and t != "avx512_vnni":
        return "lanes=16 needs avx512_vnni"
    if c["plane"] not in planes_for(r.bits):
        return f"plane={c['plane']} is not valid for {r.bits}-bit codes (legal: {planes_for(r.bits)})"
    if c["kblock"] % r.period or c["kblock"] > 256:
        return f"kblock must be a multiple of the scale period {r.period} and <= 256"
    if c["unpack"] not in unpacks_for(r, t, c["plane"]):
        return f"unpack={c['unpack']} is not legal here (legal: {unpacks_for(r, t, c['plane'])})"
    if c["unpack"] == "none" and r.bits != 8 or r.bits == 8 and c["unpack"] != "none":
        return "unpack=none is for 8-bit codes only"
    two = r.two_level
    if two and c["correction"] == "pair" and c["scales"] != "unpacked":
        return "correction=pair needs scales=unpacked (pair-interleaved u8 mins)"
    if two and (c["correction"] == "float") != (c["scales"] == "fold"):
        return "correction=float goes with scales=fold (both precomputed in f32)"
    if c["scales"] == "fold" and c["accum"] != "float":
        return "scales=fold needs accum=float"
    if c["correction"] == "weight" and c["unpack"] == "perm":
        return "correction=weight uses signed codes: unpack=perm is not supported"
    lpv = (8 // r.bits) if c["plane"] == "rows" else 1
    nlg = c["rows"] * lpv
    if nlg * c["cols"] > 8:
        return "lane groups per pass (rows x planes) x cols must be <= 8"
    if c["rgroup"] > c["rows"] or c["rows"] % c["rgroup"]:
        return "rgroup must divide rows"
    if (c.get("kpanel", 0) > 0) != (c.get("rpanel", 0) > 0):
        return "kpanel and rpanel go together (row panel x K panel tiling)"
    if c.get("kpanel", 0) % c["kblock"]:
        return f"kpanel must be a multiple of kblock ({c['kblock']})"
    if c.get("stages", 1) > 1 and not c["prefetch"]:
        return "stages > 1 is a prefetch cascade: needs prefetch > 0"
    if c["swizzle"]:
        try:
            build_layout(c)
        except ValueError as e:
            return str(e)
    return None


def build_layout(c):
    r = generic.RECIPES[c["weights"]]
    corr = (("wsum", "i16"),) if c["correction"] == "weight" else ()
    meta = recipe_meta(r, c["scales"])
    if r.two_level and c["correction"] == "pair":
        meta = tuple(m if m[0] != "mn" else ("mnp", "u8", 8, 2) for m in meta)
    return record_layout(r.bits, meta, lanes=c["lanes"], plane=c["plane"], kblock=c["kblock"], period=r.period,
                         rgroup=c["rgroup"], place=c["meta"], corr_fields=corr, align=64 if c["align"] == 64 else 0,
                         rg_pad=c["rgpad"], swizzle=c["swizzle"])


def preset_config(name, c):
    """Composed-layout values that reproduce a fixed layout (for tests and docs)."""
    r = generic.RECIPES[c["weights"]]
    if name == "vnni16":
        return dict(lanes=16, plane="none", kblock=32, meta="tail", rgroup=1, rgpad=0, swizzle=0)
    if name in ("i16", "i8"):
        return dict(lanes=16 if name == "i16" else 8, plane=planes_for(r.bits)[0], kblock=r.period, meta="head",
                    rgroup=1, rgpad=0, swizzle=0)
    raise KeyError(name)


# --------------------------------------------------------------------------- lowering
def lower(target, c):
    reason = check(c)
    if reason:
        raise LoweringError(reason)
    r = generic.RECIPES[c["weights"]]
    lay = build_layout(c)
    return _prelude(target) + _prepare(r, lay, c) + _kernel(target, r, lay, c)


def _prelude(target):
    return """// Generated by kurn (layout algebra + schedule lowering, sched.py). Do not edit; edit the .kurn spec instead.
#include "kurn.h"
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>
static inline float f16f(uint16_t h) { return _cvtsh_ss(h); }
"""


def _ustore(r, c):
    """Stored code byte for raw code q."""
    if c["correction"] == "weight":
        return f"(uint8_t)(int8_t)({r.signed_c})" if r.bits == 8 else "q"
    if r.bits == 8:
        return r.maps["none"][0]
    return "q"


def _umap(r, c):
    """(u expression, alpha, beta) of the unpack method."""
    u = c["unpack"]
    if u == "perm":
        u = "lut" if "lut" in r.maps else "mask"
    if u in r.maps:
        return u, r.maps[u]
    return u, r.maps[next(iter(r.maps))]


def _prepare(r, lay, c):
    bits, epb = r.bits, 8 // r.bits
    R, KB, per = lay.rows, lay.kblock, r.period
    two = r.two_level
    corr_w = c["correction"] == "weight"
    kv = ("static const int8_t KV[16] __attribute__((unused)) = {" + ", ".join(str(v) for v in generic.KV_IQ4NL) + "};\n"
          if "KV[" in (r.signed_c + "".join(m[0] for m in r.maps.values())) else "")
    code_e = lay.code.c_expr(("row", "kk"))
    if lay.swizzle is not None:
        sw = lay.swizzle
        nb_lg = lay.lg_code_bytes
        # swizzle the vector index inside its (lane group, K-group) chunk by the row-group bits
        swz = (f"{{ const int64_t rel = byte - {lay.code_base}, kgi = rel / {lay.kg_bytes}, rem = rel % {lay.kg_bytes};\n"
               f"                      if (rem < {nb_lg * lay.params['rgroup']}) {{\n"
               f"                        const int64_t lgi = rem / {nb_lg}, off = rem % {nb_lg};\n"
               f"                        byte = {lay.code_base} + kgi * {lay.kg_bytes} + lgi * {nb_lg} + "
               f"(off ^ ((gr & {(1 << sw.bits) - 1}) << {sw.base})); }} }}")
    else:
        swz = ""
    fill = []
    f = lay.field
    if two:
        fill.append(f"memcpy(rec + {f('d').layout.c_expr(('row', '0', 'q'))}, &b->d, 2);" if f("d") else "")
        fill.append(f"memcpy(rec + {f('dmin').layout.c_expr(('row', '0', 'q'))}, &b->dmin, 2);" if f("dmin") else "")
        dec = ("const uint8_t *q6 = b->scales; int sc[8], mn[8];\n"
               "                    for (int s = 0; s < 8; s++) {\n"
               "                        sc[s] = s < 4 ? q6[s] & 63 : (q6[s + 4] & 0xF) | ((q6[s - 4] >> 6) << 4);\n"
               "                        mn[s] = s < 4 ? q6[s + 4] & 63 : (q6[s + 4] >> 4) | ((q6[s] >> 6) << 4);\n"
               "                    }\n                    (void)sc; (void)mn;")
        fill.append(dec)
        if f("sc"):
            fill.append(f"for (int s = 0; s < 8; s++) rec[{f('sc').layout.c_expr(('row', 's', 'q'))}] = (uint8_t)sc[s];")
        if f("mn"):
            fill.append(f"for (int s = 0; s < 8; s++) rec[{f('mn').layout.c_expr(('row', 's', 'q'))}] = (uint8_t)mn[s];")
        if f("mnp"):
            fill.append(f"for (int s = 0; s < 8; s++) rec[{f('mnp').layout.c_expr(('row', 's', 'q'))}] = (uint8_t)mn[s];")
        if f("raw"):
            fill.append(f"for (int t = 0; t < 12; t++) rec[{f('raw').layout.c_expr(('row', 't', 'q'))}] = b->scales[t];")
        if f("dsc"):
            fill.append(f"for (int s = 0; s < 8; s++) {{ const float v = f16f(b->d) * (float)sc[s]; "
                        f"memcpy(rec + {f('dsc').layout.c_expr(('row', 's', 'q'))}, &v, 4); }}")
            fill.append(f"for (int s = 0; s < 8; s++) {{ const float v = f16f(b->dmin) * (float)mn[s]; "
                        f"memcpy(rec + {f('dmn').layout.c_expr(('row', 's', 'q'))}, &v, 4); }}")
    else:
        fill.append(f"memcpy(rec + {f('d').layout.c_expr(('row', '0', 'q'))}, &{r.d_c}, 2);")
    fill = "\n                    ".join(x for x in fill if x)
    wsum = ""
    if corr_w:
        wsum = (f"if ((kk & 31) == 31) {{ const int16_t ws = (int16_t)wsum; "
                f"memcpy(rec + {f('wsum').layout.c_expr(('row', 'kk / 32'))}, &ws, 2); wsum = 0; }}")
    ustore = _ustore(r, c)
    return f"""
typedef struct {{ {r.struct} }} nblock;
{kv}typedef struct {{ int64_t nrec_k, ngroups; uint8_t *buf; }} packed_t;
/* record layout: {lay.describe()} */
#define REC_BYTES {lay.rec_bytes}
#define RG_STRIDE(nrec) ((size_t)(nrec) * REC_BYTES + {lay.rg_pad})

void *{c['entry']}_prepare(const void *W, int64_t K, int64_t N) {{
    const nblock *w = (const nblock *)W;
    const int64_t nb = K / {r.block};
    packed_t *pk = malloc(sizeof *pk);
    pk->nrec_k = K / {KB}; pk->ngroups = (N + {R - 1}) / {R} + {c['rows'] // c['rgroup']};
    const size_t bytes = (size_t)pk->ngroups * RG_STRIDE(pk->nrec_k);
    pk->buf = aligned_alloc(64, (bytes + 63) & ~(size_t)63);
    memset(pk->buf, 0, bytes);
    for (int64_t gr = 0; gr < pk->ngroups; gr++)
        for (int64_t p = 0; p < pk->nrec_k; p++) {{
            uint8_t *rec = pk->buf + gr * RG_STRIDE(pk->nrec_k) + p * REC_BYTES;
            for (int64_t row = 0; row < {R}; row++) {{
                const int64_t n = gr * {R} + row;
                if (n >= N) continue;  /* padding rows stay zero: d = 0 */
                for (int64_t q = 0; q < {KB // per}; q++) {{
                    const nblock *b = w + n * nb + (p * {KB} + q * {per}) / {r.block};
                    {fill}
                }}
                int32_t wsum = 0;
                (void)wsum;
                for (int64_t kk = 0; kk < {KB}; kk++) {{
                    const int64_t vv = p * {KB} + kk;
                    const nblock *b = w + n * nb + vv / {r.block};
                    const int v = (int)(vv % {r.block});
                    const int q = {r.code_c};
                    {"wsum += " + r.signed_c + ";" if corr_w else ""}
                    const uint8_t u = {ustore};
                    const int64_t e = {code_e};
                    int64_t byte = e / {epb};
                    {swz}
                    rec[byte] |= (uint8_t)(u << ((e % {epb}) * {bits}));
                    {wsum}
                }}
            }}
        }}
    return pk;
}}
"""


def _atoms(r, lay, c, nlg):
    """For each (lane group, K step) inside one K-group chunk: (byte offset relative to
    the chunk of K-group 0, plane, record index in the pass), derived by evaluating the
    code layout; verifies that every atom is a contiguous rows-into-lanes block."""
    epb = 8 // r.bits
    L = lay.lanes
    out = {}
    for lg in range(nlg):
        row0 = lg * L
        t, rr = divmod(row0, lay.rows)
        for kk in range(8):
            e0 = lay.code(rr, kk * 4)
            for i in range(L):
                for j in range(4):
                    e = lay.code(rr + i, kk * 4 + j)
                    if lay.params["plane"] == "atom":
                        ok = e == e0 + 4 * i + j
                    else:
                        ok = e == e0 + epb * (4 * i + j)
                    if not ok:
                        raise LoweringError(f"layout {lay.code} is not a rows-into-lanes atom at lane group {lg}, step {kk}")
            if lay.params["plane"] == "atom":
                if e0 % 8:
                    raise LoweringError("bit atoms must start on a byte")
                out[(lg, kk)] = (e0 // 8, 0, t)
            else:
                out[(lg, kk)] = (e0 // epb, e0 % epb, t)
    return out


def _kernel(target, r, lay, c):
    P = prims(target, lay.lanes)
    L, V, F = P["L"], P["V"], P["F"]
    epb = 8 // r.bits
    two = r.two_level
    G, M, PF = c["rows"], c["cols"], c["prefetch"]
    lpv = epb if c["plane"] == "rows" else 1
    nlg = G * lpv  # lane groups per pass (one float accumulator vector each per column)
    R = lay.rows
    recs = (nlg * L) // R  # records per pass (row direction)
    gemm = M > 1
    entry = c["entry"]
    unpack, ch = c["unpack"], max(1, c["chains"])
    corr = c["correction"]
    umode, (u_expr, alpha, beta) = _umap(r, c)
    per_kg = r.period // 32
    periods = lay.kblock // r.period
    atoms = _atoms(r, lay, c, nlg)
    vb = lay.vec_bytes
    hint = HINTS[c["pfhint"]]
    sw = lay.swizzle

    # lookup tables
    tables = ""
    lut_decl = ""
    use_lut = unpack == "lut" or (corr == "weight" and r.lut and r.bits < 8)
    if use_lut:
        vals = r.lut
        entries = [v & 0xFF for v in vals] if corr == "weight" else [(v + 128) & 0xFF for v in vals]
        entries = (entries * (16 // len(entries) + 1))[:16]
        tables += "static const uint8_t LUT[16] = {" + ", ".join(map(str, entries)) + "};\n"
        lut_decl = f"const {V} lut = {P['tbl16']('LUT')};"
    if unpack == "perm":
        mask = (1 << r.bits) - 1
        if umode == "lut":
            base = [(v + 128) & 0xFF for v in r.lut]
        else:
            base = list(range(mask + 1))
        nt = 64 if L == 16 else 32
        entries = [base[i & mask] for i in range(nt)]
        tables += f"static const uint8_t PTBL[{nt}] __attribute__((aligned(64))) = {{" + ", ".join(map(str, entries)) + "};\n"
        lut_decl = f"const {V} ptbl = {P['tbl64']('PTBL')};"

    mvar = {4: "m4", 2: "m3", 1: "m1"}.get(r.bits, "m4")

    def unpack_expr(var, plane):
        if r.bits == 8:
            return var
        if c["plane"] == "atom":
            return P["bits"](var)
        sh = r.bits * plane
        v = P["srli16"](var, sh) if sh else var
        if unpack == "perm":
            return P["perm8"](v, "ptbl")
        v = P["and_"](v, mvar)
        if use_lut:
            v = P["shuf"]("lut", v)
        return v

    ind = "                "
    lines = []
    rec_of = {}
    for t in range(recs):
        rec_of[t] = f"rec{t}"
        lines.append(f"{ind}const uint8_t *rec{t} = pk->buf + (size_t)(gr + {t}) * rgs + p * REC_BYTES;")
    stages = c.get("stages", 1)
    cascade = [(PF * (1 << (stages - 1)), hint)] + [(PF * (1 << i), HINTS["t1" if i else "t0"]) for i in range(stages - 2, -1, -1)]
    if PF:
        for t in range(recs):
            for dist, h in cascade:
                if c["pfgran"] == "rec":
                    lines.append(f"{ind}_mm_prefetch((const char *)(rec{t} + {dist} * REC_BYTES), {h});")
                else:
                    hdr_lines = (lay.code_base + 63) // 64 if c["meta"] == "head" else 0
                    for li in range(hdr_lines):
                        lines.append(f"{ind}_mm_prefetch((const char *)(rec{t} + {dist} * REC_BYTES + {64 * li}), {h});")

    # metadata vectors per lane group
    def fld(name, lg, idx=0, q=0):
        """Address of metadata `name` entry `idx` (int: evaluated; str: affine) for the
        L rows of lane group lg, verified row-contiguous (pair-interleaved for mnp)."""
        t, rr = divmod(lg * L, R)
        fl = lay.field(name)
        i0 = idx if isinstance(idx, int) else 0
        off = fl.layout(rr, i0, q)
        size = {"f16": 2, "f32": 4, "u8": 1, "i16": 2}[fl.ctype]
        il = 2 if name == "mnp" else 1
        for i in range(L):
            if fl.layout(rr + i, i0, q) != off + size * i * il:
                raise LoweringError(f"field {name} is not row-contiguous")
        if isinstance(idx, int):
            return f"rec{t} + {off}"
        step = fl.layout(rr, 1, q) - off
        return f"rec{t} + {off} + ({idx}) * {step}"

    for q in range(periods):
        sfx = f"_{q}" if periods > 1 else ""
        for lg in range(nlg):
            if two and c["scales"] != "fold":
                lines.append(f"{ind}const {F} d{lg}{sfx} = {P['h2f'](fld('d', lg, q=q))}, dm{lg}{sfx} = {P['h2f'](fld('dmin', lg, q=q))};")
            elif not two:
                lines.append(f"{ind}const {F} d{lg}{sfx} = {P['h2f'](fld('d', lg, q=q))};")
        if two and c["scales"] == "packed":
            for lg in range(nlg):
                lines.append(f"{ind}{V} q{lg}{sfx}[12]; for (int t = 0; t < 12; t++) q{lg}{sfx}[t] = {P['u8x'](fld('raw', lg, 't', q))};")

    int_period = c["accum"] == "int"
    for q in range(periods):
        sfx = f"_{q}" if periods > 1 else ""
        k0 = f"p * {lay.kblock // 32} + {q * per_kg}"
        for lg in range(nlg):
            for m in range(M):
                decl = []
                if int_period:
                    decl.append(f"{V} ia{lg}_{m} = {P['zero']};")
                if two and corr in ("act", "pair"):
                    decl.append(f"{V} ma{lg}_{m} = {P['zero']};")
                if two and (c["accum"] == "float" or c["scales"] == "fold"):
                    decl.append(f"{F} af{lg}_{m} = {P['fzero']};")
                if decl:
                    lines.append(ind + " ".join(decl))
        lines.append(f"{ind}for (int kg = 0; kg < {per_kg}; kg++) {{")
        lines.append(f"{ind}    const int64_t k = {k0} + kg;")
        lines.append(f"{ind}    const size_t kgo = (size_t)({q * per_kg} + kg) * {lay.kg_bytes};")
        # loads: distinct (record, byte) of this K-group
        loads = {}
        for _, (byte, _plane, t) in sorted(atoms.items()):
            key = (t, byte)
            if key not in loads:
                loads[key] = f"c{t}_{byte}"
        if PF and c["pfgran"] == "line":
            seen = set()
            for (t, byte) in loads:
                line = byte // 64
                if (t, line) not in seen:
                    seen.add((t, line))
                    for dist, h in cascade:
                        lines.append(f"{ind}    _mm_prefetch((const char *)(rec{t} + {dist} * REC_BYTES + kgo + {64 * line}), {h});")
        for (t, byte), var in loads.items():
            addr = f"rec{t} + kgo + {byte}"
            if sw is not None:
                rel = byte - lay.code_base
                lgc = rel % lay.kg_bytes
                lgi, off = divmod(lgc, lay.lg_code_bytes)
                vi = off // vb
                base = lay.code_base + lgi * lay.lg_code_bytes + (off % vb)
                addr = f"rec{t} + kgo + {base} + ((size_t)({vi} ^ swz{t}) * {vb})"
            if r.bits == 1 and c["plane"] == "atom":
                lines.append(f"{ind}    const uint8_t *{var} = {addr};")
            else:
                lines.append(f"{ind}    const {V} {var} = {P['loadu'](addr)};")
        # seeds
        for lg in range(nlg):
            t, rr = divmod(lg * L, R)
            for m in range(M):
                if corr == "weight":
                    seed = P["mullo"](P["i16x"](f"rec{t} + kgo + {lay.field('wsum').layout(rr, 0) }"),
                                      P["set1_32"]("-128"))
                elif not two and beta and alpha == 1:
                    seed = P["set1_32"](f"{beta} * sx[{m} * nk + k]")
                else:
                    seed = P["zero"]
                accs = [f"s{lg}_{m}_0 = {seed}"] + [f"s{lg}_{m}_{i} = {P['zero']}" for i in range(1, ch)]
                lines.append(f"{ind}    {V} " + ", ".join(accs) + ";")
        for kk in range(8):
            for m in range(M):
                lines.append(f"{ind}    const {V} x{m}_{kk} = {P['set1_32'](f'xw[({m} * nk + k) * 8 + {kk}]')};")
            for lg in range(nlg):
                byte, plane, t = atoms[(lg, kk)]
                var = loads[(t, byte)]
                lines.append(f"{ind}    {{ const {V} u = {unpack_expr(var, plane)};")
                for m in range(M):
                    acc = f"s{lg}_{m}_{kk % ch}"
                    ops = (f"x{m}_{kk}", "u") if corr == "weight" else ("u", f"x{m}_{kk}")
                    lines.append(f"{ind}      {acc} = {P['dp'](acc, *ops)};")
                lines.append(f"{ind}    }}")
        # apply scales
        for lg in range(nlg):
            for m in range(M):
                dsum = f"s{lg}_{m}_0"
                for i in range(1, ch):
                    dsum = P["add"](dsum, f"s{lg}_{m}_{i}")
                if alpha != 1:
                    dsum = P["add"](P["mullo"](dsum, P["set1_32"](str(alpha))), P["set1_32"](f"{beta} * sx[{m} * nk + k]"))
                if two:
                    sfx = f"_{q}" if periods > 1 else ""
                    if c["scales"] == "fold":
                        dsc = P["f32"](fld("dsc", lg, "kg", q))
                        dmn = P["f32"](fld("dmn", lg, "kg", q))
                        lines.append(f"{ind}    af{lg}_{m} = {P['fma'](P['cvt'](dsum), dsc, f'af{lg}_{m}')};")
                        lines.append(f"{ind}    af{lg}_{m} = {P['fnma'](dmn, P['fset1'](f'(float)sx[{m} * nk + k]'), f'af{lg}_{m}')};")
                        continue
                    if c["scales"] == "unpacked":
                        scv = P["u8x"](fld("sc", lg, "kg", q))
                    else:
                        scv = f"q4k_sc(q{lg}{sfx}, kg)"
                    if c["accum"] == "int":
                        lines.append(f"{ind}    ia{lg}_{m} = {P['add'](f'ia{lg}_{m}', P['mullo'](dsum, scv))};")
                    else:
                        lines.append(f"{ind}    af{lg}_{m} = {P['fma'](P['cvt'](dsum), P['cvt'](scv), f'af{lg}_{m}')};")
                    if corr == "act":
                        mnv = P["u8x"](fld("mn", lg, "kg", q)) if c["scales"] == "unpacked" else f"q4k_mn(q{lg}{sfx}, kg)"
                        mterm = P["mullo"](mnv, P["set1_32"](f"sx[{m} * nk + k]"))
                        lines.append(f"{ind}    ma{lg}_{m} = {P['add'](f'ma{lg}_{m}', mterm)};")
                elif int_period:
                    lines.append(f"{ind}    ia{lg}_{m} = {P['add'](f'ia{lg}_{m}', dsum)};")
                else:
                    sc = P["mul"](f"d{lg}{sfx}", P["fset1"](f"dx[{m} * nk + k]"))
                    lines.append(f"{ind}    a{lg}_{m} = {P['fma'](P['cvt'](dsum), sc, f'a{lg}_{m}')};")
        lines.append(f"{ind}}}")
        # end of period
        sfx = f"_{q}" if periods > 1 else ""
        for lg in range(nlg):
            for m in range(M):
                kx = f"{m} * nk + {k0}"
                if two:
                    if corr == "pair":
                        for s2 in range(per_kg // 2):
                            mnp = P["u8w"](fld("mnp", lg, 2 * s2, q))
                            lines.append(f"{ind}ma{lg}_{m} = {P['dpw'](f'ma{lg}_{m}', mnp, P['set1_32'](f'sxp[({kx}) / 2 + {s2}]'))};")
                    if c["scales"] == "fold":
                        lines.append(f"{ind}a{lg}_{m} = {P['fma'](f'af{lg}_{m}', P['fset1'](f'dx[{kx}]'), f'a{lg}_{m}')};")
                        continue
                    dd = P["mul"](f"d{lg}{sfx}", P["fset1"](f"dx[{kx}]"))
                    if c["accum"] == "int":
                        lines.append(f"{ind}a{lg}_{m} = {P['fma'](P['cvt'](f'ia{lg}_{m}'), dd, f'a{lg}_{m}')};")
                    else:
                        lines.append(f"{ind}a{lg}_{m} = {P['fma'](f'af{lg}_{m}', dd, f'a{lg}_{m}')};")
                    dmx = P["mul"](f"dm{lg}{sfx}", P["fset1"](f"-dx[{kx}]"))
                    lines.append(f"{ind}a{lg}_{m} = {P['fma'](P['cvt'](f'ma{lg}_{m}'), dmx, f'a{lg}_{m}')};")
                elif int_period:
                    ddx = P["mul"](f"d{lg}{sfx}", P["fset1"](f"dx[{kx}]"))
                    lines.append(f"{ind}a{lg}_{m} = {P['fma'](P['cvt'](f'ia{lg}_{m}'), ddx, f'a{lg}_{m}')};")
    inner = "\n".join(lines)

    KP = c.get("kpanel", 0) // lay.kblock
    RP = c.get("rpanel", 0)
    tiled = KP > 0
    nacc = nlg * M
    if tiled:
        decl = " ".join(f"{F} a{lg}_{m} = kp ? {P['f32'](f'pt + {(lg * M + m) * L}')} : {P['fzero']};"
                        for lg in range(nlg) for m in range(M))
    else:
        decl = " ".join(f"{F} a{lg}_{m} = {P['fzero']};" for lg in range(nlg) for m in range(M))
    swz_decl = ""
    if sw is not None:
        swz_decl = " ".join(f"const size_t swz{t} = (size_t)((gr + {t}) & {(1 << sw.bits) - 1});" for t in range(recs))
    stores = "\n".join((f"        if ({m} < M) " if m else "        ") + f"storev(Y + {m} * N, gr * {R} + {lg * L}, r0, r1, a{lg}_{m});"
                       for lg in range(nlg) for m in range(M))
    if L == 16:
        storev = """
static inline void storev(float *y, int64_t row0, int64_t r0, int64_t r1, __m512 v) {
    if (row0 >= r0 && row0 + 16 <= r1) { _mm512_storeu_ps(y + row0, v); return; }
    uint32_t m = 0xFFFF;
    if (row0 < r0) m &= r0 - row0 >= 16 ? 0 : 0xFFFFu << (r0 - row0);
    if (row0 + 16 > r1) m &= (1u << (r1 > row0 ? (r1 - row0 > 16 ? 16 : r1 - row0) : 0)) - 1;
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
    const {V} m63 = {P['set1_32'](63)}, m15 = {P['set1_32'](15)};
    if (s < 4) return {P['and_']('q[s]', 'm63')};
    return {P['add'](P['and_']('q[s + 4]', 'm15'), P['mullo'](P['srli16']('q[s - 4]', 6), P['set1_32'](16)))};
}}
static inline {V} q4k_mn(const {V} *q, int s) {{
    const {V} m63 = {P['set1_32'](63)};
    if (s < 4) return {P['and_']('q[s + 4]', 'm63')};
    return {P['add'](P['srli16']('q[s + 4]', 4), P['mullo'](P['srli16']('q[s]', 6), P['set1_32'](16)))};
}}"""
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
    if corr == "pair":
        act_prep += """
        for (int64_t k = 0; k < nk; k += 2)
            sxp[(m * nk + k) / 2] = (int32_t)(((uint32_t)(uint16_t)(int16_t)sx[m * nk + k])
                                            | ((uint32_t)(uint16_t)(int16_t)sx[m * nk + k + 1] << 16));"""
    act_bias = ""
    if corr == "weight":
        act_bias = f"\n    for (int64_t i = 0; i < {M} * nk * 8; i++) xw[i] ^= (int32_t)0x80808080;  /* x + 128 as u8 */"
    sig = (f"void {entry}_packed(const void *pv, const void *X, float *Y, int64_t K, int64_t N, int64_t M, int64_t r0, int64_t r1)"
           if gemm else f"void {entry}_packed(const void *pv, const void *X, float *Y, int64_t K, int64_t r0, int64_t r1)")
    mdecl = "" if gemm else "    const int64_t M = 1, N = 0;\n"
    fallback = (f"void {entry}(const void *W, const void *X, float *Y, int64_t K, int64_t N, int64_t M, int64_t n0, int64_t n1) {{\n"
                "    (void)W; (void)X; (void)Y; (void)K; (void)N; (void)M; (void)n0; (void)n1; abort();\n}\n") if gemm else ""
    attr = '__attribute__((target("avx512vbmi"))) ' if unpack == "perm" else ""
    if not tiled:
        loop = f"""    for (int64_t gr = gr0; gr < gr1; gr += {recs}) {{
        {decl}
        {swz_decl}
        for (int64_t p = 0; p < pk->nrec_k; p++) {{
{inner}
        }}
{stores}
    }}"""
    else:
        # row panel x K panel: the activation slice of one K panel stays in L1 while RP passes
        # of row groups stream through it; float partial sums wait in `part` between K panels
        parks = "\n".join(f"                {P['store'](f'pt + {(lg * M + m) * L}', f'a{lg}_{m}')};"
                           for lg in range(nlg) for m in range(M))
        stores_t = "\n".join("        " + ln for ln in stores.splitlines())
        loop = f"""    float part[{RP} * {nacc} * {L}] __attribute__((aligned(64)));
    for (int64_t gp = gr0; gp < gr1; gp += {RP * recs}) {{
        const int64_t gpe = gp + {RP * recs} < gr1 ? gp + {RP * recs} : gr1;
        for (int64_t kp = 0; kp < pk->nrec_k; kp += {KP}) {{
            const int64_t kpe = kp + {KP} < pk->nrec_k ? kp + {KP} : pk->nrec_k;
            for (int64_t gr = gp; gr < gpe; gr += {recs}) {{
                float *const pt = part + (gr - gp) / {recs} * {nacc * L};
                {decl}
                {swz_decl}
                for (int64_t p = kp; p < kpe; p++) {{
{inner}
                }}
                if (kpe < pk->nrec_k) {{
{parks}
                    continue;
                }}
{stores_t}
            }}
        }}
    }}"""
    return f"""
{act_struct}
{tables}{storev}
{q4k_packed}
{attr}{sig} {{
    const packed_t *pk = (const packed_t *)pv;
{mdecl}    const int64_t nk = K / 32, nbx = {nbx};
    if (M > {M}) abort();
    int32_t xw[{M} * 1024 * 8], sx[{M} * 1024], sxp[{M} * 512];
    float dx[{M} * 1024];
    (void)sx; (void)sxp; (void)nbx;
    for (int64_t m = 0; m < {M}; m++) {{{act_prep}
    }}{act_bias}
    const {V} m4 = {P['set1_8']('0x0F')}, m3 = {P['set1_8']('0x03')}, m1 = {P['set1_8']('0x01')};
    (void)m4; (void)m3; (void)m1;
    {lut_decl}
    const size_t rgs = RG_STRIDE(pk->nrec_k);
    const int64_t gr0 = r0 / {R}, gr1 = (r1 + {R - 1}) / {R};
{loop}
}}
{fallback}"""
