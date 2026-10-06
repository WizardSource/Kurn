"""Fused GEMV epilogues (CUTLASS-style) for the vnni16 Q8_0 decode kernel.

The vnni16 GEMV leaves 16 output rows per zmm accumulator. An epilogue decides what
happens to those registers before they are stored, so work that would otherwise be a
separate pass over the output (and, in a multi-threaded decode step, often a barrier)
runs while the values are still in registers:

    store         y[row] = acc                                 (plain GEMV)
    store_sumsq   y[row] = acc, returns sum(acc^2)             (RMSNorm partial sums)
    axpy          y[row] += s * acc                            (residual add / weighted MoE expert sum)
    swiglu_q8     gate/up rows interleaved per 16 -> silu(gate) * up -> Q8_0 blocks
                  (the next GEMV's activation, bit-identical to ggml's ggml_vec_swiglu_f32 +
                  quantize_row_q8_0 on AVX-512), optionally also the float activation

Every entry point also takes a block range [b0, b1) of K (column-split GEMVs: each thread
multiplies only its slice of the input vector and produces a partial output) and a group
range [g0, g1) of 16-row groups. Activation preparation (`kq8e_prep`) is done once per
input vector and shared by every call that uses it.

Registered through kurn.hooks as the schedule key `epilogue` (`none` | `fused`) of the
gemv/q8_0/avx512_vnni vnni16 kernel. `fused` appends the kq8e_* entry points (declared in
model/kq8e.h) to the unchanged base kernel, so the kurn.h entry points and everything
the harness checks are byte-identical to `epilogue=none`; the key is therefore declared
as not changing the (harness-visible) codegen. The model engine links these kernels.

`engine_kernels(..., fmts=("q8_0", "q4_0"))` also emits the same four epilogues as kq4e_* for
Q4_0 weights in a vnni16 nibble layout (two K steps per 64-byte vector, low/high nibble), used by
the engine for all-Q4_0 models; activations stay Q8_0 (kq8e_act).
"""

from . import codegen, hooks

EPILOGUES = ("store", "store_sumsq", "axpy", "swiglu_q8")

_ARGS = {
    "store": "float *y",
    "store_sumsq": "float *y",
    "axpy": "float *y, float s",
    "swiglu_q8": "void *yqv, float *yact",
}
_RET = {"store_sumsq": "float"}

_HELPERS = r"""
// ggml's AVX-512 expf / silu (ggml-cpu/vec.h), copied so the fused SwiGLU is bit-identical to
// ggml_vec_swiglu_f32 on the same inputs.
static inline __m512 kq8e_v_expf(__m512 x) {
    const __m512 r = _mm512_set1_ps(0x1.8p23f);
    const __m512 z = _mm512_fmadd_ps(x, _mm512_set1_ps(0x1.715476p+0f), r);
    const __m512 n = _mm512_sub_ps(z, r);
    const __m512 b = _mm512_fnmadd_ps(n, _mm512_set1_ps(0x1.7f7d1cp-20f), _mm512_fnmadd_ps(n, _mm512_set1_ps(0x1.62e4p-1f), x));
    const __mmask16 d = _mm512_cmp_ps_mask(_mm512_abs_ps(n), _mm512_set1_ps(192), _CMP_GT_OQ);
    const __m512 u = _mm512_mul_ps(b, b);
    const __m512 j = _mm512_fmadd_ps(
        _mm512_fmadd_ps(_mm512_fmadd_ps(_mm512_set1_ps(0x1.0e4020p-7f), b, _mm512_set1_ps(0x1.573e2ep-5f)), u,
                        _mm512_fmadd_ps(_mm512_set1_ps(0x1.555e66p-3f), b, _mm512_set1_ps(0x1.fffdb6p-2f))),
        u, _mm512_fmadd_ps(_mm512_set1_ps(0x1.ffffecp-1f), b, _mm512_set1_ps(1.0F)));
    const __m512 res = _mm512_scalef_ps(j, n);
    if (_mm512_kortestz(d, d)) return res;
    const __m512 zero = _mm512_setzero_ps();
    const __m512 alt = _mm512_mask_blend_ps(_mm512_cmp_ps_mask(n, zero, _CMP_LE_OQ), _mm512_set1_ps(__builtin_inff()), zero);
    return _mm512_mask_blend_ps(d, res, alt);
}
static inline __m512 kq8e_v_silu(__m512 x) {
    const __m512 neg = _mm512_sub_ps(_mm512_setzero_ps(), x);
    return _mm512_div_ps(x, _mm512_add_ps(_mm512_set1_ps(1), kq8e_v_expf(neg)));
}
// 32 floats -> one Q8_0 block, the arithmetic of ggml's x86 quantize_row_q8_0
// (amax/127 scale stored as fp16, x * (127/amax) rounded half-to-even).
static inline void kq8e_q8_block(__m512 lo, __m512 hi, block_q8_0 *out) {
    const float amax = _mm512_reduce_max_ps(_mm512_max_ps(_mm512_abs_ps(lo), _mm512_abs_ps(hi)));
    const float d = amax / 127.f, id = amax != 0.0f ? 127.f / amax : 0.0f;
    out->d = _cvtss_sh(d, 0);
    const __m512 m = _mm512_set1_ps(id);
    const __m512i a = _mm512_cvtps_epi32(_mm512_roundscale_ps(_mm512_mul_ps(lo, m), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    const __m512i b = _mm512_cvtps_epi32(_mm512_roundscale_ps(_mm512_mul_ps(hi, m), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    _mm_storeu_si128((__m128i *)out->qs, _mm512_cvtepi32_epi8(a));
    _mm_storeu_si128((__m128i *)(out->qs + 16), _mm512_cvtepi32_epi8(b));
}
"""


def _pass(n, ld512, ld256, P, fmt="q8_0"):
    """Code for one pass over n 16-row groups starting at g: accumulators a0..a{n-1}.
    q8_0: 8 x 64-byte loads per block (+128-biased bytes, correction -128 * sum(x));
    q4_0: 4 x 64-byte loads, each holding two k-steps as low / high nibbles (0..15, correction -8 * sum(x))."""
    vt = "vblk" if fmt == "q8_0" else "vblk4"
    L = [" ".join(f"__m512 a{i} = _mm512_setzero_ps();" for i in range(n))]
    L.append("for (int64_t b = b0; b < b1; b++) {")
    L.append("    const int64_t xb = b - b0;")
    ncs = "A->ncs[xb]" if fmt == "q8_0" else "(A->ncs[xb] >> 4)"
    L.append(f"    const __m512i nc = _mm512_set1_epi32({ncs}); const __m512 dx = _mm512_set1_ps(A->xd[xb]);")
    L += [f"    const {vt} *blk{i} = W + (g + {i}) * nb + b;" for i in range(n)]
    if P:
        L += [
            f"    _mm_prefetch((const char *)(blk{i} + {P}), _MM_HINT_T0); _mm_prefetch((const char *)(blk{i} + {P}) + 256, _MM_HINT_T0);"
            for i in range(n)
        ]
    L += [f"    __m512i e{i} = nc, o{i} = _mm512_setzero_si512();" for i in range(n)]
    if fmt == "q8_0":
        for kk in range(8):
            acc = "e" if kk % 2 == 0 else "o"
            L.append(f"    {{ const __m512i xk = _mm512_set1_epi32(A->xw[xb * 8 + {kk}]);")
            L += [f"      {acc}{i} = _mm512_dpbusd_epi32({acc}{i}, {ld512}(blk{i}->q + {64 * kk}), xk);" for i in range(n)]
            L.append("    }")
    else:
        for h in range(4):
            L.append(
                f"    {{ const __m512i xl = _mm512_set1_epi32(A->xw[xb * 8 + {2 * h}]),"
                f" xh = _mm512_set1_epi32(A->xw[xb * 8 + {2 * h + 1}]);"
            )
            for i in range(n):
                L.append(f"      const __m512i v{i} = {ld512}(blk{i}->q + {64 * h});")
                L.append(f"      e{i} = _mm512_dpbusd_epi32(e{i}, _mm512_and_si512(v{i}, m4), xl);")
                L.append(f"      o{i} = _mm512_dpbusd_epi32(o{i}, _mm512_and_si512(_mm512_srli_epi16(v{i}, 4), m4), xh);")
            L.append("    }")
    L += [
        f"    a{i} = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_add_epi32(e{i}, o{i})), "
        f"_mm512_mul_ps(_mm512_cvtph_ps({ld256}((const __m256i *)blk{i}->d)), dx), a{i});"
        for i in range(n)
    ]
    L.append("}")
    return L


def _epi(kind, n):
    if kind == "store":
        return [f"_mm512_storeu_ps(y + (g + {i}) * 16, a{i});" for i in range(n)]
    if kind == "store_sumsq":
        return [f"_mm512_storeu_ps(y + (g + {i}) * 16, a{i}); ss = _mm512_fmadd_ps(a{i}, a{i}, ss);" for i in range(n)]
    if kind == "axpy":
        return [f"_mm512_storeu_ps(y + (g + {i}) * 16, _mm512_fmadd_ps(sv, a{i}, _mm512_loadu_ps(y + (g + {i}) * 16)));" for i in range(n)]
    if kind == "swiglu_q8":
        L = []
        for q in range(n // 4):
            g0, u0, g1, u1 = 4 * q, 4 * q + 1, 4 * q + 2, 4 * q + 3
            L.append(f"{{ const __m512 lo = _mm512_mul_ps(kq8e_v_silu(a{g0}), a{u0}), hi = _mm512_mul_ps(kq8e_v_silu(a{g1}), a{u1});")
            L.append(f"  kq8e_q8_block(lo, hi, yq + (g + {4 * q}) / 4);")
            L.append(
                f"  if (yact) {{ _mm512_storeu_ps(yact + (g + {4 * q}) * 8, lo); _mm512_storeu_ps(yact + (g + {4 * q}) * 8 + 16, hi); }} }}"
            )
        return L
    raise ValueError(kind)


def _function(kind, G, ld512, ld256, P, fmt="q8_0"):
    ret = _RET.get(kind, "void")
    pre, vt = ("kq8e", "vblk") if fmt == "q8_0" else ("kq4e", "vblk4")
    head = (
        f"{ret} {pre}_{kind}(const void *Wv, int64_t nb, const kq8e_act *A, int64_t b0, int64_t b1, "
        f"int64_t g0, int64_t g1, {_ARGS[kind]}) {{"
    )
    L = [head, f"    const {vt} *W = (const {vt} *)Wv;", "    int64_t g = g0;"]
    if fmt == "q4_0":
        L.append("    const __m512i m4 = _mm512_set1_epi8(0x0F);")
    if kind == "store_sumsq":
        L.append("    __m512 ss = _mm512_setzero_ps();")
    if kind == "axpy":
        L.append("    const __m512 sv = _mm512_set1_ps(s);")
    if kind == "swiglu_q8":
        L.append("    block_q8_0 *yq = (block_q8_0 *)yqv;")
    widths = [G] + [w for w in (4, 1) if w < G]
    if kind == "swiglu_q8":  # gate/up interleave: 4 groups = 32 activation values = one Q8_0 block
        widths = sorted({w for w in (G - G % 4, 4) if w >= 4}, reverse=True)
    for w in widths:
        L.append(f"    for (; g + {w} <= g1; g += {w}) {{")
        L += ["        " + s for s in _pass(w, ld512, ld256, P, fmt)]
        L += ["        " + s for s in _epi(kind, w)]
        L.append("    }")
    if kind == "store_sumsq":
        L.append("    return _mm512_reduce_add_ps(ss);")
    L.append("}")
    return "\n".join(L)


_Q4_0 = r"""// Q4_0 weights (block_q4_0: fp16 d; 16 bytes, value j in the low nibble of qs[j], j + 16 in the high
// nibble) in a vnni16 layout: per 16-row group and 32-column block, 4 x 64 bytes, byte h*64 + r*4 + j
// holding k-step 2h (low nibble) and 2h+1 (high nibble) of row r, unsigned 0..15.
size_t kq4e_blk_bytes(void) { return sizeof(vblk4); }
void kq4e_pack(void *dst, const void *src, int64_t nb, int64_t N) {
    const kq4e_block_q4_0 *w = (const kq4e_block_q4_0 *)src;
    vblk4 *out = (vblk4 *)dst;
    const int64_t ng = (N + 15) / 16;
    for (int64_t g = 0; g < ng; g++)
        for (int64_t b = 0; b < nb; b++) {
            vblk4 *o = out + g * nb + b;
            memset(o, 0, sizeof *o);
            for (int r = 0; r < 16; r++) {
                if (g * 16 + r >= N) {  // padding rows: value 8 (= 0 after the -8 offset), scale 0
                    for (int h = 0; h < 4; h++) for (int j = 0; j < 4; j++) o->q[h * 64 + r * 4 + j] = 0x88;
                    continue;
                }
                const kq4e_block_q4_0 *s = w + (g * 16 + r) * nb + b;
                o->d[r] = s->d;
                for (int kk = 0; kk < 8; kk++)
                    for (int j = 0; j < 4; j++) {
                        const int k = kk * 4 + j;
                        const uint8_t v = k < 16 ? (s->qs[k] & 0x0F) : (s->qs[k - 16] >> 4);
                        o->q[(kk / 2) * 64 + r * 4 + j] |= (uint8_t)(kk % 2 ? v << 4 : v);
                    }
            }
        }
}"""


def engine_kernels(c, standalone=True, fmts=("q8_0",)):
    """C source of the kq8e_* entry points (see model/kq8e.h) for a resolved vnni16 config.
    standalone=False: to append to the base vnni16 kernel (which already has the includes and vblk).
    fmts: weight formats to emit GEMVs for ("q8_0" -> kq8e_*, "q4_0" -> kq4e_*)."""
    if c.get("layout") != "vnni16":
        raise ValueError("fused epilogues need layout=vnni16")
    G, P = c["rows"], c["prefetch"]
    aligned = c.get("align") == 64
    ld512, ld256 = ("_mm512_load_si512", "_mm256_load_si256") if aligned else ("_mm512_loadu_si512", "_mm256_loadu_si256")
    vblk = (
        f"typedef struct {{ uint8_t q[8 * 64]; uint16_t d[16];{' uint8_t pad[32];' if aligned else ''} }} vblk;"
        f" // {576 if aligned else 544} B: 16 rows x 32 values, rows interleaved per 4 bytes, +128 biased"
    )
    head = [
        "// Generated by kurn (epilogue.py). Fused-epilogue vnni16 Q8_0 GEMVs; ABI in model/kq8e.h.",
        '#include "kq8e.h"',
        "#include <string.h>",
        "#include <immintrin.h>",
        vblk,
    ]
    body = (head if standalone else [_ABI]) + [
        _HELPERS,
        "size_t kq8e_blk_bytes(void) { return sizeof(vblk); }",
        r"""void kq8e_pack(void *dst, const void *src, int64_t nb, int64_t N) {
    const block_q8_0 *w = (const block_q8_0 *)src;
    vblk *out = (vblk *)dst;
    const int64_t ng = (N + 15) / 16;
    for (int64_t g = 0; g < ng; g++)
        for (int64_t b = 0; b < nb; b++) {
            vblk *o = out + g * nb + b;
            memset(o, 0, sizeof *o);
            for (int r = 0; r < 16; r++) {
                if (g * 16 + r >= N) {
                    for (int kk = 0; kk < 8; kk++) for (int j = 0; j < 4; j++) o->q[kk * 64 + r * 4 + j] = 0x80;
                    continue;
                }
                const block_q8_0 *s = w + (g * 16 + r) * nb + b;
                o->d[r] = s->d;
                for (int kk = 0; kk < 8; kk++)
                    for (int j = 0; j < 4; j++) o->q[kk * 64 + r * 4 + j] = (uint8_t)(s->qs[kk * 4 + j] ^ 0x80);
            }
        }
}""",
        r"""void kq8e_prep(const void *xv, int64_t nb, kq8e_act *A) {
    const block_q8_0 *x = (const block_q8_0 *)xv;
    for (int64_t b = 0; b < nb; b++) {
        int32_t s = 0;
        for (int l = 0; l < 32; l++) s += x[b].qs[l];
        memcpy(A->xw + 8 * b, x[b].qs, 32);
        A->ncs[b] = -128 * s;
        A->xd[b] = _cvtsh_ss(x[b].d);
    }
}""",
        r"""void kq8e_quantize(const float *x, void *yv, int64_t k) {
    block_q8_0 *y = (block_q8_0 *)yv;
    for (int64_t b = 0; b < k / 32; b++) kq8e_q8_block(_mm512_loadu_ps(x + 32 * b), _mm512_loadu_ps(x + 32 * b + 16), y + b);
}""",
        r"""void kq8e_swiglu(const float *g, const float *u, float *y, int64_t n) {
    for (int64_t i = 0; i < n; i += 16) _mm512_storeu_ps(y + i, _mm512_mul_ps(kq8e_v_silu(_mm512_loadu_ps(g + i)), _mm512_loadu_ps(u + i)));
}""",
    ]
    if "q8_0" in fmts:
        body += [_function(k, G, ld512, ld256, P) for k in EPILOGUES]
    if "q4_0" in fmts:
        body.append(
            f"typedef struct {{ uint8_t q[4 * 64]; uint16_t d[16];{' uint8_t pad[32];' if aligned else ''} }} vblk4;"
            f" // {320 if aligned else 288} B: 16 rows x 32 values, two k-steps per byte"
        )
        body.append(_Q4_0)
        body += [_function(k, G, ld512, ld256, P, "q4_0") for k in EPILOGUES]
    return "\n".join(body) + "\n"


# ------------------------------------------------------------------ kurn.hooks registration
def _legal(op, f, t):
    return ("none", "fused") if (op, f, t) == ("gemv", "q8_0", "avx512_vnni") else ("none",)


def _with_epilogues(base):
    def lower(target, c):
        src = base(target, c)
        if c.get("epilogue") != "fused":
            return src
        return src + "\n// --- fused epilogues (kurn.epilogue) ---\n" + engine_kernels(c, standalone=False)

    return lower


_ABI = r"""#ifndef KQ8E_MAXNB
#define KQ8E_MAXNB 1024
typedef struct { int32_t xw[8 * KQ8E_MAXNB]; int32_t ncs[KQ8E_MAXNB]; float xd[KQ8E_MAXNB]; } kq8e_act;
#endif
"""


def register():
    if "epilogue" in hooks.NEW_KEYS:
        return
    hooks.new_key("epilogue", _legal, "none", changes_codegen=False)
    hooks.EXTRA_INVALID.append(
        (lambda c: c.get("epilogue", "none") != "none" and c.get("layout") != "vnni16", "epilogue=fused needs layout=vnni16")
    )
    hooks.LOWERINGS["vnni16"] = _with_epilogues(hooks.LOWERINGS.get("vnni16", codegen.q8_0_gemv))
