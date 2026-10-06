/* kurn attention kernel template. kurn.attention emits the KA_* configuration
 * #defines in front of this file; see kurn_attn.h for the ABI.
 *
 *   KA_DK, KA_DV      head dims (multiples of 64)
 *   KA_KV             KATTN_KV_F16 | KATTN_KV_BF16 | KATTN_KV_Q8_0
 *   KA_ENGINE         0 = f32 FMA, 1 = AVX-512 BF16 (vdpbf16ps), 2 = AMX-BF16      (tile engine)
 *   KA_TQ, KA_TK      query tokens / KV tokens per tile (tile engine)
 *   KA_SPLIT          KV splits per (kv head, query tile): 0 = auto
 *   KA_DEC_ROWS       use the decode (row) engine when n_q * G <= KA_DEC_ROWS (<= 8; 0 = never)
 *
 * Scores are kept in the log2 domain (q is pre-scaled by scale * log2 e). The running
 * max is only raised when a tile's max exceeds it by more than KA_RESCALE (lazy
 * rescaling): probabilities stay <= 2^KA_RESCALE, which is exact enough in f32/bf16
 * and skips almost every rescale of the output accumulator after the first tiles.
 *
 *   KA_KCENTER        bf16 engines: subtract a per-item K mean (of the first K tile it packs, offset-
 *                     dominated channels only) before rounding K to bf16. q.mu is constant per row, so softmax is unchanged;
 *                     it is added back to the running max for the split-KV merge. Removes the
 *                     rounding noise of constant K channel offsets (e.g. Qwen2 key bias).
 */
#ifndef KA_KCENTER
#define KA_KCENTER 1
#endif
/*   KA_QK8            amx_bf16: Q K^T on AMX-INT8 (q int8 per row, k int8 per token after centering),
 *                     P V stays bf16. About 6x the bf16 score rounding error; no kattn_pack support. */
#ifndef KA_QK8
#define KA_QK8 0
#endif
#if KA_ENGINE == 0
#undef KA_KCENTER
#define KA_KCENTER 0
#endif
#if KA_ENGINE != 2
#undef KA_QK8
#define KA_QK8 0
#endif
#include <immintrin.h>
#include <math.h>
#include <stdatomic.h>
#include <stdint.h>
#include <string.h>

#include "kurn_attn.h"

#if KA_ENGINE == 2
#include <sys/syscall.h>
#include <unistd.h>
#include <x86intrin.h>
#endif

#define KA_LOG2E 1.4426950408889634f
#define KA_RESCALE 8.0f
#define KA_DEC_TK 64
#define KA_MAX_SPLIT 64
#define KA_ALIGN(x) (((x) + 63) & ~(size_t)63)
#define KA_MIN(a, b) ((a) < (b) ? (a) : (b))
#define KA_MAX(a, b) ((a) > (b) ? (a) : (b))
#define KA_INLINE static inline __attribute__((always_inline, unused)) /* shared by all engines; each uses a subset */

/* pre-RoPE formats: K and V are dequantized (and K rotated) into f32 staging rows per tile */
#define KA_PRE_ROPE (KA_KV == KATTN_KV_K4C_Q4 || KA_KV == KATTN_KV_K4C_Q8)

#if KA_KV == KATTN_KV_Q8_0
#define KA_ROW_BYTES(d) ((int64_t)(d) / 32 * 34)
#elif KA_PRE_ROPE
#define KA_ROW_BYTES(d) ((int64_t)(d) * 4)
#else
#define KA_ROW_BYTES(d) ((int64_t)(d) * 2)
#endif

int64_t kattn_config(int *dk, int *dv, int *kv_format) {
    if (dk) *dk = KA_DK;
    if (dv) *dv = KA_DV;
    if (kv_format) *kv_format = KA_KV;
#if KA_PRE_ROPE
    return KATTN_K4C_BLOCK_BYTES(KA_DK);
#else
    return KA_ROW_BYTES(KA_DK);
#endif
}

/* ---------------------------------------------------------------- helpers */

KA_INLINE __m512 ka_exp2(__m512 x) {
    x = _mm512_max_ps(x, _mm512_set1_ps(-127.0f));
    const __m512 n = _mm512_roundscale_ps(x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    const __m512 f = _mm512_sub_ps(x, n); /* [-0.5, 0.5]: degree-6 Taylor of 2^f, rel. err ~1.3e-7 */
    __m512 p = _mm512_set1_ps(1.5403530393381606e-4f);
    p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(1.3333558146428443e-3f));
    p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(9.6181291076284772e-3f));
    p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(5.5504108664821580e-2f));
    p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(2.4022650695910071e-1f));
    p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(6.9314718055994531e-1f));
    p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(1.0f));
    return _mm512_scalef_ps(p, n);
}

KA_INLINE float ka_f16(uint16_t h) { return _cvtsh_ss(h); }

/* 16 consecutive KV elements [i, i+16) of one row as f32 (i multiple of 16). */
KA_INLINE __m512 ka_row16(const uint8_t *row, int i) {
#if KA_KV == KATTN_KV_F16
    return _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(row + 2 * i)));
#elif KA_KV == KATTN_KV_BF16
    return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i *)(row + 2 * i))), 16));
#elif KA_PRE_ROPE
    return _mm512_loadu_ps((const float *)row + i);
#else
    const uint8_t *b = row + (i >> 5) * 34;
    const __m512 d = _mm512_set1_ps(ka_f16(*(const uint16_t *)b));
    return _mm512_mul_ps(d, _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i *)(b + 2 + (i & 31))))));
#endif
}

/* lane j of the result = sum of all lanes of v[slot(j)], slot(j) = ((j & 3) << 2) | (j >> 2) */
KA_INLINE __m512 ka_hsum16(const float *v /* [16][16] */) {
#define LD(i) _mm512_load_ps(v + 16 * (i))
    __m512 a[8], b[4], c[2];
    for (int k = 0; k < 8; k++) {
        const __m512 x = LD(2 * k), y = LD(2 * k + 1);
        a[k] = _mm512_add_ps(_mm512_shuffle_f32x4(x, y, 0x44), _mm512_shuffle_f32x4(x, y, 0xEE));
    }
    for (int k = 0; k < 4; k++)
        b[k] = _mm512_add_ps(_mm512_shuffle_f32x4(a[2 * k], a[2 * k + 1], 0x88), _mm512_shuffle_f32x4(a[2 * k], a[2 * k + 1], 0xDD));
    for (int k = 0; k < 2; k++)
        c[k] = _mm512_add_ps(_mm512_shuffle_ps(b[2 * k], b[2 * k + 1], 0x44), _mm512_shuffle_ps(b[2 * k], b[2 * k + 1], 0xEE));
    return _mm512_add_ps(_mm512_shuffle_ps(c[0], c[1], 0x88), _mm512_shuffle_ps(c[0], c[1], 0xDD));
#undef LD
}

KA_INLINE void ka_transpose16(__m512i r[16]) {
    __m512i t[16];
    for (int i = 0; i < 16; i += 2) {
        t[i] = _mm512_unpacklo_epi32(r[i], r[i + 1]);
        t[i + 1] = _mm512_unpackhi_epi32(r[i], r[i + 1]);
    }
    for (int i = 0; i < 16; i += 4) {
        r[i] = _mm512_unpacklo_epi64(t[i], t[i + 2]);
        r[i + 1] = _mm512_unpackhi_epi64(t[i], t[i + 2]);
        r[i + 2] = _mm512_unpacklo_epi64(t[i + 1], t[i + 3]);
        r[i + 3] = _mm512_unpackhi_epi64(t[i + 1], t[i + 3]);
    }
    for (int i = 0; i < 4; i++) {
        t[i] = _mm512_shuffle_i32x4(r[i], r[i + 4], 0x88);
        t[i + 4] = _mm512_shuffle_i32x4(r[i], r[i + 4], 0xDD);
        t[i + 8] = _mm512_shuffle_i32x4(r[i + 8], r[i + 12], 0x88);
        t[i + 12] = _mm512_shuffle_i32x4(r[i + 8], r[i + 12], 0xDD);
    }
    for (int i = 0; i < 4; i++) {
        r[i] = _mm512_shuffle_i32x4(t[i], t[i + 8], 0x88);
        r[i + 4] = _mm512_shuffle_i32x4(t[i + 4], t[i + 12], 0x88);
        r[i + 8] = _mm512_shuffle_i32x4(t[i], t[i + 8], 0xDD);
        r[i + 12] = _mm512_shuffle_i32x4(t[i + 4], t[i + 12], 0xDD);
    }
}

/* ---------------------------------------------------------------- plan */

/* Work items are (query tile, block of hb kv heads, KV split). The tile engine uses one kv
 * head per item; the decode engine walks a block of heads chunk by chunk so that each
 * thread streams a contiguous token range of the (token-major, heads-inner) KV cache. */
typedef struct {
    int64_t G, tq, nqt, R, Rpad, nsplit, hb, nhb, ngroups, nitems, chunk;
    int dec;
    size_t thr_bytes, part_off, thr_off;
} ka_plan;

#if KA_PRE_ROPE
static size_t ka_stage_bytes(int dec);
#endif

typedef struct {
    atomic_int next, done;
    atomic_int group[]; /* per (kv head, query tile): splits finished */
} ka_hdr;

static void ka_make_plan(const kattn_args *a, int nth, ka_plan *p) {
    p->G = a->n_head / a->n_head_kv;
    p->dec = KA_DEC_ROWS > 0 && a->n_q * p->G <= KA_DEC_ROWS;
    p->tq = p->dec ? a->n_q : KA_MIN((int64_t)KA_TQ, a->n_q);
    p->nqt = (a->n_q + p->tq - 1) / p->tq;
    p->R = p->G * p->tq;
    p->Rpad = (p->R + 31) & ~(int64_t)31;
    p->ngroups = (int64_t)a->n_head_kv * p->nqt;
    int64_t ns = KA_SPLIT;
    if (p->dec) {
        if (ns <= 0) ns = KA_MIN((int64_t)2 * nth, KA_MAX((int64_t)1, a->n_kv / 256));
        ns = KA_MIN(KA_MAX(ns, (int64_t)1), (int64_t)KA_MAX_SPLIT);
        int64_t nhb = KA_MIN((int64_t)a->n_head_kv, KA_MAX((int64_t)1, (2 * nth + ns - 1) / ns));
        p->hb = (a->n_head_kv + nhb - 1) / nhb;
    } else {
        if (ns <= 0) {
            ns = 1;
            if (p->ngroups < 2 * nth) ns = (2 * nth + p->ngroups - 1) / p->ngroups;
            ns = KA_MIN(ns, KA_MAX((int64_t)1, a->n_kv / 512));
        }
        ns = KA_MIN(KA_MAX(ns, (int64_t)1), (int64_t)KA_MAX_SPLIT);
        p->hb = 1;
    }
    p->nsplit = ns;
    p->nhb = (a->n_head_kv + p->hb - 1) / p->hb;
    p->nitems = p->nqt * p->nhb * p->nsplit;
    const int64_t gran = p->dec ? KA_DEC_TK : KA_TK;
    p->chunk = ((a->n_kv + p->nsplit - 1) / p->nsplit + gran - 1) / gran * gran;
    size_t t;
    if (p->dec) {
        t = KA_ALIGN(sizeof(float) * KA_DEC_ROWS * KA_DEC_TK)        /* scores / probabilities */
          + KA_ALIGN(sizeof(float) * KA_DEC_ROWS * 256)              /* per-token partial dot products */
          + KA_ALIGN(sizeof(float) * p->hb * KA_DEC_ROWS * KA_DK)    /* scaled q */
          + KA_ALIGN(sizeof(float) * p->hb * KA_DEC_ROWS * KA_DV)    /* O */
          + 2 * KA_ALIGN(sizeof(float) * p->hb * KA_DEC_ROWS);      /* m, l */
    } else {
#if KA_ENGINE == 0
        const size_t qb = 4, kb = 4;
#else
        const size_t qb = 2, kb = 2;
#endif
        t = KA_ALIGN(qb * p->Rpad * KA_DK)                           /* Q */
          + KA_ALIGN(sizeof(float) * p->Rpad * KA_TK)                /* S */
          + KA_ALIGN(2 * (size_t)p->Rpad * KA_TK)                    /* P (bf16) */
          + KA_ALIGN(kb * KA_TK * KA_DK)                             /* packed K tile */
          + KA_ALIGN(kb * KA_TK * KA_DV)                             /* packed V tile */
          + KA_ALIGN(sizeof(float) * p->Rpad * KA_DV)                /* O */
          + 2 * KA_ALIGN(sizeof(float) * p->Rpad)                    /* m, l */
          + KA_ALIGN(sizeof(float) * KA_DK)                          /* K mean */
          + 64;
    }
#if KA_PRE_ROPE
    t = KA_ALIGN(t) + ka_stage_bytes(p->dec); /* staging area: the last bytes of each thread's share */
#endif
    p->thr_bytes = KA_ALIGN(t);
    p->part_off = KA_ALIGN(sizeof(ka_hdr) + sizeof(atomic_int) * p->ngroups);
    const size_t part = p->nsplit > 1 ? sizeof(float) * p->ngroups * p->nsplit * p->R * (KA_DV + 2) : 0;
    p->thr_off = KA_ALIGN(p->part_off + part);
}

size_t kattn_workspace(const kattn_args *a, int nth) {
    ka_plan p;
    ka_make_plan(a, nth, &p);
    return p.thr_off + p.thr_bytes * (size_t)nth + 64;
}

typedef struct {
    const kattn_args *a;
    const ka_plan *p;
    int64_t g, ng, qt, t0, nt, k0, k1; /* first kv head, heads, query tile, first token, tokens, kv range */
    float *m, *l, *O;              /* per row: running max (log2), sum, unnormalized output [R][DV] */
    const uint8_t *kvp;            /* kattn_packed: AMX-ready K/V (NULL = pack per tile) */
    int64_t cap;                   /* its token capacity (multiple of KA_TK) */
} ka_item;

KA_INLINE const float *ka_qrow_g(const ka_item *it, int64_t g, int64_t r) {
    const int64_t G = it->p->G;
    return it->a->q + (it->t0 + r / G) * it->a->q_s_tok + (g * G + r % G) * it->a->q_s_head;
}

KA_INLINE const float *ka_qrow(const ka_item *it, int64_t r) { return ka_qrow_g(it, it->g, r); }

#if KA_PRE_ROPE
static const kattn_args *ka_staged(const kattn_args *a, int64_t g, int64_t j);
#endif

KA_INLINE const uint8_t *ka_krow(const kattn_args *a, int64_t g, int64_t j) {
#if KA_PRE_ROPE
    a = ka_staged(a, g, j);
#endif
    return (const uint8_t *)a->k + j * a->k_s_tok + g * a->k_s_head;
}

KA_INLINE const uint8_t *ka_vrow(const kattn_args *a, int64_t g, int64_t j) {
#if KA_PRE_ROPE
    a = ka_staged(a, g, j);
#endif
    return (const uint8_t *)a->v + j * a->v_s_tok + g * a->v_s_head;
}

#if KA_PRE_ROPE
/* ---------------------------------------------------------------- pre-RoPE staging
 * Per thread: f32 K [tk][DK] (dequantized, rotated), f32 V [tk][DV], cos / sin [tk][DK/2] of
 * the tile's positions and the key they were computed for. The engines then read the staged
 * rows through a copy of the arguments whose k / v point at them (ka_row16 reads f32). */
#define KA_STK(dec) ((dec) ? (int64_t)KA_DEC_TK : (int64_t)KA_TK)

static size_t ka_stage_bytes(int dec) {
    const size_t tk = (size_t)KA_STK(dec);
    return KA_ALIGN(4 * tk * KA_DK) + KA_ALIGN(4 * tk * KA_DV) + 2 * KA_ALIGN(4 * tk * (KA_DK / 2)) + 64;
}

/* cos / sin of pos * f[i], i < 16: angle and range reduction in double, minimax-style f32
 * polynomials on [-pi/4, pi/4] (Cephes sinf / cosf), ~1e-7 absolute */
KA_INLINE void ka_sincos16(double pos, const float *f, float *co, float *si) {
    __m256 rh[2];
    __m256i qh[2];
    for (int h = 0; h < 2; h++) {
        const __m512d x = _mm512_mul_pd(_mm512_set1_pd(pos), _mm512_cvtps_pd(_mm256_loadu_ps(f + 8 * h)));
        const __m512d k = _mm512_roundscale_pd(_mm512_mul_pd(x, _mm512_set1_pd(0.63661977236758134)), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        __m512d r = _mm512_fnmadd_pd(k, _mm512_set1_pd(1.5707963267948966), x);
        r = _mm512_fnmadd_pd(k, _mm512_set1_pd(6.123233995736766e-17), r);
        rh[h] = _mm512_cvtpd_ps(r);
        qh[h] = _mm512_cvtpd_epi32(k);
    }
    const __m512 r = _mm512_insertf32x8(_mm512_castps256_ps512(rh[0]), rh[1], 1);
    const __m512i q = _mm512_inserti32x8(_mm512_castsi256_si512(qh[0]), qh[1], 1);
    const __m512 z = _mm512_mul_ps(r, r);
    __m512 s = _mm512_fmadd_ps(_mm512_set1_ps(-1.9515295891e-4f), z, _mm512_set1_ps(8.3321608736e-3f));
    s = _mm512_fmadd_ps(s, z, _mm512_set1_ps(-1.6666654611e-1f));
    s = _mm512_fmadd_ps(_mm512_mul_ps(s, z), r, r);
    __m512 c = _mm512_fmadd_ps(_mm512_set1_ps(2.443315711809948e-5f), z, _mm512_set1_ps(-1.388731625493765e-3f));
    c = _mm512_fmadd_ps(c, z, _mm512_set1_ps(4.166664568298827e-2f));
    c = _mm512_fmadd_ps(_mm512_mul_ps(c, z), z, _mm512_fnmadd_ps(_mm512_set1_ps(0.5f), z, _mm512_set1_ps(1.0f)));
    const __mmask16 sw = _mm512_test_epi32_mask(q, _mm512_set1_epi32(1));
    const __mmask16 ns = _mm512_test_epi32_mask(q, _mm512_set1_epi32(2));
    const __mmask16 nc = _mm512_test_epi32_mask(_mm512_add_epi32(q, _mm512_set1_epi32(1)), _mm512_set1_epi32(2));
    __m512 so = _mm512_mask_blend_ps(sw, s, c), cof = _mm512_mask_blend_ps(sw, c, s);
    const __m512 neg = _mm512_set1_ps(-0.0f);
    so = _mm512_mask_xor_ps(so, ns, so, neg);
    cof = _mm512_mask_xor_ps(cof, nc, cof, neg);
    _mm512_storeu_ps(si, so);
    _mm512_storeu_ps(co, cof);
}

/* cos / sin rows [nk][DK/2] for positions pos0 .. pos0 + nk - 1: exact every 16 positions, the
 * positions in between by rotating with (cos f, sin f), so the error stays ~1e-6 */
static void ka_rope_table(int64_t pos0, int nk, const float *f, int rd, float *C, float *Sn) {
    for (int i = 0; i < rd / 2; i += 16) {
        float cf[16], sf[16];
        ka_sincos16(1.0, f + i, cf, sf);
        const __m512 vcf = _mm512_loadu_ps(cf), vsf = _mm512_loadu_ps(sf);
        __m512 c = _mm512_setzero_ps(), s = _mm512_setzero_ps();
        for (int t = 0; t < nk; t++) {
            float *ct = C + t * (KA_DK / 2) + i, *st = Sn + t * (KA_DK / 2) + i;
            if (t % 16 == 0) {
                ka_sincos16((double)(pos0 + t), f + i, ct, st);
                c = _mm512_loadu_ps(ct);
                s = _mm512_loadu_ps(st);
            } else {
                const __m512 c2 = _mm512_fmsub_ps(c, vcf, _mm512_mul_ps(s, vsf));
                s = _mm512_fmadd_ps(s, vcf, _mm512_mul_ps(c, vsf));
                c = c2;
                _mm512_storeu_ps(ct, c);
                _mm512_storeu_ps(st, s);
            }
        }
    }
}

/* rotate the f32 row k[DK] by the cos / sin row (rope_dim % 32 == 0) */
KA_INLINE void ka_rope_row(__m512 *k, const float *co, const float *si, int rd, int mode) {
    if (mode == KATTN_ROPE_NEOX) {
        const int hb = rd / 32; /* 16-lane blocks per half */
        for (int b = 0; b < hb; b++) {
            const __m512 c = _mm512_loadu_ps(co + 16 * b), s = _mm512_loadu_ps(si + 16 * b);
            const __m512 x0 = k[b], x1 = k[b + hb];
            k[b] = _mm512_fnmadd_ps(x1, s, _mm512_mul_ps(x0, c));
            k[b + hb] = _mm512_fmadd_ps(x0, s, _mm512_mul_ps(x1, c));
        }
    } else {
        const __m512i dup = _mm512_set_epi32(7, 7, 6, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 0, 0);
        const __m512 alt = _mm512_set_ps(1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1);
        for (int b = 0; b < rd / 16; b++) {
            const __m512 c = _mm512_permutexvar_ps(dup, _mm512_castps256_ps512(_mm256_loadu_ps(co + 8 * b)));
            const __m512 s = _mm512_mul_ps(alt, _mm512_permutexvar_ps(dup, _mm512_castps256_ps512(_mm256_loadu_ps(si + 8 * b))));
            k[b] = _mm512_fmadd_ps(_mm512_permute_ps(k[b], 0xB1), s, _mm512_mul_ps(k[b], c));
        }
    }
}

KA_INLINE __m512 ka_v16(const uint8_t *row, int i) {
    const uint8_t *b = row + (i >> 5) * (KA_KV == KATTN_KV_K4C_Q4 ? 18 : 34);
    const __m512 d = _mm512_set1_ps(_cvtsh_ss(*(const uint16_t *)b));
#if KA_KV == KATTN_KV_K4C_Q4
    __m512i x = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *)(b + 2)));
    x = (i & 31) ? _mm512_srli_epi32(x, 4) : _mm512_and_si512(x, _mm512_set1_epi32(15));
    return _mm512_mul_ps(d, _mm512_cvtepi32_ps(_mm512_sub_epi32(x, _mm512_set1_epi32(8))));
#else
    return _mm512_mul_ps(d, _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i *)(b + 2 + (i & 31))))));
#endif
}

KA_INLINE const uint8_t *ka_vrow_raw(const kattn_args *a, int64_t g, int64_t j) {
    return (const uint8_t *)a->v + j * a->v_s_tok + g * a->v_s_head;
}

static const kattn_args *ka_stage(const kattn_args *a, int64_t g, int64_t kv0, int nk, uint8_t *st, int dec, kattn_args *out) {
    const int64_t tk = KA_STK(dec);
    float *Kf = (float *)st;
    float *Vf = (float *)(st + KA_ALIGN(4 * tk * KA_DK));
    float *C = (float *)((uint8_t *)Vf + KA_ALIGN(4 * tk * KA_DV));
    float *Sn = (float *)((uint8_t *)C + KA_ALIGN(4 * tk * (KA_DK / 2)));
    int64_t *tag = (int64_t *)((uint8_t *)Sn + KA_ALIGN(4 * tk * (KA_DK / 2)));
    const int rd = a->rope_dim > KA_DK ? KA_DK : a->rope_dim;
    if (rd > 0) {
        const int64_t pos0 = a->k_pos0 + kv0;
        int64_t fbits = 0;
        memcpy(&fbits, a->rope_freq + rd / 2 - 1, 4);
        if (tag[0] != pos0 || tag[1] < nk || tag[2] != (int64_t)(uintptr_t)a->rope_freq || tag[3] != rd || tag[4] != fbits + 1) {
            ka_rope_table(pos0, nk, a->rope_freq, rd, C, Sn);
            tag[0] = pos0; tag[1] = nk; tag[2] = (int64_t)(uintptr_t)a->rope_freq; tag[3] = rd; tag[4] = fbits + 1;
        }
    }
    const int64_t nfull = a->n_kv / KATTN_K4C_GROUP * KATTN_K4C_GROUP;
    __m512 sc[KA_DK / 16], mn[KA_DK / 16];
    for (int v = 0; v < KA_DK / 16; v++) sc[v] = mn[v] = _mm512_setzero_ps();
    int64_t cur = -1;
    for (int t = 0; t < nk; t++) {
        const int64_t j = kv0 + t, jn = j + tk;
        __m512 kf[KA_DK / 16];
        if (j < nfull) {
            const uint8_t *blk = (const uint8_t *)a->k + (j / KATTN_K4C_GROUP) * a->k_s_tok + g * a->k_s_head;
            if (j / KATTN_K4C_GROUP != cur) {
                cur = j / KATTN_K4C_GROUP;
                for (int v = 0; v < KA_DK / 16; v++) {
                    sc[v] = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(blk + 32 * v)));
                    mn[v] = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(blk + 2 * KA_DK + 32 * v)));
                }
            }
            const uint8_t *qs = blk + 4 * KA_DK + (j % KATTN_K4C_GROUP) * (KA_DK / 2);
            for (int c = 0; c < KA_DK; c += 32) {
                const __m512i x = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *)(qs + c / 2)));
                kf[c / 16] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_and_si512(x, _mm512_set1_epi32(15))), sc[c / 16], mn[c / 16]);
                kf[c / 16 + 1] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_srli_epi32(x, 4)), sc[c / 16 + 1], mn[c / 16 + 1]);
            }
        } else {
            const uint16_t *tr = (const uint16_t *)((const uint8_t *)a->k_tail + (j % KATTN_K4C_GROUP) * a->kt_s_tok + g * a->kt_s_head);
            for (int v = 0; v < KA_DK / 16; v++) kf[v] = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(tr + 16 * v)));
        }
        if (jn < nfull) {
            const uint8_t *nb = (const uint8_t *)a->k + (jn / KATTN_K4C_GROUP) * a->k_s_tok + g * a->k_s_head;
            if (jn % KATTN_K4C_GROUP == 0)
                for (int c = 0; c < 4 * KA_DK; c += 64) _mm_prefetch((const char *)nb + c, _MM_HINT_T0);
            for (int c = 0; c < KA_DK / 2; c += 64) _mm_prefetch((const char *)nb + 4 * KA_DK + (jn % KATTN_K4C_GROUP) * (KA_DK / 2) + c, _MM_HINT_T0);
        }
        if (rd > 0) ka_rope_row(kf, C + t * (KA_DK / 2), Sn + t * (KA_DK / 2), rd, a->rope_mode);
        for (int v = 0; v < KA_DK / 16; v++) _mm512_store_ps(Kf + t * KA_DK + 16 * v, kf[v]);
        const uint8_t *vr = ka_vrow_raw(a, g, j);
        if (jn < a->n_kv)
            for (int c = 0; c < (KA_KV == KATTN_KV_K4C_Q4 ? 18 : 34) * (KA_DV / 32); c += 64) _mm_prefetch((const char *)vr + tk * a->v_s_tok + c, _MM_HINT_T0);
        for (int v = 0; v < KA_DV / 16; v++) _mm512_store_ps(Vf + t * KA_DV + 16 * v, ka_v16(vr, 16 * v));
    }
    *out = *a;
    out->k = (const void *)((uintptr_t)Kf - (uintptr_t)(kv0 * 4 * KA_DK));
    out->k_s_tok = 4 * KA_DK;
    out->k_s_head = 0;
    out->v = (const void *)((uintptr_t)Vf - (uintptr_t)(kv0 * 4 * KA_DV));
    out->v_s_tok = 4 * KA_DV;
    out->v_s_head = 0;
    return out;
}

/* The tile engine (and anything else that reads K / V through ka_krow / ka_vrow) sees staged rows:
 * the KA_TK-token tile holding j is staged on first touch. Tiles start at multiples of KA_TK (item
 * ranges are multiples of the chunk). kattn() points st at this thread's staging bytes per call. */
static __thread struct {
    uint8_t *st;
    const kattn_args *a;
    int64_t g, j0, j1;
    kattn_args sa;
} ka_tls;

static const kattn_args *ka_staged(const kattn_args *a, int64_t g, int64_t j) {
    if (a != ka_tls.a || g != ka_tls.g || j < ka_tls.j0 || j >= ka_tls.j1) {
        const int64_t j0 = j / KA_TK * KA_TK;
        const int nk = (int)KA_MIN((int64_t)KA_TK, a->n_kv - j0);
        ka_stage(a, g, j0, nk, ka_tls.st, 0, &ka_tls.sa);
        ka_tls.a = a;
        ka_tls.g = g;
        ka_tls.j0 = j0;
        ka_tls.j1 = j0 + nk;
    }
    return &ka_tls.sa;
}
#endif

/* Online-softmax update of one score row (log2 domain, n valid of `width` columns, masked
 * entries already -inf). Writes probabilities to p (f32) and returns the factor the row's
 * output accumulator must be multiplied by (1 if the max did not move). */
KA_INLINE float ka_softmax_row(float *s, int width, float *m, float *l, int masked) {
    __m512 mx = _mm512_set1_ps(-INFINITY);
    for (int j = 0; j < width; j += 16) mx = _mm512_max_ps(mx, _mm512_loadu_ps(s + j));
    const float tmax = _mm512_reduce_max_ps(mx);
    float alpha = 1.0f;
    if (tmax == -INFINITY) { /* nothing visible in this tile */
        for (int j = 0; j < width; j += 16) _mm512_storeu_ps(s + j, _mm512_setzero_ps());
        return alpha;
    }
    if (*m == -INFINITY || tmax > *m + KA_RESCALE) {
        alpha = *m == -INFINITY ? 0.0f : exp2f(*m - tmax);
        *l *= alpha;
        *m = tmax;
    }
    const __m512 vm = _mm512_set1_ps(*m), ninf = _mm512_set1_ps(-INFINITY);
    __m512 sum = _mm512_setzero_ps();
    for (int j = 0; j < width; j += 16) {
        const __m512 x = _mm512_loadu_ps(s + j);
        __m512 e = ka_exp2(_mm512_sub_ps(x, vm));
        if (masked) e = _mm512_maskz_mov_ps(_mm512_cmp_ps_mask(x, ninf, _CMP_NEQ_OQ), e);
        sum = _mm512_add_ps(sum, e);
        _mm512_storeu_ps(s + j, e);
    }
    *l += _mm512_reduce_add_ps(sum);
    return alpha;
}

KA_INLINE void ka_scale_row(float *o, int n, float alpha) {
    const __m512 va = _mm512_set1_ps(alpha);
    for (int d = 0; d < n; d += 16) _mm512_storeu_ps(o + d, _mm512_mul_ps(va, _mm512_loadu_ps(o + d)));
}

#define KA_MASK_SKIP 0  /* every entry -inf: the tile contributes nothing */
#define KA_MASK_NONE 1  /* every entry 0: nothing to add */
#define KA_MASK_MIXED 2

/* Classify the mask entries [kv0, kv0 + nk) of the item's query rows. */
static int ka_mask_scan(const kattn_args *a, const ka_item *it, int64_t kv0, int nk) {
    __mmask16 inf = 0xFFFF, zero = 0xFFFF;
    const __m256i vinf = _mm256_set1_epi16((short)0xFC00), vabs = _mm256_set1_epi16(0x7FFF);
    for (int64_t t = 0; t < it->nt; t++) {
        const uint16_t *mr = a->mask + (it->t0 + t) * a->mask_s_tok + kv0;
        for (int j = 0; j < nk; j += 16) {
            const __mmask16 live = nk - j >= 16 ? 0xFFFF : (__mmask16)((1u << (nk - j)) - 1);
            const __m256i x = _mm256_maskz_loadu_epi16(live, mr + j);
            inf &= _mm256_cmpeq_epi16_mask(x, vinf) | (__mmask16)~live;
            zero &= _mm256_testn_epi16_mask(x, vabs) | (__mmask16)~live;
        }
        if (inf != 0xFFFF && zero != 0xFFFF) return KA_MASK_MIXED;
    }
    return inf == 0xFFFF ? KA_MASK_SKIP : zero == 0xFFFF ? KA_MASK_NONE : KA_MASK_MIXED;
}

/* Add the fp16 mask (x log2 e) to the scores of a tile; columns >= nk become -inf. */
static void ka_mask_add(const kattn_args *a, const ka_item *it, float *S, int64_t sstride, int64_t kv0, int nk, int width) {
    const int64_t G = it->p->G;
    const __m512 l2e = _mm512_set1_ps(KA_LOG2E), ninf = _mm512_set1_ps(-INFINITY);
    for (int64_t t = 0; t < it->nt; t++) {
        const uint16_t *mr = a->mask + (it->t0 + t) * a->mask_s_tok + kv0;
        for (int j = 0; j < width; j += 16) {
            const __mmask16 live = nk - j >= 16 ? 0xFFFF : nk <= j ? 0 : (__mmask16)((1u << (nk - j)) - 1);
            const __m512 m = _mm512_mask_mul_ps(ninf, live, _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(live, mr + j)), l2e);
            for (int64_t r = t * G; r < (t + 1) * G; r++) {
                float *s = S + r * sstride + j;
                _mm512_storeu_ps(s, _mm512_add_ps(_mm512_loadu_ps(s), m));
            }
        }
    }
}

/* ---------------------------------------------------------------- decode (row) engine */
#if KA_DEC_ROWS > 0

#define KA_NVK (KA_DK / 16)
#define KA_NVV (KA_DV / 16)

KA_INLINE void ka_dec_run(ka_item *it, uint8_t *scratch, const int NR) {
    const kattn_args *a = it->a;
    float *S = (float *)scratch;                                                          /* [NR][KA_DEC_TK] */
    float *acc = (float *)(scratch + KA_ALIGN(sizeof(float) * KA_DEC_ROWS * KA_DEC_TK)); /* [NR][16][16] */
    float *Qall = acc + KA_DEC_ROWS * 256;                                                /* [hb][8][DK] scaled */
    const float qs = a->scale * KA_LOG2E;
    for (int64_t hl = 0; hl < it->ng; hl++) {
        for (int r = 0; r < NR; r++) {
            const float *q = ka_qrow_g(it, it->g + hl, r);
            float *Q = Qall + (hl * KA_DEC_ROWS + r) * KA_DK;
            for (int d = 0; d < KA_DK; d += 16) _mm512_store_ps(Q + d, _mm512_mul_ps(_mm512_set1_ps(qs), _mm512_loadu_ps(q + d)));
            it->m[hl * KA_DEC_ROWS + r] = -INFINITY;
            it->l[hl * KA_DEC_ROWS + r] = 0.0f;
            memset(it->O + (hl * KA_DEC_ROWS + r) * KA_DV, 0, sizeof(float) * KA_DV);
        }
    }
    for (int64_t kv0 = it->k0; kv0 < it->k1; kv0 += KA_DEC_TK) {
        const int nk = (int)KA_MIN((int64_t)KA_DEC_TK, it->k1 - kv0);
        const int width = (nk + 15) & ~15;
        const int mclass = a->mask ? ka_mask_scan(a, it, kv0, nk) : KA_MASK_NONE;
        if (mclass == KA_MASK_SKIP) continue;
        for (int64_t hl = 0; hl < it->ng; hl++) {
            const int64_t g = it->g + hl;
            const float *Q = Qall + hl * KA_DEC_ROWS * KA_DK;
            float *O = it->O + hl * KA_DEC_ROWS * KA_DV, *m = it->m + hl * KA_DEC_ROWS, *l = it->l + hl * KA_DEC_ROWS;
            /* scores: one K row at a time, NR dot products, reduced 16 tokens at a time */
            for (int j0 = 0; j0 < width; j0 += 16) {
                for (int jj = 0; jj < 16; jj++) {
                    const int slot = ((jj & 3) << 2) | (jj >> 2);
                    if (j0 + jj >= nk) {
                        for (int r = 0; r < NR; r++) _mm512_store_ps(acc + r * 256 + slot * 16, _mm512_setzero_ps());
                        continue;
                    }
                    const uint8_t *kr = ka_krow(a, g, kv0 + j0 + jj);
                    for (int c = 0; c < KA_ROW_BYTES(KA_DK); c += 64) _mm_prefetch((const char *)kr + KA_DEC_TK * a->k_s_tok + c, _MM_HINT_T0);
                    __m512 kf[KA_NVK];
                    for (int v = 0; v < KA_NVK; v++) kf[v] = ka_row16(kr, 16 * v);
                    for (int r = 0; r < NR; r++) {
                        const float *q = Q + r * KA_DK;
                        __m512 s0 = _mm512_mul_ps(_mm512_load_ps(q), kf[0]);
                        __m512 s1 = _mm512_mul_ps(_mm512_load_ps(q + 16), kf[1]);
                        for (int v = 2; v < KA_NVK; v += 2) {
                            s0 = _mm512_fmadd_ps(_mm512_load_ps(q + 16 * v), kf[v], s0);
                            s1 = _mm512_fmadd_ps(_mm512_load_ps(q + 16 * v + 16), kf[v + 1], s1);
                        }
                        _mm512_store_ps(acc + r * 256 + slot * 16, _mm512_add_ps(s0, s1));
                    }
                }
                for (int r = 0; r < NR; r++) _mm512_storeu_ps(S + r * KA_DEC_TK + j0, ka_hsum16(acc + r * 256));
            }
            int masked = nk < width;
            for (int r = 0; r < NR; r++)
                for (int j = nk; j < width; j++) S[r * KA_DEC_TK + j] = -INFINITY;
            if (a->causal && kv0 + nk - 1 > a->q_pos0 + it->t0) {
                masked = 1;
                for (int r = 0; r < NR; r++) {
                    const int64_t lim = a->q_pos0 + it->t0 + r / it->p->G - kv0; /* last visible column */
                    for (int j = (int)KA_MAX((int64_t)0, lim + 1); j < nk; j++) S[r * KA_DEC_TK + j] = -INFINITY;
                }
            }
            if (mclass == KA_MASK_MIXED) {
                masked = 1;
                ka_mask_add(a, it, S, KA_DEC_TK, kv0, nk, width);
            }
            for (int r = 0; r < NR; r++) {
                const float alpha = ka_softmax_row(S + r * KA_DEC_TK, width, &m[r], &l[r], masked);
                if (alpha != 1.0f) ka_scale_row(O + r * KA_DV, KA_DV, alpha);
            }
            /* O += P V, output block of NR rows x DC vectors in registers */
#define KA_DC (KA_MIN(KA_NVV, (NR <= 2 ? 8 : NR <= 4 ? 4 : 2)))
            for (int d0 = 0; d0 < KA_NVV; d0 += KA_DC) {
                __m512 o[KA_DEC_ROWS][8];
                for (int r = 0; r < NR; r++)
                    for (int c = 0; c < KA_DC; c++) o[r][c] = _mm512_loadu_ps(O + r * KA_DV + 16 * (d0 + c));
                for (int j = 0; j < nk; j++) {
                    const uint8_t *vr = ka_vrow(a, g, kv0 + j);
                    if (d0 == 0)
                        for (int c = 0; c < KA_ROW_BYTES(KA_DV); c += 64) _mm_prefetch((const char *)vr + KA_DEC_TK * a->v_s_tok + c, _MM_HINT_T0);
                    __m512 vf[8];
                    for (int c = 0; c < KA_DC; c++) vf[c] = ka_row16(vr, 16 * (d0 + c));
                    for (int r = 0; r < NR; r++) {
                        const __m512 pj = _mm512_set1_ps(S[r * KA_DEC_TK + j]);
                        for (int c = 0; c < KA_DC; c++) o[r][c] = _mm512_fmadd_ps(pj, vf[c], o[r][c]);
                    }
                }
                for (int r = 0; r < NR; r++)
                    for (int c = 0; c < KA_DC; c++) _mm512_storeu_ps(O + r * KA_DV + 16 * (d0 + c), o[r][c]);
            }
#undef KA_DC
        }
    }
}

#if KA_PRE_ROPE
/* The decode engine for pre-RoPE K: K is dequantized and rotated in registers (scales of a
 * 32-token group loaded once per 16 tokens) and V read from its Q4_0 / Q8_0 rows, with no
 * staging pass; scores, masking, softmax and the PV block are as in ka_dec_run. */
KA_INLINE void ka_dec_run_k4c(ka_item *it, uint8_t *scratch, const int NR) {
    const kattn_args *a = it->a;
    float *S = (float *)scratch;
    float *acc = (float *)(scratch + KA_ALIGN(sizeof(float) * KA_DEC_ROWS * KA_DEC_TK));
    float *Qall = acc + KA_DEC_ROWS * 256;
    float *Cs = (float *)(scratch + it->p->thr_bytes - ka_stage_bytes(1)), *Sn = Cs + KA_DEC_TK * (KA_DK / 2);
    const float qs = a->scale * KA_LOG2E;
    const int rd = a->rope_dim > KA_DK ? KA_DK : a->rope_dim;
    const int64_t nfull = a->n_kv / KATTN_K4C_GROUP * KATTN_K4C_GROUP;
    const int vbytes = (KA_KV == KATTN_KV_K4C_Q4 ? 18 : 34) * (KA_DV / 32);
    for (int64_t hl = 0; hl < it->ng; hl++) {
        for (int r = 0; r < NR; r++) {
            const float *q = ka_qrow_g(it, it->g + hl, r);
            float *Q = Qall + (hl * KA_DEC_ROWS + r) * KA_DK;
            for (int d = 0; d < KA_DK; d += 16) _mm512_store_ps(Q + d, _mm512_mul_ps(_mm512_set1_ps(qs), _mm512_loadu_ps(q + d)));
            it->m[hl * KA_DEC_ROWS + r] = -INFINITY;
            it->l[hl * KA_DEC_ROWS + r] = 0.0f;
            memset(it->O + (hl * KA_DEC_ROWS + r) * KA_DV, 0, sizeof(float) * KA_DV);
        }
    }
    for (int64_t kv0 = it->k0; kv0 < it->k1; kv0 += KA_DEC_TK) {
        const int nk = (int)KA_MIN((int64_t)KA_DEC_TK, it->k1 - kv0);
        const int width = (nk + 15) & ~15;
        const int mclass = a->mask ? ka_mask_scan(a, it, kv0, nk) : KA_MASK_NONE;
        if (mclass == KA_MASK_SKIP) continue;
        if (rd > 0) ka_rope_table(a->k_pos0 + kv0, nk, a->rope_freq, rd, Cs, Sn);
        for (int64_t hl = 0; hl < it->ng; hl++) {
            const int64_t g = it->g + hl;
            const float *Q = Qall + hl * KA_DEC_ROWS * KA_DK;
            float *O = it->O + hl * KA_DEC_ROWS * KA_DV, *m = it->m + hl * KA_DEC_ROWS, *l = it->l + hl * KA_DEC_ROWS;
            for (int j0 = 0; j0 < width; j0 += 16) {
                const int64_t jb = kv0 + j0; /* a 16-token block never straddles nfull (kv0 % 64 == 0) */
                const uint8_t *blk = (const uint8_t *)a->k + (jb / KATTN_K4C_GROUP) * a->k_s_tok + g * a->k_s_head;
                __m512 sc[KA_DK / 16], mn[KA_DK / 16];
                for (int v = 0; v < KA_DK / 16; v++) {
                    sc[v] = jb < nfull ? _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(blk + 32 * v))) : _mm512_setzero_ps();
                    mn[v] = jb < nfull ? _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(blk + 2 * KA_DK + 32 * v))) : _mm512_setzero_ps();
                }
                if (jb + KA_DEC_TK < nfull && (jb + KA_DEC_TK) % KATTN_K4C_GROUP == 0)
                    for (int c = 0; c < 4 * KA_DK; c += 64)
                        _mm_prefetch((const char *)blk + KA_DEC_TK / KATTN_K4C_GROUP * a->k_s_tok + c, _MM_HINT_T0);
                for (int jj = 0; jj < 16; jj++) {
                    const int slot = ((jj & 3) << 2) | (jj >> 2);
                    const int64_t j = jb + jj;
                    if (j0 + jj >= nk) {
                        for (int r = 0; r < NR; r++) _mm512_store_ps(acc + r * 256 + slot * 16, _mm512_setzero_ps());
                        continue;
                    }
                    __m512 kf[KA_NVK];
                    if (jb < nfull) {
                        const uint8_t *qr = blk + 4 * KA_DK + (j % KATTN_K4C_GROUP) * (KA_DK / 2);
                        if (j + KA_DEC_TK < nfull)
                            for (int c = 0; c < KA_DK / 2; c += 64)
                                _mm_prefetch((const char *)qr + KA_DEC_TK / KATTN_K4C_GROUP * a->k_s_tok + c, _MM_HINT_T0);
                        for (int c = 0; c < KA_DK; c += 32) {
                            const __m512i x = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *)(qr + c / 2)));
                            kf[c / 16] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_and_si512(x, _mm512_set1_epi32(15))), sc[c / 16], mn[c / 16]);
                            kf[c / 16 + 1] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_srli_epi32(x, 4)), sc[c / 16 + 1], mn[c / 16 + 1]);
                        }
                    } else {
                        const uint16_t *tr = (const uint16_t *)((const uint8_t *)a->k_tail + (j % KATTN_K4C_GROUP) * a->kt_s_tok + g * a->kt_s_head);
                        for (int v = 0; v < KA_NVK; v++) kf[v] = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(tr + 16 * v)));
                    }
                    if (rd > 0) ka_rope_row(kf, Cs + (j0 + jj) * (KA_DK / 2), Sn + (j0 + jj) * (KA_DK / 2), rd, a->rope_mode);
                    for (int r = 0; r < NR; r++) {
                        const float *q = Q + r * KA_DK;
                        __m512 s0 = _mm512_mul_ps(_mm512_load_ps(q), kf[0]);
                        __m512 s1 = _mm512_mul_ps(_mm512_load_ps(q + 16), kf[1]);
                        for (int v = 2; v < KA_NVK; v += 2) {
                            s0 = _mm512_fmadd_ps(_mm512_load_ps(q + 16 * v), kf[v], s0);
                            s1 = _mm512_fmadd_ps(_mm512_load_ps(q + 16 * v + 16), kf[v + 1], s1);
                        }
                        _mm512_store_ps(acc + r * 256 + slot * 16, _mm512_add_ps(s0, s1));
                    }
                }
                for (int r = 0; r < NR; r++) _mm512_storeu_ps(S + r * KA_DEC_TK + j0, ka_hsum16(acc + r * 256));
            }
            int masked = nk < width;
            for (int r = 0; r < NR; r++)
                for (int j = nk; j < width; j++) S[r * KA_DEC_TK + j] = -INFINITY;
            if (a->causal && kv0 + nk - 1 > a->q_pos0 + it->t0) {
                masked = 1;
                for (int r = 0; r < NR; r++) {
                    const int64_t lim = a->q_pos0 + it->t0 + r / it->p->G - kv0;
                    for (int j = (int)KA_MAX((int64_t)0, lim + 1); j < nk; j++) S[r * KA_DEC_TK + j] = -INFINITY;
                }
            }
            if (mclass == KA_MASK_MIXED) {
                masked = 1;
                ka_mask_add(a, it, S, KA_DEC_TK, kv0, nk, width);
            }
            for (int r = 0; r < NR; r++) {
                const float alpha = ka_softmax_row(S + r * KA_DEC_TK, width, &m[r], &l[r], masked);
                if (alpha != 1.0f) ka_scale_row(O + r * KA_DV, KA_DV, alpha);
            }
#define KA_DC (KA_MIN(KA_NVV, (NR <= 2 ? 8 : NR <= 4 ? 4 : 2)))
            for (int d0 = 0; d0 < KA_NVV; d0 += KA_DC) {
                __m512 o[KA_DEC_ROWS][8];
                for (int r = 0; r < NR; r++)
                    for (int c = 0; c < KA_DC; c++) o[r][c] = _mm512_loadu_ps(O + r * KA_DV + 16 * (d0 + c));
                for (int j = 0; j < nk; j++) {
                    const uint8_t *vr = ka_vrow_raw(a, g, kv0 + j);
                    if (d0 == 0 && kv0 + j + KA_DEC_TK < a->n_kv)
                        for (int c = 0; c < vbytes; c += 64) _mm_prefetch((const char *)vr + KA_DEC_TK * a->v_s_tok + c, _MM_HINT_T0);
                    __m512 vf[8];
#if KA_KV == KATTN_KV_K4C_Q4
                    for (int c = 0; c < KA_DC; c += 2) { /* one Q4_0 block = two 16-lane vectors */
                        const uint8_t *b = vr + (d0 + c) / 2 * 18;
                        const __m512 d = _mm512_set1_ps(_cvtsh_ss(*(const uint16_t *)b)), d8 = _mm512_mul_ps(d, _mm512_set1_ps(-8.0f));
                        const __m512i x = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *)(b + 2)));
                        vf[c] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_and_si512(x, _mm512_set1_epi32(15))), d, d8);
                        vf[c + 1] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_srli_epi32(x, 4)), d, d8);
                    }
#else
                    for (int c = 0; c < KA_DC; c++) vf[c] = ka_v16(vr, 16 * (d0 + c));
#endif
                    for (int r = 0; r < NR; r++) {
                        const __m512 pj = _mm512_set1_ps(S[r * KA_DEC_TK + j]);
                        for (int c = 0; c < KA_DC; c++) o[r][c] = _mm512_fmadd_ps(pj, vf[c], o[r][c]);
                    }
                }
                for (int r = 0; r < NR; r++)
                    for (int c = 0; c < KA_DC; c++) _mm512_storeu_ps(O + r * KA_DV + 16 * (d0 + c), o[r][c]);
            }
#undef KA_DC
        }
    }
}
#define KA_DEC_RUN ka_dec_run_k4c
#else
#define KA_DEC_RUN ka_dec_run
#endif

static void ka_dec(ka_item *it, uint8_t *scratch) {
    switch (it->p->R) {
#define C(n) case n: KA_DEC_RUN(it, scratch, n < KA_DEC_ROWS ? n : KA_DEC_ROWS); break;
        C(1) C(2) C(3) C(4) C(5) C(6) C(7) C(8)
#undef C
    }
}
#endif

/* ---------------------------------------------------------------- tile engine */

typedef struct {
    void *Q;     /* [Rpad][DK]  bf16 or f32, pre-scaled */
    float *S;    /* [Rpad][TK] */
    uint16_t *P; /* [Rpad][TK]  bf16 probabilities */
    void *Kp;    /* bf16: [TK/16][DK/2][16] pairs; f32: [DK][TK] */
    void *Vp;    /* bf16: [DV/16][TK/2][16] pairs; f32: [TK][DV] */
    float *mu;   /* [DK] KA_KCENTER mean */
} ka_tile_buf;

#if KA_ENGINE != 0
KA_INLINE __m512i ka_bf16x32(__m512 lo, __m512 hi) { return (__m512i)_mm512_cvtne2ps_pbh(hi, lo); }

/* Per-channel mean of K rows [kv0, kv0 + nk), kept only where it exceeds half the channel's spread
 * (|mu| > 0.5 sd): only offset-dominated channels gain from centering, and zero-mean channels then
 * round exactly as without it. */
static __attribute__((unused)) void ka_kmean(const kattn_args *a, int64_t g, int64_t kv0, int nk, float *mu) {
    const __m512 inv = _mm512_set1_ps(1.0f / (float)nk);
    for (int d = 0; d < KA_DK; d += 16) {
        __m512 s = _mm512_setzero_ps(), ss = _mm512_setzero_ps();
        for (int t = 0; t < nk; t++) {
            const __m512 x = ka_row16(ka_krow(a, g, kv0 + t), d);
            s = _mm512_add_ps(s, x);
            ss = _mm512_fmadd_ps(x, x, ss);
        }
        const __m512 m = _mm512_mul_ps(s, inv), m2 = _mm512_mul_ps(m, m);
        const __m512 var = _mm512_fmsub_ps(ss, inv, m2);
        _mm512_store_ps(mu + d, _mm512_maskz_mov_ps(_mm512_cmp_ps_mask(_mm512_mul_ps(_mm512_set1_ps(4.0f), m2), var, _CMP_GT_OQ), m));
    }
}

/* 16 K rows [kv0, kv0 + 16) (nk valid, the rest zero) as bf16 pairs [DK/2][16] */
static __attribute__((unused)) void ka_pack_k16(const kattn_args *a, int64_t g, int64_t kv0, int nk, uint32_t *dst, const float *mu) {
    for (int d0 = 0; d0 < KA_DK; d0 += 32) {
        __m512i r[16];
#if KA_KCENTER
        const __m512 m0 = _mm512_load_ps(mu + d0), m1 = _mm512_load_ps(mu + d0 + 16);
#else
        (void)mu;
#endif
        for (int t = 0; t < 16; t++) {
            if (t < nk) {
                const uint8_t *kr = ka_krow(a, g, kv0 + t);
                __m512 x0 = ka_row16(kr, d0), x1 = ka_row16(kr, d0 + 16);
#if KA_KCENTER
                x0 = _mm512_sub_ps(x0, m0);
                x1 = _mm512_sub_ps(x1, m1);
#endif
                r[t] = ka_bf16x32(x0, x1);
            } else {
                r[t] = _mm512_setzero_si512();
            }
        }
        ka_transpose16(r);
        for (int i = 0; i < 16; i++) _mm512_store_si512(dst + (size_t)(d0 / 2 + i) * 16, r[i]);
    }
}

static __attribute__((unused)) void ka_pack_k(const kattn_args *a, int64_t g, int64_t kv0, int nk, uint32_t *Kp, const float *mu) {
    for (int t0 = 0; t0 < KA_TK; t0 += 16) ka_pack_k16(a, g, kv0 + t0, nk - t0, Kp + (size_t)(t0 / 16) * (KA_DK / 2) * 16, mu);
}

/* ntok V rows [kv0, kv0 + ntok) (nk valid, the rest zero) as bf16 pairs [DV/16][vs][16] */
static void ka_pack_vn(const kattn_args *a, int64_t g, int64_t kv0, int nk, int ntok, uint32_t *Vp, int64_t vs) {
    const __m512i idx = _mm512_set_epi16(31, 15, 30, 14, 29, 13, 28, 12, 27, 11, 26, 10, 25, 9, 24, 8,
                                         23, 7, 22, 6, 21, 5, 20, 4, 19, 3, 18, 2, 17, 1, 16, 0);
    for (int t = 0; t < ntok; t += 2) {
        const uint8_t *va = t < nk ? ka_vrow(a, g, kv0 + t) : NULL;
        const uint8_t *vb = t + 1 < nk ? ka_vrow(a, g, kv0 + t + 1) : NULL;
        for (int d0 = 0; d0 < KA_DV; d0 += 16) {
            const __m512 x = va ? ka_row16(va, d0) : _mm512_setzero_ps();
            const __m512 y = vb ? ka_row16(vb, d0) : _mm512_setzero_ps();
            const __m512i pr = _mm512_permutexvar_epi16(idx, ka_bf16x32(x, y)); /* (x0,y0,x1,y1,...) */
            _mm512_store_si512(Vp + ((size_t)(d0 / 16) * vs + t / 2) * 16, pr);
        }
    }
}

static void ka_pack_v(const kattn_args *a, int64_t g, int64_t kv0, int nk, uint32_t *Vp) { ka_pack_vn(a, g, kv0, nk, KA_TK, Vp, KA_TK / 2); }

static __attribute__((unused)) void ka_pack_q(const ka_item *it, uint16_t *Q) {
    const float qs = it->a->scale * KA_LOG2E;
    const __m512 vs = _mm512_set1_ps(qs);
    for (int64_t r = 0; r < it->p->Rpad; r++) {
        uint16_t *dst = Q + r * KA_DK;
        if (r < it->nt * it->p->G) {
            const float *q = ka_qrow(it, r);
            for (int d = 0; d < KA_DK; d += 32)
                _mm512_store_si512(dst + d, ka_bf16x32(_mm512_mul_ps(vs, _mm512_loadu_ps(q + d)), _mm512_mul_ps(vs, _mm512_loadu_ps(q + d + 16))));
        } else {
            memset(dst, 0, 2 * KA_DK);
        }
    }
}

/* 2^x to 7.5e-5 relative (degree-3 relative-minimax on [-0.5, 0.5]): far below the bf16 rounding of P.
 * No clamp: unmasked tiles hold finite scores (scalef flushes very negative n to 0), and the NaN that
 * -inf produces only appears in masked tiles, whose -inf lanes the caller zeroes. */
KA_INLINE __m512 ka_exp2_lo(__m512 x) {
    const __m512 n = _mm512_roundscale_ps(x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    const __m512 f = _mm512_sub_ps(x, n);
    __m512 p = _mm512_set1_ps(5.517134442925453e-2f);
    p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(2.4261033535003662e-1f));
    p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(6.932609677314758e-1f));
    p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(9.999281167984009e-1f));
    return _mm512_scalef_ps(p, n);
}

/* ka_softmax_row writing the probabilities as bf16 to p (width = KA_TK); l sums the rounded values */
KA_INLINE float ka_softmax_row_bf16(const float *s, uint16_t *p, float *m, float *l, int masked) {
    __m512 mx = _mm512_set1_ps(-INFINITY);
    for (int j = 0; j < KA_TK; j += 16) mx = _mm512_max_ps(mx, _mm512_loadu_ps(s + j));
    const float tmax = _mm512_reduce_max_ps(mx);
    float alpha = 1.0f;
    if (tmax == -INFINITY) {
        for (int j = 0; j < KA_TK; j += 32) _mm512_store_si512(p + j, _mm512_setzero_si512());
        return alpha;
    }
    if (*m == -INFINITY || tmax > *m + KA_RESCALE) {
        alpha = *m == -INFINITY ? 0.0f : exp2f(*m - tmax);
        *l *= alpha;
        *m = tmax;
    }
    const __m512 vm = _mm512_set1_ps(*m), ninf = _mm512_set1_ps(-INFINITY);
    __m512 sum = _mm512_setzero_ps();
    for (int j = 0; j < KA_TK; j += 32) {
        const __m512 x0 = _mm512_loadu_ps(s + j), x1 = _mm512_loadu_ps(s + j + 16);
        __m512 e0 = ka_exp2_lo(_mm512_sub_ps(x0, vm)), e1 = ka_exp2_lo(_mm512_sub_ps(x1, vm));
        if (masked) {
            e0 = _mm512_maskz_mov_ps(_mm512_cmp_ps_mask(x0, ninf, _CMP_NEQ_OQ), e0);
            e1 = _mm512_maskz_mov_ps(_mm512_cmp_ps_mask(x1, ninf, _CMP_NEQ_OQ), e1);
        }
        const __m512i pb = ka_bf16x32(e0, e1);
        _mm512_store_si512(p + j, pb);
        /* bf16 -> f32 is a 16-bit shift: sum exactly what PV will multiply */
        sum = _mm512_add_ps(sum, _mm512_castsi512_ps(_mm512_slli_epi32(pb, 16)));
        sum = _mm512_add_ps(sum, _mm512_castsi512_ps(_mm512_and_si512(pb, _mm512_set1_epi32((int)0xFFFF0000))));
    }
    *l += _mm512_reduce_add_ps(sum);
    return alpha;
}
#endif

#if KA_ENGINE == 1
/* S[R][TK] = Q Kᵀ with vdpbf16ps: 4 rows x 64 tokens per register block */
static void ka_qk(const uint16_t *Q, const uint32_t *Kp, float *S, int64_t R) {
    for (int64_t r0 = 0; r0 < R; r0 += 4) {
        for (int c0 = 0; c0 < KA_TK; c0 += 64) {
            __m512 acc[4][4];
            for (int i = 0; i < 4; i++)
                for (int c = 0; c < 4; c++) acc[i][c] = _mm512_setzero_ps();
            const uint32_t *q32 = (const uint32_t *)(Q + r0 * KA_DK);
            for (int kp = 0; kp < KA_DK / 2; kp++) {
                __m512bh kv[4];
                for (int c = 0; c < 4; c++) kv[c] = (__m512bh)_mm512_load_si512(Kp + ((size_t)(c0 / 16 + c) * (KA_DK / 2) + kp) * 16);
                for (int i = 0; i < 4; i++) {
                    const __m512bh qb = (__m512bh)_mm512_set1_epi32((int)q32[i * (KA_DK / 2) + kp]);
                    for (int c = 0; c < 4; c++) acc[i][c] = _mm512_dpbf16_ps(acc[i][c], qb, kv[c]);
                }
            }
            for (int i = 0; i < 4; i++)
                for (int c = 0; c < 4; c++) _mm512_storeu_ps(S + (r0 + i) * KA_TK + c0 + 16 * c, acc[i][c]);
        }
    }
}

/* O[R][DV] += P V: 4 rows x 64 dims per register block; vs = pair rows per 16-dim group of Vp */
static void ka_pv(const uint16_t *P, const uint32_t *Vp, float *O, int64_t R, int64_t vs) {
    for (int64_t r0 = 0; r0 < R; r0 += 4) {
        for (int d0 = 0; d0 < KA_DV; d0 += 64) {
            __m512 acc[4][4];
            for (int i = 0; i < 4; i++)
                for (int c = 0; c < 4; c++) acc[i][c] = _mm512_loadu_ps(O + (r0 + i) * KA_DV + d0 + 16 * c);
            const uint32_t *p32 = (const uint32_t *)(P + r0 * KA_TK);
            for (int kp = 0; kp < KA_TK / 2; kp++) {
                __m512bh vv[4];
                for (int c = 0; c < 4; c++) vv[c] = (__m512bh)_mm512_load_si512(Vp + ((size_t)(d0 / 16 + c) * vs + kp) * 16);
                for (int i = 0; i < 4; i++) {
                    const __m512bh pb = (__m512bh)_mm512_set1_epi32((int)p32[i * (KA_TK / 2) + kp]);
                    for (int c = 0; c < 4; c++) acc[i][c] = _mm512_dpbf16_ps(acc[i][c], pb, vv[c]);
                }
            }
            for (int i = 0; i < 4; i++)
                for (int c = 0; c < 4; c++) _mm512_storeu_ps(O + (r0 + i) * KA_DV + d0 + 16 * c, acc[i][c]);
        }
    }
}
#endif

#if KA_ENGINE == 2
typedef struct __attribute__((aligned(64))) {
    uint8_t palette, start_row, reserved[14];
    uint16_t colsb[16];
    uint8_t rows[16];
} ka_tilecfg;

static atomic_int ka_amx_perm;

static int ka_amx_begin(ka_tilecfg *saved) {
    if (!atomic_load_explicit(&ka_amx_perm, memory_order_acquire)) {
        syscall(SYS_arch_prctl, 0x1023 /* ARCH_REQ_XCOMP_PERM */, 18 /* XFEATURE_XTILEDATA */);
        atomic_store_explicit(&ka_amx_perm, 1, memory_order_release);
    }
    _tile_storeconfig(saved);
    ka_tilecfg c;
    memset(&c, 0, sizeof c);
    c.palette = 1;
    for (int i = 0; i < 8; i++) {
        c.rows[i] = 16;
        c.colsb[i] = 64;
    }
    _tile_loadconfig(&c);
    return 0;
}

static void ka_amx_end(const ka_tilecfg *saved) {
    if (saved->palette) _tile_loadconfig(saved);
    else _tile_release();
}

/*
 * Some KVM guests (seen on kernel 6.12 / Firecracker) drop AMX tile data when the thread is
 * descheduled: a 32x32 block that spans a preemption comes back as a partial or zero sum.
 * Every observed corruption coincided with a >= 16K-cycle gap and none with shorter gaps, so
 * each block is timed and recomputed when it took longer than KA_AMX_GUARD cycles. Blocks
 * never accumulate into live output: PV sums into zeroed tiles and is added to O afterwards.
 */
#ifndef KA_AMX_GUARD
#define KA_AMX_GUARD 8192
#endif

static inline int ka_amx_slow(uint64_t t0) {
    unsigned aux;
    return __rdtscp(&aux) - t0 > KA_AMX_GUARD;
}

/* S = Q Kᵀ: 32 rows x 32 tokens per step (tiles 0-3 accumulate, 4-5 Q, 6-7 K) */
static __attribute__((unused)) void ka_qk(const uint16_t *Q, const uint32_t *Kp, float *S, int64_t R) {
    for (int64_t r0 = 0; r0 < R; r0 += 32) {
        for (int c0 = 0; c0 < KA_TK; c0 += 32) {
            uint64_t t0;
            do {
                t0 = __rdtsc();
                _tile_zero(0); _tile_zero(1); _tile_zero(2); _tile_zero(3);
                for (int k = 0; k < KA_DK; k += 32) {
                    _tile_loadd(4, Q + r0 * KA_DK + k, 2 * KA_DK);
                    _tile_loadd(5, Q + (r0 + 16) * KA_DK + k, 2 * KA_DK);
                    _tile_loadd(6, Kp + ((size_t)(c0 / 16) * (KA_DK / 2) + k / 2) * 16, 64);
                    _tile_loadd(7, Kp + ((size_t)(c0 / 16 + 1) * (KA_DK / 2) + k / 2) * 16, 64);
                    _tile_dpbf16ps(0, 4, 6);
                    _tile_dpbf16ps(1, 4, 7);
                    _tile_dpbf16ps(2, 5, 6);
                    _tile_dpbf16ps(3, 5, 7);
                }
                _tile_stored(0, S + r0 * KA_TK + c0, 4 * KA_TK);
                _tile_stored(1, S + r0 * KA_TK + c0 + 16, 4 * KA_TK);
                _tile_stored(2, S + (r0 + 16) * KA_TK + c0, 4 * KA_TK);
                _tile_stored(3, S + (r0 + 16) * KA_TK + c0 + 16, 4 * KA_TK);
            } while (ka_amx_slow(t0));
        }
    }
}

/* O += P V: 32 rows x 32 dims per step; vs = pair rows per 16-dim group of Vp */
static void ka_pv(const uint16_t *P, const uint32_t *Vp, float *O, int64_t R, int64_t vs) {
    float T[32 * 32] __attribute__((aligned(64)));
    for (int64_t r0 = 0; r0 < R; r0 += 32) {
        for (int d0 = 0; d0 < KA_DV; d0 += 32) {
            uint64_t t0;
            do {
                t0 = __rdtsc();
                _tile_zero(0); _tile_zero(1); _tile_zero(2); _tile_zero(3);
                for (int k = 0; k < KA_TK; k += 32) {
                    _tile_loadd(4, P + r0 * KA_TK + k, 2 * KA_TK);
                    _tile_loadd(5, P + (r0 + 16) * KA_TK + k, 2 * KA_TK);
                    _tile_loadd(6, Vp + ((size_t)(d0 / 16) * vs + k / 2) * 16, 64);
                    _tile_loadd(7, Vp + ((size_t)(d0 / 16 + 1) * vs + k / 2) * 16, 64);
                    _tile_dpbf16ps(0, 4, 6);
                    _tile_dpbf16ps(1, 4, 7);
                    _tile_dpbf16ps(2, 5, 6);
                    _tile_dpbf16ps(3, 5, 7);
                }
                _tile_stored(0, T, 128);
                _tile_stored(1, T + 16, 128);
                _tile_stored(2, T + 16 * 32, 128);
                _tile_stored(3, T + 16 * 32 + 16, 128);
            } while (ka_amx_slow(t0));
            for (int i = 0; i < 32; i++) {
                float *o = O + (r0 + i) * KA_DV + d0;
                _mm512_storeu_ps(o, _mm512_add_ps(_mm512_loadu_ps(o), _mm512_load_ps(T + 32 * i)));
                _mm512_storeu_ps(o + 16, _mm512_add_ps(_mm512_loadu_ps(o + 16), _mm512_load_ps(T + 32 * i + 16)));
            }
        }
    }
}

#if KA_QK8
/* Scaled q rows -> int8, one scale per row: Q8 [Rpad][DK], sq [Rpad] */
static void ka_pack_q8(const ka_item *it, int8_t *Q8, float *sq) {
    const __m512 vs = _mm512_set1_ps(it->a->scale * KA_LOG2E);
    for (int64_t r = 0; r < it->p->Rpad; r++) {
        int8_t *dst = Q8 + r * KA_DK;
        if (r >= it->nt * it->p->G) {
            memset(dst, 0, KA_DK);
            sq[r] = 0.0f;
            continue;
        }
        const float *q = ka_qrow(it, r);
        __m512 mx = _mm512_setzero_ps();
        for (int d = 0; d < KA_DK; d += 16) mx = _mm512_max_ps(mx, _mm512_abs_ps(_mm512_mul_ps(vs, _mm512_loadu_ps(q + d))));
        const float amax = _mm512_reduce_max_ps(mx);
        sq[r] = amax / 127.0f;
        const __m512 id = _mm512_set1_ps(amax > 0.0f ? 127.0f / amax : 0.0f);
        for (int d = 0; d < KA_DK; d += 16)
            _mm_storeu_si128((__m128i *)(dst + d), _mm512_cvtsepi32_epi8(_mm512_cvtps_epi32(_mm512_mul_ps(id, _mm512_mul_ps(vs, _mm512_loadu_ps(q + d))))));
    }
}

/* K rows [kv0, kv0 + TK) (nk valid) minus mu -> int8, one scale per token (sk [TK]). Per 16 tokens and
 * 64 dims, 16 rows (dim quads) x 16 tokens x 4 bytes: the AMX-INT8 B operand. */
static void ka_pack_k8(const kattn_args *a, int64_t g, int64_t kv0, int nk, uint32_t *K8, float *sk, const float *mu) {
    for (int t0 = 0; t0 < KA_TK; t0 += 16) {
        float id[16];
        for (int t = 0; t < 16; t++) {
            id[t] = 0.0f;
            sk[t0 + t] = 0.0f;
            if (t0 + t >= nk) continue;
            const uint8_t *kr = ka_krow(a, g, kv0 + t0 + t);
            __m512 mx = _mm512_setzero_ps();
            for (int d = 0; d < KA_DK; d += 16) {
                __m512 x = ka_row16(kr, d);
#if KA_KCENTER
                x = _mm512_sub_ps(x, _mm512_load_ps(mu + d));
#endif
                mx = _mm512_max_ps(mx, _mm512_abs_ps(x));
            }
            const float amax = _mm512_reduce_max_ps(mx);
            sk[t0 + t] = amax / 127.0f;
            id[t] = amax > 0.0f ? 127.0f / amax : 0.0f;
        }
        for (int d0 = 0; d0 < KA_DK; d0 += 64) {
            __m512i r[16];
            for (int t = 0; t < 16; t++) {
                r[t] = _mm512_setzero_si512();
                if (t0 + t >= nk) continue;
                const uint8_t *kr = ka_krow(a, g, kv0 + t0 + t);
                const __m512 vi = _mm512_set1_ps(id[t]);
                __m128i b[4];
                for (int i = 0; i < 4; i++) {
                    __m512 x = ka_row16(kr, d0 + 16 * i);
#if KA_KCENTER
                    x = _mm512_sub_ps(x, _mm512_load_ps(mu + d0 + 16 * i));
#endif
                    b[i] = _mm512_cvtsepi32_epi8(_mm512_cvtps_epi32(_mm512_mul_ps(vi, x)));
                }
                r[t] = _mm512_inserti32x4(_mm512_inserti32x4(_mm512_inserti32x4(_mm512_castsi128_si512(b[0]), b[1], 1), b[2], 2), b[3], 3);
            }
            ka_transpose16(r);
            uint32_t *dst = K8 + ((size_t)(t0 / 16) * (KA_DK / 64) + d0 / 64) * 256;
            for (int i = 0; i < 16; i++) _mm512_store_si512(dst + 16 * i, r[i]);
        }
    }
#if !KA_KCENTER
    (void)mu;
#endif
}

/* S = (Q8 K8^T) * sq[r] * sk[j] as f32 in place: 32 rows x 32 tokens per AMX step */
static void ka_qk8(const int8_t *Q8, const float *sq, const uint32_t *K8, const float *sk, float *S, int64_t R) {
    for (int64_t r0 = 0; r0 < R; r0 += 32) {
        for (int c0 = 0; c0 < KA_TK; c0 += 32) {
            uint64_t t0;
            do {
                t0 = __rdtsc();
                _tile_zero(0); _tile_zero(1); _tile_zero(2); _tile_zero(3);
                for (int k = 0; k < KA_DK; k += 64) {
                    _tile_loadd(4, Q8 + r0 * KA_DK + k, KA_DK);
                    _tile_loadd(5, Q8 + (r0 + 16) * KA_DK + k, KA_DK);
                    _tile_loadd(6, K8 + ((size_t)(c0 / 16) * (KA_DK / 64) + k / 64) * 256, 64);
                    _tile_loadd(7, K8 + ((size_t)(c0 / 16 + 1) * (KA_DK / 64) + k / 64) * 256, 64);
                    _tile_dpbssd(0, 4, 6);
                    _tile_dpbssd(1, 4, 7);
                    _tile_dpbssd(2, 5, 6);
                    _tile_dpbssd(3, 5, 7);
                }
                _tile_stored(0, S + r0 * KA_TK + c0, 4 * KA_TK);
                _tile_stored(1, S + r0 * KA_TK + c0 + 16, 4 * KA_TK);
                _tile_stored(2, S + (r0 + 16) * KA_TK + c0, 4 * KA_TK);
                _tile_stored(3, S + (r0 + 16) * KA_TK + c0 + 16, 4 * KA_TK);
            } while (ka_amx_slow(t0));
        }
        for (int64_t r = r0; r < r0 + 32; r++) {
            const __m512 vq = _mm512_set1_ps(sq[r]);
            float *s = S + r * KA_TK;
            for (int j = 0; j < KA_TK; j += 16)
                _mm512_storeu_ps(s + j, _mm512_mul_ps(_mm512_mul_ps(vq, _mm512_loadu_ps(sk + j)), _mm512_cvtepi32_ps(_mm512_loadu_si512(s + j))));
        }
    }
}
#endif
#endif

#if KA_ENGINE == 0
static void ka_pack_k(const kattn_args *a, int64_t g, int64_t kv0, int nk, float *Kt, const float *mu __attribute__((unused))) { /* [DK][TK] */
    for (int t0 = 0; t0 < KA_TK; t0 += 16) {
        for (int d0 = 0; d0 < KA_DK; d0 += 16) {
            __m512i r[16];
            for (int t = 0; t < 16; t++)
                r[t] = t0 + t < nk ? _mm512_castps_si512(ka_row16(ka_krow(a, g, kv0 + t0 + t), d0)) : _mm512_setzero_si512();
            ka_transpose16(r);
            for (int i = 0; i < 16; i++) _mm512_store_si512(Kt + (size_t)(d0 + i) * KA_TK + t0, r[i]);
        }
    }
}

static void ka_pack_v(const kattn_args *a, int64_t g, int64_t kv0, int nk, float *V) { /* [TK][DV] */
    for (int t = 0; t < KA_TK; t++) {
        const uint8_t *vr = t < nk ? ka_vrow(a, g, kv0 + t) : NULL;
        for (int d = 0; d < KA_DV; d += 16) _mm512_store_ps(V + (size_t)t * KA_DV + d, vr ? ka_row16(vr, d) : _mm512_setzero_ps());
    }
}

static void ka_pack_q(const ka_item *it, float *Q) {
    const __m512 vs = _mm512_set1_ps(it->a->scale * KA_LOG2E);
    for (int64_t r = 0; r < it->p->Rpad; r++) {
        float *dst = Q + r * KA_DK;
        if (r < it->nt * it->p->G) {
            const float *q = ka_qrow(it, r);
            for (int d = 0; d < KA_DK; d += 16) _mm512_store_ps(dst + d, _mm512_mul_ps(vs, _mm512_loadu_ps(q + d)));
        } else {
            memset(dst, 0, 4 * KA_DK);
        }
    }
}

static void ka_qk(const float *Q, const float *Kt, float *S, int64_t R) {
    for (int64_t r0 = 0; r0 < R; r0 += 4) {
        for (int c0 = 0; c0 < KA_TK; c0 += 64) {
            __m512 acc[4][4];
            for (int i = 0; i < 4; i++)
                for (int c = 0; c < 4; c++) acc[i][c] = _mm512_setzero_ps();
            for (int d = 0; d < KA_DK; d++) {
                __m512 kv[4];
                for (int c = 0; c < 4; c++) kv[c] = _mm512_load_ps(Kt + (size_t)d * KA_TK + c0 + 16 * c);
                for (int i = 0; i < 4; i++) {
                    const __m512 qb = _mm512_set1_ps(Q[(r0 + i) * KA_DK + d]);
                    for (int c = 0; c < 4; c++) acc[i][c] = _mm512_fmadd_ps(qb, kv[c], acc[i][c]);
                }
            }
            for (int i = 0; i < 4; i++)
                for (int c = 0; c < 4; c++) _mm512_storeu_ps(S + (r0 + i) * KA_TK + c0 + 16 * c, acc[i][c]);
        }
    }
}

static void ka_pv(const float *P, const float *V, float *O, int64_t R) {
    for (int64_t r0 = 0; r0 < R; r0 += 4) {
        for (int d0 = 0; d0 < KA_DV; d0 += 64) {
            __m512 acc[4][4];
            for (int i = 0; i < 4; i++)
                for (int c = 0; c < 4; c++) acc[i][c] = _mm512_loadu_ps(O + (r0 + i) * KA_DV + d0 + 16 * c);
            for (int t = 0; t < KA_TK; t++) {
                __m512 vv[4];
                for (int c = 0; c < 4; c++) vv[c] = _mm512_load_ps(V + (size_t)t * KA_DV + d0 + 16 * c);
                for (int i = 0; i < 4; i++) {
                    const __m512 pb = _mm512_set1_ps(P[(r0 + i) * KA_TK + t]);
                    for (int c = 0; c < 4; c++) acc[i][c] = _mm512_fmadd_ps(pb, vv[c], acc[i][c]);
                }
            }
            for (int i = 0; i < 4; i++)
                for (int c = 0; c < 4; c++) _mm512_storeu_ps(O + (r0 + i) * KA_DV + d0 + 16 * c, acc[i][c]);
        }
    }
}
#endif

/* kattn_pack buffer: per kv head a K mean [DK] (KA_KCENTER, frozen once >= KA_MU_MIN tokens are packed),
 * then K as bf16 pairs [cap/16][DK/2][16] per head, then V as bf16 pairs [DV/16][cap/2][16] per head:
 * a KV tile at kv0 is K + kv0/16 groups and V + kv0/2 pair rows (V stride cap/2), read in place. */
#define KA_MU_MIN 32
static int64_t ka_cap(int64_t cap) { return (cap + KA_TK - 1) / KA_TK * KA_TK; }
static size_t ka_kvp_koff(const kattn_args *a) { return KA_ALIGN(sizeof(float) * a->n_head_kv * KA_DK); }
static size_t ka_kvp_voff(const kattn_args *a, int64_t cap) { return ka_kvp_koff(a) + (size_t)a->n_head_kv * cap * KA_DK * 2; }
static inline __attribute__((unused)) const float *ka_kvp_mu(const uint8_t *kvp, int64_t g) { return (const float *)kvp + g * KA_DK; }
static inline __attribute__((unused)) const uint32_t *ka_kvp_k(const kattn_args *a, const uint8_t *kvp, int64_t cap, int64_t g) {
    return (const uint32_t *)(kvp + ka_kvp_koff(a) + (size_t)g * cap * KA_DK * 2);
}
static inline __attribute__((unused)) const uint32_t *ka_kvp_v(const kattn_args *a, const uint8_t *kvp, int64_t cap, int64_t g) {
    return (const uint32_t *)(kvp + ka_kvp_voff(a, cap) + (size_t)g * cap * KA_DV * 2);
}

/* Not for the pre-RoPE formats: packing would need K dequantized and rotated at fixed positions. */
#define KA_CAN_PACK (KA_ENGINE != 0 && !KA_QK8 && !KA_PRE_ROPE)

size_t kattn_pack_bytes(const kattn_args *a, int64_t cap) {
    if (!KA_CAN_PACK) return 0;
    const int64_t C = ka_cap(cap);
    return ka_kvp_voff(a, C) + (size_t)a->n_head_kv * C * KA_DV * 2;
}

void kattn_pack(const kattn_args *a, void *kvp, int64_t cap, int64_t j0, int64_t j1, int ith, int nth) {
#if KA_CAN_PACK
    const int64_t C = ka_cap(cap);
    uint8_t *base = (uint8_t *)kvp;
    /* while fewer than KA_MU_MIN tokens were packed before, recompute the K mean and repack from 0 */
    const int remu = KA_KCENTER && j0 < KA_MU_MIN;
    if (remu) j0 = 0;
    j0 &= ~(int64_t)15;
    if (j1 > C || j1 <= j0) return;
    const int64_t ng = (j1 - j0 + 15) / 16, units = (int64_t)a->n_head_kv * ng;
    float mul[KA_DK] __attribute__((aligned(64)));
    int64_t gl = -1;
    const float *mu = NULL;
    for (int64_t u = units * ith / nth; u < units * (ith + 1) / nth; u++) {
        const int64_t g = u / ng, t = j0 + (u % ng) * 16;
        if (g != gl) {
            gl = g;
            mu = ka_kvp_mu(base, g);
            if (remu) {
                const int64_t n = KA_MIN(j1, (int64_t)KA_TK);
                if (n >= KA_MU_MIN) ka_kmean(a, g, 0, (int)n, mul);
                else memset(mul, 0, sizeof mul);
                if (u % ng == 0) memcpy((float *)base + g * KA_DK, mul, sizeof mul); /* one writer per head */
                mu = mul;
            }
        }
        const int nk = (int)KA_MIN((int64_t)16, j1 - t);
        ka_pack_k16(a, g, t, nk, (uint32_t *)ka_kvp_k(a, base, C, g) + (size_t)(t / 16) * (KA_DK / 2) * 16, mu);
        ka_pack_vn(a, g, t, nk, 16, (uint32_t *)ka_kvp_v(a, base, C, g) + (size_t)(t / 2) * 16, C / 2);
    }
#else
    (void)a, (void)kvp, (void)cap, (void)j0, (void)j1, (void)ith, (void)nth;
#endif
}

static void ka_tile(ka_item *it, const ka_tile_buf *b) {
    const kattn_args *a = it->a;
    const ka_plan *p = it->p;
    const int64_t Rv = it->nt * p->G;            /* valid rows */
    const int64_t Rc = (Rv + 31) & ~(int64_t)31; /* rows computed */
#if KA_QK8
    int8_t *Q8 = (int8_t *)b->Q; /* int8 rows, then the row scales (the bf16 Q buffer has room for both) */
    float *sq = (float *)(Q8 + p->Rpad * KA_DK), *sk = (float *)((uint8_t *)b->Kp + KA_TK * KA_DK);
    ka_pack_q8(it, Q8, sq);
#else
    ka_pack_q(it, b->Q);
#endif
    for (int64_t r = 0; r < Rc; r++) {
        it->m[r] = -INFINITY;
        it->l[r] = 0.0f;
    }
    memset(it->O, 0, sizeof(float) * Rc * KA_DV);
#if KA_ENGINE != 0
    memset(b->P + Rv * KA_TK, 0, 2 * (Rc - Rv) * KA_TK);
#endif
    const float *mu = NULL;
    for (int64_t kv0 = it->k0; kv0 < it->k1; kv0 += KA_TK) {
        const int nk = (int)KA_MIN((int64_t)KA_TK, it->k1 - kv0);
        float *S = b->S;
        int masked = nk < KA_TK;
        const int mclass = a->mask ? ka_mask_scan(a, it, kv0, nk) : KA_MASK_NONE;
        if (mclass == KA_MASK_SKIP) continue;
#if KA_KCENTER
        if (!mu) {
            if (it->kvp) {
                mu = ka_kvp_mu(it->kvp, it->g);
            } else {
                ka_kmean(a, it->g, kv0, nk, b->mu);
                mu = b->mu;
            }
        }
#endif
#if KA_ENGINE != 0
        const uint32_t *Kt = b->Kp, *Vt = b->Vp;
        int64_t vs = KA_TK / 2;
#if KA_QK8
        (void)Kt;
        ka_pack_k8(a, it->g, kv0, nk, b->Kp, sk, mu);
        ka_qk8(Q8, sq, b->Kp, sk, S, Rc);
#else
        if (it->kvp) {
            Kt = ka_kvp_k(a, it->kvp, it->cap, it->g) + (size_t)(kv0 / 16) * (KA_DK / 2) * 16;
            Vt = ka_kvp_v(a, it->kvp, it->cap, it->g) + (size_t)(kv0 / 2) * 16;
            vs = it->cap / 2;
        } else {
            ka_pack_k(a, it->g, kv0, nk, b->Kp, mu);
        }
        ka_qk(b->Q, Kt, S, Rc);
#endif
#else
        ka_pack_k(a, it->g, kv0, nk, b->Kp, mu);
        ka_qk(b->Q, b->Kp, S, Rc);
#endif
        if (masked)
            for (int64_t r = 0; r < Rv; r++)
                for (int j = nk; j < KA_TK; j++) S[r * KA_TK + j] = -INFINITY;
        if (a->causal && kv0 + nk - 1 > a->q_pos0 + it->t0) {
            masked = 1;
            for (int64_t r = 0; r < Rv; r++) {
                const int64_t lim = a->q_pos0 + it->t0 + r / p->G - kv0;
                for (int64_t j = KA_MAX((int64_t)0, lim + 1); j < nk; j++) S[r * KA_TK + j] = -INFINITY;
            }
        }
        if (mclass == KA_MASK_MIXED) {
            masked = 1;
            ka_mask_add(a, it, S, KA_TK, kv0, nk, KA_TK);
        }
#if KA_ENGINE == 0
        for (int64_t r = 0; r < Rv; r++) {
            const float alpha = ka_softmax_row(S + r * KA_TK, KA_TK, &it->m[r], &it->l[r], masked);
            if (alpha != 1.0f) ka_scale_row(it->O + r * KA_DV, KA_DV, alpha);
        }
        for (int64_t r = Rv; r < Rc; r++) memset(S + r * KA_TK, 0, 4 * KA_TK);
        ka_pack_v(a, it->g, kv0, nk, b->Vp);
        ka_pv(S, b->Vp, it->O, Rc);
#else
        for (int64_t r = 0; r < Rv; r++) {
            const float alpha = ka_softmax_row_bf16(S + r * KA_TK, b->P + r * KA_TK, &it->m[r], &it->l[r], masked);
            if (alpha != 1.0f) ka_scale_row(it->O + r * KA_DV, KA_DV, alpha);
        }
        if (!it->kvp) ka_pack_v(a, it->g, kv0, nk, b->Vp);
        ka_pv(b->P, Vt, it->O, Rc, vs);
#endif
    }
    if (mu) {
        const float qs = a->scale * KA_LOG2E;
        for (int64_t r = 0; r < Rv; r++) {
            if (it->m[r] == -INFINITY) continue;
            const float *q = ka_qrow(it, r);
            __m512 s = _mm512_setzero_ps();
            for (int d = 0; d < KA_DK; d += 16) s = _mm512_fmadd_ps(_mm512_loadu_ps(q + d), _mm512_load_ps(mu + d), s);
            it->m[r] += qs * _mm512_reduce_add_ps(s);
        }
    }
}

/* ---------------------------------------------------------------- driver */

static void ka_write_out(const kattn_args *a, const ka_item *it, int64_t r, const float *o, float inv) {
    const int64_t G = it->p->G;
    float *dst = a->out + (it->t0 + r / G) * a->o_s_tok + (it->g * G + r % G) * a->o_s_head;
    const __m512 vi = _mm512_set1_ps(inv);
    for (int d = 0; d < KA_DV; d += 16) _mm512_storeu_ps(dst + d, _mm512_mul_ps(vi, _mm512_loadu_ps(o + d)));
}

static void ka_merge(const kattn_args *a, const ka_plan *p, const ka_item *it, const float *part /* [nsplit][R][DV+2] */) {
    const int64_t Rv = it->nt * p->G, rs = p->R * (KA_DV + 2);
    for (int64_t r = 0; r < Rv; r++) {
        float M = -INFINITY;
        for (int64_t s = 0; s < p->nsplit; s++) M = KA_MAX(M, part[s * rs + r * (KA_DV + 2)]);
        __m512 acc[KA_DV / 16];
        for (int d = 0; d < KA_DV / 16; d++) acc[d] = _mm512_setzero_ps();
        float L = 0.0f;
        if (M != -INFINITY) {
            for (int64_t s = 0; s < p->nsplit; s++) {
                const float *ps = part + s * rs + r * (KA_DV + 2);
                if (ps[0] == -INFINITY) continue;
                const float w = exp2f(ps[0] - M);
                L += w * ps[1];
                const __m512 vw = _mm512_set1_ps(w);
                for (int d = 0; d < KA_DV / 16; d++) acc[d] = _mm512_fmadd_ps(vw, _mm512_loadu_ps(ps + 2 + 16 * d), acc[d]);
            }
        }
        const int64_t G = p->G;
        float *dst = a->out + (it->t0 + r / G) * a->o_s_tok + (it->g * G + r % G) * a->o_s_head;
        const __m512 vi = _mm512_set1_ps(L > 0.0f ? 1.0f / L : 0.0f);
        for (int d = 0; d < KA_DV / 16; d++) _mm512_storeu_ps(dst + 16 * d, _mm512_mul_ps(vi, acc[d]));
    }
}

static void ka_run(const kattn_args *a, const uint8_t *kvp, int64_t cap, void *ws, int ith, int nth) {
    ka_plan p;
    ka_make_plan(a, nth, &p);
    uint8_t *base = (uint8_t *)(((uintptr_t)ws + 63) & ~(uintptr_t)63);
    ka_hdr *h = (ka_hdr *)base;
    float *parts = (float *)(base + p.part_off);
    uint8_t *mine = base + p.thr_off + p.thr_bytes * (size_t)ith;
#if KA_PRE_ROPE
    ka_tls.st = mine + p.thr_bytes - ka_stage_bytes(p.dec);
    ka_tls.a = NULL;
#endif
    ka_tile_buf tb = {0};
    float *m, *l, *O;
    if (p.dec) {
        uint8_t *q = mine;
        tb.S = (float *)q;  q += KA_ALIGN(sizeof(float) * KA_DEC_ROWS * KA_DEC_TK) + KA_ALIGN(sizeof(float) * KA_DEC_ROWS * 256)
                               + KA_ALIGN(sizeof(float) * p.hb * KA_DEC_ROWS * KA_DK);
        O = (float *)q;     q += KA_ALIGN(sizeof(float) * p.hb * KA_DEC_ROWS * KA_DV);
        m = (float *)q;     q += KA_ALIGN(sizeof(float) * p.hb * KA_DEC_ROWS);
        l = (float *)q;
    } else {
#if KA_ENGINE == 0
        const size_t qb = 4;
#else
        const size_t qb = 2;
#endif
        uint8_t *q = mine;
        tb.Q = q;                                      q += KA_ALIGN(qb * p.Rpad * KA_DK);
        tb.S = (float *)q;                             q += KA_ALIGN(sizeof(float) * p.Rpad * KA_TK);
        tb.P = (uint16_t *)q;                          q += KA_ALIGN(2 * (size_t)p.Rpad * KA_TK);
        tb.Kp = q;                                     q += KA_ALIGN(qb * KA_TK * KA_DK);
        tb.Vp = q;                                     q += KA_ALIGN(qb * KA_TK * KA_DV);
        O = (float *)q;                                q += KA_ALIGN(sizeof(float) * p.Rpad * KA_DV);
        m = (float *)q;                                q += KA_ALIGN(sizeof(float) * p.Rpad);
        l = (float *)q;                                q += KA_ALIGN(sizeof(float) * p.Rpad);
        tb.mu = (float *)q;
    }
#if KA_ENGINE == 2
    ka_tilecfg saved;
    if (!p.dec) ka_amx_begin(&saved);
#endif
    const int64_t per_qt = p.nhb * p.nsplit, hrows = p.dec ? KA_DEC_ROWS : 0;
    for (;;) {
        const int64_t i = atomic_fetch_add_explicit(&h->next, 1, memory_order_relaxed);
        if (i >= p.nitems) break;
        ka_item it = {a, &p, 0, 0, 0, 0, 0, 0, 0, m, l, O, kvp, cap};
        it.qt = p.nqt - 1 - i / per_qt; /* most expensive (last) query tiles first */
        it.g = (i % per_qt) / p.nsplit * p.hb;
        it.ng = KA_MIN(p.hb, a->n_head_kv - it.g);
        const int64_t s = i % p.nsplit;
        it.t0 = it.qt * p.tq;
        it.nt = KA_MIN(p.tq, a->n_q - it.t0);
        int64_t hi = a->n_kv;
        if (a->causal) hi = KA_MAX((int64_t)0, KA_MIN(hi, a->q_pos0 + it.t0 + it.nt));
        it.k0 = KA_MIN(s * p.chunk, hi);
        it.k1 = KA_MIN((s + 1) * p.chunk, hi);
#if KA_DEC_ROWS > 0
        if (p.dec) ka_dec(&it, (uint8_t *)tb.S);
        else
#endif
            ka_tile(&it, &tb);
        const int64_t Rv = it.nt * p.G, g0 = it.g;
        for (int64_t hl = 0; hl < it.ng; hl++) {
            const float *mh = m + hl * hrows, *lh = l + hl * hrows, *Oh = O + hl * hrows * KA_DV;
            it.g = g0 + hl;
            if (p.nsplit == 1) {
                for (int64_t r = 0; r < Rv; r++) ka_write_out(a, &it, r, Oh + r * KA_DV, lh[r] > 0.0f ? 1.0f / lh[r] : 0.0f);
                continue;
            }
            const int64_t grp = it.qt * a->n_head_kv + it.g;
            float *gp = parts + (size_t)grp * p.nsplit * p.R * (KA_DV + 2);
            float *ps = gp + (size_t)s * p.R * (KA_DV + 2);
            for (int64_t r = 0; r < Rv; r++) {
                float *dst = ps + r * (KA_DV + 2);
                dst[0] = it.k1 > it.k0 ? mh[r] : -INFINITY;
                dst[1] = lh[r];
                if (it.k1 > it.k0) memcpy(dst + 2, Oh + r * KA_DV, sizeof(float) * KA_DV);
            }
            if (atomic_fetch_add_explicit(&h->group[grp], 1, memory_order_acq_rel) == p.nsplit - 1) {
                ka_merge(a, &p, &it, gp);
                atomic_store_explicit(&h->group[grp], 0, memory_order_relaxed);
            }
        }
    }
#if KA_ENGINE == 2
    if (!p.dec) ka_amx_end(&saved);
#endif
    if (atomic_fetch_add_explicit(&h->done, 1, memory_order_acq_rel) == nth - 1) {
        atomic_store_explicit(&h->next, 0, memory_order_relaxed);
        atomic_store_explicit(&h->done, 0, memory_order_release);
    }
}

void kattn(const kattn_args *a, void *ws, int ith, int nth) { ka_run(a, NULL, 0, ws, ith, nth); }

void kattn_packed(const kattn_args *a, const void *kvp, int64_t cap, void *ws, int ith, int nth) {
    const int use = KA_CAN_PACK && kvp && a->n_kv <= ka_cap(cap);
    ka_run(a, use ? (const uint8_t *)kvp : NULL, ka_cap(cap), ws, ith, nth);
}
