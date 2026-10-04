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
 */
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

#if KA_KV == KATTN_KV_Q8_0
#define KA_ROW_BYTES(d) ((int64_t)(d) / 32 * 34)
#else
#define KA_ROW_BYTES(d) ((int64_t)(d) * 2)
#endif

int64_t kattn_config(int *dk, int *dv, int *kv_format) {
    if (dk) *dk = KA_DK;
    if (dv) *dv = KA_DV;
    if (kv_format) *kv_format = KA_KV;
    return KA_ROW_BYTES(KA_DK);
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
          + 64;
    }
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
} ka_item;

KA_INLINE const float *ka_qrow_g(const ka_item *it, int64_t g, int64_t r) {
    const int64_t G = it->p->G;
    return it->a->q + (it->t0 + r / G) * it->a->q_s_tok + (g * G + r % G) * it->a->q_s_head;
}

KA_INLINE const float *ka_qrow(const ka_item *it, int64_t r) { return ka_qrow_g(it, it->g, r); }

KA_INLINE const uint8_t *ka_krow(const kattn_args *a, int64_t g, int64_t j) {
    return (const uint8_t *)a->k + j * a->k_s_tok + g * a->k_s_head;
}

KA_INLINE const uint8_t *ka_vrow(const kattn_args *a, int64_t g, int64_t j) {
    return (const uint8_t *)a->v + j * a->v_s_tok + g * a->v_s_head;
}

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

static void ka_dec(ka_item *it, uint8_t *scratch) {
    switch (it->p->R) {
#define C(n) case n: ka_dec_run(it, scratch, n < KA_DEC_ROWS ? n : KA_DEC_ROWS); break;
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
} ka_tile_buf;

#if KA_ENGINE != 0
KA_INLINE __m512i ka_bf16x32(__m512 lo, __m512 hi) { return (__m512i)_mm512_cvtne2ps_pbh(hi, lo); }

static void ka_pack_k(const kattn_args *a, int64_t g, int64_t kv0, int nk, uint32_t *Kp) {
    for (int t0 = 0; t0 < KA_TK; t0 += 16) {
        for (int d0 = 0; d0 < KA_DK; d0 += 32) {
            __m512i r[16];
            for (int t = 0; t < 16; t++) {
                if (t0 + t < nk) {
                    const uint8_t *kr = ka_krow(a, g, kv0 + t0 + t);
                    r[t] = ka_bf16x32(ka_row16(kr, d0), ka_row16(kr, d0 + 16));
                } else {
                    r[t] = _mm512_setzero_si512();
                }
            }
            ka_transpose16(r);
            uint32_t *dst = Kp + ((size_t)(t0 / 16) * (KA_DK / 2) + d0 / 2) * 16;
            for (int i = 0; i < 16; i++) _mm512_store_si512(dst + 16 * i, r[i]);
        }
    }
}

static void ka_pack_v(const kattn_args *a, int64_t g, int64_t kv0, int nk, uint32_t *Vp) {
    const __m512i idx = _mm512_set_epi16(31, 15, 30, 14, 29, 13, 28, 12, 27, 11, 26, 10, 25, 9, 24, 8,
                                         23, 7, 22, 6, 21, 5, 20, 4, 19, 3, 18, 2, 17, 1, 16, 0);
    for (int t = 0; t < KA_TK; t += 2) {
        const uint8_t *va = t < nk ? ka_vrow(a, g, kv0 + t) : NULL;
        const uint8_t *vb = t + 1 < nk ? ka_vrow(a, g, kv0 + t + 1) : NULL;
        for (int d0 = 0; d0 < KA_DV; d0 += 16) {
            const __m512 x = va ? ka_row16(va, d0) : _mm512_setzero_ps();
            const __m512 y = vb ? ka_row16(vb, d0) : _mm512_setzero_ps();
            const __m512i pr = _mm512_permutexvar_epi16(idx, ka_bf16x32(x, y)); /* (x0,y0,x1,y1,...) */
            _mm512_store_si512(Vp + ((size_t)(d0 / 16) * (KA_TK / 2) + t / 2) * 16, pr);
        }
    }
}

static void ka_pack_q(const ka_item *it, uint16_t *Q) {
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

/* 2^x to ~4e-5 relative (degree-4 Taylor on [-0.5, 0.5]): far below the bf16 rounding of P */
KA_INLINE __m512 ka_exp2_lo(__m512 x) {
    x = _mm512_max_ps(x, _mm512_set1_ps(-127.0f));
    const __m512 n = _mm512_roundscale_ps(x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    const __m512 f = _mm512_sub_ps(x, n);
    __m512 p = _mm512_set1_ps(9.6181291076284772e-3f);
    p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(5.5504108664821580e-2f));
    p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(2.4022650695910071e-1f));
    p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(6.9314718055994531e-1f));
    p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(1.0f));
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

/* O[R][DV] += P V: 4 rows x 64 dims per register block */
static void ka_pv(const uint16_t *P, const uint32_t *Vp, float *O, int64_t R) {
    for (int64_t r0 = 0; r0 < R; r0 += 4) {
        for (int d0 = 0; d0 < KA_DV; d0 += 64) {
            __m512 acc[4][4];
            for (int i = 0; i < 4; i++)
                for (int c = 0; c < 4; c++) acc[i][c] = _mm512_loadu_ps(O + (r0 + i) * KA_DV + d0 + 16 * c);
            const uint32_t *p32 = (const uint32_t *)(P + r0 * KA_TK);
            for (int kp = 0; kp < KA_TK / 2; kp++) {
                __m512bh vv[4];
                for (int c = 0; c < 4; c++) vv[c] = (__m512bh)_mm512_load_si512(Vp + ((size_t)(d0 / 16 + c) * (KA_TK / 2) + kp) * 16);
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
static void ka_qk(const uint16_t *Q, const uint32_t *Kp, float *S, int64_t R) {
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

/* O += P V: 32 rows x 32 dims per step */
static void ka_pv(const uint16_t *P, const uint32_t *Vp, float *O, int64_t R) {
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
                    _tile_loadd(6, Vp + ((size_t)(d0 / 16) * (KA_TK / 2) + k / 2) * 16, 64);
                    _tile_loadd(7, Vp + ((size_t)(d0 / 16 + 1) * (KA_TK / 2) + k / 2) * 16, 64);
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
#endif

#if KA_ENGINE == 0
static void ka_pack_k(const kattn_args *a, int64_t g, int64_t kv0, int nk, float *Kt) { /* [DK][TK] */
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

static void ka_tile(ka_item *it, const ka_tile_buf *b) {
    const kattn_args *a = it->a;
    const ka_plan *p = it->p;
    const int64_t Rv = it->nt * p->G;            /* valid rows */
    const int64_t Rc = (Rv + 31) & ~(int64_t)31; /* rows computed */
    ka_pack_q(it, b->Q);
    for (int64_t r = 0; r < Rc; r++) {
        it->m[r] = -INFINITY;
        it->l[r] = 0.0f;
    }
    memset(it->O, 0, sizeof(float) * Rc * KA_DV);
#if KA_ENGINE != 0
    memset(b->P + Rv * KA_TK, 0, 2 * (Rc - Rv) * KA_TK);
#endif
    for (int64_t kv0 = it->k0; kv0 < it->k1; kv0 += KA_TK) {
        const int nk = (int)KA_MIN((int64_t)KA_TK, it->k1 - kv0);
        float *S = b->S;
        int masked = nk < KA_TK;
        const int mclass = a->mask ? ka_mask_scan(a, it, kv0, nk) : KA_MASK_NONE;
        if (mclass == KA_MASK_SKIP) continue;
        ka_pack_k(a, it->g, kv0, nk, b->Kp);
        ka_qk(b->Q, b->Kp, S, Rc);
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
        ka_pack_v(a, it->g, kv0, nk, b->Vp);
        ka_pv(b->P, b->Vp, it->O, Rc);
#endif
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

void kattn(const kattn_args *a, void *ws, int ith, int nth) {
    ka_plan p;
    ka_make_plan(a, nth, &p);
    uint8_t *base = (uint8_t *)(((uintptr_t)ws + 63) & ~(uintptr_t)63);
    ka_hdr *h = (ka_hdr *)base;
    float *parts = (float *)(base + p.part_off);
    uint8_t *mine = base + p.thr_off + p.thr_bytes * (size_t)ith;
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
        l = (float *)q;
    }
#if KA_ENGINE == 2
    ka_tilecfg saved;
    if (!p.dec) ka_amx_begin(&saved);
#endif
    const int64_t per_qt = p.nhb * p.nsplit, hrows = p.dec ? KA_DEC_ROWS : 0;
    for (;;) {
        const int64_t i = atomic_fetch_add_explicit(&h->next, 1, memory_order_relaxed);
        if (i >= p.nitems) break;
        ka_item it = {a, &p, 0, 0, 0, 0, 0, 0, 0, m, l, O};
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
