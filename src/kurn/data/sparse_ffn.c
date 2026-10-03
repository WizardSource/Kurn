/*
 * kurn sparse SwiGLU FFN for one decode token (Deja Vu / PowerInfer-style row skipping), Q8_0.
 *
 *   y = W_down (silu(W_gate x) * (W_up x)),   only rows i of the d_ff dimension that are active:
 *   KS_GATE  the gate product is computed for every row; row i is active iff |silu(g_i)| >= thr
 *   KS_PRED  a rank-r predictor g~ = B (A x) picks the rows (|silu(g~_i)| >= thr); gate and up
 *            rows are computed only for active rows
 *   KS_DENSE every row active (same code path; the dense reference point)
 * A bench/test caller may force the active set with `force` (1 byte per row, 1 = active).
 *
 * Weights are ggml Q8_0 blocks (fp16 scale + 32 int8), row-major: W_gate, W_up [d_ff][d];
 * W_down is stored transposed ([d_ff][d], blocks along d) so an active row is one contiguous
 * run of d/32 blocks. A [r][d] and B [d_ff][r] are Q8_0 too (r a multiple of 32).
 * x is quantized to Q8_0 like ggml's activations; dot products are exact int32 per block.
 *
 * Threading: ks_ffn(args, ws, ith, nth) is called by nth threads; each owns a slice of d_ff
 * rows and a private partial y; one barrier (two with the predictor) then a sliced reduction.
 * The workspace (ks_workspace bytes, zeroed once) holds the barrier and the partials.
 */
#include <immintrin.h>
#include <math.h>
#include <stdatomic.h>
#include <stdint.h>
#include <string.h>

#define KS_DENSE 0
#define KS_GATE 1
#define KS_PRED 2

#ifndef KS_PF
#define KS_PF 2 /* prefetch the active row this many rows ahead (0 = off) */
#endif

typedef struct {
    int32_t d, dff, rank, mode;
    float thr;
    const void *wg, *wu, *wdt, *pa, *pb;
    const uint8_t *force;
    const float *x;
    float *y;
    int64_t *n_active; /* optional out: active rows */
} ks_args;

typedef struct {
    uint16_t d;
    int8_t q[32];
} __attribute__((packed)) ks_blk;

typedef struct {
    float d;
    int8_t q[32];
    uint8_t a[32]; /* |q| */
} ks_xblk;

typedef struct {
    atomic_int count, gen;
} ks_barrier;

#define KS_ALIGN(n) (((n) + 63) & ~(size_t)63)

static size_t ks_part_off(void) { return KS_ALIGN(sizeof(ks_barrier)); }

size_t ks_workspace(const ks_args *a, int nth) {
    /* barrier | partial y [nth][d] | z [rank] | per-thread x/z blocks + row scratch */
    size_t per_thr = KS_ALIGN(sizeof(ks_xblk) * (size_t)(a->d / 32)) + KS_ALIGN(sizeof(ks_xblk) * (size_t)(a->rank / 32 + 1)) +
                     2 * KS_ALIGN(sizeof(float) * (size_t)(a->dff / nth + 64)) + KS_ALIGN(sizeof(int32_t) * (size_t)(a->dff / nth + 64));
    return ks_part_off() + KS_ALIGN(sizeof(float) * (size_t)nth * a->d) + KS_ALIGN(sizeof(float) * (size_t)(a->rank + 16)) +
           (size_t)nth * per_thr + 64;
}

static void ks_wait(ks_barrier *b, int nth) {
    const int g = atomic_load_explicit(&b->gen, memory_order_acquire);
    if (atomic_fetch_add_explicit(&b->count, 1, memory_order_acq_rel) == nth - 1) {
        atomic_store_explicit(&b->count, 0, memory_order_relaxed);
        atomic_store_explicit(&b->gen, g + 1, memory_order_release);
    } else {
        while (atomic_load_explicit(&b->gen, memory_order_acquire) == g) _mm_pause();
    }
}

static void ks_quant(const float *x, int n, ks_xblk *out) {
    for (int b = 0; b < n / 32; b++) {
        float amax = 0.0f;
        for (int i = 0; i < 32; i++) amax = fmaxf(amax, fabsf(x[32 * b + i]));
        const float d = amax / 127.0f, id = d > 0.0f ? 1.0f / d : 0.0f;
        out[b].d = d;
        for (int i = 0; i < 32; i++) {
            const int q = (int)roundf(x[32 * b + i] * id);
            out[b].q[i] = (int8_t)q;
            out[b].a[i] = (uint8_t)(q < 0 ? -q : q);
        }
    }
}

/* dot of one Q8_0 row (nb blocks) with the quantized activation */
static inline float ks_dot(const ks_blk *w, const ks_xblk *x, int nb) {
    __m256 acc = _mm256_setzero_ps();
    for (int b = 0; b < nb; b++) {
        const __m256i wq = _mm256_loadu_si256((const __m256i *)w[b].q);
        const __m256i xq = _mm256_loadu_si256((const __m256i *)x[b].q);
        const __m256i xa = _mm256_loadu_si256((const __m256i *)x[b].a);
        const __m256i p = _mm256_dpbusd_epi32(_mm256_setzero_si256(), xa, _mm256_sign_epi8(wq, xq));
        acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p), _mm256_set1_ps(_cvtsh_ss(w[b].d) * x[b].d), acc);
    }
    const __m128 s = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
    const __m128 t = _mm_add_ps(s, _mm_movehl_ps(s, s));
    return _mm_cvtss_f32(_mm_add_ss(t, _mm_movehdup_ps(t)));
}

static inline void ks_prefetch_row(const ks_blk *w, int nb) {
    const char *p = (const char *)w;
    for (int o = 0; o < nb * 34; o += 64) _mm_prefetch(p + o, _MM_HINT_T0);
}

/* y += c * row (Q8_0 row of nb blocks) */
static inline void ks_axpy(const ks_blk *w, float c, float *y, int nb) {
    for (int b = 0; b < nb; b++) {
        const __m512 s = _mm512_set1_ps(c * _cvtsh_ss(w[b].d));
        const __m512 lo = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i *)w[b].q)));
        const __m512 hi = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i *)(w[b].q + 16))));
        _mm512_storeu_ps(y + 32 * b, _mm512_fmadd_ps(s, lo, _mm512_loadu_ps(y + 32 * b)));
        _mm512_storeu_ps(y + 32 * b + 16, _mm512_fmadd_ps(s, hi, _mm512_loadu_ps(y + 32 * b + 16)));
    }
}

static inline float ks_silu(float g) { return g / (1.0f + expf(-g)); }

void ks_ffn(const ks_args *a, void *ws, int ith, int nth) {
    const int d = a->d, dff = a->dff, nbd = d / 32, r = a->rank;
    uint8_t *base = (uint8_t *)(((uintptr_t)ws + 63) & ~(uintptr_t)63);
    ks_barrier *bar = (ks_barrier *)base;
    float *part = (float *)(base + ks_part_off());
    float *z = (float *)((uint8_t *)part + KS_ALIGN(sizeof(float) * (size_t)nth * d));
    const size_t per_thr = KS_ALIGN(sizeof(ks_xblk) * (size_t)nbd) + KS_ALIGN(sizeof(ks_xblk) * (size_t)(r / 32 + 1)) +
                           2 * KS_ALIGN(sizeof(float) * (size_t)(dff / nth + 64)) + KS_ALIGN(sizeof(int32_t) * (size_t)(dff / nth + 64));
    uint8_t *mine = (uint8_t *)z + KS_ALIGN(sizeof(float) * (size_t)(r + 16)) + (size_t)ith * per_thr;
    ks_xblk *xq = (ks_xblk *)mine;
    ks_xblk *zq = (ks_xblk *)(mine + KS_ALIGN(sizeof(ks_xblk) * (size_t)nbd));
    float *gv = (float *)((uint8_t *)zq + KS_ALIGN(sizeof(ks_xblk) * (size_t)(r / 32 + 1)));
    float *hv = (float *)((uint8_t *)gv + KS_ALIGN(sizeof(float) * (size_t)(dff / nth + 64)));
    int32_t *idx = (int32_t *)((uint8_t *)hv + KS_ALIGN(sizeof(float) * (size_t)(dff / nth + 64)));
    float *y = part + (size_t)ith * d;

    ks_quant(a->x, d, xq);
    const ks_blk *wg = (const ks_blk *)a->wg, *wu = (const ks_blk *)a->wu, *wdt = (const ks_blk *)a->wdt;
    const int64_t chunk = ((dff + nth - 1) / nth + 15) / 16 * 16;
    const int i0 = (int)(ith * chunk < dff ? ith * chunk : dff), i1 = (int)((ith + 1) * chunk < dff ? (ith + 1) * chunk : dff);

    if (a->mode == KS_PRED) {
        const ks_blk *pa = (const ks_blk *)a->pa;
        const int per = (r + nth - 1) / nth;
        for (int j = ith * per; j < r && j < (ith + 1) * per; j++) z[j] = ks_dot(pa + (size_t)j * nbd, xq, nbd);
        ks_wait(bar, nth);
        ks_quant(z, r, zq);
    }

    /* select the active rows of this slice */
    int n = 0;
    for (int i = i0; i < i1; i++) {
        int on;
        if (a->mode == KS_DENSE) {
            on = 1;
        } else if (a->mode == KS_GATE) {
            if (KS_PF && i + KS_PF < i1) ks_prefetch_row(wg + (size_t)(i + KS_PF) * nbd, nbd);
            const float g = ks_dot(wg + (size_t)i * nbd, xq, nbd);
            gv[n] = g;
            on = a->force ? a->force[i] : fabsf(ks_silu(g)) >= a->thr;
        } else {
            const float gp = ks_dot((const ks_blk *)a->pb + (size_t)i * (r / 32), zq, r / 32);
            on = a->force ? a->force[i] : fabsf(ks_silu(gp)) >= a->thr;
        }
        if (on) idx[n++] = i;
    }
    if (a->n_active) __atomic_fetch_add(a->n_active, n, __ATOMIC_RELAXED);

    /* h_i = silu(g_i) * u_i on the active rows */
    for (int j = 0; j < n; j++) {
        const int i = idx[j];
        if (KS_PF && j + KS_PF < n) {
            ks_prefetch_row(wu + (size_t)idx[j + KS_PF] * nbd, nbd);
            if (a->mode != KS_GATE) ks_prefetch_row(wg + (size_t)idx[j + KS_PF] * nbd, nbd);
        }
        const float g = a->mode == KS_GATE ? gv[j] : ks_dot(wg + (size_t)i * nbd, xq, nbd);
        hv[j] = ks_silu(g) * ks_dot(wu + (size_t)i * nbd, xq, nbd);
    }

    /* y_partial = sum over active rows of h_i * W_down^T[i] */
    memset(y, 0, sizeof(float) * d);
    for (int j = 0; j < n; j++) {
        if (KS_PF && j + KS_PF < n) ks_prefetch_row(wdt + (size_t)idx[j + KS_PF] * nbd, nbd);
        ks_axpy(wdt + (size_t)idx[j] * nbd, hv[j], y, nbd);
    }
    ks_wait(bar, nth);

    /* reduce the partials over a slice of d */
    const int per = ((d + nth - 1) / nth + 15) / 16 * 16;
    for (int c = ith * per; c < d && c < (ith + 1) * per; c += 16) {
        __m512 s = _mm512_loadu_ps(part + c);
        for (int t = 1; t < nth; t++) s = _mm512_add_ps(s, _mm512_loadu_ps(part + (size_t)t * d + c));
        _mm512_storeu_ps(a->y + c, s);
    }
    ks_wait(bar, nth); /* the next call overwrites the partials */
}
