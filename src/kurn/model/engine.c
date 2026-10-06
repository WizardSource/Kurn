// kurn model engine v2: the whole decode step of ONE fixed model compiled as one program
// ("the model is the program"), with the fewest synchronization points the data flow allows.
//
// Schedule (computed once at load = "captured", then replayed every token):
//   * every thread owns whole KV-head groups: its QKV rows, QK-norm, RoPE, KV-cache rows and
//     attention are thread-local; the O projection is split by COLUMNS (the thread's own head
//     outputs) and yields a partial residual vector;
//   * every thread owns an FF slice (dense) or a slice of the (selected expert, FF block) list
//     (MoE): gate/up rows -> SwiGLU -> Q8_0 in the GEMV epilogue -> down-projection column
//     slice -> partial residual vector;
//   * after each partial, threads publish an epoch (one cache line each) and every thread
//     reduces all partials in the same order into its PRIVATE copy of the residual stream,
//     waiting point-to-point on each producer's epoch (no central barrier counter). RMSNorm's
//     sum of squares is fused into that reduction; norms and activation quantization are
//     recomputed per thread instead of synchronized.
//   Sync points per token: 2 per dense layer (3 per OLMoE layer: its full-vector QK-norm needs
//   one sum-of-squares exchange) + 1 for the output argmax. Weights are repacked at load into
//   per-thread arenas in execution order (THP-backed), so each core streams its own memory.
//
//   engine MODEL.gguf gen  T  N_GEN  tok,tok,...     greedy decode; token ids, tok/s, CPU s, sync stats
//   engine MODEL.gguf ppl  T  CTX  tokens.txt        perplexity over the 2nd half of each CTX chunk
// Decode output also reports runq_share: the fraction of thread time spent runnable but descheduled.
// Env: KURN_WAIT=spin|futex:N|yield:N (or per sync kind: attn=...,qk=...,ffn=...,out=...),
//      KURN_THP=0 (no huge pages), KURN_PIN=0 (threads not pinned 1:1 to cores), KURN_PFWAIT=bytes (prefetch-in-wait budget, default 0), PPL_CHUNKS=n (default 4), KURN_PROF=1 (per-thread waits to stderr),
//      KURN_DUMP_LOGITS=file (gen: float32 logits of every step, prompt included), KURN_PERF_CTL=fifo (below),
//      KURN_KV=f16|q8_0|q4_0|k4c_q4|k4c_q8|vqk_q4|vqk_q8 (KV cache format, below; vqk_* also KURN_VQ=codebooks),
//      KURN_KLD=ref:file|cmp:file (ppl: KL vs a reference run), KURN_DUMP_KQ=prefix (calibration dump, below).
//      -DMAX_CTX=n raises the context limit (default 2048).
#define _GNU_SOURCE
#include "model_config.h"
#include "kq8e.h"
#include <fcntl.h>
#include <immintrin.h>
#include <linux/futex.h>
#include <math.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#ifndef KURN_EPILOGUE
#define KURN_EPILOGUE 1  // 0: gate/up stored as floats, then separate SwiGLU and quantize passes
#endif

#define MAXT 64
#ifndef MAX_CTX
#define MAX_CTX 2048
#endif
#define QB(K) ((K) / 32)
#define D HEAD_DIM
#define NG(n) (((n) + 15) / 16)
#define G_QPK (N_HEAD / N_KV)
#define ALIGN64(x) (((x) + 63) & ~(size_t)63)

static uint8_t *G_file;
static int T = 8;
static size_t BLK;  // bytes per packed (16-row group, 32-column block)

// matmul weight format, fixed per model (W_Q4_0 from model_config.h); activations are Q8_0 either way
#if W_Q4_0
typedef kq4e_block_q4_0 wblock_t;
#define W_BLK_BYTES kq4e_blk_bytes
#define W_PACK kq4e_pack
#define W_STORE kq4e_store
#define W_AXPY kq4e_axpy
#define W_SWIGLU_Q8 kq4e_swiglu_q8
static inline float w_val(const wblock_t *b, int i) {
    return _cvtsh_ss(b->d) * (float)((i < 16 ? b->qs[i] & 0x0F : b->qs[i - 16] >> 4) - 8);
}
#else
typedef block_q8_0 wblock_t;
#define W_BLK_BYTES kq8e_blk_bytes
#define W_PACK kq8e_pack
#define W_STORE kq8e_store
#define W_AXPY kq8e_axpy
#define W_SWIGLU_Q8 kq8e_swiglu_q8
static inline float w_val(const wblock_t *b, int i) { return _cvtsh_ss(b->d) * b->qs[i]; }
#endif

static const float *F32(int64_t off) { return (const float *)(G_file + off); }
static const wblock_t *QW(int64_t off) { return (const wblock_t *)(G_file + off); }

// ---------------------------------------------------------------- sync: per-thread epochs
enum { K_ATTN, K_QK, K_FFN, K_OUT, K_N };
static const char *K_NAME[K_N] = {"attn", "qk", "ffn", "out"};
typedef struct {
    _Alignas(64) atomic_int epoch;
    atomic_int sleepers;
    double part[2];   // OLMoE QK-norm partial sums of squares
    int arg;          // output: local argmax
    float max;
} arrive_t;
static _Alignas(64) arrive_t ARR[MAXT];
static int WAIT_MODE[K_N];   // 0 spin, 1 futex, 2 yield
static long WAIT_SPINS[K_N];
typedef struct { _Alignas(64) uint64_t wait[K_N], n[K_N], step; } prof_t;
static prof_t PROF[MAXT];

static inline void publish(int ith, int epoch) {
    atomic_store_explicit(&ARR[ith].epoch, epoch, memory_order_seq_cst);
    if (atomic_load_explicit(&ARR[ith].sleepers, memory_order_seq_cst))
        syscall(SYS_futex, &ARR[ith].epoch, FUTEX_WAKE_PRIVATE, MAXT, NULL, NULL, 0);
}
// prefetch-in-wait (KURN_PFWAIT=bytes): a thread that has to wait streams the start of the weights
// it runs next into L2 instead of only pausing
static long PF_MAX;
static _Thread_local const char *pf_p;
static _Thread_local long pf_left;
static inline void pf_next(const void *p, size_t n) { pf_p = p; pf_left = PF_MAX < (long)n ? PF_MAX : (long)n; }
__attribute__((noinline)) static void await_slow(int s, int epoch, int kind);
static inline void await_one(int s, int epoch, int kind) {
    if (atomic_load_explicit(&ARR[s].epoch, memory_order_acquire) < epoch) await_slow(s, epoch, kind);
}
__attribute__((noinline)) static void await_slow(int s, int epoch, int kind) {  // not inlined: perf attributes spinning here
    for (long i = 0; atomic_load_explicit(&ARR[s].epoch, memory_order_acquire) < epoch; i++) {
        if (pf_left > 0) {
            for (int j = 0; j < 16; j++) _mm_prefetch(pf_p + 64 * j, _MM_HINT_T1);
            pf_p += 1024; pf_left -= 1024;
            continue;
        }
        if (WAIT_MODE[kind] && i >= WAIT_SPINS[kind]) {
            if (WAIT_MODE[kind] == 2) { sched_yield(); continue; }
            atomic_fetch_add(&ARR[s].sleepers, 1);
            int v;
            while ((v = atomic_load(&ARR[s].epoch)) < epoch)
                syscall(SYS_futex, &ARR[s].epoch, FUTEX_WAIT_PRIVATE, v, NULL, NULL, 0);
            atomic_fetch_sub(&ARR[s].sleepers, 1);
            return;
        }
        _mm_pause();
    }
}

// ---------------------------------------------------------------- numerics (ggml-matching)
static inline float h2f(uint16_t h) { return _cvtsh_ss(h); }
static float rms_scale(const float *x, int n) {  // ggml_compute_forward_rms_norm_f32
    double ss = 0;
    for (int i = 0; i < n; i++) ss += (double)(x[i] * x[i]);
    const float mean = (float)(ss / n);
    return 1.0f / sqrtf(mean + RMS_EPS);
}
static void rmsnorm(const float *x, const float *w, float *o, int n) {
    const float sc = rms_scale(x, n);
    for (int i = 0; i < n; i++) o[i] = (x[i] * sc) * w[i];
}
static void rope_cache(int pos, float *cs) {  // ggml_rope_cache_init, NEOX, no scaling
    const float ts = powf(ROPE_BASE, -2.0f / D);
    float th = (float)pos;
    for (int i = 0; i < D / 2; i++) { cs[2 * i] = cosf(th); cs[2 * i + 1] = sinf(th); th *= ts; }
}
static void rope_neox(float *v, const float *cs) {
    for (int i = 0; i < D / 2; i++) {
        const float a = v[i], b = v[i + D / 2], c = cs[2 * i], s = cs[2 * i + 1];
        v[i] = a * c - b * s;
        v[i + D / 2] = a * s + b * c;
    }
}
static inline float dot_f32(const float *a, const float *b, int n) {
    __m512 s0 = _mm512_setzero_ps(), s1 = _mm512_setzero_ps();
    int i = 0;
    for (; i + 32 <= n; i += 32) {
        s0 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i), s0);
        s1 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i + 16), _mm512_loadu_ps(b + i + 16), s1);
    }
    float s = _mm512_reduce_add_ps(_mm512_add_ps(s0, s1));
    for (; i < n; i++) s += a[i] * b[i];
    return s;
}

// ---------------------------------------------------------------- per-thread schedule
typedef struct { int slot; int fb0, fb1; } unit_t;   // MoE: FF blocks [fb0, fb1) of selected slot
typedef struct {
    int kh0, kh1, qh0, qh1, leader;   // KV heads owned / Q heads computed / writes the KV cache
    int nq_rows, nqkv_rows;
    int fb0, fb1;                     // dense FF blocks (32 values each)
    int vg0, vg1;                     // output vocab groups (16 rows each)
    int nunits;
    unit_t unit[N_USED_OR1 + 1];
    void *qkv[N_LAYER], *o[N_LAYER];
#if !N_EXPERT
    void *gu[N_LAYER], *down[N_LAYER];
#endif
    void *out;
    uint8_t *arena; size_t arena_sz, arena_used;
} sched_t;
static sched_t S[MAXT];

#if N_EXPERT
static void *E_GU[N_LAYER][N_EXPERT], *E_DOWN[N_LAYER][N_EXPERT];
static uint8_t *E_POOL;
#endif

static float *P[2];     // [2][T][N_EMBD] partial residual vectors
static uint16_t *KC, *VC;  // [layer][kv head][pos][D], f16 (llama.cpp's default KV type)
static float *LOGITS;
static int USE_THP = 1;

static inline size_t kv_at(int l, int h, int pos) { return (((size_t)l * N_KV + h) * MAX_CTX + pos) * D; }
static inline float *part(int buf, int t) { return P[buf] + (size_t)t * N_EMBD; }

static void *big_alloc(size_t sz) {
    void *p = mmap(NULL, sz, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) { perror("mmap"); exit(1); }
    if (USE_THP) madvise(p, sz, MADV_HUGEPAGE);
    return p;
}
static void *arena_take(sched_t *s, size_t sz) {
    void *p = s->arena + s->arena_used;
    s->arena_used += ALIGN64(sz);
    if (s->arena_used > s->arena_sz) { fprintf(stderr, "arena overflow\n"); exit(1); }
    return p;
}

static void plan(void) {
    for (int t = 0; t < T; t++) {
        sched_t *s = &S[t];
        if (N_KV % T == 0) {
            s->kh0 = t * (N_KV / T); s->kh1 = s->kh0 + N_KV / T; s->qh0 = s->kh0 * G_QPK; s->qh1 = s->kh1 * G_QPK; s->leader = 1;
        } else if (T % N_KV == 0 && G_QPK % (T / N_KV) == 0) {
            const int tpk = T / N_KV, sub = t % tpk;
            s->kh0 = t / tpk; s->kh1 = s->kh0 + 1; s->leader = sub == 0;
            s->qh0 = s->kh0 * G_QPK + sub * (G_QPK / tpk); s->qh1 = s->qh0 + G_QPK / tpk;
        } else {
            s->kh0 = t * N_KV / T; s->kh1 = (t + 1) * N_KV / T; s->qh0 = s->kh0 * G_QPK; s->qh1 = s->kh1 * G_QPK; s->leader = 1;
        }
        s->nq_rows = (s->qh1 - s->qh0) * D;
        s->nqkv_rows = s->nq_rows + 2 * (s->kh1 - s->kh0) * D;
        const int nfb = N_FF / 32;
        s->fb0 = t * nfb / T; s->fb1 = (t + 1) * nfb / T;
        const int nvg = NG(N_VOCAB);
        s->vg0 = (int)((int64_t)t * nvg / T); s->vg1 = (int)((int64_t)(t + 1) * nvg / T);
#if N_EXPERT
        const int W = N_USED * nfb, w0 = t * W / T, w1 = (t + 1) * W / T;
        s->nunits = 0;
        for (int w = w0; w < w1;) {
            const int slot = w / nfb, end = (slot + 1) * nfb < w1 ? (slot + 1) * nfb : w1;
            if (s->nunits == N_USED_OR1 + 1) { fprintf(stderr, "too many MoE units per thread\n"); exit(1); }
            s->unit[s->nunits++] = (unit_t){slot, w - slot * nfb, end - slot * nfb};
            w = end;
        }
#endif
        size_t sz = 0;
        sz += N_LAYER * ALIGN64((size_t)NG(s->nqkv_rows) * QB(N_EMBD) * BLK);
        sz += N_LAYER * ALIGN64((size_t)NG(N_EMBD) * (s->nq_rows / 32) * BLK);
#if !N_EXPERT
        sz += N_LAYER * ALIGN64((size_t)NG(2 * 32 * (s->fb1 - s->fb0)) * QB(N_EMBD) * BLK);
        sz += N_LAYER * ALIGN64((size_t)NG(N_EMBD) * (s->fb1 - s->fb0) * BLK);
#endif
        sz += ALIGN64((size_t)(s->vg1 - s->vg0) * QB(N_EMBD) * BLK);
        s->arena_sz = sz + 4096;
    }
}

// pack rows [r0, r0+n) x blocks [c0, c0+nc) of a row-major weight matrix with `nb_src` blocks per row
static void pack_sub(void *dst, const wblock_t *src, int64_t nb_src, int64_t r0, int64_t n, int64_t c0, int64_t nc,
                     wblock_t *tmp) {
    for (int64_t r = 0; r < n; r++) memcpy(tmp + r * nc, src + (r0 + r) * nb_src + c0, nc * sizeof(wblock_t));
    W_PACK(dst, tmp, nc, n);
}

static void pack_thread(int t) {  // runs on thread t: per-thread arena is first-touched by its owner
    sched_t *s = &S[t];
    s->arena = big_alloc(s->arena_sz);
    const int64_t nbe = QB(N_EMBD);
    size_t tmpn = (size_t)(N_FF > N_EMBD ? N_FF : N_EMBD) * 2 * nbe;
    if ((size_t)(s->vg1 - s->vg0) * 16 * nbe > tmpn) tmpn = (size_t)(s->vg1 - s->vg0) * 16 * nbe;
    wblock_t *tmp = malloc(tmpn * sizeof(wblock_t));
    for (int l = 0; l < N_LAYER; l++) {
        // QKV: own Q heads, then own K heads, then own V heads
        wblock_t *p = tmp;
        const int nqr = s->nq_rows, nkr = (s->kh1 - s->kh0) * D;
        memcpy(p, QW(OFF_Q[l]) + (int64_t)s->qh0 * D * nbe, nqr * nbe * sizeof *p); p += nqr * nbe;
        memcpy(p, QW(OFF_K[l]) + (int64_t)s->kh0 * D * nbe, nkr * nbe * sizeof *p); p += nkr * nbe;
        memcpy(p, QW(OFF_V[l]) + (int64_t)s->kh0 * D * nbe, nkr * nbe * sizeof *p);
        s->qkv[l] = arena_take(s, (size_t)NG(s->nqkv_rows) * nbe * BLK);
        W_PACK(s->qkv[l], tmp, nbe, s->nqkv_rows);
        // O: all N_EMBD rows, columns of own Q heads
        s->o[l] = arena_take(s, (size_t)NG(N_EMBD) * (nqr / 32) * BLK);
        if (nqr) pack_sub(s->o[l], QW(OFF_O[l]), QB(N_HEAD * D), 0, N_EMBD, s->qh0 * D / 32, nqr / 32, tmp);
#if !N_EXPERT
        // gate/up: own FF slice, interleaved per 16 rows (gate 16j.., up 16j..)
        const int f0 = s->fb0 * 32, nf = (s->fb1 - s->fb0) * 32;
        for (int j = 0; j < nf / 16; j++) {
            memcpy(tmp + (int64_t)(32 * j) * nbe, QW(OFF_GATE[l]) + (int64_t)(f0 + 16 * j) * nbe, 16 * nbe * sizeof *tmp);
            memcpy(tmp + (int64_t)(32 * j + 16) * nbe, QW(OFF_UP[l]) + (int64_t)(f0 + 16 * j) * nbe, 16 * nbe * sizeof *tmp);
        }
        s->gu[l] = arena_take(s, (size_t)NG(2 * nf) * nbe * BLK);
        W_PACK(s->gu[l], tmp, nbe, 2 * nf);
        // down: all N_EMBD rows, columns of own FF slice
        s->down[l] = arena_take(s, (size_t)NG(N_EMBD) * (s->fb1 - s->fb0) * BLK);
        pack_sub(s->down[l], QW(OFF_DOWN[l]), QB(N_FF), 0, N_EMBD, s->fb0, s->fb1 - s->fb0, tmp);
#endif
    }
    s->out = arena_take(s, (size_t)(s->vg1 - s->vg0) * nbe * BLK);
    {
        const int64_t r0 = (int64_t)s->vg0 * 16, n = ((int64_t)s->vg1 * 16 < N_VOCAB ? (int64_t)s->vg1 * 16 : N_VOCAB) - r0;
        if (n > 0) pack_sub(s->out, QW(OFF_OUTPUT), nbe, r0, n, 0, nbe, tmp);
    }
#if N_EXPERT
    const size_t gsz = (size_t)NG(2 * N_FF) * nbe * BLK, dsz = (size_t)NG(N_EMBD) * QB(N_FF) * BLK;
    for (int i = t; i < N_LAYER * N_EXPERT; i += T) {
        const int l = i / N_EXPERT, e = i % N_EXPERT;
        uint8_t *base = E_POOL + (size_t)i * (gsz + dsz);
        const wblock_t *g = QW(OFF_GATE[l]) + (int64_t)e * N_FF * nbe, *u = QW(OFF_UP[l]) + (int64_t)e * N_FF * nbe;
        for (int j = 0; j < N_FF / 16; j++) {
            memcpy(tmp + (int64_t)(32 * j) * nbe, g + (int64_t)16 * j * nbe, 16 * nbe * sizeof *tmp);
            memcpy(tmp + (int64_t)(32 * j + 16) * nbe, u + (int64_t)16 * j * nbe, 16 * nbe * sizeof *tmp);
        }
        E_GU[l][e] = base;
        W_PACK(base, tmp, nbe, 2 * N_FF);
        E_DOWN[l][e] = base + gsz;
        W_PACK(base + gsz, QW(OFF_DOWN[l]) + (int64_t)e * N_EMBD * QB(N_FF), QB(N_FF), N_EMBD);
    }
#endif
    free(tmp);
}

// ---------------------------------------------------------------- per-thread working state
typedef struct {
    float H[N_EMBD], xn[N_EMBD], y[N_EMBD];
    _Alignas(64) block_q8_0 xq[QB(N_EMBD)];
    _Alignas(64) block_q8_0 aq[QB(N_HEAD * D)];
    _Alignas(64) block_q8_0 fq[QB(N_FF)];
    _Alignas(64) float qkv[N_QKV];
    _Alignas(64) float att[N_HEAD * D];
    _Alignas(64) float gu[2 * N_FF];
    _Alignas(64) float act[N_FF];
    float sc[MAX_CTX];
    float rope[D];
    kq8e_act ax, aa, af;
} work_t;
static work_t *WK[MAXT];

static int G_ntok, *G_toks, G_mode_ppl, G_ctx, G_ngen, G_nprompt;
static double G_nll; static long G_cnt;
static double *G_dt;
static uint64_t *G_wq, *G_cq;  // thread 0's sync-wait cycles and total cycles per generated token
static FILE *G_dump;  // KURN_DUMP_LOGITS=path: gen mode writes every step's logits (float32)

static inline void wait_kind(int ith, int epoch, int kind) {
    const uint64_t t0 = __rdtsc();
    for (int s = 0; s < T; s++) if (s != ith) await_one(s, epoch, kind);
    PROF[ith].wait[kind] += __rdtsc() - t0;
    PROF[ith].n[kind]++;
}
// H += sum_s P[buf][s] (same order on every thread, waiting for each producer); returns rms scale of H
static float reduce_into(int ith, work_t *w, int buf, int epoch, int kind) {
    const uint64_t t0 = __rdtsc();
    uint64_t waited = 0;
    float *y = w->y;
    for (int s = 0; s < T; s++) {
        if (s != ith) { const uint64_t a = __rdtsc(); await_one(s, epoch, kind); waited += __rdtsc() - a; }
        const float *p = part(buf, s);
        if (s == 0) memcpy(y, p, sizeof(float) * N_EMBD);
        else for (int i = 0; i < N_EMBD; i += 16) _mm512_storeu_ps(y + i, _mm512_add_ps(_mm512_loadu_ps(y + i), _mm512_loadu_ps(p + i)));
    }
    double ss = 0;
    for (int i = 0; i < N_EMBD; i++) { w->H[i] += y[i]; ss += (double)(w->H[i] * w->H[i]); }
    PROF[ith].wait[kind] += waited;
    PROF[ith].n[kind]++;
    (void)t0;
    return 1.0f / sqrtf((float)(ss / N_EMBD) + RMS_EPS);
}
static void norm_quant(work_t *w, float sc, int64_t off_w) {
    const float *nw = F32(off_w);
    for (int i = 0; i < N_EMBD; i++) w->xn[i] = (w->H[i] * sc) * nw[i];
    kq8e_quantize(w->xn, w->xq, N_EMBD);
    kq8e_prep(w->xq, QB(N_EMBD), &w->ax);
}

static void attention(int ith, work_t *w, int l, int pos) {
    sched_t *s = &S[ith];
    float *q = w->qkv, *k = q + s->nq_rows, *v = k + (s->kh1 - s->kh0) * D;
    const float scale = 1.0f / sqrtf((float)D);
    for (int kh = s->kh0; kh < s->kh1; kh++) {
        float *kk = k + (kh - s->kh0) * D, *vv = v + (kh - s->kh0) * D;
        rope_neox(kk, w->rope);
        uint16_t *kc = KC + kv_at(l, kh, 0), *vc = VC + kv_at(l, kh, 0);
        if (s->leader)
            for (int i = 0; i < D; i += 16) {
                _mm256_storeu_si256((__m256i *)(kc + (size_t)pos * D + i), _mm512_cvtps_ph(_mm512_loadu_ps(kk + i), 0));
                _mm256_storeu_si256((__m256i *)(vc + (size_t)pos * D + i), _mm512_cvtps_ph(_mm512_loadu_ps(vv + i), 0));
            }
        const int h0 = kh * G_QPK > s->qh0 ? kh * G_QPK : s->qh0, h1 = (kh + 1) * G_QPK < s->qh1 ? (kh + 1) * G_QPK : s->qh1;
        for (int h = h0; h < h1; h++) {
            float *qq = q + (h - s->qh0) * D;
            rope_neox(qq, w->rope);
            __m512 qv[D / 16];
            for (int i = 0; i < D / 16; i++) qv[i] = _mm512_loadu_ps(qq + 16 * i);
            float mx = -INFINITY;
            for (int t = 0; t <= pos; t++) {
                const uint16_t *kt = (t == pos) ? NULL : kc + (size_t)t * D;
                __m512 acc = _mm512_setzero_ps();
                if (kt)
                    for (int i = 0; i < D / 16; i++) acc = _mm512_fmadd_ps(qv[i], _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(kt + 16 * i))), acc);
                else  // current position: the f16-rounded k, as stored
                    for (int i = 0; i < D / 16; i++)
                        acc = _mm512_fmadd_ps(qv[i], _mm512_cvtph_ps(_mm512_cvtps_ph(_mm512_loadu_ps(kk + 16 * i), 0)), acc);
                const float sc = _mm512_reduce_add_ps(acc) * scale;
                w->sc[t] = sc;
                mx = sc > mx ? sc : mx;
            }
            float den = 0;
            for (int t = 0; t <= pos; t++) { w->sc[t] = expf(w->sc[t] - mx); den += w->sc[t]; }
            const float inv = 1.0f / den;
            __m512 o[D / 16];
            for (int i = 0; i < D / 16; i++) o[i] = _mm512_setzero_ps();
            for (int t = 0; t <= pos; t++) {
                const __m512 p = _mm512_set1_ps(w->sc[t] * inv);
                if (t < pos || s->leader) {
                    const uint16_t *vt = vc + (size_t)t * D;
                    for (int i = 0; i < D / 16; i++) o[i] = _mm512_fmadd_ps(p, _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(vt + 16 * i))), o[i]);
                } else {
                    for (int i = 0; i < D / 16; i++)
                        o[i] = _mm512_fmadd_ps(p, _mm512_cvtph_ps(_mm512_cvtps_ph(_mm512_loadu_ps(vv + 16 * i), 0)), o[i]);
                }
            }
            float *ao = w->att + (h - s->qh0) * D;
            for (int i = 0; i < D / 16; i++) _mm512_storeu_ps(ao + 16 * i, o[i]);
        }
    }
}

// ---------------------------------------------------------------- quantized KV cache (KURN_KV)
// KURN_KV=q8_0|q4_0 store post-RoPE K and V as ggml rows (llama.cpp -ctk/-ctv semantics);
// k4c_q4|k4c_q8 store K before RoPE, 4-bit per channel in 32-token groups (f16 scale and min per
// channel; the group being filled stays f16, as in kurn_attn.h) and V as Q4_0 / Q8_0. K is
// rotated when read with the same rope_cache values the f16 path applies. Every format writes
// the current token and reads it back, so attention sees exactly what the cache holds.
enum { KVF_F16, KVF_Q8_0, KVF_Q4_0, KVF_K4C_Q4, KVF_K4C_Q8, KVF_VQK_Q4, KVF_VQK_Q8, KVF_N };
static const char *KVF_NAME[KVF_N] = {"f16", "q8_0", "q4_0", "k4c_q4", "k4c_q8", "vqk_q4", "vqk_q8"};
static int KVF;
static uint8_t *KQ, *VQ;  // K rows or k4c group blocks; V rows
static uint16_t *KT;      // k4c: f16 pre-RoPE rows of the group being filled, [layer][kv head][32][D]
static float *RT;         // rope_cache(t) for every position, [MAX_CTX][D]
static float *SCQ[MAXT];  // per thread scores, [G_QPK][MAX_CTX]
static FILE *G_kld;
static void kld_step(double lse);
#define K4C_G 32
#define K4C_BLK (20 * D)
// vqk_*: TaSQ-style vector-quantized pre-RoPE K (KURN_VQ=codebook file from
// benchmarks/v0.2/kvformat/calibrate_vq.py): k is weighted per channel (sqrt of the query second
// moment of its RoPE pair), divided by its per-head RMS (stored f16), permuted so that groups of 8
// channels (whole RoPE pairs, covariance-aware) are contiguous, and each group stored as the index of
// the nearest of 1024 codewords. Groups of 32 tokens are encoded when complete (f16 tail before).
#define VQ_G 8
#define VQ_K 1024
#define VQ_NG (D / VQ_G)
typedef struct { float wsq[D]; int32_t perm[D]; float cb[VQ_NG][VQ_K][VQ_G]; float cn[VQ_NG][VQ_K]; } vq_head_t;
static vq_head_t *VQB;  // [N_LAYER][N_KV]
#define VQ_TOK (VQ_NG * 2 + 2)  // bytes per token in the engine (uint16 indices + f16 scale; 10-bit packing would be 22)
static inline int kvf_vq(void) { return KVF == KVF_VQK_Q4 || KVF == KVF_VQK_Q8; }
static inline int kvf_v_q4(void) { return KVF == KVF_Q4_0 || KVF == KVF_K4C_Q4 || KVF == KVF_VQK_Q4; }
static inline size_t kq_row(void) { return KVF == KVF_Q8_0 ? D / 32 * 34 : D / 32 * 18; }
static inline size_t vq_row(void) { return kvf_v_q4() ? D / 32 * 18 : D / 32 * 34; }
static inline uint8_t *kq_at(int l, int h, int pos) {
    if (kvf_vq()) return KQ + (((size_t)l * N_KV + h) * MAX_CTX + pos) * VQ_TOK;
    if (KVF >= KVF_K4C_Q4) return KQ + (((size_t)l * N_KV + h) * (MAX_CTX / K4C_G) + pos / K4C_G) * K4C_BLK;
    return KQ + (((size_t)l * N_KV + h) * MAX_CTX + pos) * kq_row();
}
static inline uint8_t *vq_at(int l, int h, int pos) { return VQ + (((size_t)l * N_KV + h) * MAX_CTX + pos) * vq_row(); }
static inline uint16_t *kt_at(int l, int h, int pos) { return KT + (((size_t)l * N_KV + h) * K4C_G + pos % K4C_G) * D; }

static void q4_0_row(const float *x, uint8_t *dst) {  // ggml quantize_row_q4_0_ref
    for (int b = 0; b < D / 32; b++) {
        float amax = 0, mx = 0;
        for (int i = 0; i < 32; i++) if (amax < fabsf(x[32 * b + i])) { amax = fabsf(x[32 * b + i]); mx = x[32 * b + i]; }
        const float d = mx / -8, id = d ? 1.0f / d : 0.0f;
        uint8_t *blk = dst + 18 * b;
        *(uint16_t *)blk = _cvtss_sh(d, 0);
        for (int i = 0; i < 16; i++) {
            const int lo = (int8_t)(x[32 * b + i] * id + 8.5f), hi = (int8_t)(x[32 * b + 16 + i] * id + 8.5f);
            blk[2 + i] = (uint8_t)((lo < 15 ? lo : 15) | ((hi < 15 ? hi : 15) << 4));
        }
    }
}
static void row_put(const float *x, uint8_t *dst, int q4) { if (q4) q4_0_row(x, dst); else kq8e_quantize(x, dst, D); }
static void row_get(const uint8_t *src, float *out, int q4) {
    for (int b = 0; b < D / 32; b++) {
        if (q4) {
            const float d = _cvtsh_ss(*(const uint16_t *)(src + 18 * b));
            for (int i = 0; i < 16; i++) {
                out[32 * b + i] = d * ((src[18 * b + 2 + i] & 15) - 8);
                out[32 * b + 16 + i] = d * ((src[18 * b + 2 + i] >> 4) - 8);
            }
        } else {
            const float d = _cvtsh_ss(*(const uint16_t *)(src + 34 * b));
            for (int i = 0; i < 32; i++) out[32 * b + i] = d * ((const int8_t *)(src + 34 * b + 2))[i];
        }
    }
}
static void vq_put(int l, int h, int t, const float *k) {
    const vq_head_t *b = VQB + (size_t)l * N_KV + h;
    float x[D], ss = 0;
    for (int c = 0; c < D; c++) { x[c] = b->wsq[c] * k[c]; ss += x[c] * x[c]; }
    const uint16_t sh = _cvtss_sh(sqrtf(ss / D), 0);
    const float s = _cvtsh_ss(sh), is = s > 0 ? 1.0f / s : 0.0f;
    uint16_t *z = (uint16_t *)kq_at(l, h, t);
    for (int g = 0; g < VQ_NG; g++) {
        float v[VQ_G];
        for (int i = 0; i < VQ_G; i++) v[i] = x[b->perm[g * VQ_G + i]] * is;
        int best = 0;
        float bd = INFINITY;
        for (int j = 0; j < VQ_K; j++) {
            float d = b->cn[g][j];
            for (int i = 0; i < VQ_G; i++) d -= 2.0f * b->cb[g][j][i] * v[i];
            if (d < bd) { bd = d; best = j; }
        }
        z[g] = (uint16_t)best;
    }
    z[VQ_NG] = sh;
}
static void vq_get(int l, int h, int t, float *out) {
    const vq_head_t *b = VQB + (size_t)l * N_KV + h;
    const uint16_t *z = (const uint16_t *)kq_at(l, h, t);
    const float s = _cvtsh_ss(z[VQ_NG]);
    for (int g = 0; g < VQ_NG; g++)
        for (int i = 0; i < VQ_G; i++) {
            const int c = b->perm[g * VQ_G + i];
            out[c] = s * b->cb[g][z[g]][i] / b->wsq[c];
        }
}

// quantize the complete group of tokens [32 g, 32 g + 32) from its f16 tail rows
static void k4c_seal(int l, int h, int pos) {
    if (kvf_vq()) {
        float k[D];
        for (int t = 0; t < K4C_G; t++) {
            for (int c = 0; c < D; c++) k[c] = _cvtsh_ss(kt_at(l, h, t)[c]);
            vq_put(l, h, pos - K4C_G + 1 + t, k);
        }
        return;
    }
    uint8_t *blk = kq_at(l, h, pos);
    uint16_t *sc = (uint16_t *)blk, *mn = sc + D;
    uint8_t *qs = blk + 4 * D;
    memset(qs, 0, 16 * D);
    for (int c = 0; c < D; c++) {
        float lo = INFINITY, hi = -INFINITY;
        for (int t = 0; t < K4C_G; t++) { const float x = _cvtsh_ss(kt_at(l, h, t)[c]); lo = fminf(lo, x); hi = fmaxf(hi, x); }
        sc[c] = _cvtss_sh((hi - lo) / 15.0f, 0);
        mn[c] = _cvtss_sh(lo, 0);
        const float s = _cvtsh_ss(sc[c]), m = _cvtsh_ss(mn[c]);
        for (int t = 0; t < K4C_G; t++) {
            long q = s > 0 ? lrintf((_cvtsh_ss(kt_at(l, h, t)[c]) - m) / s) : 0;
            q = q < 0 ? 0 : q > 15 ? 15 : q;
            qs[t * (D / 2) + (c / 32) * 16 + (c % 16)] |= (uint8_t)(q << ((c % 32) >= 16 ? 4 : 0));
        }
    }
}
// K of position t as attention sees it (post-RoPE), with pos + 1 tokens in the cache
static void k_get(int l, int h, int t, int pos, float *out) {
    if (KVF < KVF_K4C_Q4) { row_get(kq_at(l, h, t), out, KVF == KVF_Q4_0); return; }
    if (kvf_vq() && t < (pos + 1) / K4C_G * K4C_G) {
        vq_get(l, h, t, out);
    } else if (t < (pos + 1) / K4C_G * K4C_G) {
        const uint8_t *blk = kq_at(l, h, t);
        const uint16_t *sc = (const uint16_t *)blk, *mn = sc + D;
        const uint8_t *qs = blk + 4 * D + (t % K4C_G) * (D / 2);
        for (int c = 0; c < D; c++) out[c] = (float)((qs[(c / 32) * 16 + (c % 16)] >> ((c % 32) >= 16 ? 4 : 0)) & 15) * _cvtsh_ss(sc[c]) + _cvtsh_ss(mn[c]);
    } else {
        const uint16_t *tr = kt_at(l, h, t);
        for (int c = 0; c < D; c++) out[c] = _cvtsh_ss(tr[c]);
    }
    rope_neox(out, RT + (size_t)t * D);
}

static void attention_q(int ith, work_t *w, int l, int pos) {
    sched_t *s = &S[ith];
    float *q = w->qkv, *k = q + s->nq_rows, *v = k + (s->kh1 - s->kh0) * D;
    const float scale = 1.0f / sqrtf((float)D);
    float *sc = SCQ[ith];
    _Alignas(64) float kr[D], vr[D];
    for (int kh = s->kh0; kh < s->kh1; kh++) {
        float *kk = k + (kh - s->kh0) * D, *vv = v + (kh - s->kh0) * D;
        if (KVF >= KVF_K4C_Q4) {
            for (int i = 0; i < D; i++) kt_at(l, kh, pos)[i] = _cvtss_sh(kk[i], 0);
            if (pos % K4C_G == K4C_G - 1) k4c_seal(l, kh, pos);
        } else {
            rope_neox(kk, w->rope);
            row_put(kk, kq_at(l, kh, pos), KVF == KVF_Q4_0);
        }
        row_put(vv, vq_at(l, kh, pos), kvf_v_q4());
        const int h0 = kh * G_QPK > s->qh0 ? kh * G_QPK : s->qh0, h1 = (kh + 1) * G_QPK < s->qh1 ? (kh + 1) * G_QPK : s->qh1;
        for (int h = h0; h < h1; h++) rope_neox(q + (h - s->qh0) * D, w->rope);
        for (int t = 0; t <= pos; t++) {
            k_get(l, kh, t, pos, kr);
            for (int h = h0; h < h1; h++) sc[(size_t)(h - h0) * MAX_CTX + t] = dot_f32(q + (h - s->qh0) * D, kr, D) * scale;
        }
        float inv[G_QPK];
        for (int h = h0; h < h1; h++) {
            float *sh = sc + (size_t)(h - h0) * MAX_CTX, mx = -INFINITY, den = 0;
            for (int t = 0; t <= pos; t++) mx = sh[t] > mx ? sh[t] : mx;
            for (int t = 0; t <= pos; t++) { sh[t] = expf(sh[t] - mx); den += sh[t]; }
            inv[h - h0] = 1.0f / den;
            memset(w->att + (h - s->qh0) * D, 0, sizeof(float) * D);
        }
        for (int t = 0; t <= pos; t++) {
            row_get(vq_at(l, kh, t), vr, kvf_v_q4());
            for (int h = h0; h < h1; h++) {
                const float p = sc[(size_t)(h - h0) * MAX_CTX + t] * inv[h - h0];
                float *ao = w->att + (h - s->qh0) * D;
                for (int i = 0; i < D; i++) ao[i] += p * vr[i];
            }
        }
    }
}

// KURN_DUMP_KQ=prefix: calibration data for KV quantizers. Thread t appends, per step and layer,
// the pre-RoPE K rows of its kv heads as f16 to prefix.t (kh0..kh1, D each) and accumulates the
// per-channel second moment of its pre-RoPE queries per kv head, written to prefix.qq.t at exit as
// float64 [N_LAYER][kh1 - kh0][D] followed by the number of steps.
static FILE *G_dkq[MAXT];
static double *G_qq[MAXT];
static long G_qq_n[MAXT];
static void dump_kq(int ith, work_t *w, int l) {
    sched_t *s = &S[ith];
    const float *q = w->qkv, *k = q + s->nq_rows;
    uint16_t row[D];
    for (int kh = s->kh0; kh < s->kh1; kh++) {
        for (int i = 0; i < D; i++) row[i] = _cvtss_sh(k[(kh - s->kh0) * D + i], 0);
        if (fwrite(row, sizeof row, 1, G_dkq[ith]) != 1) { perror("dump kq"); exit(1); }
        double *acc = G_qq[ith] + ((size_t)l * (s->kh1 - s->kh0) + (kh - s->kh0)) * D;
        for (int h = kh * G_QPK; h < (kh + 1) * G_QPK; h++)
            if (h >= s->qh0 && h < s->qh1)
                for (int i = 0; i < D; i++) acc[i] += (double)q[(h - s->qh0) * D + i] * q[(h - s->qh0) * D + i];
    }
    if (l == 0) G_qq_n[ith]++;
}

// one token on thread ith; `ep` is this thread's running epoch (identical on every thread)
static int step(int ith, int tok, int pos, int *ep) {
    sched_t *s = &S[ith];
    work_t *w = WK[ith];
    const int64_t nbe = QB(N_EMBD);
    const wblock_t *e = QW(OFF_TOK_EMBD) + (int64_t)tok * nbe;
    for (int i = 0; i < N_EMBD; i++) w->H[i] = w_val(e + i / 32, i % 32);
    float sc = rms_scale(w->H, N_EMBD);
    rope_cache(pos, w->rope);
    int buf = 0;
    for (int l = 0; l < N_LAYER; l++) {
        // ---------------- attention block: QKV (own heads) -> attention -> O column slice -> partial
        norm_quant(w, sc, OFF_ATTN_NORM[l]);
        W_STORE(s->qkv[l], nbe, &w->ax, 0, nbe, 0, NG(s->nqkv_rows), w->qkv);
        {
            float *q = w->qkv, *k = q + s->nq_rows;
            const int nkr = (s->kh1 - s->kh0) * D;
#if QK_NORM_HEAD
            for (int h = 0; h < s->qh1 - s->qh0; h++) rmsnorm(q + h * D, F32(OFF_Q_NORM[l]), q + h * D, D);
            for (int h = 0; h < s->kh1 - s->kh0; h++) rmsnorm(k + h * D, F32(OFF_K_NORM[l]), k + h * D, D);
#endif
#if QK_NORM_FULL
            double sq = 0, sk = 0;
            for (int i = 0; i < s->nq_rows; i++) sq += (double)(q[i] * q[i]);
            if (s->leader) for (int i = 0; i < nkr; i++) sk += (double)(k[i] * k[i]);
            ARR[ith].part[0] = sq; ARR[ith].part[1] = sk;
            pf_next(s->o[l], (size_t)NG(N_EMBD) * (s->nq_rows / 32) * BLK);
            publish(ith, ++*ep);
            wait_kind(ith, *ep, K_QK);
            sq = sk = 0;
            for (int t = 0; t < T; t++) { sq += ARR[t].part[0]; sk += ARR[t].part[1]; }
            const float scq = 1.0f / sqrtf((float)(sq / (N_HEAD * D)) + RMS_EPS), sck = 1.0f / sqrtf((float)(sk / (N_KV * D)) + RMS_EPS);
            const float *wq = F32(OFF_Q_NORM[l]) + s->qh0 * D, *wk = F32(OFF_K_NORM[l]) + s->kh0 * D;
            for (int i = 0; i < s->nq_rows; i++) q[i] = (q[i] * scq) * wq[i];
            for (int i = 0; i < nkr; i++) k[i] = (k[i] * sck) * wk[i];
#endif
            (void)nkr;
        }
        if (G_dkq[ith]) dump_kq(ith, w, l);
        if (KVF) attention_q(ith, w, l, pos);
        else attention(ith, w, l, pos);
        float *pt = part(buf, ith);
        if (s->nq_rows) {
            kq8e_quantize(w->att, w->aq, s->nq_rows);
            kq8e_prep(w->aq, s->nq_rows / 32, &w->aa);
            W_STORE(s->o[l], s->nq_rows / 32, &w->aa, 0, s->nq_rows / 32, 0, NG(N_EMBD), pt);
        } else memset(pt, 0, sizeof(float) * N_EMBD);
#if N_EXPERT
        pf_next(F32(OFF_ROUTER[l]), sizeof(float) * N_EXPERT * N_EMBD);
#else
        pf_next(s->gu[l], (size_t)NG(64 * (s->fb1 - s->fb0)) * nbe * BLK);
#endif
        publish(ith, ++*ep);
        sc = reduce_into(ith, w, buf, *ep, K_ATTN);
        buf ^= 1;
        // ---------------- feed-forward block
        norm_quant(w, sc, OFF_FFN_NORM[l]);
        pt = part(buf, ith);
#if N_EXPERT
        {
            float lg[N_EXPERT], mx = -INFINITY, den = 0;
            const float *wr = F32(OFF_ROUTER[l]);
            for (int x = 0; x < N_EXPERT; x++) { lg[x] = dot_f32(wr + (size_t)x * N_EMBD, w->xn, N_EMBD); mx = lg[x] > mx ? lg[x] : mx; }
            for (int x = 0; x < N_EXPERT; x++) { lg[x] = expf(lg[x] - mx); den += lg[x]; }
            int sel[N_USED], used[N_EXPERT] = {0};
            float sw[N_USED], wsum = 0;
            for (int j = 0; j < N_USED; j++) {
                int best = -1;
                for (int x = 0; x < N_EXPERT; x++) if (!used[x] && (best < 0 || lg[x] > lg[best])) best = x;
                used[best] = 1; sel[j] = best; sw[j] = lg[best] / den; wsum += sw[j];
            }
            if (EXPERT_WEIGHTS_NORM) for (int j = 0; j < N_USED; j++) sw[j] /= wsum;
            memset(pt, 0, sizeof(float) * N_EMBD);
            for (int u = 0; u < s->nunits; u++) {
                const unit_t un = s->unit[u];
                const int ex = sel[un.slot], nfb = un.fb1 - un.fb0;
#if KURN_EPILOGUE
                W_SWIGLU_Q8(E_GU[l][ex], nbe, &w->ax, 0, nbe, 4 * un.fb0, 4 * un.fb1, w->fq, NULL);
#else
                W_STORE(E_GU[l][ex], nbe, &w->ax, 0, nbe, 4 * un.fb0, 4 * un.fb1, w->gu);
                for (int j = 2 * un.fb0; j < 2 * un.fb1; j++) kq8e_swiglu(w->gu + 32 * j, w->gu + 32 * j + 16, w->act + 16 * j, 16);
                kq8e_quantize(w->act + 32 * un.fb0, w->fq + un.fb0, 32 * nfb);
#endif
                kq8e_prep(w->fq + un.fb0, nfb, &w->af);
                W_AXPY(E_DOWN[l][ex], QB(N_FF), &w->af, un.fb0, un.fb1, 0, NG(N_EMBD), pt, sw[un.slot]);
            }
        }
#else
        {
            const int nfb = s->fb1 - s->fb0;
#if KURN_EPILOGUE
            W_SWIGLU_Q8(s->gu[l], nbe, &w->ax, 0, nbe, 0, 4 * nfb, w->fq, NULL);
#else
            W_STORE(s->gu[l], nbe, &w->ax, 0, nbe, 0, 4 * nfb, w->gu);
            for (int j = 0; j < 2 * nfb; j++) kq8e_swiglu(w->gu + 32 * j, w->gu + 32 * j + 16, w->act + 16 * j, 16);
            kq8e_quantize(w->act, w->fq, 32 * nfb);
#endif
            kq8e_prep(w->fq, nfb, &w->af);
            W_STORE(s->down[l], nfb, &w->af, 0, nfb, 0, NG(N_EMBD), pt);
        }
#endif
        if (l + 1 < N_LAYER) pf_next(s->qkv[l + 1], (size_t)NG(s->nqkv_rows) * nbe * BLK);
        else pf_next(s->out, (size_t)(s->vg1 - s->vg0) * nbe * BLK);
        publish(ith, ++*ep);
        sc = reduce_into(ith, w, buf, *ep, K_FFN);
        buf ^= 1;
    }
    // ---------------- output: own vocab rows, local argmax, exchange
    norm_quant(w, sc, OFF_OUT_NORM);
    int best = -1; float bmax = -INFINITY;
    if (s->vg1 > s->vg0) {
        float *lg = LOGITS + (int64_t)s->vg0 * 16;
        W_STORE(s->out, nbe, &w->ax, 0, nbe, 0, s->vg1 - s->vg0, lg);
        const int n = (int)(((int64_t)s->vg1 * 16 < N_VOCAB ? (int64_t)s->vg1 * 16 : N_VOCAB) - (int64_t)s->vg0 * 16);
        for (int i = 0; i < n; i++) if (lg[i] > bmax) { bmax = lg[i]; best = s->vg0 * 16 + i; }
    }
    ARR[ith].arg = best; ARR[ith].max = bmax;
    pf_next(s->qkv[0], (size_t)NG(s->nqkv_rows) * nbe * BLK);
    publish(ith, ++*ep);
    wait_kind(ith, *ep, K_OUT);
    int gb = -1; float gm = -INFINITY;
    for (int t = 0; t < T; t++) if (ARR[t].arg >= 0 && (gb < 0 || ARR[t].max > gm)) { gm = ARR[t].max; gb = ARR[t].arg; }
    return gb;
}

static double now(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec + ts.tv_nsec * 1e-9; }
static double cpu_s(void) {
    struct rusage r; getrusage(RUSAGE_SELF, &r);
    return r.ru_utime.tv_sec + r.ru_utime.tv_usec * 1e-6 + r.ru_stime.tv_sec + r.ru_stime.tv_usec * 1e-6;
}

static pthread_barrier_t START;
static int G_fd, G_drop_cache;
static size_t G_file_sz;
static uint64_t G_rdtsc0, G_rdtsc1;
static double G_w0, G_c0, G_w1, G_c1;
// time this thread was runnable but not running (preempted by other work), ns: schedstat field 2
static uint64_t runq_ns(void) {
    unsigned long long run = 0, wait = 0;
    FILE *f = fopen("/proc/thread-self/schedstat", "r");
    if (f) { if (fscanf(f, "%llu %llu", &run, &wait) != 2) wait = 0; fclose(f); }
    return wait;
}
static uint64_t G_runq[MAXT];

// KURN_PERF_CTL=fifo: `perf record -D -1 --control fifo:FIFO` then profiles the decode loop only
static void perf_ctl(const char *cmd) {
    const char *p = getenv("KURN_PERF_CTL");
    if (!p) return;
    int fd = open(p, O_WRONLY);
    if (fd >= 0) { if (write(fd, cmd, strlen(cmd)) < 0) perror("perf ctl"); close(fd); }
}

static void run(int ith) {
    int ep = 0;
    if (!getenv("KURN_PIN") || strcmp(getenv("KURN_PIN"), "0")) {
        cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(ith % CPU_SETSIZE, &cs); sched_setaffinity(0, sizeof cs, &cs);
    }
    pack_thread(ith);
    WK[ith] = aligned_alloc(64, ALIGN64(sizeof(work_t)));
    memset(WK[ith], 0, sizeof(work_t));
    pthread_barrier_wait(&START);
    if (ith == 0 && G_drop_cache) {  // the packed copy is all we stream; let the file's page cache go
        madvise(G_file, G_file_sz, MADV_DONTNEED);
        posix_fadvise(G_fd, 0, 0, POSIX_FADV_DONTNEED);
    }
    if (!G_mode_ppl) {
        int next = 0;
        for (int i = 0; i < G_nprompt; i++) {
            next = step(ith, G_toks[i], i, &ep);
            if (ith == 0 && G_dump) fwrite(LOGITS, sizeof(float), N_VOCAB, G_dump);
        }
        pthread_barrier_wait(&START);
        G_runq[ith] = runq_ns();
        if (ith == 0) { perf_ctl("enable\n"); memset(PROF, 0, sizeof PROF); G_w0 = now(); G_c0 = cpu_s(); G_rdtsc0 = __rdtsc(); printf("gen:"); }
        pthread_barrier_wait(&START);
        for (int i = 0; i < G_ngen; i++) {
            double t0 = ith == 0 ? now() : 0;
            uint64_t c0 = 0, w0 = 0;
            if (ith == 0) { printf(" %d", next); c0 = __rdtsc(); for (int k = 0; k < K_N; k++) w0 += PROF[0].wait[k]; }
            next = step(ith, next, G_nprompt + i, &ep);
            if (ith == 0) {
                G_dt[i] = now() - t0;
                G_cq[i] = __rdtsc() - c0;
                G_wq[i] = -w0;
                for (int k = 0; k < K_N; k++) G_wq[i] += PROF[0].wait[k];
            }
            if (ith == 0 && G_dump) fwrite(LOGITS, sizeof(float), N_VOCAB, G_dump);
        }
        G_runq[ith] = runq_ns() - G_runq[ith];
        pthread_barrier_wait(&START);
        if (ith == 0) { G_rdtsc1 = __rdtsc(); G_w1 = now(); G_c1 = cpu_s(); perf_ctl("disable\n"); }
    } else {
        if (ith == 0) { memset(PROF, 0, sizeof PROF); G_w0 = now(); G_c0 = cpu_s(); G_rdtsc0 = __rdtsc(); }
        int nchunks = getenv("PPL_CHUNKS") ? atoi(getenv("PPL_CHUNKS")) : 4;
        long steps = 0;
        for (int c = 0; (c + 1) * G_ctx <= G_ntok && c < nchunks; c++)
            for (int i = 0; i < G_ctx - 1; i++) {
                step(ith, G_toks[c * G_ctx + i], i, &ep);
                steps++;
                if (ith == 0 && i >= G_ctx / 2) {
                    double mx = -INFINITY, den = 0;
                    for (int v = 0; v < N_VOCAB; v++) mx = fmax(mx, LOGITS[v]);
                    for (int v = 0; v < N_VOCAB; v++) den += exp(LOGITS[v] - mx);
                    G_nll += -(LOGITS[G_toks[c * G_ctx + i + 1]] - mx - log(den));
                    G_cnt++;
                    if (G_kld) kld_step(mx + log(den));
                }
            }
        pthread_barrier_wait(&START);
        if (ith == 0) { G_rdtsc1 = __rdtsc(); G_w1 = now(); G_c1 = cpu_s(); G_ngen = (int)steps; }
    }
}
static void *worker(void *arg) { run((int)(intptr_t)arg); return NULL; }

// KURN_KLD=ref:PATH (ppl mode) writes the top-32 log-probs of every scored token; cmp:PATH reads
// them back and reports KL(ref || this run) over the top 32 plus one bucket for the rest (a lower
// bound of the full-vocabulary KL) and how often the argmax matches the reference's.
#define KLD_TOP 32
static int G_kld_cmp;
static double G_kl;
static long G_kl_n, G_top1;
static void kld_step(double lse) {
    int32_t id[KLD_TOP];
    float lp[KLD_TOP];
    if (!G_kld_cmp) {
        int n = 0;
        for (int v = 0; v < N_VOCAB; v++) {
            const float x = LOGITS[v];
            if (n == KLD_TOP && x <= lp[n - 1]) continue;
            int i = n < KLD_TOP ? n++ : n - 1;
            while (i > 0 && lp[i - 1] < x) { lp[i] = lp[i - 1]; id[i] = id[i - 1]; i--; }
            lp[i] = x; id[i] = v;
        }
        for (int i = 0; i < KLD_TOP; i++) lp[i] = (float)(lp[i] - lse);
        if (fwrite(id, sizeof id, 1, G_kld) != 1 || fwrite(lp, sizeof lp, 1, G_kld) != 1) { perror("kld write"); exit(1); }
        return;
    }
    if (fread(id, sizeof id, 1, G_kld) != 1 || fread(lp, sizeof lp, 1, G_kld) != 1) { fprintf(stderr, "kld: reference too short\n"); exit(1); }
    double ps = 0, qs = 0, kl = 0;
    for (int i = 0; i < KLD_TOP; i++) {
        const double p = exp(lp[i]), lq = LOGITS[id[i]] - lse;
        kl += p * (lp[i] - lq);
        ps += p;
        qs += exp(lq);
    }
    const double rp = fmax(1 - ps, 1e-12), rq = fmax(1 - qs, 1e-12);
    kl += rp * log(rp / rq);
    int best = 0;
    for (int v = 1; v < N_VOCAB; v++) if (LOGITS[v] > LOGITS[best]) best = v;
    G_kl += kl;
    G_top1 += best == id[0];
    G_kl_n++;
}

static void parse_wait(void) {
    const char *e = getenv("KURN_WAIT");
    if (!e) return;
    char buf[256]; snprintf(buf, sizeof buf, "%s", e);
    for (char *tok = strtok(buf, ","); tok; tok = strtok(NULL, ",")) {
        int kinds[K_N], nk = 0;
        char *val = strchr(tok, '=');
        if (val) { *val++ = 0; for (int k = 0; k < K_N; k++) if (!strcmp(tok, K_NAME[k])) kinds[nk++] = k; }
        else { val = tok; for (int k = 0; k < K_N; k++) kinds[nk++] = k; }
        int mode = 0; long spins = 0;
        if (!strncmp(val, "futex", 5)) { mode = 1; spins = val[5] == ':' ? atol(val + 6) : 0; }
        else if (!strncmp(val, "yield", 5)) { mode = 2; spins = val[5] == ':' ? atol(val + 6) : 0; }
        for (int k = 0; k < nk; k++) { WAIT_MODE[kinds[k]] = mode; WAIT_SPINS[kinds[k]] = spins; }
    }
}

int main(int argc, char **argv) {
    if (argc < 6) { fprintf(stderr, "usage: %s model.gguf gen|ppl THREADS N_GEN|CTX tokens\n", argv[0]); return 2; }
    T = atoi(argv[3]);
    if (T < 1 || T > MAXT) return 2;
    parse_wait();
    if (getenv("KURN_THP") && !strcmp(getenv("KURN_THP"), "0")) USE_THP = 0;
    if (getenv("KURN_PFWAIT")) PF_MAX = atol(getenv("KURN_PFWAIT"));
    BLK = W_BLK_BYTES();
    int fd = open(argv[1], O_RDONLY);
    if (fd < 0) { perror("open"); return 1; }
    struct stat st; fstat(fd, &st);
    G_file = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (G_file == MAP_FAILED) { perror("mmap"); return 1; }
    madvise(G_file, st.st_size, MADV_WILLNEED);
    G_fd = fd; G_file_sz = st.st_size; G_drop_cache = st.st_size > (4LL << 30);
    double t0 = now();
    plan();
#if N_EXPERT
    {
        const size_t gsz = (size_t)NG(2 * N_FF) * QB(N_EMBD) * BLK, dsz = (size_t)NG(N_EMBD) * QB(N_FF) * BLK;
        E_POOL = big_alloc((size_t)N_LAYER * N_EXPERT * (gsz + dsz));
    }
#endif
    for (int b = 0; b < 2; b++) P[b] = aligned_alloc(64, sizeof(float) * N_EMBD * T);
    KC = big_alloc(sizeof(uint16_t) * (size_t)N_LAYER * N_KV * MAX_CTX * D);
    VC = big_alloc(sizeof(uint16_t) * (size_t)N_LAYER * N_KV * MAX_CTX * D);
    if (getenv("KURN_KV")) {
        KVF = -1;
        for (int f = 0; f < KVF_N; f++) if (!strcmp(getenv("KURN_KV"), KVF_NAME[f])) KVF = f;
        if (KVF < 0) { fprintf(stderr, "KURN_KV=%s: expected f16, q8_0, q4_0, k4c_q4, k4c_q8, vqk_q4 or vqk_q8\n", getenv("KURN_KV")); return 2; }
        if (KVF && T > N_KV) { fprintf(stderr, "KURN_KV=%s needs THREADS <= %d (one owner per kv head)\n", KVF_NAME[KVF], N_KV); return 2; }
        if (KVF && (MAX_CTX % K4C_G || D % 32)) { fprintf(stderr, "KURN_KV needs MAX_CTX %% 32 == 0 and head_dim %% 32 == 0\n"); return 2; }
    }
    if (kvf_vq()) {
        const char *p = getenv("KURN_VQ");
        FILE *f = p ? fopen(p, "rb") : NULL;
        int32_t hdr[6];
        if (!f || fread(hdr, sizeof hdr, 1, f) != 1 || hdr[0] != 0x4B565131 || hdr[1] != N_LAYER || hdr[2] != N_KV || hdr[3] != D || hdr[4] != VQ_G || hdr[5] != VQ_K) {
            fprintf(stderr, "KURN_KV=%s needs KURN_VQ=codebook file for this model (calibrate_vq.py)\n", KVF_NAME[KVF]);
            return 2;
        }
        VQB = big_alloc(sizeof(vq_head_t) * N_LAYER * N_KV);
        for (int i = 0; i < N_LAYER * N_KV; i++) {
            vq_head_t *b = VQB + i;
            if (fread(b->wsq, sizeof b->wsq, 1, f) != 1 || fread(b->perm, sizeof b->perm, 1, f) != 1 || fread(b->cb, sizeof b->cb, 1, f) != 1) {
                fprintf(stderr, "%s: truncated\n", p);
                return 2;
            }
            for (int g = 0; g < VQ_NG; g++)
                for (int j = 0; j < VQ_K; j++) {
                    float n = 0;
                    for (int c = 0; c < VQ_G; c++) n += b->cb[g][j][c] * b->cb[g][j][c];
                    b->cn[g][j] = n;
                }
        }
        fclose(f);
    }
    if (KVF) {
        const size_t kbytes = kvf_vq() ? (size_t)MAX_CTX * VQ_TOK : KVF >= KVF_K4C_Q4 ? (size_t)MAX_CTX / K4C_G * K4C_BLK : (size_t)MAX_CTX * kq_row();
        KQ = big_alloc((size_t)N_LAYER * N_KV * kbytes);
        VQ = big_alloc((size_t)N_LAYER * N_KV * MAX_CTX * vq_row());
        KT = big_alloc(sizeof(uint16_t) * (size_t)N_LAYER * N_KV * K4C_G * D);
        RT = big_alloc(sizeof(float) * (size_t)MAX_CTX * D);
        for (int t = 0; t < MAX_CTX; t++) rope_cache(t, RT + (size_t)t * D);
        for (int t = 0; t < T; t++) SCQ[t] = aligned_alloc(64, ALIGN64(sizeof(float) * G_QPK * MAX_CTX));
    }
    if (getenv("KURN_DUMP_KQ")) {
        if (T > N_KV) { fprintf(stderr, "KURN_DUMP_KQ needs THREADS <= %d\n", N_KV); return 2; }
        for (int t = 0; t < T; t++) {
            char p[4096];
            snprintf(p, sizeof p, "%s.%d", getenv("KURN_DUMP_KQ"), t);
            if (!(G_dkq[t] = fopen(p, "wb"))) { perror(p); return 1; }
            G_qq[t] = calloc((size_t)N_LAYER * (S[t].kh1 - S[t].kh0) * D, sizeof(double));
        }
    }
    if (getenv("KURN_KLD")) {
        const char *e = getenv("KURN_KLD");
        G_kld_cmp = !strncmp(e, "cmp:", 4);
        if ((!G_kld_cmp && strncmp(e, "ref:", 4)) || !(G_kld = fopen(e + 4, G_kld_cmp ? "rb" : "wb"))) { fprintf(stderr, "KURN_KLD=ref:PATH|cmp:PATH\n"); return 2; }
    }
    LOGITS = aligned_alloc(64, sizeof(float) * ((size_t)NG(N_VOCAB) * 16 + 64));
    if (!strcmp(argv[2], "gen")) {
        G_ngen = atoi(argv[4]);
        G_toks = malloc(sizeof(int) * MAX_CTX);
        for (char *p = strtok(argv[5], ","); p; p = strtok(NULL, ",")) G_toks[G_nprompt++] = atoi(p);
        if (G_nprompt + G_ngen > MAX_CTX) { fprintf(stderr, "context too long\n"); return 2; }
        G_dt = malloc(sizeof(double) * (G_ngen + 1));
        G_wq = malloc(sizeof(uint64_t) * (G_ngen + 1));
        G_cq = malloc(sizeof(uint64_t) * (G_ngen + 1));
        if (getenv("KURN_DUMP_LOGITS") && !(G_dump = fopen(getenv("KURN_DUMP_LOGITS"), "wb"))) { perror("dump"); return 1; }
    } else {
        G_mode_ppl = 1; G_ctx = atoi(argv[4]);
        FILE *f = fopen(argv[5], "r");
        if (!f) { perror("tokens"); return 1; }
        G_toks = malloc(sizeof(int) << 20);
        int x;
        while (G_ntok < (1 << 20) && fscanf(f, "%d", &x) == 1) G_toks[G_ntok++] = x;
    }
    pthread_barrier_init(&START, NULL, T);
    pthread_t th[MAXT];
    for (int t = 1; t < T; t++) pthread_create(&th[t], NULL, worker, (void *)(intptr_t)t);
    run(0);  // thread 0 runs inline; every thread packs its own weights before the first START barrier
    for (int t = 1; t < T; t++) pthread_join(th[t], NULL);
    fprintf(stderr, "load+run %.1f s\n", now() - t0);
    if (G_dump) fclose(G_dump);
    if (G_kld) fclose(G_kld);
    for (int t = 0; t < T && G_dkq[t]; t++) {
        fclose(G_dkq[t]);
        char p[4096];
        snprintf(p, sizeof p, "%s.qq.%d", getenv("KURN_DUMP_KQ"), t);
        FILE *f = fopen(p, "wb");
        const double n = (double)G_qq_n[t];
        if (!f || fwrite(G_qq[t], sizeof(double), (size_t)N_LAYER * (S[t].kh1 - S[t].kh0) * D, f) == 0 || fwrite(&n, sizeof n, 1, f) != 1) perror(p);
        if (f) fclose(f);
    }

    const double wall = G_w1 - G_w0, cpu = G_c1 - G_c0, cyc = (double)(G_rdtsc1 - G_rdtsc0);
    uint64_t wk[K_N] = {0}, nk[K_N] = {0}, wsum = 0, nsum = 0;
    for (int t = 0; t < T; t++) for (int k = 0; k < K_N; k++) { wk[k] += PROF[t].wait[k]; nk[k] += PROF[t].n[k]; }
    for (int k = 0; k < K_N; k++) { wsum += wk[k]; nsum += nk[k]; }
    if (!G_mode_ppl) {
        int cmpd(const void *a, const void *b) { double x = *(const double *)a, y = *(const double *)b; return (x > y) - (x < y); }
        // "quiet" wait share: thread 0, over the faster half of the tokens (those no other process interrupted)
        double *sorted = malloc(sizeof(double) * G_ngen), qw = 0, qc = 0;
        memcpy(sorted, G_dt, sizeof(double) * G_ngen);
        qsort(sorted, G_ngen, sizeof(double), cmpd);
        for (int i = 0; i < G_ngen; i++) if (G_dt[i] <= sorted[G_ngen / 2]) { qw += G_wq[i]; qc += G_cq[i]; }
        free(sorted);
        qsort(G_dt, G_ngen, sizeof(double), cmpd);
        printf("\ndecode_tok_s %.2f cpu_s_per_tok %.4f wall_s %.2f med_ms %.3f p10_ms %.3f p90_ms %.3f barriers_per_tok %.1f wait_share %.3f",
               G_ngen / wall, cpu / G_ngen, wall, G_dt[G_ngen / 2] * 1e3, G_dt[G_ngen / 10] * 1e3, G_dt[G_ngen * 9 / 10] * 1e3,
               (double)nsum / T / G_ngen, wsum / (cyc * T));
        for (int k = 0; k < K_N; k++) if (nk[k]) printf(" wait_%s %.3f", K_NAME[k], wk[k] / (cyc * T));
        double rq = 0;
        for (int t = 0; t < T; t++) rq += G_runq[t] * 1e-9;
        printf(" quiet_wait_share %.3f runq_share %.4f\n", qc > 0 ? qw / qc : 0.0, rq / (wall * T));
        if (getenv("KURN_PROF"))
            for (int t = 0; t < T; t++) {
                fprintf(stderr, "thread %d wait", t);
                for (int k = 0; k < K_N; k++) if (nk[k]) fprintf(stderr, " %s %.3f", K_NAME[k], PROF[t].wait[k] / cyc);
                fprintf(stderr, " runq %.4f\n", G_runq[t] * 1e-9 / wall);
            }
    } else {
        printf("ppl %.4f over %ld tokens; %.2f tok/s, cpu_s/tok %.4f barriers_per_tok %.1f wait_share %.3f\n", exp(G_nll / G_cnt), G_cnt,
               G_ngen / wall, cpu / G_ngen, (double)nsum / T / G_ngen, wsum / (cyc * T));
        if (KVF || G_kld) printf("kv %s ctx %d", KVF_NAME[KVF], G_ctx);
        if (G_kld && G_kld_cmp) printf(" kld_top32 %.6f top1_agree %.4f over %ld", G_kl / G_kl_n, (double)G_top1 / G_kl_n, G_kl_n);
        if (KVF || G_kld) printf("\n");
    }
    return 0;
}
