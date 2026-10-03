// kurn model engine: the whole decode step of ONE fixed model compiled as one program
// (Taalas lesson "the model is the program"). Generated config (model_config.h) fixes
// every dimension and tensor offset at compile time. One persistent, pinned thread
// pool; every thread owns the same rows of every matrix on every token; RMSNorm and
// activation quantization are recomputed per thread instead of synchronized; QKV and
// gate/up are fused GEMVs; MoE layers run all selected experts in one gate/up pass and
// one down pass. Barriers: 1 per token + 5 per layer + 1 for the output.
//
//   engine MODEL.gguf gen  T  N_GEN  tok,tok,...     greedy decode; prints token ids, tok/s, CPU s
//   engine MODEL.gguf ppl  T  CTX  tokens.txt        llama-perplexity-style score (2nd half of each chunk)
#define _GNU_SOURCE
#include "model_config.h"
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

void *kq8_gemv_prepare(const void *W, int64_t K, int64_t N);
void kq8_gemv_packed(const void *pk, const void *x, float *y, int64_t K, int64_t r0, int64_t r1);

typedef struct { uint16_t d; int8_t qs[32]; } q8blk;
#define QB(K) ((K) / 32)
#define MAXT 64
#define MAX_CTX 2048

static uint8_t *G_file;
static int T = 8;
static long n_barriers;

// ---------------------------------------------------------------- barrier (sense-reversing)
static atomic_int bar_count, bar_gen;
static long bar_spins = -1;  // -1 spin (ggml-like); >= 0 spin then futex
static __thread int my_ith;
static struct { uint64_t wait, n; char pad[48]; } PROF[MAXT];
static void barrier_(void);
static inline void barrier(void) {
    const uint64_t t0 = __rdtsc();
    barrier_();
    PROF[my_ith].wait += __rdtsc() - t0;
    PROF[my_ith].n++;
}
__attribute__((noinline)) static void barrier_(void) {
    int gen = atomic_load_explicit(&bar_gen, memory_order_acquire);
    if (atomic_fetch_add_explicit(&bar_count, 1, memory_order_acq_rel) == T - 1) {
        atomic_store_explicit(&bar_count, 0, memory_order_relaxed);
        atomic_fetch_add_explicit(&bar_gen, 1, memory_order_release);
        if (bar_spins >= 0) syscall(SYS_futex, &bar_gen, FUTEX_WAKE_PRIVATE, MAXT, NULL, NULL, 0);
    } else {
        for (long i = 0; atomic_load_explicit(&bar_gen, memory_order_acquire) == gen; i++) {
            if (bar_spins >= 0 && i >= bar_spins) {
                while (atomic_load_explicit(&bar_gen, memory_order_acquire) == gen)
                    syscall(SYS_futex, &bar_gen, FUTEX_WAIT_PRIVATE, gen, NULL, NULL, 0);
                break;
            }
            _mm_pause();
        }
    }
}

// ---------------------------------------------------------------- helpers
static inline float h2f(uint16_t h) { return _cvtsh_ss(h); }
static void quantize_q8(const float *x, q8blk *y, int64_t K) {  // ggml quantize_row_q8_0 (x86 path)
    for (int64_t b = 0; b < K / 32; b++) {
        float amax = 0;
        for (int l = 0; l < 32; l++) amax = fmaxf(amax, fabsf(x[32 * b + l]));
        const float d = amax / 127.f, id = amax != 0.f ? 127.f / amax : 0.f;
        y[b].d = _cvtss_sh(d, 0);
        for (int l = 0; l < 32; l++) y[b].qs[l] = (int8_t)nearbyintf(x[32 * b + l] * id);
    }
}
static void rmsnorm(const float *x, const float *w, float *o, int n, float eps) {
    double ss = 0;  // ggml sums in float; double here is a small, deliberate difference
    for (int i = 0; i < n; i++) ss += (double)x[i] * x[i];
    const float sc = 1.0f / sqrtf((float)(ss / n) + eps);
    for (int i = 0; i < n; i++) o[i] = x[i] * sc * w[i];
}
static const float *F32(int64_t off) { return (const float *)(G_file + off); }
static void slice(int64_t n, int align, int ith, int64_t *a, int64_t *b) {
    int64_t per = ((n + T - 1) / T + align - 1) / align * align;
    *a = ith * per < n ? ith * per : n;
    *b = (ith + 1) * per < n ? (ith + 1) * per : n;
}
static void rope_neox(float *v, int d, int pos) {
    for (int i = 0; i < d / 2; i++) {
        const float th = pos * powf(ROPE_BASE, -2.0f * i / d);
        const float c = cosf(th), s = sinf(th), a = v[i], b = v[i + d / 2];
        v[i] = a * c - b * s;
        v[i + d / 2] = a * s + b * c;
    }
}

// ---------------------------------------------------------------- weights (packed at load)
typedef struct {
    void *qkv, *o;
#if N_EXPERT
    void *gu[N_EXPERT], *down[N_EXPERT];
#else
    void *gu, *down;
#endif
} layer_w;
static layer_w LW[N_LAYER];
static void *W_out;

// concatenate row blocks of Q8_0 matrices (each `rows[i]` x K) and pack them
static void *pack_cat(int n, const int64_t *offs, const int64_t *rows, int64_t K) {
    int64_t tot = 0;
    for (int i = 0; i < n; i++) tot += rows[i];
    uint8_t *tmp = malloc(QB(K) * 34 * tot), *p = tmp;
    for (int i = 0; i < n; i++) { memcpy(p, G_file + offs[i], QB(K) * 34 * rows[i]); p += QB(K) * 34 * rows[i]; }
    void *pk = kq8_gemv_prepare(tmp, K, tot);
    free(tmp);
    return pk;
}
// gate/up interleaved in 16-row blocks: rows [32j, 32j+16) = gate rows [16j, 16j+16), next 16 = up
static void *pack_gate_up(int64_t g_off, int64_t u_off, int64_t F, int64_t K) {
    const size_t rb = QB(K) * 34;
    uint8_t *tmp = malloc(rb * 2 * F);
    for (int64_t j = 0; j < F / 16; j++) {
        memcpy(tmp + rb * (32 * j), G_file + g_off + rb * 16 * j, rb * 16);
        memcpy(tmp + rb * (32 * j + 16), G_file + u_off + rb * 16 * j, rb * 16);
    }
    void *pk = kq8_gemv_prepare(tmp, K, 2 * F);
    free(tmp);
    return pk;
}

// ---------------------------------------------------------------- state
static float H[N_EMBD];                       // residual stream (rows owned by GEMV slices)
static float QKV[N_HEAD * HEAD_DIM + 2 * N_KV * HEAD_DIM];
static float ATT[N_HEAD * HEAD_DIM];
static float *KC, *VC;                        // [layer][pos][N_KV*HEAD_DIM]
#if N_EXPERT
static float GU[N_USED][2 * N_FF];
static float ACT[N_USED][N_FF];
static int SEL[N_USED];
static float SELW[N_USED];
#else
static float GU[2 * N_FF];
static float ACT[N_FF];
#endif
static float *LOGITS;
static int LOCAL_ARG[MAXT];
static float LOCAL_MAX[MAXT];
static int cur_tok, cur_pos, want_logits;
static atomic_int stop_flag;

static inline size_t kv_at(int l, int pos) { return ((size_t)l * MAX_CTX + pos) * (N_KV * HEAD_DIM); }

static void step(int ith) {  // one token, every thread
    int64_t a, b;
    float xn[N_EMBD > N_FF ? N_EMBD : N_FF];
    static __thread q8blk *xq;
    if (!xq) xq = aligned_alloc(64, sizeof(q8blk) * (QB(N_EMBD) + QB(N_FF) + QB(N_HEAD * HEAD_DIM)) + 64);
    // embedding (dequantize row cur_tok of token_embd)
    slice(N_EMBD, 32, ith, &a, &b);
    {
        const q8blk *e = (const q8blk *)(G_file + OFF_TOK_EMBD) + (int64_t)cur_tok * QB(N_EMBD);
        for (int64_t i = a; i < b; i++) H[i] = h2f(e[i / 32].d) * e[i / 32].qs[i % 32];
    }
    barrier();
    for (int l = 0; l < N_LAYER; l++) {
        // ---- attention
        rmsnorm(H, F32(OFF_ATTN_NORM[l]), xn, N_EMBD, RMS_EPS);
        quantize_q8(xn, xq, N_EMBD);
        slice(N_QKV, 16, ith, &a, &b);
        if (a < b) kq8_gemv_packed(LW[l].qkv, xq, QKV, N_EMBD, a, b);
        barrier();
        {
            float *q = QKV, *k = QKV + N_HEAD * HEAD_DIM, *v = k + N_KV * HEAD_DIM;
#if QK_NORM_FULL  // OLMoE: RMSNorm over the whole q / k vectors (every thread computes the factors)
            float qn[N_HEAD * HEAD_DIM], kn[N_KV * HEAD_DIM];
            rmsnorm(q, F32(OFF_Q_NORM[l]), qn, N_HEAD * HEAD_DIM, RMS_EPS);
            rmsnorm(k, F32(OFF_K_NORM[l]), kn, N_KV * HEAD_DIM, RMS_EPS);
            q = qn; k = kn;
#endif
            for (int kvh = ith; kvh < N_KV; kvh += T) {
                float kk[HEAD_DIM];
                memcpy(kk, k + kvh * HEAD_DIM, sizeof kk);
#if QK_NORM_HEAD  // Qwen3: per-head RMSNorm
                rmsnorm(k + kvh * HEAD_DIM, F32(OFF_K_NORM[l]), kk, HEAD_DIM, RMS_EPS);
#endif
                rope_neox(kk, HEAD_DIM, cur_pos);
                memcpy(KC + kv_at(l, cur_pos) + kvh * HEAD_DIM, kk, sizeof kk);
                memcpy(VC + kv_at(l, cur_pos) + kvh * HEAD_DIM, v + kvh * HEAD_DIM, sizeof kk);
                for (int h = kvh * (N_HEAD / N_KV); h < (kvh + 1) * (N_HEAD / N_KV); h++) {
                    float qq[HEAD_DIM], sc[MAX_CTX], mx = -INFINITY, den = 0;
                    memcpy(qq, q + h * HEAD_DIM, sizeof qq);
#if QK_NORM_HEAD
                    rmsnorm(q + h * HEAD_DIM, F32(OFF_Q_NORM[l]), qq, HEAD_DIM, RMS_EPS);
#endif
                    rope_neox(qq, HEAD_DIM, cur_pos);
                    for (int t = 0; t <= cur_pos; t++) {
                        const float *kt = KC + kv_at(l, t) + kvh * HEAD_DIM;
                        float s = 0;
                        for (int i = 0; i < HEAD_DIM; i++) s += qq[i] * kt[i];
                        sc[t] = s / sqrtf((float)HEAD_DIM);
                        mx = fmaxf(mx, sc[t]);
                    }
                    for (int t = 0; t <= cur_pos; t++) { sc[t] = expf(sc[t] - mx); den += sc[t]; }
                    float *o = ATT + h * HEAD_DIM;
                    for (int i = 0; i < HEAD_DIM; i++) o[i] = 0;
                    for (int t = 0; t <= cur_pos; t++) {
                        const float *vt = VC + kv_at(l, t) + kvh * HEAD_DIM, p = sc[t] / den;
                        for (int i = 0; i < HEAD_DIM; i++) o[i] += p * vt[i];
                    }
                }
            }
        }
        barrier();
        quantize_q8(ATT, xq, N_HEAD * HEAD_DIM);
        {
            float y[N_EMBD];
            slice(N_EMBD, 16, ith, &a, &b);
            if (a < b) kq8_gemv_packed(LW[l].o, xq, y, N_HEAD * HEAD_DIM, a, b);
            for (int64_t i = a; i < b; i++) H[i] += y[i];
        }
        barrier();
        // ---- feed-forward
        rmsnorm(H, F32(OFF_FFN_NORM[l]), xn, N_EMBD, RMS_EPS);
        quantize_q8(xn, xq, N_EMBD);
#if N_EXPERT
        {   // router (every thread, tiny): softmax over experts, top-k, weights not renormalized (OLMoE)
            float lg[N_EXPERT], mx = -INFINITY, den = 0;
            const float *wr = F32(OFF_ROUTER[l]);
            for (int e = 0; e < N_EXPERT; e++) {
                float s = 0;
                for (int i = 0; i < N_EMBD; i++) s += wr[(size_t)e * N_EMBD + i] * xn[i];
                lg[e] = s;
                mx = fmaxf(mx, s);
            }
            for (int e = 0; e < N_EXPERT; e++) { lg[e] = expf(lg[e] - mx); den += lg[e]; }
            int used[N_EXPERT] = {0};
            for (int j = 0; j < N_USED; j++) {
                int best = -1;
                for (int e = 0; e < N_EXPERT; e++) if (!used[e] && (best < 0 || lg[e] > lg[best])) best = e;
                used[best] = 1;
                if (ith == 0) { SEL[j] = best; SELW[j] = lg[best] / den; }
            }
            int sel[N_USED];
            {   // every thread derives the same selection; thread 0's copy is used after the barrier
                int u2[N_EXPERT] = {0};
                for (int j = 0; j < N_USED; j++) {
                    int best = -1;
                    for (int e = 0; e < N_EXPERT; e++) if (!u2[e] && (best < 0 || lg[e] > lg[best])) best = e;
                    u2[best] = 1; sel[j] = best;
                }
            }
            // all selected experts' gate/up rows as one work list of 32-row units
            const int64_t units = (int64_t)N_USED * (2 * N_FF / 32);
            int64_t ua, ub;
            slice(units, 1, ith, &ua, &ub);
            for (int64_t u = ua; u < ub; ) {
                const int j = (int)(u / (2 * N_FF / 32));
                const int64_t uend = ((int64_t)(j + 1) * (2 * N_FF / 32)) < ub ? (int64_t)(j + 1) * (2 * N_FF / 32) : ub;
                const int64_t r0 = (u - (int64_t)j * (2 * N_FF / 32)) * 32, r1 = (uend - (int64_t)j * (2 * N_FF / 32)) * 32;
                kq8_gemv_packed(LW[l].gu[sel[j]], xq, GU[j], N_EMBD, r0, r1);
                for (int64_t r = r0; r < r1; r += 32)
                    for (int i = 0; i < 16; i++) {
                        const float g = GU[j][r + i], up = GU[j][r + 16 + i];
                        ACT[j][r / 2 + i] = g / (1.0f + expf(-g)) * up;
                    }
                u = uend;
            }
            barrier();
            float y[N_EMBD];
            slice(N_EMBD, 16, ith, &a, &b);
            for (int j = 0; j < N_USED; j++) {
                quantize_q8(ACT[j], xq, N_FF);
                if (a < b) kq8_gemv_packed(LW[l].down[SEL[j]], xq, y, N_FF, a, b);
                for (int64_t i = a; i < b; i++) H[i] += SELW[j] * y[i];
            }
        }
#else
        slice(2 * N_FF, 32, ith, &a, &b);
        if (a < b) kq8_gemv_packed(LW[l].gu, xq, GU, N_EMBD, a, b);
        for (int64_t r = a; r < b; r += 32)
            for (int i = 0; i < 16; i++) {
                const float g = GU[r + i], up = GU[r + 16 + i];
                ACT[r / 2 + i] = g / (1.0f + expf(-g)) * up;
            }
        barrier();
        quantize_q8(ACT, xq, N_FF);
        {
            float y[N_EMBD];
            slice(N_EMBD, 16, ith, &a, &b);
            if (a < b) kq8_gemv_packed(LW[l].down, xq, y, N_FF, a, b);
            for (int64_t i = a; i < b; i++) H[i] += y[i];
        }
#endif
        barrier();
    }
    // ---- output
    rmsnorm(H, F32(OFF_OUT_NORM), xn, N_EMBD, RMS_EPS);
    quantize_q8(xn, xq, N_EMBD);
    slice(N_VOCAB, 16, ith, &a, &b);
    if (a < b) kq8_gemv_packed(W_out, xq, LOGITS, N_EMBD, a, b);
    int best = (int)a;
    for (int64_t i = a; i < b; i++) if (LOGITS[i] > LOGITS[best]) best = (int)i;
    LOCAL_ARG[ith] = best; LOCAL_MAX[ith] = a < b ? LOGITS[best] : -INFINITY;
    barrier();
}

static void *worker(void *arg) {
    int ith = (int)(intptr_t)arg;
    my_ith = ith;
    cpu_set_t s; CPU_ZERO(&s); CPU_SET(ith, &s); sched_setaffinity(0, sizeof s, &s);
    for (;;) {
        barrier();  // wait for a token
        if (atomic_load(&stop_flag)) break;
        step(ith);
    }
    return NULL;
}

static int run_token(int tok, int pos) {  // main thread = thread 0
    cur_tok = tok; cur_pos = pos;
    barrier();
    step(0);
    int best = 0;
    for (int t = 1; t < T; t++) if (LOCAL_MAX[t] > LOCAL_MAX[best]) best = t;
    return LOCAL_ARG[best];
}

static double now(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec + ts.tv_nsec * 1e-9; }
static double cpu_s(void) { struct rusage r; getrusage(RUSAGE_SELF, &r); return r.ru_utime.tv_sec + r.ru_utime.tv_usec * 1e-6 + r.ru_stime.tv_sec + r.ru_stime.tv_usec * 1e-6; }

int main(int argc, char **argv) {
    if (argc < 6) { fprintf(stderr, "usage: %s model.gguf gen|ppl THREADS N_GEN|CTX tokens\n", argv[0]); return 2; }
    T = atoi(argv[3]);
    if (getenv("KURN_WAIT_SLEEP")) bar_spins = 2000;
    int fd = open(argv[1], O_RDONLY);
    struct stat st; fstat(fd, &st);
    G_file = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (G_file == MAP_FAILED) { perror("mmap"); return 1; }
    double t0 = now();
    for (int l = 0; l < N_LAYER; l++) {
        const int64_t offs[3] = {OFF_Q[l], OFF_K[l], OFF_V[l]}, rows[3] = {N_HEAD * HEAD_DIM, N_KV * HEAD_DIM, N_KV * HEAD_DIM};
        LW[l].qkv = pack_cat(3, offs, rows, N_EMBD);
        const int64_t oo[1] = {OFF_O[l]}, orow[1] = {N_EMBD};
        LW[l].o = pack_cat(1, oo, orow, N_HEAD * HEAD_DIM);
#if N_EXPERT
        for (int e = 0; e < N_EXPERT; e++) {
            const size_t gsz = (size_t)QB(N_EMBD) * 34 * N_FF, dsz = (size_t)QB(N_FF) * 34 * N_EMBD;
            LW[l].gu[e] = pack_gate_up(OFF_GATE[l] + e * gsz, OFF_UP[l] + e * gsz, N_FF, N_EMBD);
            const int64_t d1[1] = {OFF_DOWN[l] + (int64_t)(e * dsz)}, dr[1] = {N_EMBD};
            LW[l].down[e] = pack_cat(1, d1, dr, N_FF);
        }
#else
        LW[l].gu = pack_gate_up(OFF_GATE[l], OFF_UP[l], N_FF, N_EMBD);
        const int64_t d1[1] = {OFF_DOWN[l]}, dr[1] = {N_EMBD};
        LW[l].down = pack_cat(1, d1, dr, N_FF);
#endif
    }
    { const int64_t oo[1] = {OFF_OUTPUT}, orow[1] = {N_VOCAB}; W_out = pack_cat(1, oo, orow, N_EMBD); }
    KC = calloc((size_t)N_LAYER * MAX_CTX * N_KV * HEAD_DIM, sizeof(float));
    VC = calloc((size_t)N_LAYER * MAX_CTX * N_KV * HEAD_DIM, sizeof(float));
    LOGITS = aligned_alloc(64, sizeof(float) * (N_VOCAB + 64));
    fprintf(stderr, "packed in %.1f s\n", now() - t0);
    cpu_set_t s0; CPU_ZERO(&s0); CPU_SET(0, &s0); sched_setaffinity(0, sizeof s0, &s0);
    pthread_t th[MAXT];
    for (int t = 1; t < T; t++) pthread_create(&th[t], NULL, worker, (void *)(intptr_t)t);

    if (!strcmp(argv[2], "gen")) {
        const char *ctl = getenv("KURN_PERF_CTL");  // `perf record -D -1 --control fifo:F`: decode loop only
        const int ngen = atoi(argv[4]);
        int toks[MAX_CTX], n = 0;
        for (char *p = strtok(argv[5], ","); p; p = strtok(NULL, ",")) toks[n++] = atoi(p);
        int next = 0;
        for (int i = 0; i < n; i++) next = run_token(toks[i], i);
        double *dt = malloc(sizeof(double) * ngen);
        uint64_t *wq = malloc(sizeof(uint64_t) * ngen), *cq = malloc(sizeof(uint64_t) * ngen);
        memset(PROF, 0, sizeof PROF);
        double w0 = now(), c0 = cpu_s();
        const uint64_t r0 = __rdtsc();
        if (ctl) { int fd = open(ctl, O_WRONLY); if (fd >= 0) { if (write(fd, "enable\n", 7) < 0) perror("perf ctl"); close(fd); } }
        printf("gen:");
        for (int i = 0; i < ngen; i++) {
            printf(" %d", next);
            double t0 = now();
            const uint64_t c0 = __rdtsc(), w0 = PROF[0].wait;
            next = run_token(next, n + i);
            dt[i] = now() - t0;
            cq[i] = __rdtsc() - c0; wq[i] = PROF[0].wait - w0;
        }
        const uint64_t rcyc = __rdtsc() - r0;
        double wall = now() - w0, cpu = cpu_s() - c0;
        if (ctl) { int fd = open(ctl, O_WRONLY); if (fd >= 0) { if (write(fd, "disable\n", 8) < 0) perror("perf ctl"); close(fd); } }
        int cmpd(const void *x, const void *y) { double p = *(const double *)x, q = *(const double *)y; return (p > q) - (p < q); }
        double *sorted = malloc(sizeof(double) * ngen), qw = 0, qc = 0;  // thread 0, faster half of the tokens
        memcpy(sorted, dt, sizeof(double) * ngen);
        qsort(sorted, ngen, sizeof(double), cmpd);
        for (int i = 0; i < ngen; i++) if (dt[i] <= sorted[ngen / 2]) { qw += wq[i]; qc += cq[i]; }
        qsort(dt, ngen, sizeof(double), cmpd);
        uint64_t wsum = 0, nb = 0;
        for (int t = 0; t < T; t++) { wsum += PROF[t].wait; nb += PROF[t].n; }
        printf("\ndecode_tok_s %.2f cpu_s_per_tok %.4f wall_s %.2f med_ms %.3f p10_ms %.3f p90_ms %.3f barriers_per_tok %.1f wait_share %.3f "
               "quiet_wait_share %.3f\n",
               ngen / wall, cpu / ngen, wall, dt[ngen / 2] * 1e3, dt[ngen / 10] * 1e3, dt[ngen * 9 / 10] * 1e3,
               (double)nb / T / ngen, (double)wsum / ((double)rcyc * T), qc > 0 ? qw / qc : 0.0);
    } else {  // ppl: chunks of CTX tokens, score positions [CTX/2, CTX-1)
        const int ctx = atoi(argv[4]);
        FILE *f = fopen(argv[5], "r");
        int *toks = malloc(sizeof(int) * 1 << 20), n = 0, x;
        while (n < (1 << 20) && fscanf(f, "%d", &x) == 1) toks[n++] = x;
        double nll = 0; long cnt = 0;
        double w0 = now(), c0 = cpu_s(); long steps = 0;
        for (int c = 0; (c + 1) * ctx <= n && c < 4; c++) {
            for (int i = 0; i < ctx - 1; i++) {
                run_token(toks[c * ctx + i], i);
                steps++;
                if (i >= ctx / 2) {
                    double mx = -INFINITY, den = 0;
                    for (int v = 0; v < N_VOCAB; v++) mx = fmax(mx, LOGITS[v]);
                    for (int v = 0; v < N_VOCAB; v++) den += exp(LOGITS[v] - mx);
                    nll += -(LOGITS[toks[c * ctx + i + 1]] - mx - log(den));
                    cnt++;
                }
            }
        }
        printf("ppl %.4f over %ld tokens; %.2f tok/s, cpu_s/tok %.4f\n", exp(nll / cnt), cnt, steps / (now() - w0), (cpu_s() - c0) / steps);
    }
    atomic_store(&stop_flag, 1);
    barrier();
    return 0;
}
