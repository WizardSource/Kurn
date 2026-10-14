// Checks, benchmarks and a synthetic quaternion-MLP demo for kurn-hc against the real expansion run by kurn's
// production kernels (ggml MUL_MAT with the expanded Q8_0 weights in the KURN buffer; F32 on ggml's CPU path).
//
//   hc_bench quant                                     kurn_hc_quantize_q8_0 == ggml's quantize_row_q8_0, bytewise
//   hc_bench check D FMT Nq Kq "M..." [T]              compact vs expanded per M (Q8: bitwise; F32: relerr + oracle),
//                                                      expansion vs the Hamilton / complex product in double, and the
//                                                      Freivalds oracle on the AMX-BF16 path (exactness from the env)
//   hc_bench bench D FMT Nq Kq M T REPS LAYERS WHICH  WHICH = compact | expanded: median ms per graph of LAYERS layers
//   hc_bench mlp D FMT Hq Fq LAYERS M T REPS WHICH     WHICH = compact | expanded | check: x += Wd relu(Wu x) per layer
//
// FMT: q8 | f32. GGML_KURN_EXACT=1 selects exact mode on both sides. Build: see README.md.
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "kurn-hc.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static uint64_t rng = 0x9e3779b97f4a7c15ull;
static float frand(void) {
    rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
    return (float)((rng >> 40) * (1.0 / (1ull << 23))) - 1.0f;
}
static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}
static int cmpd(const void *a, const void *b) {
    const double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : x > y;
}
static int exact_env(void) { const char *e = getenv("GGML_KURN_EXACT"); return e && atoi(e); }

static ggml_backend_buffer_type_t find_buft(const char *name) {
    if (!strcmp(name, "CPU")) return ggml_backend_cpu_buffer_type();
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU));
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    typedef ggml_backend_buffer_type_t *(*fn_t)(ggml_backend_dev_t);
    fn_t fn = (fn_t)ggml_backend_reg_get_proc_address(reg, "ggml_backend_dev_get_extra_bufts");
    for (ggml_backend_buffer_type_t *b = fn ? fn(dev) : NULL; b && *b; b++)
        if (!strcmp(ggml_backend_buft_name(*b), name)) return *b;
    return NULL;
}

static ggml_backend_t make_backend(int threads) {
    ggml_backend_t be = ggml_backend_cpu_init();
    struct ggml_threadpool_params tpp = ggml_threadpool_params_default(threads);
    for (int i = 0; i < threads && i < GGML_MAX_N_THREADS; i++) tpp.cpumask[i] = true;
    tpp.strict_cpu = true;
    ggml_backend_cpu_set_threadpool(be, ggml_threadpool_new(&tpp));
    ggml_backend_cpu_set_n_threads(be, threads);
    return be;
}

// compact weights with a per-row, per-component magnitude spread (like trained layers)
static float *rand_compact(int D, int64_t Nq, int64_t Kq, float scale) {
    float *W = malloc(sizeof(float) * D * Nq * Kq);
    for (int64_t i = 0; i < (int64_t)D * Nq * Kq; i++) W[i] = scale * (frand() + frand() + frand()) * (1 + (i / 32) % 5);
    return W;
}

static int fmt_of(const char *s) { return !strcmp(s, "q8") ? KURN_HC_Q8 : KURN_HC_F32; }

// ---------------------------------------------------------------------------------------------------------
// expanded layer graph: Y = Wexp X for M columns, weights in KURN (Q8_0) or CPU (F32) buffer
typedef struct {
    struct ggml_context *cw, *ca;
    ggml_backend_buffer_t bw, ba;
    struct ggml_tensor *x, *y;
    struct ggml_cgraph *gf;
} xg;

static xg expanded_graph(ggml_backend_t be, const kurn_hc *w, int D, int64_t Nq, int64_t Kq, int64_t M, int fmt, int layers) {
    xg g = {0};
    const int64_t N = D * Nq, K = D * Kq;
    struct ggml_init_params ip = {(size_t)(layers + 8) * ggml_tensor_overhead(), NULL, true};
    struct ggml_init_params ia = {(size_t)(2 * layers + 8) * ggml_tensor_overhead() + ggml_graph_overhead_custom(4 * layers + 16, false), NULL, true};
    g.cw = ggml_init(ip); g.ca = ggml_init(ia);
    struct ggml_tensor **wt = malloc(sizeof(*wt) * layers);
    for (int l = 0; l < layers; l++) wt[l] = ggml_new_tensor_2d(g.cw, fmt == KURN_HC_Q8 ? GGML_TYPE_Q8_0 : GGML_TYPE_F32, K, N);
    g.x = ggml_new_tensor_2d(g.ca, GGML_TYPE_F32, K, M);
    g.gf = ggml_new_graph_custom(g.ca, 4 * layers + 16, false);
    for (int l = 0; l < layers; l++) {
        g.y = ggml_mul_mat(g.ca, wt[l], g.x);
        ggml_build_forward_expand(g.gf, g.y);
    }
    g.bw = ggml_backend_alloc_ctx_tensors_from_buft(g.cw, find_buft(fmt == KURN_HC_Q8 ? "KURN" : "CPU"));
    ggml_backend_buffer_set_usage(g.bw, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    g.ba = ggml_backend_alloc_ctx_tensors(g.ca, be);
    void *ex = malloc(kurn_hc_expanded_bytes(w));
    kurn_hc_expand(w, ex);
    for (int l = 0; l < layers; l++) ggml_backend_tensor_set(wt[l], ex, 0, kurn_hc_expanded_bytes(w));
    free(ex); free(wt);
    return g;
}

static void xg_free(xg *g) {
    ggml_backend_buffer_free(g->ba); ggml_backend_buffer_free(g->bw); ggml_free(g->ca); ggml_free(g->cw);
}

// dequantized compact components: Wd[(p * Nq + n) * Kq + k]
static double *dequant_compact(const kurn_hc *w, int D, int64_t Nq, int64_t Kq) {
    // via the expansion: row (n, o = 0), block (kb, q) holds component P(0, q) = q with sign +1 (q = 0) or -1
    const int64_t N = D * Nq, K = D * Kq;
    double *Wd = malloc(sizeof(double) * D * Nq * Kq);
    if (kurn_hc_expanded_bytes(w) == (size_t)N * (K / 32) * 34) {
        uint8_t *ex = malloc(kurn_hc_expanded_bytes(w));
        kurn_hc_expand(w, ex);
        for (int64_t n = 0; n < Nq; n++) {
            const int64_t R = (n / 32) * 32 * D + n % 32;  // output component 0
            for (int64_t b = 0; b < K / 32; b++) {
                const int64_t kb = b / D, q = b % D;
                const uint8_t *blk = ex + (R * (K / 32) + b) * 34;
                uint16_t dh; memcpy(&dh, blk, 2);
                const float d = ggml_fp16_to_fp32(dh);
                const double s = q == 0 ? 1 : -1;
                for (int i = 0; i < 32; i++) Wd[(q * Nq + n) * Kq + kb * 32 + i] = s * (double)((int8_t)blk[2 + i]) * d;
            }
        }
        free(ex);
    } else {
        float *ex = malloc(kurn_hc_expanded_bytes(w));
        kurn_hc_expand(w, ex);
        for (int64_t n = 0; n < Nq; n++) {
            const int64_t R = (n / 32) * 32 * D + n % 32;
            for (int64_t b = 0; b < K / 32; b++) {
                const int64_t kb = b / D, q = b % D;
                const double s = q == 0 ? 1 : -1;
                for (int i = 0; i < 32; i++) Wd[(q * Nq + n) * Kq + kb * 32 + i] = s * ex[R * K + b * 32 + i];
            }
        }
        free(ex);
    }
    return Wd;
}

static int64_t bp(int D, int64_t k, int c) { return (k / 32) * 32 * D + c * 32 + k % 32; }

// y = W x in double, as a Hamilton / complex product of the compact components (independent of the expansion)
static void hc_ref(int D, int64_t Nq, int64_t Kq, const double *Wd, const float *x, double *y) {
    for (int64_t n = 0; n < Nq; n++) {
        double acc[4] = {0, 0, 0, 0};
        for (int64_t k = 0; k < Kq; k++) {
            double a[4], v[4];
            for (int c = 0; c < D; c++) { a[c] = Wd[(c * Nq + n) * Kq + k]; v[c] = x[bp(D, k, c)]; }
            if (D == 4) {
                acc[0] += a[0] * v[0] - a[1] * v[1] - a[2] * v[2] - a[3] * v[3];
                acc[1] += a[0] * v[1] + a[1] * v[0] + a[2] * v[3] - a[3] * v[2];
                acc[2] += a[0] * v[2] - a[1] * v[3] + a[2] * v[0] + a[3] * v[1];
                acc[3] += a[0] * v[3] + a[1] * v[2] - a[2] * v[1] + a[3] * v[0];
            } else {
                acc[0] += a[0] * v[0] - a[1] * v[1];
                acc[1] += a[0] * v[1] + a[1] * v[0];
            }
        }
        for (int c = 0; c < D; c++) y[bp(D, n, c)] = acc[c];
    }
}

// Freivalds oracle (as test_kurn_buft's): C ~ X Wexp^T, bf16 inputs, statistical mode, alpha = 2^-40 per check;
// returns the worst |C r - X (W r)| / tau over all checks (> 1: rejected). Wexp rows from the double expansion.
static double freivalds(int D, int64_t Nq, int64_t Kq, const double *Wd, const float *X, const float *C, int64_t M) {
    const int64_t N = D * Nq, K = D * Kq, B = 64, G = N / B;
    const int t = 2;
    const double lam = sqrt(2 * log(2.0 * M * G * t / ldexp(1.0, -40))), u = ldexp(1.0, -8) + K * ldexp(1.0, -24);
    double *wrow = malloc(sizeof(double) * K), *wr = malloc(sizeof(double) * K * t), *w2 = malloc(sizeof(double) * K);
    signed char sg[2][64];
    double worst = 0;
    uint64_t s = (uint64_t)time(NULL) * 2654435761u | 1;
    for (int64_t g = 0; g < G; g++) {
        for (int p = 0; p < t; p++) for (int j = 0; j < B; j++) { s ^= s << 13; s ^= s >> 7; s ^= s << 17; sg[p][j] = (s >> 40) & 1 ? 1 : -1; }
        memset(wr, 0, sizeof(double) * K * t); memset(w2, 0, sizeof(double) * K);
        for (int64_t j = 0; j < B; j++) {
            const int64_t R = g * B + j, o = (R % (32 * D)) / 32, n = (R / (32 * D)) * 32 + R % 32;
            for (int64_t k = 0; k < Kq; k++)
                for (int q = 0; q < D; q++) {
                    static const int QP[4][4] = {{0, 1, 2, 3}, {1, 0, 3, 2}, {2, 3, 0, 1}, {3, 2, 1, 0}};
                    static const int QS[4][4] = {{1, -1, -1, -1}, {1, 1, -1, 1}, {1, 1, 1, -1}, {1, -1, 1, 1}};
                    static const int CP[2][2] = {{0, 1}, {1, 0}}, CS[2][2] = {{1, -1}, {1, 1}};
                    const int pc = D == 4 ? QP[o][q] : CP[o][q], sc = D == 4 ? QS[o][q] : CS[o][q];
                    wrow[bp(D, k, q)] = sc * Wd[(pc * Nq + n) * Kq + k];
                }
            for (int64_t k = 0; k < K; k++) {
                for (int p = 0; p < t; p++) wr[p * K + k] += sg[p][j] * wrow[k];
                w2[k] += wrow[k] * wrow[k];
            }
        }
        for (int64_t m = 0; m < M; m++) {
            double q2 = 0;
            for (int64_t k = 0; k < K; k++) q2 += (double)X[m * K + k] * X[m * K + k] * w2[k];
            const double tau = lam * u * sqrt(q2) + 1e-30;
            for (int p = 0; p < t; p++) {
                double l = 0, r = 0;
                for (int64_t j = 0; j < B; j++) l += sg[p][j] * (double)C[m * N + g * B + j];
                for (int64_t k = 0; k < K; k++) r += X[m * K + k] * wr[p * K + k];
                const double v = fabs(l - r) / tau;
                if (v > worst || v != v) worst = v;
            }
        }
    }
    free(wrow); free(wr); free(w2);
    return worst;
}

static int nfail, npass;
static void report(int ok, const char *what) { ok ? npass++ : nfail++; printf("%s %s\n", ok ? "ok  " : "FAIL", what); }

static int mode_quant(void) {
    const struct ggml_type_traits_cpu *tt = ggml_get_type_traits_cpu(GGML_TYPE_Q8_0);
    const int64_t n = 32 * 4096;
    float *x = malloc(sizeof(float) * n);
    for (int64_t i = 0; i < n; i++) x[i] = frand() * (1 + (i / 32) % 7);
    for (int64_t b = 0; b < 512; b++)  // exact .5 ties after scaling: values k / 2 with amax 127 / 2
        for (int i = 0; i < 32; i++) x[b * 32 + i] = (i == 0 ? 63.5f : (float)((int)(frand() * 127)) * 0.5f);
    uint8_t *a = malloc(n / 32 * 34), *b = malloc(n / 32 * 34);
    tt->from_float(x, a, n);
    kurn_hc_quantize_q8_0(x, b, n);
    report(!memcmp(a, b, n / 32 * 34), "kurn_hc_quantize_q8_0 == ggml quantize_row_q8_0 (131072 values incl. ties)");
    free(x); free(a); free(b);
    return nfail;
}

static int mode_check(int D, int fmt, int64_t Nq, int64_t Kq, const char *ms, int threads) {
    ggml_backend_t be = make_backend(threads);
    const int ex = exact_env();
    float *W = rand_compact(D, Nq, Kq, 0.05f);
    kurn_hc *w = kurn_hc_pack(D, Nq, Kq, W, fmt);
    const int64_t N = D * Nq, K = D * Kq;
    double *Wd = dequant_compact(w, D, Nq, Kq);
    // the expansion is the Hamilton / complex left-multiplication matrix: expanded f32 rows x x (double) vs hc_ref
    {
        float *x = malloc(sizeof(float) * K);
        for (int64_t i = 0; i < K; i++) x[i] = frand();
        double *yr = malloc(sizeof(double) * N), *ye = calloc(N, sizeof(double));
        hc_ref(D, Nq, Kq, Wd, x, yr);
        static const int QP[4][4] = {{0, 1, 2, 3}, {1, 0, 3, 2}, {2, 3, 0, 1}, {3, 2, 1, 0}};
        static const int QS[4][4] = {{1, -1, -1, -1}, {1, 1, -1, 1}, {1, 1, 1, -1}, {1, -1, 1, 1}};
        static const int CP[2][2] = {{0, 1}, {1, 0}}, CS[2][2] = {{1, -1}, {1, 1}};
        uint8_t *exq = malloc(kurn_hc_expanded_bytes(w));
        kurn_hc_expand(w, exq);
        double err = 0, mag = 1e-30;
        for (int64_t R = 0; R < N; R++) {
            double s = 0;
            for (int64_t b = 0; b < K / 32; b++)
                for (int i = 0; i < 32; i++) {
                    double v;
                    if (fmt == KURN_HC_Q8) {
                        const uint8_t *blk = exq + (R * (K / 32) + b) * 34;
                        uint16_t dh; memcpy(&dh, blk, 2);
                        v = (double)(int8_t)blk[2 + i] * ggml_fp16_to_fp32(dh);
                    } else v = ((float *)exq)[R * K + b * 32 + i];
                    s += v * x[b * 32 + i];
                }
            ye[R] = s;
        }
        (void)QP; (void)QS; (void)CP; (void)CS;
        for (int64_t i = 0; i < N; i++) { err = fmax(err, fabs(ye[i] - yr[i])); mag = fmax(mag, fabs(yr[i])); }
        char what[160];
        snprintf(what, sizeof what, "expansion == %s product (double): relerr %.1e", D == 4 ? "Hamilton" : "complex", err / mag);
        report(err / mag < 1e-12, what);
        free(x); free(yr); free(ye); free(exq);
    }
    for (const char *p = ms; *p;) {
        const int64_t M = strtoll(p, (char **)&p, 10);
        while (*p == ' ' || *p == ',') p++;
        if (M <= 0) break;
        float *X = malloc(sizeof(float) * K * M), *yc = malloc(sizeof(float) * N * M), *ye = malloc(sizeof(float) * N * M);
        for (int64_t i = 0; i < K * M; i++) X[i] = frand() * (1 + (i / 64) % 3);
        xg g = expanded_graph(be, w, D, Nq, Kq, M, fmt, 1);
        ggml_backend_tensor_set(g.x, X, 0, sizeof(float) * K * M);
        ggml_backend_graph_compute(be, g.gf);
        ggml_backend_tensor_get(g.y, ye, 0, sizeof(float) * N * M);
        kurn_hc_matmul(w, X, yc, M, threads, ex);
        double err = 0, mag = 1e-30;
        for (int64_t i = 0; i < N * M; i++) { err = fmax(err, fabs((double)yc[i] - ye[i])); mag = fmax(mag, fabs(ye[i])); }
        const int same = !memcmp(yc, ye, sizeof(float) * N * M);
        const int kl = !ex && M >= 32;  // AMX-BF16 on both sides
        double fv = 0;
        if (kl) fv = freivalds(D, Nq, Kq, Wd, X, yc, M);
        char what[200];
        snprintf(what, sizeof what, "%s %s Nq=%ld Kq=%ld M=%ld%s: compact %s expanded (relerr %.1e)%s", D == 4 ? "quaternion" : "complex",
                 fmt == KURN_HC_Q8 ? "q8" : "f32", (long)Nq, (long)Kq, (long)M, ex ? " exact" : "", same ? "==" : "!=", err / mag, "");
        if (kl) {
            char tail[64];
            snprintf(tail, sizeof tail, ", Freivalds worst %.3f of tau", fv);
            strncat(what, tail, sizeof what - strlen(what) - 1);
        }
        // f32: the expanded side stays f32 at every M, so from 32 columns (bf16 inputs on the compact side) only the oracle applies
        const int ok = fmt == KURN_HC_Q8 ? same && (!kl || fv <= 1) : kl ? fv <= 1 : err / mag < 1e-4;
        report(ok, what);
        xg_free(&g); free(X); free(yc); free(ye);
    }
    kurn_hc_free(w); free(W); free(Wd);
    ggml_backend_free(be);
    printf("check: %d passed, %d failed\n", npass, nfail);
    return nfail;
}

static double median_time(int reps, void (*fn)(void *), void *arg) {
    double *t = malloc(sizeof(double) * reps);
    for (int i = 0; i < 3; i++) fn(arg);
    for (int i = 0; i < reps; i++) { const double t0 = now(); fn(arg); t[i] = now() - t0; }
    qsort(t, reps, sizeof(double), cmpd);
    const double m = t[reps / 2];
    free(t);
    return m;
}

typedef struct { kurn_hc **w; int layers; const float *X; float *Y; int64_t M; int threads, exact; } cargs;
static void run_compact(void *a) {
    cargs *c = a;
    for (int l = 0; l < c->layers; l++) kurn_hc_matmul(c->w[l], c->X, c->Y, c->M, c->threads, c->exact);
}
typedef struct { ggml_backend_t be; struct ggml_cgraph *gf; } gargs;
static void run_graph(void *a) { gargs *g = a; ggml_backend_graph_compute(g->be, g->gf); }

static int mode_bench(int D, int fmt, int64_t Nq, int64_t Kq, int64_t M, int threads, int reps, int layers, const char *which) {
    const int64_t N = D * Nq, K = D * Kq;
    float *W = rand_compact(D, Nq, Kq, 0.05f);
    kurn_hc **ws = malloc(sizeof(*ws) * layers);
    for (int l = 0; l < layers; l++) ws[l] = kurn_hc_pack(D, Nq, Kq, W, fmt);  // separate copies: no cache reuse across layers
    float *X = malloc(sizeof(float) * K * M), *Y = malloc(sizeof(float) * N * M);
    for (int64_t i = 0; i < K * M; i++) X[i] = frand();
    double t, bytes;
    if (!strcmp(which, "compact")) {
        cargs c = {ws, layers, X, Y, M, threads, exact_env()};
        t = median_time(reps, run_compact, &c);
        bytes = (double)layers * kurn_hc_bytes(ws[0]);
    } else {
        ggml_backend_t be = make_backend(threads);
        xg g = expanded_graph(be, ws[0], D, Nq, Kq, M, fmt, layers);
        ggml_backend_tensor_set(g.x, X, 0, sizeof(float) * K * M);
        gargs ga = {be, g.gf};
        t = median_time(reps, run_graph, &ga);
        bytes = (double)layers * kurn_hc_expanded_bytes(ws[0]);
        xg_free(&g);
    }
    printf("hcbench %s D=%d %s Nq=%ld Kq=%ld M=%ld T=%d L=%d: %.3f ms (%.1f MB weights, %.1f GB/s)\n", which, D, fmt == KURN_HC_Q8 ? "q8" : "f32",
           (long)Nq, (long)Kq, (long)M, threads, layers, 1e3 * t, bytes / 1e6, bytes / t / 1e9);
    return 0;
}

// ---------------------------------------------------------------------------------------------------------
// synthetic quaternion MLP: x += Wd relu(Wu x) per layer (split ReLU: per real component, exact on both sides)
typedef struct { kurn_hc **wu, **wd; int layers; float *x, *h; int64_t M; int64_t H, F; int threads, exact; } margs;
static void run_mlp_compact(void *a) {
    margs *c = a;
    for (int l = 0; l < c->layers; l++) {
        kurn_hc_matmul(c->wu[l], c->x, c->h, c->M, c->threads, c->exact);
        float *h = c->h;
#pragma omp parallel for num_threads(c->threads) schedule(static)
        for (int64_t i = 0; i < c->F * c->M; i++) h[i] = h[i] > 0 ? h[i] : 0;
        float *y = c->h + c->F * c->M, *x = c->x;
        kurn_hc_matmul(c->wd[l], c->h, y, c->M, c->threads, c->exact);
#pragma omp parallel for num_threads(c->threads) schedule(static)
        for (int64_t i = 0; i < c->H * c->M; i++) x[i] = x[i] + y[i];
    }
}

static int mode_mlp(int D, int fmt, int64_t Hq, int64_t Fq, int layers, int64_t M, int threads, int reps, const char *which) {
    const int64_t H = D * Hq, F = D * Fq;
    kurn_hc **wu = malloc(sizeof(*wu) * layers), **wd = malloc(sizeof(*wd) * layers);
    for (int l = 0; l < layers; l++) {
        float *a = rand_compact(D, Fq, Hq, 0.02f), *b = rand_compact(D, Hq, Fq, 0.01f);
        wu[l] = kurn_hc_pack(D, Fq, Hq, a, fmt); wd[l] = kurn_hc_pack(D, Hq, Fq, b, fmt);
        free(a); free(b);
    }
    float *x0 = malloc(sizeof(float) * H * M);
    for (int64_t i = 0; i < H * M; i++) x0[i] = frand();
    float *xc = malloc(sizeof(float) * H * M), *h = malloc(sizeof(float) * (F + H) * M), *xe = malloc(sizeof(float) * H * M);
    const int do_c = strcmp(which, "expanded") != 0, do_e = strcmp(which, "compact") != 0;
    double tc = 0, te = 0, bc = 0, be_ = 0;
    for (int l = 0; l < layers; l++) { bc += kurn_hc_bytes(wu[l]) + kurn_hc_bytes(wd[l]); be_ += kurn_hc_expanded_bytes(wu[l]) + kurn_hc_expanded_bytes(wd[l]); }
    if (do_c) {
        margs c = {wu, wd, layers, xc, h, M, H, F, threads, exact_env()};
        memcpy(xc, x0, sizeof(float) * H * M);
        run_mlp_compact(&c);  // one pass for the check
        if (reps > 0) {
            float *keep = malloc(sizeof(float) * H * M);
            memcpy(keep, xc, sizeof(float) * H * M);
            tc = median_time(reps, run_mlp_compact, &c);
            memcpy(xc, keep, sizeof(float) * H * M);
            free(keep);
        }
    }
    if (do_e) {
        ggml_backend_t be = make_backend(threads);
        struct ggml_init_params ip = {(size_t)(2 * layers + 8) * ggml_tensor_overhead(), NULL, true};
        struct ggml_init_params ia = {(size_t)(8 * layers + 8) * ggml_tensor_overhead() + ggml_graph_overhead_custom(8 * layers + 16, false), NULL, true};
        struct ggml_context *cw = ggml_init(ip), *ca = ggml_init(ia);
        const enum ggml_type wt = fmt == KURN_HC_Q8 ? GGML_TYPE_Q8_0 : GGML_TYPE_F32;
        struct ggml_tensor **tu = malloc(sizeof(*tu) * layers), **td = malloc(sizeof(*td) * layers);
        struct ggml_tensor *x = ggml_new_tensor_2d(ca, GGML_TYPE_F32, H, M), *cur = x;
        struct ggml_cgraph *gf = ggml_new_graph_custom(ca, 8 * layers + 16, false);
        for (int l = 0; l < layers; l++) {
            tu[l] = ggml_new_tensor_2d(cw, wt, H, F); td[l] = ggml_new_tensor_2d(cw, wt, F, H);
            cur = ggml_add(ca, cur, ggml_mul_mat(ca, td[l], ggml_relu(ca, ggml_mul_mat(ca, tu[l], cur))));
        }
        ggml_build_forward_expand(gf, cur);
        ggml_backend_buffer_t bw = ggml_backend_alloc_ctx_tensors_from_buft(cw, find_buft(fmt == KURN_HC_Q8 ? "KURN" : "CPU"));
        ggml_backend_buffer_set_usage(bw, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        ggml_backend_buffer_t ba = ggml_backend_alloc_ctx_tensors(ca, be);
        for (int l = 0; l < layers; l++) {
            void *e = malloc(kurn_hc_expanded_bytes(wu[l])); kurn_hc_expand(wu[l], e); ggml_backend_tensor_set(tu[l], e, 0, kurn_hc_expanded_bytes(wu[l])); free(e);
            e = malloc(kurn_hc_expanded_bytes(wd[l])); kurn_hc_expand(wd[l], e); ggml_backend_tensor_set(td[l], e, 0, kurn_hc_expanded_bytes(wd[l])); free(e);
        }
        ggml_backend_tensor_set(x, x0, 0, sizeof(float) * H * M);
        ggml_backend_graph_compute(be, gf);
        ggml_backend_tensor_get(cur, xe, 0, sizeof(float) * H * M);
        if (reps > 0) { gargs ga = {be, gf}; te = median_time(reps, run_graph, &ga); }
        ggml_backend_buffer_free(ba); ggml_backend_buffer_free(bw); ggml_free(ca); ggml_free(cw); free(tu); free(td);
        ggml_backend_free(be);
    }
    if (do_c && do_e) {
        double err = 0, mag = 1e-30;
        for (int64_t i = 0; i < H * M; i++) { err = fmax(err, fabs((double)xc[i] - xe[i])); mag = fmax(mag, fabs(xe[i])); }
        char what[200];
        snprintf(what, sizeof what, "mlp D=%d %s Hq=%ld Fq=%ld L=%d M=%ld%s: compact %s expanded after %d layers (relerr %.1e)", D,
                 fmt == KURN_HC_Q8 ? "q8" : "f32", (long)Hq, (long)Fq, layers, (long)M, exact_env() ? " exact" : "",
                 memcmp(xc, xe, sizeof(float) * H * M) ? "!=" : "==", layers, err / mag);
        report(fmt == KURN_HC_Q8 ? !memcmp(xc, xe, sizeof(float) * H * M) : err / mag < 1e-3, what);
    }
    if (reps > 0)
        printf("hcmlp D=%d %s Hq=%ld Fq=%ld L=%d M=%ld T=%d: %s%.3f ms%s (weights %.1f MB compact, %.1f MB expanded)\n", D,
               fmt == KURN_HC_Q8 ? "q8" : "f32", (long)Hq, (long)Fq, layers, (long)M, threads, do_c ? "compact " : "", do_c ? 1e3 * tc : 1e3 * te,
               do_c && do_e ? "" : "", bc / 1e6, be_ / 1e6);
    if (reps > 0 && do_c && do_e) printf("hcmlp expanded %.3f ms\n", 1e3 * te);
    return nfail;
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "see the header of hc_bench.c\n"); return 2; }
    ggml_cpu_init();
    if (!strcmp(argv[1], "quant")) return mode_quant();
    if (!strcmp(argv[1], "check") && argc >= 7)
        return mode_check(atoi(argv[2]), fmt_of(argv[3]), atoll(argv[4]), atoll(argv[5]), argv[6], argc > 7 ? atoi(argv[7]) : 8);
    if (!strcmp(argv[1], "bench") && argc >= 11)
        return mode_bench(atoi(argv[2]), fmt_of(argv[3]), atoll(argv[4]), atoll(argv[5]), atoll(argv[6]), atoi(argv[7]), atoi(argv[8]),
                          atoi(argv[9]), argv[10]);
    if (!strcmp(argv[1], "mlp") && argc >= 11)
        return mode_mlp(atoi(argv[2]), fmt_of(argv[3]), atoll(argv[4]), atoll(argv[5]), atoi(argv[6]), atoll(argv[7]), atoi(argv[8]),
                        atoi(argv[9]), argv[10]);
    fprintf(stderr, "bad arguments; see the header of hc_bench.c\n");
    return 2;
}
