// test-backend-ops-style check of the KURN extra buffer type: MUL_MAT and MUL_MAT_ID with
// weights in the KURN buffer, for every format kurn has kernels for, against a reference
// built from ggml's own vec_dot on the same quantized activations. Also checks batch
// invariance: column 0 must be bit-identical for every batch size 1..40 (GEMV, verify,
// tiled verify and AMX paths), and buffer reuse (set_tensor twice on one buffer).
//
//   test_kurn_buft [smoke|quick|full] [type,...]     exit code = number of failing cases
//   test_kurn_buft case TYPE K N M THREADS [REPS]   one MUL_MAT case
//   test_kurn_buft determinism TYPE K N M THREADS REPS   repeat-run bit-identity (run AMX cases
//                                                     under benchlock.sh, threads pinned)
//
// Build against a llama.cpp checkout with apply.sh applied:
//   gcc -O2 -I$L/ggml/include test_kurn_buft.c -L$L/build/bin -lggml -lggml-base -lggml-cpu -lm
//       -Wl,-rpath,$L/build/bin -o test_kurn_buft      (one command line)
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint64_t rng = 88172645463325252ull;
static float frand(void) {
    rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
    return (float)((rng >> 40) * (1.0 / (1ull << 23))) - 1.0f;
}

static int nfail, npass;
static ggml_backend_buffer_type_t kurn;
static double worst;

static ggml_backend_buffer_type_t find_kurn(void) {
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU));
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    typedef ggml_backend_buffer_type_t *(*fn_t)(ggml_backend_dev_t);
    fn_t fn = (fn_t)ggml_backend_reg_get_proc_address(reg, "ggml_backend_dev_get_extra_bufts");
    for (ggml_backend_buffer_type_t *b = fn ? fn(dev) : NULL; b && *b; b++)
        if (!strcmp(ggml_backend_buft_name(*b), getenv("TEST_BUFT") ? getenv("TEST_BUFT") : "KURN")) return *b;
    return NULL;
}

static void quantize(enum ggml_type t, int64_t K, int64_t rows, void *dst, float scale) {
    float *f = malloc(sizeof(float) * K * rows);
    for (int64_t i = 0; i < K * rows; i++) f[i] = scale * (frand() + frand() + frand()) * (1 + (i / 32) % 5);
    ggml_quantize_chunk(t, f, dst, 0, rows, K, NULL);
    free(f);
}

// out[m * N + n] = vec_dot(W row n, quantized X row m)
static void ref_matmul(enum ggml_type t, const void *w, const float *x, int64_t K, int64_t N, int64_t M, float *out) {
    const struct ggml_type_traits_cpu *tt = ggml_get_type_traits_cpu(t);
    const enum ggml_type vt = tt->vec_dot_type;
    const size_t wrow = ggml_row_size(t, K), xrow = ggml_row_size(vt, K);
    void *xq = malloc(xrow * M);
    for (int64_t m = 0; m < M; m++) ggml_get_type_traits_cpu(vt)->from_float(x + m * K, (char *)xq + m * xrow, K);
    for (int64_t m = 0; m < M; m++)
        for (int64_t n = 0; n < N; n++)
            tt->vec_dot(K, &out[m * N + n], 0, (const char *)w + n * wrow, 0, (char *)xq + m * xrow, 0, 1);
    free(xq);
}

static double relerr(const float *got, const float *ref, int64_t n) {
    double err = 0, mag = 1e-30;
    for (int64_t i = 0; i < n; i++) {
        if (!isfinite(got[i])) return INFINITY;
        err = fmax(err, fabs((double)got[i] - ref[i]));
        mag = fmax(mag, fabs(ref[i]));
    }
    return err / mag;
}

static void report(int ok, const char *what, double e) {
    ok ? npass++ : nfail++;
    if (isfinite(e) && e > worst && ok) worst = e;
    if (!ok || getenv("VERBOSE")) printf("%s %s relerr=%.2e\n", ok ? "ok  " : "FAIL", what, e);
}

typedef struct {
    struct ggml_context *cw, *ca;
    ggml_backend_buffer_t bw, ba;
    struct ggml_tensor *w, *x, *ids, *out;
    struct ggml_cgraph *gf;
} graph;

static void graph_free(graph *g) {
    ggml_backend_buffer_free(g->ba); ggml_backend_buffer_free(g->bw);
    ggml_free(g->ca); ggml_free(g->cw);
}

// MUL_MAT: W [K, N] in KURN, X [K, M, B]
static graph mm_graph(ggml_backend_t be, enum ggml_type t, int64_t K, int64_t N, int64_t M, int64_t B) {
    graph g = {0};
    struct ggml_init_params ip = {16 * ggml_tensor_overhead() + ggml_graph_overhead(), NULL, true};
    g.cw = ggml_init(ip); g.ca = ggml_init(ip);
    g.w = ggml_new_tensor_2d(g.cw, t, K, N);
    g.x = ggml_new_tensor_3d(g.ca, GGML_TYPE_F32, K, M, B);
    g.out = ggml_mul_mat(g.ca, g.w, g.x);
    g.bw = ggml_backend_alloc_ctx_tensors_from_buft(g.cw, kurn);
    ggml_backend_buffer_set_usage(g.bw, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    g.ba = ggml_backend_alloc_ctx_tensors(g.ca, be);
    g.gf = ggml_new_graph(g.ca);
    ggml_build_forward_expand(g.gf, g.out);
    return g;
}

static void case_mm(ggml_backend_t be, enum ggml_type t, int64_t K, int64_t N, int64_t M, int64_t B, int threads, int reuse) {
    const size_t wrow = ggml_row_size(t, K);
    void *wq = malloc(wrow * N), *wq2 = malloc(wrow * N);
    float *x = malloc(sizeof(float) * K * M * B);
    quantize(t, K, N, wq, 0.05f);
    quantize(t, K, N, wq2, 0.02f);
    for (int64_t i = 0; i < K * M * B; i++) x[i] = frand() * (1 + (i / 64) % 3);
    graph g = mm_graph(be, t, K, N, M, B);
    ggml_backend_cpu_set_n_threads(be, threads);
    ggml_backend_tensor_set(g.w, wq, 0, ggml_nbytes(g.w));
    ggml_backend_tensor_set(g.x, x, 0, ggml_nbytes(g.x));
    const int64_t nout = N * M * B;
    float *got = malloc(sizeof(float) * nout), *ref = malloc(sizeof(float) * nout);
    for (int pass = 0; pass < (reuse ? 2 : 1); pass++) {
        if (pass) ggml_backend_tensor_set(g.w, wq2, 0, ggml_nbytes(g.w));
        ggml_backend_graph_compute(be, g.gf);
        ggml_backend_tensor_get(g.out, got, 0, sizeof(float) * nout);
        ref_matmul(t, pass ? wq2 : wq, x, K, N, M * B, ref);
        char what[160];
        snprintf(what, sizeof what, "MUL_MAT %s K=%ld N=%ld M=%ld B=%ld T=%d%s", ggml_type_name(t), (long)K, (long)N, (long)M,
                 (long)B, threads, pass ? " (reused buffer)" : "");
        const double e = relerr(got, ref, nout);
        report(e < 1e-4, what, e);
        if (e >= 1e-4 && getenv("DEBUG"))
            for (int64_t m = 0; m < M * B; m++) {
                const double em = relerr(got + m * N, ref + m * N, N);
                if (em >= 1e-4) printf("  column %ld relerr %.2e\n", (long)m, em);
            }
    }
    free(got); free(ref); graph_free(&g); free(wq); free(wq2); free(x);
}

// column 0 of X computed alone and with 1..maxm-1 other columns must be bit-identical
static void case_invariance(ggml_backend_t be, enum ggml_type t, int64_t K, int64_t N, int maxm) {
    const size_t wrow = ggml_row_size(t, K);
    void *wq = malloc(wrow * N);
    quantize(t, K, N, wq, 0.05f);
    float *x = malloc(sizeof(float) * K * maxm);
    for (int64_t i = 0; i < K * maxm; i++) x[i] = frand();
    float *y1 = malloc(sizeof(float) * N * maxm), *ym = malloc(sizeof(float) * N * maxm);
    int bad = 0, badm = 0;
    for (int M = 1; M <= maxm; M++) {
        graph g = mm_graph(be, t, K, N, M, 1);
        ggml_backend_cpu_set_n_threads(be, 1 + M % 8);
        ggml_backend_tensor_set(g.w, wq, 0, ggml_nbytes(g.w));
        ggml_backend_tensor_set(g.x, x, 0, ggml_nbytes(g.x));
        ggml_backend_graph_compute(be, g.gf);
        ggml_backend_tensor_get(g.out, M == 1 ? y1 : ym, 0, sizeof(float) * N * M);
        if (M > 1 && memcmp(y1, ym, sizeof(float) * N)) { bad++; if (!badm) badm = M; }
        graph_free(&g);
    }
    char what[160];
    snprintf(what, sizeof what, "batch invariance %s K=%ld N=%ld M=1..%d (first differing M=%d)", ggml_type_name(t), (long)K,
             (long)N, maxm, badm);
    report(bad == 0, what, (double)bad);
    free(wq); free(x); free(y1); free(ym);
}

// MUL_MAT_ID: W [K, N, E] in KURN, X [K, ne11, T] (ne11 = 1: shared input, or n_used), ids [n_used, T]
static void case_mmid(ggml_backend_t be, enum ggml_type t, int64_t K, int64_t N, int64_t E, int64_t n_used, int64_t T,
                      int bcast, int threads) {
    const size_t wrow = ggml_row_size(t, K);
    const int64_t ne11 = bcast ? 1 : n_used;
    void *wq = malloc(wrow * N * E);
    quantize(t, K, N * E, wq, 0.05f);
    float *x = malloc(sizeof(float) * K * ne11 * T);
    for (int64_t i = 0; i < K * ne11 * T; i++) x[i] = frand();
    int32_t *ids = malloc(sizeof(int32_t) * n_used * T);
    for (int64_t tk = 0; tk < T; tk++)
        for (int64_t i = 0; i < n_used; i++) {
            int32_t e;
            int dup;
            do {  // distinct experts per token, skewed so some experts get many tokens
                e = (int32_t)(fabsf(frand()) * fabsf(frand()) * E) % E;
                dup = 0;
                for (int64_t j = 0; j < i; j++) dup |= ids[tk * n_used + j] == e;
            } while (dup);
            ids[tk * n_used + i] = e;
        }
    struct ggml_init_params ip = {16 * ggml_tensor_overhead() + ggml_graph_overhead(), NULL, true};
    graph g = {0};
    g.cw = ggml_init(ip); g.ca = ggml_init(ip);
    g.w = ggml_new_tensor_3d(g.cw, t, K, N, E);
    g.x = ggml_new_tensor_3d(g.ca, GGML_TYPE_F32, K, ne11, T);
    g.ids = ggml_new_tensor_2d(g.ca, GGML_TYPE_I32, n_used, T);
    g.out = ggml_mul_mat_id(g.ca, g.w, g.x, g.ids);
    g.bw = ggml_backend_alloc_ctx_tensors_from_buft(g.cw, kurn);
    ggml_backend_buffer_set_usage(g.bw, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    g.ba = ggml_backend_alloc_ctx_tensors(g.ca, be);
    g.gf = ggml_new_graph(g.ca);
    ggml_build_forward_expand(g.gf, g.out);
    ggml_backend_cpu_set_n_threads(be, threads);
    ggml_backend_tensor_set(g.w, wq, 0, ggml_nbytes(g.w));
    ggml_backend_tensor_set(g.x, x, 0, ggml_nbytes(g.x));
    ggml_backend_tensor_set(g.ids, ids, 0, ggml_nbytes(g.ids));
    ggml_backend_graph_compute(be, g.gf);
    const int64_t nout = N * n_used * T;
    float *got = malloc(sizeof(float) * nout), *ref = malloc(sizeof(float) * nout);
    ggml_backend_tensor_get(g.out, got, 0, sizeof(float) * nout);
    for (int64_t tk = 0; tk < T; tk++)
        for (int64_t i = 0; i < n_used; i++) {
            const int32_t e = ids[tk * n_used + i];
            ref_matmul(t, (const char *)wq + e * N * wrow, x + (tk * ne11 + (bcast ? 0 : i)) * K, K, N, 1,
                       ref + (tk * n_used + i) * N);
        }
    char what[200];
    snprintf(what, sizeof what, "MUL_MAT_ID %s K=%ld N=%ld E=%ld used=%ld T=%ld bcast=%d threads=%d", ggml_type_name(t), (long)K,
             (long)N, (long)E, (long)n_used, (long)T, bcast, threads);
    const double e = relerr(got, ref, nout);
    report(e < 1e-4, what, e);
    free(got); free(ref); graph_free(&g); free(wq); free(x); free(ids);
}

// the same MUL_MAT computed reps times must be bit-identical every time (AMX tile state on VMs),
// match the reference, and its column 0 must equal the 1-column (GEMV) result bit for bit
static void case_determinism(ggml_backend_t be, enum ggml_type t, int64_t K, int64_t N, int64_t M, int threads, int reps) {
    const size_t wrow = ggml_row_size(t, K);
    void *wq = malloc(wrow * N);
    quantize(t, K, N, wq, 0.05f);
    float *x = malloc(sizeof(float) * K * M);
    for (int64_t i = 0; i < K * M; i++) x[i] = frand();
    float *first = malloc(sizeof(float) * N * M), *got = malloc(sizeof(float) * N * M), *ref = malloc(sizeof(float) * N * M);
    graph g1 = mm_graph(be, t, K, N, 1, 1);
    ggml_backend_cpu_set_n_threads(be, threads);
    ggml_backend_tensor_set(g1.w, wq, 0, ggml_nbytes(g1.w));
    ggml_backend_tensor_set(g1.x, x, 0, ggml_nbytes(g1.x));
    ggml_backend_graph_compute(be, g1.gf);
    ggml_backend_tensor_get(g1.out, ref, 0, sizeof(float) * N);
    graph_free(&g1);
    graph g = mm_graph(be, t, K, N, M, 1);
    ggml_backend_tensor_set(g.w, wq, 0, ggml_nbytes(g.w));
    ggml_backend_tensor_set(g.x, x, 0, ggml_nbytes(g.x));
    int differ = 0;
    for (int r = 0; r < reps; r++) {
        ggml_backend_graph_compute(be, g.gf);
        ggml_backend_tensor_get(g.out, r ? got : first, 0, sizeof(float) * N * M);
        if (r && memcmp(first, got, sizeof(float) * N * M)) differ++;
    }
    const int col0 = memcmp(first, ref, sizeof(float) * N) == 0;
    ref_matmul(t, wq, x, K, N, M, ref);
    const double e = relerr(first, ref, N * M);
    char what[200];
    snprintf(what, sizeof what, "determinism %s K=%ld N=%ld M=%ld T=%d: %d/%d runs differ from run 1, column 0 %s GEMV",
             ggml_type_name(t), (long)K, (long)N, (long)M, threads, differ, reps - 1, col0 ? "==" : "!=");
    report(differ == 0 && col0 && e < 1e-4, what, e);
    graph_free(&g); free(wq); free(x); free(first); free(got); free(ref);
}

static int64_t kmult(enum ggml_type t) {
    const int64_t b = ggml_blck_size(t);
    return b < 32 ? 32 : b;
}

int main(int argc, char **argv) {
    const int full = argc > 1 && !strcmp(argv[1], "full"), smoke = argc > 1 && !strcmp(argv[1], "smoke");
    kurn = find_kurn();
    if (!kurn) { printf("KURN buffer type not available (GGML_KURN=0 or no AVX-512 VNNI)\n"); return 1; }
    ggml_backend_t be = ggml_backend_cpu_init();
    if (argc > 7 && !strcmp(argv[1], "determinism")) {  // determinism TYPE K N M THREADS REPS
        enum ggml_type t = GGML_TYPE_COUNT;
        for (int i = 0; i < GGML_TYPE_COUNT; i++)
            if (ggml_type_name((enum ggml_type)i) && !strcmp(ggml_type_name((enum ggml_type)i), argv[2])) t = (enum ggml_type)i;
        // threads pinned 1:1 to CPUs 0..n-1 (strict placement), as AMX on a VM requires
        struct ggml_threadpool_params tpp = ggml_threadpool_params_default(atoi(argv[6]));
        for (int i = 0; i < tpp.n_threads; i++) tpp.cpumask[i] = true;
        tpp.strict_cpu = true;
        struct ggml_threadpool *tp = ggml_threadpool_new(&tpp);
        ggml_backend_cpu_set_threadpool(be, tp);
        case_determinism(be, t, atoll(argv[3]), atoll(argv[4]), atoll(argv[5]), atoi(argv[6]), atoi(argv[7]));
        printf("%d passed, %d failed\n", npass, nfail);
        ggml_backend_free(be);
        ggml_threadpool_free(tp);
        return nfail;
    }
    if (argc > 6 && !strcmp(argv[1], "case")) {  // case TYPE K N M THREADS [REPS]
        enum ggml_type t = GGML_TYPE_COUNT;
        for (int i = 0; i < GGML_TYPE_COUNT; i++)
            if (ggml_type_name((enum ggml_type)i) && !strcmp(ggml_type_name((enum ggml_type)i), argv[2])) t = (enum ggml_type)i;
        for (int r = 0; r < (argc > 7 ? atoi(argv[7]) : 1); r++)
            case_mm(be, t, atoll(argv[3]), atoll(argv[4]), atoll(argv[5]), 1, atoi(argv[6]), 0);
        printf("%d passed, %d failed\n", npass, nfail);
        ggml_backend_free(be);
        return nfail;
    }
    enum ggml_type all[] = {GGML_TYPE_Q8_0, GGML_TYPE_Q4_0, GGML_TYPE_IQ4_NL, GGML_TYPE_Q4_K,
                            GGML_TYPE_Q2_0, GGML_TYPE_TQ2_0, GGML_TYPE_Q1_0};
    for (size_t ti = 0; ti < sizeof all / sizeof *all; ti++) {
        const enum ggml_type t = all[ti];
        if (argc > 2 && !strstr(argv[2], ggml_type_name(t))) continue;
        const int64_t km = kmult(t) < 256 && (t == GGML_TYPE_Q4_K || t == GGML_TYPE_TQ2_0) ? 256 : kmult(t);
        const int nfail0 = nfail, npass0 = npass;
        worst = 0;
        if (smoke) {
            const int64_t Ms[] = {1, 3, 9, 40};
            for (int i = 0; i < 4; i++) {
                case_mm(be, t, km, 17, Ms[i], 1, 3, 0);
                case_mm(be, t, 2048, 1000, Ms[i], 1, 8, 0);
            }
            case_mm(be, t, 1024, 64, 2, 3, 4, 0);
            case_mm(be, t, 1024, 200, 5, 1, 8, 1);
            case_mm(be, t, 32768 + km, 64, 2, 1, 8, 0);
            case_invariance(be, t, 1024, 333, 12);
            case_mmid(be, t, 1024, 160, 16, 4, 1, 1, 8);
            case_mmid(be, t, 1024, 160, 16, 4, 9, 0, 3);
            printf("%-7s %4d passed, %d failed (worst relerr %.1e)\n", ggml_type_name(t), npass - npass0, nfail - nfail0, worst);
            continue;
        }
        const int64_t Ks[] = {km, 2 * km, 3 * km, 1024, 2048, 4096 + km};
        const int64_t Ns[] = {1, 15, 16, 17, 100, 1000, 4097};
        const int64_t Ms[] = {1, 2, 3, 5, 8, 9, 16, 17, 33, 64};
        for (size_t ki = 0; ki < sizeof Ks / sizeof *Ks; ki++)
            for (size_t ni = 0; ni < sizeof Ns / sizeof *Ns; ni++)
                for (size_t mi = 0; mi < (full ? 10 : 4); mi++)
                    case_mm(be, t, Ks[ki], Ns[ni], Ms[mi], 1, full ? 1 + (int)((ki + ni + mi) % 8) : 8, 0);
        for (int th = 1; th <= 8; th++) case_mm(be, t, 2048, 1037, 1, 1, th, 0);
        case_mm(be, t, 1024, 64, 2, 3, 4, 0);           // 2D weight broadcast over ne12 = 3
        case_mm(be, t, 2048, 512, 1, 1, 8, 1);          // same buffer, new weights
        case_mm(be, t, 2048, 512, 37, 1, 3, 1);
        case_mm(be, t, 32768, 300, 1, 1, 8, 0);         // largest K kurn accepts
        case_mm(be, t, 32768, 300, 19, 1, 8, 0);
        case_mm(be, t, 32768 + km, 64, 2, 1, 8, 0);     // above the limit: raw copy, ggml computes it
        case_mm(be, t, 2048, 512, 128, 1, 8, 0);        // prefill-sized
        if (full) case_mm(be, t, 1024, 151936, 1, 1, 8, 0);  // vocabulary-sized output layer
        case_invariance(be, t, 2048, 333, full ? 40 : 20);
        const int64_t Ts[] = {1, 2, 5, 64};
        for (size_t i = 0; i < 4; i++) {
            case_mmid(be, t, 1024, 160, 16, 4, Ts[i], 1, 8);
            case_mmid(be, t, 1024, 160, 16, 4, Ts[i], 0, 3);
        }
        case_mmid(be, t, 2048, 512, 64, 8, 128, 1, 8);  // OLMoE-like prefill batch
        printf("%-7s %4d passed, %d failed (worst relerr %.1e)\n", ggml_type_name(t), npass - npass0, nfail - nfail0, worst);
    }
    printf("%s: %d passed, %d failed\n", full ? "full" : smoke ? "smoke" : "quick", npass, nfail);
    ggml_backend_free(be);
    return nfail;
}
