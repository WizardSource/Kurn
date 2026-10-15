// test-backend-ops-style check of the KURN extra buffer type: MUL_MAT and MUL_MAT_ID with
// weights in the KURN buffer, for every format kurn has kernels for, against a reference
// built from ggml's own vec_dot on the same quantized activations. Also checks batch
// invariance: column 0 must be bit-identical for every batch size 1..40 (GEMV, verify,
// tiled verify and AMX paths), and buffer reuse (set_tensor twice on one buffer).
//
//   test_kurn_buft [smoke|quick|full] [type,...]     exit code = number of failing cases
//   test_kurn_buft native [q6_K,q5_K]               native-layout Q6_K / Q5_K below the AMX-BF16 threshold
//   test_kurn_buft tbm [smoke]                      TQ1_0 / TQ2_0 as bitmap + signs: bit-identical to TQ2_0 records
//   test_kurn_buft case TYPE K N M THREADS [REPS]   one MUL_MAT case
//   test_kurn_buft determinism TYPE K N M THREADS REPS   repeat-run bit-identity (run AMX cases
//                                                     under benchlock.sh, threads pinned)
//   test_kurn_buft bf16                             the AMX-BF16 prefill path: its error against an f32
//                                                     reference must stay within 2x of ggml's own vec_dot
//   test_kurn_buft fused [type,...]                 ffn gate + up + SwiGLU and q + k + v fused into one op each:
//                                                     bit-identical to the unfused graph (exactness from the env);
//                                                     KL-mode gate / up also pass the Freivalds oracle
//   test_kurn_buft bufts                            print the CPU backend's extra buffer types, in order (AMX
//                                                     only when kurn allows ggml's AMX buffer: not with GGML_KURN_AMX=0)
//   test_kurn_buft freivalds                        Freivalds oracle (bigVerify-style secret-probe checksums) on
//                                                     the AMX-BF16 path: no false alarm on real outputs, and planted
//                                                     faults (one entry, one 16x16 tile) rejected
//
// Every mode but bf16 checks exact results, so it runs with GGML_KURN_EXACT=1 (batch-invariant matmuls only:
// the Q8_0 AMX-INT8 kernel and the verify kernels) unless GGML_KURN_EXACT is set in the environment.
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

// every column of an M-column batch must be bit-identical to that column computed alone (GEMV),
// for M = 2..maxm and varying thread counts: exact-width and padded verify kernels, row-group
// schedules and the AMX paths all have to reproduce one-token decoding
static void case_invariance_all(ggml_backend_t be, enum ggml_type t, int64_t K, int64_t N, int maxm) {
    const size_t wrow = ggml_row_size(t, K);
    void *wq = malloc(wrow * N);
    quantize(t, K, N, wq, 0.05f);
    float *x = malloc(sizeof(float) * K * maxm);
    for (int64_t i = 0; i < K * maxm; i++) x[i] = frand();
    float *y1 = malloc(sizeof(float) * N * maxm), *ym = malloc(sizeof(float) * N * maxm);
    graph g1 = mm_graph(be, t, K, N, 1, 1);
    ggml_backend_tensor_set(g1.w, wq, 0, ggml_nbytes(g1.w));
    for (int j = 0; j < maxm; j++) {
        ggml_backend_cpu_set_n_threads(be, 1 + j % 8);
        ggml_backend_tensor_set(g1.x, x + j * K, 0, sizeof(float) * K);
        ggml_backend_graph_compute(be, g1.gf);
        ggml_backend_tensor_get(g1.out, y1 + j * N, 0, sizeof(float) * N);
    }
    graph_free(&g1);
    int bad = 0, badm = 0, badj = 0;
    for (int M = 2; M <= maxm; M++) {
        graph g = mm_graph(be, t, K, N, M, 1);
        ggml_backend_cpu_set_n_threads(be, 1 + (M * 3) % 8);
        ggml_backend_tensor_set(g.w, wq, 0, ggml_nbytes(g.w));
        ggml_backend_tensor_set(g.x, x, 0, sizeof(float) * K * M);
        ggml_backend_graph_compute(be, g.gf);
        ggml_backend_tensor_get(g.out, ym, 0, sizeof(float) * N * M);
        for (int j = 0; j < M; j++)
            if (memcmp(y1 + j * N, ym + j * N, sizeof(float) * N)) { bad++; if (!badm) { badm = M; badj = j; } }
        graph_free(&g);
    }
    char what[200];
    snprintf(what, sizeof what, "batch invariance, all columns %s K=%ld N=%ld M=2..%d (first differing M=%d column %d)",
             ggml_type_name(t), (long)K, (long)N, maxm, badm, badj);
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

// AMX-BF16 prefill (bf16 activations, weights dequantized to bf16): error against an f32 reference (dequantized
// weights x f32 activations, double accumulation) must stay within 2x of ggml's vec_dot on quantized activations
static void case_bf16(ggml_backend_t be, enum ggml_type t, int64_t K, int64_t N, int64_t M, int threads) {
    const size_t wrow = ggml_row_size(t, K);
    void *wq = malloc(wrow * N);
    quantize(t, K, N, wq, 0.05f);
    float *x = malloc(sizeof(float) * K * M), *wf = malloc(sizeof(float) * K * N);
    for (int64_t i = 0; i < K * M; i++) x[i] = frand() * (1 + (i / 64) % 3);
    for (int64_t n = 0; n < N; n++) ggml_get_type_traits(t)->to_float((const char *)wq + n * wrow, wf + n * K, K);
    graph g = mm_graph(be, t, K, N, M, 1);
    ggml_backend_cpu_set_n_threads(be, threads);
    ggml_backend_tensor_set(g.w, wq, 0, ggml_nbytes(g.w));
    ggml_backend_tensor_set(g.x, x, 0, ggml_nbytes(g.x));
    ggml_backend_graph_compute(be, g.gf);
    float *got = malloc(sizeof(float) * N * M), *refq = malloc(sizeof(float) * N * M), *reff = malloc(sizeof(float) * N * M);
    ggml_backend_tensor_get(g.out, got, 0, sizeof(float) * N * M);
    ref_matmul(t, wq, x, K, N, M, refq);
    for (int64_t m = 0; m < M; m++)
        for (int64_t n = 0; n < N; n++) {
            double s = 0;
            for (int64_t k = 0; k < K; k++) s += (double)wf[n * K + k] * x[m * K + k];
            reff[m * N + n] = (float)s;
        }
    const double eg = relerr(got, reff, N * M), eq = relerr(refq, reff, N * M);
    char what[200];
    snprintf(what, sizeof what, "bf16 path %s K=%ld N=%ld M=%ld T=%d: err vs f32 %.2e, ggml vec_dot %.2e", ggml_type_name(t), (long)K,
             (long)N, (long)M, threads, eg, eq);
    report(eg <= 2 * eq + 1e-6, what, eg);
    graph_free(&g); free(wq); free(x); free(wf); free(got); free(refq); free(reff);
}

// the same MUL_MAT computed reps times must be bit-identical and match the reference (1e-4 in exact mode, 2e-2
// otherwise); run it next to other AMX users (tools/amx_ctx.c) to check the AMX preemption guard on a VM
static void case_stress(ggml_backend_t be, enum ggml_type t, int64_t K, int64_t N, int64_t M, int threads, int reps) {
    const size_t wrow = ggml_row_size(t, K);
    void *wq = malloc(wrow * N);
    quantize(t, K, N, wq, 0.05f);
    float *x = malloc(sizeof(float) * K * M);
    for (int64_t i = 0; i < K * M; i++) x[i] = frand();
    float *first = malloc(sizeof(float) * N * M), *got = malloc(sizeof(float) * N * M), *ref = malloc(sizeof(float) * N * M);
    graph g = mm_graph(be, t, K, N, M, 1);
    ggml_backend_cpu_set_n_threads(be, threads);
    ggml_backend_tensor_set(g.w, wq, 0, ggml_nbytes(g.w));
    ggml_backend_tensor_set(g.x, x, 0, ggml_nbytes(g.x));
    int differ = 0;
    for (int r = 0; r < reps; r++) {
        ggml_backend_graph_compute(be, g.gf);
        ggml_backend_tensor_get(g.out, r ? got : first, 0, sizeof(float) * N * M);
        if (r && memcmp(first, got, sizeof(float) * N * M)) differ++;
    }
    ref_matmul(t, wq, x, K, N, M, ref);
    const double e = relerr(first, ref, N * M);
    const char * ex = getenv("GGML_KURN_EXACT");
    const double tol = ex && atoi(ex) ? 1e-4 : 2e-2;
    char what[200];
    snprintf(what, sizeof what, "stress %s K=%ld N=%ld M=%ld T=%d: %d/%d runs differ from run 1", ggml_type_name(t), (long)K, (long)N,
             (long)M, threads, differ, reps - 1);
    report(differ == 0 && e < tol, what, e);
    graph_free(&g); free(wq); free(x); free(first); free(got); free(ref);
}

// ---------------------------------------------------------------------------------------------------------
// Freivalds oracle for KL-mode (non-exact) kernels, after bigVerify's tensor checks (github.com/WizardSource/bigVerify,
// docs/bigverify-tensor.md): for a claim C ~ X W^T (X: M x K activations, W: N x K dequantized weights), split the N
// output columns into blocks of 64, draw t = 2 secret +-1 probe vectors r per block, and compare C r with X (W r)
// computed in double from the f32 inputs. Error model of the AMX-BF16 path: activations and dequantized weights
// rounded to bf16 (each product off by <= 2^-8 relative), fp32 accumulation over K. Statistical mode as in
// bigVerify's monitoring default: the rounding errors of one output sum like a random walk, so a correct output is
// within (2^-8 + K u_32) * q_mj (q_mj = sqrt(sum_k x_mk^2 w_jk^2)), and over the secret signs the block
// checksum of correct outputs exceeds lambda * sqrt(sum_j b_mj^2) with probability <= alpha (Hoeffding, alpha = 2^-40
// per check). sum_j q_mj^2 = x_m^2 . (sum_j w_j^2), so the threshold costs one more fingerprint-sized pass: the check
// is O(M K t N / 64 + M N), against O(M N K) for the product. Test-only; nothing here runs in llama.cpp.
typedef struct { int rejected, checks; double worst, tau_max; } fv_result;

static uint64_t fv_seed(void) {
    uint64_t s = 0;
    FILE *f = fopen("/dev/urandom", "rb");
    if (!f || fread(&s, sizeof s, 1, f) != 1) s = 0x9e3779b97f4a7c15ull ^ (uint64_t)(size_t)&s;
    if (f) fclose(f);
    return s | 1;
}

static fv_result freivalds_check(const float *x, const float *w, const float *c, int64_t K, int64_t N, int64_t M) {
    const int64_t B = 64, G = (N + B - 1) / B;
    const int t = 2;
    uint64_t r = fv_seed();
    const double alpha = ldexp(1.0, -40), lam = sqrt(2 * log(2.0 * (double)M * G * t / alpha));
    const double u_in = ldexp(1.0, -8), u_acc = ldexp(1.0, -24);
    double *wr = malloc(sizeof(double) * K * t), *w2 = malloc(sizeof(double) * K);
    signed char *sg = malloc((size_t)B * t);
    fv_result res = {0, 0, 0, 0};
    for (int64_t g = 0; g < G; g++) {
        const int64_t j0 = g * B, j1 = j0 + B < N ? j0 + B : N;
        for (int64_t i = 0; i < (j1 - j0) * t; i++) { r ^= r << 13; r ^= r >> 7; r ^= r << 17; sg[i] = (r >> 40) & 1 ? 1 : -1; }
        for (int64_t k = 0; k < K; k++) {
            double s2 = 0;
            for (int p = 0; p < t; p++) wr[p * K + k] = 0;
            for (int64_t j = j0; j < j1; j++) {
                const double v = w[j * K + k];
                for (int p = 0; p < t; p++) wr[p * K + k] += sg[p * B + (j - j0)] * v;
                s2 += v * v;
            }
            w2[k] = s2;
        }
        for (int64_t m = 0; m < M; m++) {
            const float *xm = x + m * K;
            double q2 = 0;
            for (int64_t k = 0; k < K; k++) q2 += (double)xm[k] * xm[k] * w2[k];
            // per-output bound b_mj = (u_in + K u_acc) q_mj (accumulation: a_mj <= sqrt(K) q_mj and a sqrt(K) u_acc random
            // walk), summed in quadrature over the block: sqrt(sum_j b_mj^2) = (u_in + K u_acc) sqrt(x_m^2 . sum_j w_j^2)
            const double bq = (u_in + (double)K * u_acc) * sqrt(q2);
            const double tau = lam * bq + 1e-30;
            if (tau > res.tau_max) res.tau_max = tau;
            for (int p = 0; p < t; p++) {
                double lhs = 0, rhs = 0;
                for (int64_t j = j0; j < j1; j++) lhs += sg[p * B + (j - j0)] * (double)c[m * N + j];
                for (int64_t k = 0; k < K; k++) rhs += (double)xm[k] * wr[p * K + k];
                const double ratio = fabs(lhs - rhs) / tau;
                res.checks++;
                if (!(ratio <= 1)) res.rejected++;
                if (ratio > res.worst || ratio != ratio) res.worst = ratio;
            }
        }
    }
    free(wr); free(w2); free(sg);
    return res;
}

static void dequant_rows(enum ggml_type t, const void *wq, int64_t K, int64_t N, float *wf) {
    const size_t wrow = ggml_row_size(t, K);
    for (int64_t n = 0; n < N; n++) ggml_get_type_traits(t)->to_float((const char *)wq + n * wrow, wf + n * K, K);
}

// AMX-BF16 path through the oracle: every real output accepted; one entry moved by the output RMS (above the 2 tau floor), and one 16 x 16
// tile replaced by another tile's values (what lost AMX tile state looks like), both rejected
static void case_freivalds(ggml_backend_t be, enum ggml_type t, int64_t K, int64_t N, int64_t M, int threads) {
    const size_t wrow = ggml_row_size(t, K);
    void *wq = malloc(wrow * N);
    quantize(t, K, N, wq, 0.05f);
    float *x = malloc(sizeof(float) * K * M), *wf = malloc(sizeof(float) * K * N), *got = malloc(sizeof(float) * N * M);
    for (int64_t i = 0; i < K * M; i++) x[i] = frand() * (1 + (i / 64) % 3);
    dequant_rows(t, wq, K, N, wf);
    graph g = mm_graph(be, t, K, N, M, 1);
    ggml_backend_cpu_set_n_threads(be, threads);
    ggml_backend_tensor_set(g.w, wq, 0, ggml_nbytes(g.w));
    ggml_backend_tensor_set(g.x, x, 0, ggml_nbytes(g.x));
    ggml_backend_graph_compute(be, g.gf);
    ggml_backend_tensor_get(g.out, got, 0, sizeof(float) * N * M);
    const fv_result ok = freivalds_check(x, wf, got, K, N, M);
    double rms = 0;
    for (int64_t i = 0; i < N * M; i++) rms += (double)got[i] * got[i];
    rms = sqrt(rms / (N * M));
    const int64_t mi = M / 2, ni = N / 3;
    got[mi * N + ni] += (float)rms;
    const fv_result f1 = freivalds_check(x, wf, got, K, N, M);
    got[mi * N + ni] -= (float)rms;
    for (int64_t m = 0; m < 16 && m < M; m++)
        for (int64_t n = 0; n < 16; n++) got[m * N + 32 + n] = got[((m + 16) % M) * N + 64 + n];
    const fv_result f2 = freivalds_check(x, wf, got, K, N, M);
    char what[240];
    snprintf(what, sizeof what, "freivalds %s K=%ld N=%ld M=%ld T=%d: clean %d/%d rejected (worst %.3f of threshold), "
             "floor 2*tau = %.2f x output RMS, entry fault (1x RMS) %s, tile fault %s", ggml_type_name(t), (long)K, (long)N, (long)M,
             threads, ok.rejected, ok.checks, ok.worst, 2 * ok.tau_max / rms, f1.rejected ? "caught" : "MISSED", f2.rejected ? "caught" : "MISSED");
    report(ok.rejected == 0 && f1.rejected > 0 && f2.rejected > 0, what, ok.worst);
    graph_free(&g); free(wq); free(x); free(wf); free(got);
}

// ffn block: gate = Wg x, up = Wu x, s = swiglu(gate, up). Graph A lets ggml-cpu fuse the three nodes; graph B marks
// gate as an output, which blocks the fusion. s must be bit-identical; in KL mode (AMX-BF16, M >= the AMX threshold
// outside exact mode) gate and up also go through the Freivalds oracle.
static void case_fused(ggml_backend_t be, enum ggml_type t, int64_t K, int64_t N, int64_t M, int threads, int kl) {
    const size_t wrow = ggml_row_size(t, K);
    void *wg = malloc(wrow * N), *wu = malloc(wrow * N);
    quantize(t, K, N, wg, 0.05f);
    quantize(t, K, N, wu, 0.04f);
    float *x = malloc(sizeof(float) * K * M);
    for (int64_t i = 0; i < K * M; i++) x[i] = frand() * (1 + (i / 64) % 3);
    float *s[2], *gate = malloc(sizeof(float) * N * M), *up = malloc(sizeof(float) * N * M);
    for (int v = 0; v < 2; v++) {
        struct ggml_init_params ip = {16 * ggml_tensor_overhead() + ggml_graph_overhead(), NULL, true};
        struct ggml_context *cw = ggml_init(ip), *ca = ggml_init(ip);
        struct ggml_tensor *tg = ggml_new_tensor_2d(cw, t, K, N), *tu = ggml_new_tensor_2d(cw, t, K, N);
        struct ggml_tensor *tx = ggml_new_tensor_2d(ca, GGML_TYPE_F32, K, M);
        struct ggml_tensor *g = ggml_mul_mat(ca, tg, tx), *u = ggml_mul_mat(ca, tu, tx);
        struct ggml_tensor *out = ggml_swiglu_split(ca, g, u);
        if (v == 1) { ggml_set_output(g); ggml_set_output(u); }
        ggml_backend_buffer_t bw = ggml_backend_alloc_ctx_tensors_from_buft(cw, kurn);
        ggml_backend_buffer_set_usage(bw, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        ggml_backend_buffer_t ba = ggml_backend_alloc_ctx_tensors(ca, be);
        struct ggml_cgraph *gf = ggml_new_graph(ca);
        ggml_build_forward_expand(gf, out);
        ggml_backend_cpu_set_n_threads(be, threads);
        ggml_backend_tensor_set(tg, wg, 0, ggml_nbytes(tg));
        ggml_backend_tensor_set(tu, wu, 0, ggml_nbytes(tu));
        ggml_backend_tensor_set(tx, x, 0, ggml_nbytes(tx));
        ggml_backend_graph_compute(be, gf);
        s[v] = malloc(sizeof(float) * N * M);
        ggml_backend_tensor_get(out, s[v], 0, sizeof(float) * N * M);
        if (v == 1) {
            ggml_backend_tensor_get(g, gate, 0, sizeof(float) * N * M);
            ggml_backend_tensor_get(u, up, 0, sizeof(float) * N * M);
        }
        ggml_backend_buffer_free(ba); ggml_backend_buffer_free(bw); ggml_free(ca); ggml_free(cw);
    }
    const int same = memcmp(s[0], s[1], sizeof(float) * N * M) == 0;
    // the unfused members against their own reference: vec_dot in exact mode, the Freivalds oracle in KL mode
    int okm = 1;
    double e = 0;
    if (kl) {
        float *wf = malloc(sizeof(float) * K * N);
        dequant_rows(t, wg, K, N, wf);
        const fv_result a = freivalds_check(x, wf, gate, K, N, M);
        dequant_rows(t, wu, K, N, wf);
        const fv_result b = freivalds_check(x, wf, up, K, N, M);
        okm = a.rejected == 0 && b.rejected == 0;
        e = fmax(a.worst, b.worst);
        free(wf);
    } else {
        float *ref = malloc(sizeof(float) * N * M);
        ref_matmul(t, wg, x, K, N, M, ref);
        e = relerr(gate, ref, N * M);
        ref_matmul(t, wu, x, K, N, M, ref);
        e = fmax(e, relerr(up, ref, N * M));
        okm = e < 1e-4;
        free(ref);
    }
    char what[200];
    snprintf(what, sizeof what, "fused gate+up+swiglu %s K=%ld N=%ld M=%ld T=%d: fused %s unfused, members %s (%s %.2e)",
             ggml_type_name(t), (long)K, (long)N, (long)M, threads, same ? "==" : "!=", okm ? "ok" : "BAD",
             kl ? "freivalds worst" : "relerr", e);
    report(same && okm, what, kl ? 0 : e);
    free(wg); free(wu); free(x); free(s[0]); free(s[1]); free(gate); free(up);
}

// attention q, k, v on the same activations: graph A builds the three MUL_MATs as consecutive nodes (ggml-cpu fuses
// them into one op), graph B puts a node between each (no fusion). All three outputs must be bit-identical.
static void case_qkv(ggml_backend_t be, enum ggml_type t, int64_t K, int64_t Nq, int64_t Nkv, int64_t M, int threads) {
    const size_t wrow = ggml_row_size(t, K);
    const int64_t Ns[3] = {Nq, Nkv, Nkv};
    void *wq[3];
    for (int i = 0; i < 3; i++) { wq[i] = malloc(wrow * Ns[i]); quantize(t, K, Ns[i], wq[i], 0.05f - 0.01f * i); }
    float *x = malloc(sizeof(float) * K * M);
    for (int64_t i = 0; i < K * M; i++) x[i] = frand() * (1 + (i / 64) % 3);
    float *out[2][3];
    for (int v = 0; v < 2; v++) {
        struct ggml_init_params ip = {32 * ggml_tensor_overhead() + ggml_graph_overhead(), NULL, true};
        struct ggml_context *cw = ggml_init(ip), *ca = ggml_init(ip);
        struct ggml_tensor *w[3], *y[3];
        struct ggml_tensor *tx = ggml_new_tensor_2d(ca, GGML_TYPE_F32, K, M);
        struct ggml_cgraph *gf = ggml_new_graph(ca);
        for (int i = 0; i < 3; i++) {
            w[i] = ggml_new_tensor_2d(cw, t, K, Ns[i]);
            y[i] = ggml_mul_mat(ca, w[i], tx);
            ggml_set_output(y[i]);
            ggml_build_forward_expand(gf, y[i]);
            if (v == 1) ggml_build_forward_expand(gf, ggml_scale(ca, y[i], 0.5f));
        }
        ggml_backend_buffer_t bw = ggml_backend_alloc_ctx_tensors_from_buft(cw, kurn);
        ggml_backend_buffer_set_usage(bw, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        ggml_backend_buffer_t ba = ggml_backend_alloc_ctx_tensors(ca, be);
        ggml_backend_cpu_set_n_threads(be, threads);
        for (int i = 0; i < 3; i++) ggml_backend_tensor_set(w[i], wq[i], 0, ggml_nbytes(w[i]));
        ggml_backend_tensor_set(tx, x, 0, ggml_nbytes(tx));
        ggml_backend_graph_compute(be, gf);
        for (int i = 0; i < 3; i++) {
            out[v][i] = malloc(sizeof(float) * Ns[i] * M);
            ggml_backend_tensor_get(y[i], out[v][i], 0, sizeof(float) * Ns[i] * M);
        }
        ggml_backend_buffer_free(ba); ggml_backend_buffer_free(bw); ggml_free(ca); ggml_free(cw);
    }
    int same = 1;
    for (int i = 0; i < 3; i++) same &= memcmp(out[0][i], out[1][i], sizeof(float) * Ns[i] * M) == 0;
    char what[200];
    snprintf(what, sizeof what, "fused q+k+v %s K=%ld N=%ld+%ld+%ld M=%ld T=%d: fused %s unfused", ggml_type_name(t), (long)K,
             (long)Nq, (long)Nkv, (long)Nkv, (long)M, threads, same ? "==" : "!=");
    report(same, what, 0);
    for (int i = 0; i < 3; i++) { free(wq[i]); free(out[0][i]); free(out[1][i]); }
    free(x);
}

// exact ternary weights d * {-1, 0, +1} with zero fraction z (one +-d per 256 values so that d is that value)
static void quantize_ternary(enum ggml_type t, int64_t K, int64_t rows, void *dst, float z) {
    float *f = malloc(sizeof(float) * K * rows);
    for (int64_t i = 0; i < K * rows; i++) {
        const float sc = 0.01f * (float)(1 + (i / K) % 7) * (1 + (i / 256) % 3);
        f[i] = 0.5f * (frand() + 1.0f) < z ? 0.0f : (frand() < 0 ? -sc : sc);
        if (i % 256 == 0) f[i] = frand() < 0 ? -sc : sc;
    }
    ggml_quantize_chunk(t, f, dst, 0, rows, K, NULL);
    free(f);
}

// TQ1_0 / TQ2_0 as presence bitmap + signs (GGML_KURN_TBM=1) must give bit-identical results to the TQ2_0
// records (GGML_KURN_TBM=0; TQ1_0 converted to TQ2_0 blocks): same codes, same per-column arithmetic
static void case_tbm_exact(ggml_backend_t be, enum ggml_type t, int64_t K, int64_t N, int64_t M, int threads, float z) {
    const size_t wrow = ggml_row_size(t, K);
    void *wq = malloc(wrow * N);
    quantize_ternary(t, K, N, wq, z);
    float *x = malloc(sizeof(float) * K * M);
    for (int64_t i = 0; i < K * M; i++) x[i] = frand() * (1 + (i / 64) % 3);
    float *y[2], *ref = malloc(sizeof(float) * N * M);
    for (int mode = 0; mode < 2; mode++) {
        setenv("GGML_KURN_TBM", mode ? "1" : "0", 1);
        graph g = mm_graph(be, t, K, N, M, 1);
        ggml_backend_cpu_set_n_threads(be, threads);
        ggml_backend_tensor_set(g.w, wq, 0, ggml_nbytes(g.w));
        ggml_backend_tensor_set(g.x, x, 0, ggml_nbytes(g.x));
        ggml_backend_graph_compute(be, g.gf);
        y[mode] = malloc(sizeof(float) * N * M);
        ggml_backend_tensor_get(g.out, y[mode], 0, sizeof(float) * N * M);
        graph_free(&g);
    }
    setenv("GGML_KURN_TBM", "1", 1);
    ref_matmul(t, wq, x, K, N, M, ref);
    const double e = relerr(y[1], ref, N * M);
    const int same = memcmp(y[0], y[1], sizeof(float) * N * M) == 0;
    char what[200];
    snprintf(what, sizeof what, "tbm %s K=%ld N=%ld M=%ld T=%d z=%.2f: %s TQ2_0 records", ggml_type_name(t), (long)K, (long)N, (long)M,
             threads, z, same ? "==" : "!=");
    report(same && e < 1e-4, what, e);
    free(wq); free(x); free(y[0]); free(y[1]); free(ref);
}

static int64_t kmult(enum ggml_type t) {
    const int64_t b = ggml_blck_size(t);
    return b < 32 ? 32 : b;
}

static int list_bufts(void) {
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU));
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    typedef ggml_backend_buffer_type_t *(*fn_t)(ggml_backend_dev_t);
    fn_t fn = (fn_t)ggml_backend_reg_get_proc_address(reg, "ggml_backend_dev_get_extra_bufts");
    printf("extra buffer types:");
    for (ggml_backend_buffer_type_t *b = fn ? fn(dev) : NULL; b && *b; b++) printf(" %s", ggml_backend_buft_name(*b));
    printf("\n");
    return 0;
}

int main(int argc, char **argv) {
    if (argc > 1 && !strcmp(argv[1], "bufts")) return list_bufts();
    const int full = argc > 1 && !strcmp(argv[1], "full"), smoke = argc > 1 && !strcmp(argv[1], "smoke");
    const int bf16 = argc > 1 && !strcmp(argv[1], "bf16");
    const int native = argc > 1 && !strcmp(argv[1], "native");  // native [types]: Q6_K / Q5_K in ggml's layout, not exact mode
    const int stress = argc > 7 && !strcmp(argv[1], "stress");  // stress TYPE K N M THREADS REPS (exactness from the env)
    const int fused = argc > 1 && !strcmp(argv[1], "fused");    // exactness from the env
    const int fv = argc > 1 && !strcmp(argv[1], "freivalds");
    if (!bf16 && !stress && !native && !fused && !fv && !getenv("GGML_KURN_EXACT")) setenv("GGML_KURN_EXACT", "1", 1);
    kurn = find_kurn();
    if (!kurn) { printf("KURN buffer type not available (GGML_KURN=0 or no AVX-512 VNNI)\n"); return 1; }
    ggml_backend_t be = ggml_backend_cpu_init();
    if (stress) {
        enum ggml_type t = GGML_TYPE_COUNT;
        for (int i = 0; i < GGML_TYPE_COUNT; i++)
            if (ggml_type_name((enum ggml_type)i) && !strcmp(ggml_type_name((enum ggml_type)i), argv[2])) t = (enum ggml_type)i;
        case_stress(be, t, atoll(argv[3]), atoll(argv[4]), atoll(argv[5]), atoi(argv[6]), atoi(argv[7]));
        printf("%d passed, %d failed\n", npass, nfail);
        ggml_backend_free(be);
        return nfail;
    }
    if (fv) {
        if (!__builtin_cpu_supports("amx-bf16") || (getenv("GGML_KURN_EXACT") && atoi(getenv("GGML_KURN_EXACT")))) {
            printf("freivalds: needs the AMX-BF16 path (AMX-BF16 CPU, not exact mode), skipped\n");
            return 0;
        }
        const enum ggml_type ts[] = {GGML_TYPE_Q8_0, GGML_TYPE_Q4_K, GGML_TYPE_Q6_K, GGML_TYPE_Q5_K};
        const int64_t ms[] = {32, 37, 64, 129};
        for (size_t ti = 0; ti < sizeof ts / sizeof *ts; ti++)
            for (size_t mi = 0; mi < sizeof ms / sizeof *ms; mi++)
                case_freivalds(be, ts[ti], 2048 + 2048 * (int64_t)(mi & 1), 1024 + 64 * (int64_t)mi, ms[mi], 1 + (int)mi * 2);
        printf("freivalds: %d passed, %d failed\n", npass, nfail);
        ggml_backend_free(be);
        return nfail;
    }
    if (fused) {
        const int exact = getenv("GGML_KURN_EXACT") && atoi(getenv("GGML_KURN_EXACT"));
        const int amx = __builtin_cpu_supports("amx-bf16");
        // TQ2_0 runs as TQ2_0 records, or as bitmap + signs with GGML_KURN_TBM=1 (kurn-tbm.cpp through run())
        const enum ggml_type ts[] = {GGML_TYPE_Q8_0, GGML_TYPE_Q4_K, GGML_TYPE_Q4_0, GGML_TYPE_Q6_K, GGML_TYPE_TQ2_0};
        const int64_t ms[] = {1, 2, 3, 5, 8, 9, 16, 31, 32, 64, 129};
        for (size_t ti = 0; ti < sizeof ts / sizeof *ts; ti++) {
            const enum ggml_type t = ts[ti];
            if (argc > 2 && !strstr(argv[2], ggml_type_name(t))) continue;
            if (t == GGML_TYPE_Q6_K && !amx) continue;  // Q6_K stays with ggml without AMX-BF16
            const int nfail0 = nfail, npass0 = npass;
            for (size_t mi = 0; mi < sizeof ms / sizeof *ms; mi++) {
                // KL mode: the AMX-BF16 path (Q8_0 / Q4_K from 32 columns, Q6_K from 16) outside exact mode
                const int kl = amx && !exact && t != GGML_TYPE_Q4_0 && t != GGML_TYPE_TQ2_0 && ms[mi] >= (t == GGML_TYPE_Q6_K ? 16 : 32);
                case_fused(be, t, t == GGML_TYPE_Q4_0 || t == GGML_TYPE_TQ2_0 ? 2048 : 4096, 1536, ms[mi], 1 + (int)(mi * 3) % 8, kl);
            }
            case_fused(be, t, 2048, 6144, 1, 8, 0);
            for (size_t mi = 0; mi < sizeof ms / sizeof *ms; mi += 2)
                case_qkv(be, t, t == GGML_TYPE_Q4_0 || t == GGML_TYPE_TQ2_0 ? 2048 : 4096, 2048, 768, ms[mi], 1 + (int)(mi * 5) % 8);
            printf("%-7s %4d passed, %d failed (fused gate+up+swiglu, q+k+v%s)\n", ggml_type_name(t), npass - npass0, nfail - nfail0,
                   exact ? ", exact mode" : "");
        }
        printf("fused: %d passed, %d failed\n", npass, nfail);
        ggml_backend_free(be);
        return nfail;
    }
    if (bf16) {
        if (!__builtin_cpu_supports("amx-bf16")) { printf("bf16: no AMX-BF16 on this CPU, skipped\n"); return 0; }
        const enum ggml_type ts[] = {GGML_TYPE_Q8_0, GGML_TYPE_Q4_K, GGML_TYPE_Q6_K, GGML_TYPE_Q5_K};
        const int64_t ms[] = {16, 37, 64, 129};
        for (size_t ti = 0; ti < sizeof ts / sizeof *ts; ti++)
            for (size_t mi = 0; mi < sizeof ms / sizeof *ms; mi++)
                case_bf16(be, ts[ti], 2048, 544 + 32 * (int64_t)mi, ms[mi], 1 + (int)mi * 2);  // N > 512: no per-tensor fallback
        printf("bf16: %d passed, %d failed\n", npass, nfail);
        ggml_backend_free(be);
        return nfail;
    }
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
    if (argc > 1 && !strcmp(argv[1], "tbm")) {
        // TQ1_0 / TQ2_0 as presence bitmap + signs: bit-identical to the TQ2_0 records at every width (GEMV,
        // verify kernels, row-chunked prefill, AMX-INT8 prefill from 32 columns), dense data (two-load
        // decode), batch invariance and MUL_MAT_ID; `tbm smoke` runs a subset
        const int sm = argc > 2 && !strcmp(argv[2], "smoke");
        const enum ggml_type ts[] = {GGML_TYPE_TQ2_0, GGML_TYPE_TQ1_0};
        const float zs[] = {0.42f, 0.02f, 0.0f, 0.9f};  // 0.02: near-full words at unaligned offsets (two-load records)
        const int64_t Ms[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 31, 32, 33, 48, 64, 129};
        for (size_t ti = 0; ti < 2; ti++) {
            const enum ggml_type t = ts[ti];
            const int nfail0 = nfail, npass0 = npass;
            worst = 0;
            for (size_t zi = 0; zi < 4; zi++)
                for (size_t mi = 0; mi < sizeof Ms / sizeof *Ms; mi += sm ? 3 : 1) {
                    case_tbm_exact(be, t, 256, 17, Ms[mi], 3, zs[zi]);
                    case_tbm_exact(be, t, 2560, 1000, Ms[mi], 1 + (int)(mi + zi) % 8, zs[zi]);
                    if (!sm) case_tbm_exact(be, t, 6912, 333, Ms[mi], 8, zs[zi]);
                }
            case_tbm_exact(be, t, 32768, 48, 1, 8, 0.42f);
            case_tbm_exact(be, t, 32768, 48, 40, 8, 0.02f);
            case_tbm_exact(be, t, 1024, 1, 3, 2, 0.42f);
            setenv("GGML_KURN_TBM", "1", 1);
            case_mm(be, t, 2048, 1000, 3, 1, 8, 1);  // reused buffer
            case_mm(be, t, 1024, 64, 2, 3, 4, 0);    // broadcast over ne12
            case_invariance_all(be, t, 2048, 1000, sm ? 20 : 40);
            case_mmid(be, t, 1024, 160, 16, 4, 1, 1, 8);
            case_mmid(be, t, 1024, 160, 16, 4, 9, 0, 3);
            if (!sm) case_mmid(be, t, 2048, 512, 16, 8, 64, 1, 8);
            printf("%-7s %4d passed, %d failed (worst relerr %.1e, bitmap + signs)\n", ggml_type_name(t), npass - npass0,
                   nfail - nfail0, worst);
        }
        printf("tbm: %d passed, %d failed\n", npass, nfail);
        ggml_backend_free(be);
        return nfail;
    }
    if (native) {
        // Q6_K / Q5_K stay in ggml's block layout in the KURN buffer on AMX-BF16 CPUs: kurn-native-vfy.cpp
        // computes them below the AMX-BF16 threshold (16 columns), and at every width with GGML_KURN_EXACT=1
        const enum ggml_type natives[] = {GGML_TYPE_Q6_K, GGML_TYPE_Q5_K};
        const int exact = getenv("GGML_KURN_EXACT") && atoi(getenv("GGML_KURN_EXACT"));
        if (!__builtin_cpu_supports("amx-bf16")) {
            printf("native: needs AMX-BF16 (Q6_K / Q5_K go to ggml otherwise), skipped\n");
            return 0;
        }
        for (size_t ti = 0; ti < sizeof natives / sizeof *natives; ti++) {
            const enum ggml_type t = natives[ti];
            if (argc > 2 && !strstr(argv[2], ggml_type_name(t))) continue;
            const int nfail0 = nfail, npass0 = npass;
            worst = 0;
            const int64_t Ms[] = {1, 2, 3, 5, 8, 9, 15, exact ? 40 : 1};
            for (size_t mi = 0; mi < sizeof Ms / sizeof *Ms; mi++) {
                case_mm(be, t, 256, 32, Ms[mi], 1, 3, 0);
                case_mm(be, t, 4096, 1008, Ms[mi], 1, 1 + (int)mi % 8, 0);
            }
            case_invariance_all(be, t, 2048, 1008, exact ? 40 : 15);
            printf("%-7s %4d passed, %d failed (worst relerr %.1e, native layout%s)\n", ggml_type_name(t), npass - npass0, nfail - nfail0, worst,
                   exact ? ", exact mode" : "");
        }
        printf("native: %d passed, %d failed\n", npass, nfail);
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
            case_invariance_all(be, t, 1024, 333, 12);
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
        case_invariance_all(be, t, 2048, 1000, full ? 40 : 20);
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
