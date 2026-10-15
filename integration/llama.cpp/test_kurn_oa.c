// Outlier-aware (OA) weights in the KURN buffer: a sidecar GGUF gives weight "w" k salient input channels (idx), an
// F16 side matrix and an F32 bias; the weight itself holds the rest with those columns zeroed. Checks, per format:
//   oa      MUL_MAT(w, x) == MUL_MAT(w_plain, x with x[idx] = 0) + bias + side . x[idx]   (relerr < 1e-5)
//           w_plain is the same data without a sidecar entry, so the main product runs the same kernel
//   inv     batch invariance: every column of an M-column batch is bit-identical to that column alone (M = 2..40),
//           i.e. the OA epilogue keeps exact mode intact on the GEMV, 2-8 verify, 9-16 and AMX-INT8 paths
//   fused   ffn gate + up + SwiGLU with OA members: fused graph bit-identical to the unfused graph
//
//   test_kurn_oa [type,...]    exit code = failures. Runs with GGML_KURN_EXACT=1 unless set (bf16: GGML_KURN_EXACT=0,
//                              then inv is skipped: the AMX-BF16 prefill path is not batch invariant by design)
//   gcc -O2 -I$L/ggml/include test_kurn_oa.c -L$L/build/bin -lggml -lggml-base -lggml-cpu -lm -Wl,-rpath,$L/build/bin
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static uint64_t rng = 0x9E3779B97F4A7C15ull;
static float frand(void) {
    rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
    return (float)((rng >> 40) * (1.0 / (1ull << 23))) - 1.0f;
}

static int nfail, npass;
static ggml_backend_buffer_type_t kurn;

static ggml_backend_buffer_type_t find_kurn(void) {
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU));
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    typedef ggml_backend_buffer_type_t *(*fn_t)(ggml_backend_dev_t);
    fn_t fn = (fn_t)ggml_backend_reg_get_proc_address(reg, "ggml_backend_dev_get_extra_bufts");
    for (ggml_backend_buffer_type_t *b = fn ? fn(dev) : NULL; b && *b; b++)
        if (!strcmp(ggml_backend_buft_name(*b), "KURN")) return *b;
    return NULL;
}

static void report(int ok, const char *what, double e) {
    ok ? npass++ : nfail++;
    if (!ok || getenv("VERBOSE")) printf("%s %s (%.2e)\n", ok ? "ok  " : "FAIL", what, e);
}

#define KS 16
#define NT 8
static const int64_t K = 2048, N = 512;
static enum ggml_type types[NT];
static int ntypes;
static int32_t idx[NT][KS];
static float side[NT][2][KS * 512], bias[NT][2][512];  // [type][member][n * KS + j]
static void *wq[NT][2];

static void make_weights(int ti, enum ggml_type t) {
    const size_t wrow = ggml_row_size(t, K);
    float *w = malloc(sizeof(float) * K * N), *imat = malloc(sizeof(float) * K);
    for (int64_t i = 0; i < K; i++) imat[i] = 1.0f;
    int used[2048] = {0};
    for (int j = 0; j < KS; j++) {
        int c;
        do c = (int)((frand() * 0.5f + 0.5f) * (K - 1)); while (used[c]);
        used[c] = 1;
    }
    for (int c = 0, j = 0; c < K; c++) if (used[c]) idx[ti][j++] = c;
    for (int g = 0; g < 2; g++) {
        for (int64_t i = 0; i < K * N; i++) w[i] = 0.05f * (frand() + frand() + frand()) * (1 + (i / 32) % 5);
        for (int64_t n = 0; n < N; n++) {
            for (int j = 0; j < KS; j++) {
                side[ti][g][n * KS + j] = ggml_fp16_to_fp32(ggml_fp32_to_fp16(w[n * K + idx[ti][j]]));
                w[n * K + idx[ti][j]] = 0;
            }
            bias[ti][g][n] = 0.01f * frand();
        }
        wq[ti][g] = malloc(wrow * N);
        ggml_quantize_chunk(t, w, wq[ti][g], 0, N, K, imat);
    }
    free(w); free(imat);
}

static void write_sidecar(const char *path) {
    struct ggml_init_params ip = {64 * ggml_tensor_overhead() + (8 << 20), NULL, false};
    struct ggml_context *ctx = ggml_init(ip);
    struct gguf_context *g = gguf_init_empty();
    for (int ti = 0; ti < ntypes; ti++) {
        for (int m = 0; m < 2; m++) {
            char name[64];
            snprintf(name, sizeof name, "w%d_%d.oa_idx", ti, m);
            struct ggml_tensor *a = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, KS);
            memcpy(a->data, idx[ti], sizeof(int32_t) * KS);
            ggml_set_name(a, name);
            gguf_add_tensor(g, a);
            snprintf(name, sizeof name, "w%d_%d.oa_side", ti, m);
            struct ggml_tensor *s = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, KS, N);
            for (int64_t i = 0; i < KS * N; i++) ((ggml_fp16_t *)s->data)[i] = ggml_fp32_to_fp16(side[ti][m][i]);
            ggml_set_name(s, name);
            gguf_add_tensor(g, s);
            snprintf(name, sizeof name, "w%d_%d.oa_bias", ti, m);
            struct ggml_tensor *b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, N);
            memcpy(b->data, bias[ti][m], sizeof(float) * N);
            ggml_set_name(b, name);
            gguf_add_tensor(g, b);
        }
    }
    gguf_write_to_file(g, path, false);
    gguf_free(g);
    ggml_free(ctx);
}

// y = W x for the named weight (OA or plain), M columns
static void matmul(ggml_backend_t be, int ti, int member, int plain, const float *x, int64_t M, int threads, float *y) {
    struct ggml_init_params ip = {8 * ggml_tensor_overhead() + ggml_graph_overhead(), NULL, true};
    struct ggml_context *cw = ggml_init(ip), *ca = ggml_init(ip);
    struct ggml_tensor *w = ggml_new_tensor_2d(cw, types[ti], K, N);
    char name[64];
    snprintf(name, sizeof name, plain ? "p%d_%d" : "w%d_%d", ti, member);
    ggml_set_name(w, name);
    struct ggml_tensor *tx = ggml_new_tensor_2d(ca, GGML_TYPE_F32, K, M);
    struct ggml_tensor *out = ggml_mul_mat(ca, w, tx);
    ggml_backend_buffer_t bw = ggml_backend_alloc_ctx_tensors_from_buft(cw, kurn);
    ggml_backend_buffer_set_usage(bw, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_buffer_t ba = ggml_backend_alloc_ctx_tensors(ca, be);
    struct ggml_cgraph *gf = ggml_new_graph(ca);
    ggml_build_forward_expand(gf, out);
    ggml_backend_cpu_set_n_threads(be, threads);
    ggml_backend_tensor_set(w, wq[ti][member], 0, ggml_nbytes(w));
    ggml_backend_tensor_set(tx, x, 0, ggml_nbytes(tx));
    ggml_backend_graph_compute(be, gf);
    ggml_backend_tensor_get(out, y, 0, sizeof(float) * N * M);
    ggml_backend_buffer_free(ba); ggml_backend_buffer_free(bw); ggml_free(ca); ggml_free(cw);
}

static void make_x(int ti, float *x, int64_t M) {
    for (int64_t i = 0; i < K * M; i++) x[i] = frand() * (1 + (i / 64) % 3);
    for (int64_t m = 0; m < M; m++)
        for (int j = 0; j < KS; j++) x[m * K + idx[ti][j]] = 60.0f * frand();  // giant activation channels
}

static void case_oa(ggml_backend_t be, int ti, int64_t M, int threads) {
    float *x = malloc(sizeof(float) * K * M), *xz = malloc(sizeof(float) * K * M);
    float *y = malloc(sizeof(float) * N * M), *ym = malloc(sizeof(float) * N * M);
    make_x(ti, x, M);
    memcpy(xz, x, sizeof(float) * K * M);
    for (int64_t m = 0; m < M; m++)
        for (int j = 0; j < KS; j++) xz[m * K + idx[ti][j]] = 0;
    matmul(be, ti, 0, 0, x, M, threads, y);
    matmul(be, ti, 0, 1, xz, M, threads, ym);
    double err = 0, mag = 1e-30;
    for (int64_t m = 0; m < M; m++)
        for (int64_t n = 0; n < N; n++) {
            double r = (double)ym[m * N + n] + bias[ti][0][n];
            for (int j = 0; j < KS; j++) r += (double)side[ti][0][n * KS + j] * x[m * K + idx[ti][j]];
            err = fmax(err, fabs(r - y[m * N + n]));
            mag = fmax(mag, fabs(r));
        }
    if (getenv("DEBUG")) {
        for (int n = 0; n < 3; n++) {
            double r = bias[ti][0][n];
            for (int j = 0; j < KS; j++) r += (double)side[ti][0][n * KS + j] * x[idx[ti][j]];
            printf("  n=%d got %.5f main %.5f side+bias %.5f\n", n, y[n], ym[n], r);
        }
    }
    // q2_K / iq2_* are KURN weights only with OA data, so w_plain runs on ggml's vec_dot (int8 activations); from 16
    // columns the OA weight is on AMX-BF16 in fast mode, and the two main products differ by bf16-vs-int8 rounding
    const int generic = types[ti] == GGML_TYPE_Q2_K || types[ti] == GGML_TYPE_IQ2_XS || types[ti] == GGML_TYPE_IQ2_S;
    const int exact = atoi(getenv("GGML_KURN_EXACT")) != 0;
    const double tol = generic && !exact && M >= 16 ? 1e-2 : 1e-5;
    char what[160];
    snprintf(what, sizeof what, "oa %s K=%ld N=%ld M=%ld T=%d", ggml_type_name(types[ti]), (long)K, (long)N, (long)M, threads);
    report(err / mag < tol, what, err / mag);
    free(x); free(xz); free(y); free(ym);
}

static void case_inv(ggml_backend_t be, int ti, int maxm) {
    float *x = malloc(sizeof(float) * K * maxm), *y1 = malloc(sizeof(float) * N * maxm), *ym = malloc(sizeof(float) * N * maxm);
    make_x(ti, x, maxm);
    for (int64_t m = 0; m < maxm; m++) matmul(be, ti, 0, 0, x + m * K, 1, 1 + m % 8, y1 + m * N);
    int bad = 0, badm = 0;
    for (int M = 2; M <= maxm; M++) {
        matmul(be, ti, 0, 0, x, M, 1 + M % 8, ym);
        if (memcmp(y1, ym, sizeof(float) * N * M)) { bad++; if (!badm) badm = M; }
    }
    char what[160];
    snprintf(what, sizeof what, "batch invariance %s M=2..%d (first differing M=%d)", ggml_type_name(types[ti]), maxm, badm);
    report(bad == 0, what, bad);
    free(x); free(y1); free(ym);
}

static void case_fused(ggml_backend_t be, int ti, int64_t M, int threads) {
    float *x = malloc(sizeof(float) * K * M), *s[2];
    make_x(ti, x, M);
    for (int v = 0; v < 2; v++) {
        struct ggml_init_params ip = {16 * ggml_tensor_overhead() + ggml_graph_overhead(), NULL, true};
        struct ggml_context *cw = ggml_init(ip), *ca = ggml_init(ip);
        struct ggml_tensor *tg = ggml_new_tensor_2d(cw, types[ti], K, N), *tu = ggml_new_tensor_2d(cw, types[ti], K, N);
        char name[64];
        snprintf(name, sizeof name, "w%d_0", ti); ggml_set_name(tg, name);
        snprintf(name, sizeof name, "w%d_1", ti); ggml_set_name(tu, name);
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
        ggml_backend_tensor_set(tg, wq[ti][0], 0, ggml_nbytes(tg));
        ggml_backend_tensor_set(tu, wq[ti][1], 0, ggml_nbytes(tu));
        ggml_backend_tensor_set(tx, x, 0, ggml_nbytes(tx));
        ggml_backend_graph_compute(be, gf);
        s[v] = malloc(sizeof(float) * N * M);
        ggml_backend_tensor_get(out, s[v], 0, sizeof(float) * N * M);
        ggml_backend_buffer_free(ba); ggml_backend_buffer_free(bw); ggml_free(ca); ggml_free(cw);
    }
    char what[160];
    snprintf(what, sizeof what, "fused gate+up+swiglu %s M=%ld T=%d", ggml_type_name(types[ti]), (long)M, threads);
    report(!memcmp(s[0], s[1], sizeof(float) * N * M), what, 0);
    free(x); free(s[0]); free(s[1]);
}

int main(int argc, char **argv) {
    const char *list = argc > 1 ? argv[1] : "q8_0,q4_0,q4_K,q2_0,q6_K,q2_K,iq2_xs";
    char path[256];
    snprintf(path, sizeof path, "/tmp/test_kurn_oa_%d.oa.gguf", (int)getpid());
    setenv("GGML_KURN_EXACT", "1", 0);
    setenv("GGML_KURN_OA", path, 1);
    setenv("GGML_KURN_OA_PARTIAL", "1", 1);  // weights are created one graph at a time
    setenv("GGML_KURN_FALLBACK", "0", 1);    // w_plain (N = 512) on the KURN buffer too, same path as w
    const int exact = atoi(getenv("GGML_KURN_EXACT")) != 0;
    char buf[256];
    snprintf(buf, sizeof buf, "%s", list);
    for (char *tok = strtok(buf, ","); tok && ntypes < NT; tok = strtok(NULL, ",")) {
        for (int t = 0; t < GGML_TYPE_COUNT; t++)
            if (ggml_get_type_traits(t)->type_name && !strcmp(ggml_get_type_traits(t)->type_name, tok)) types[ntypes++] = t;
    }
    for (int ti = 0; ti < ntypes; ti++) make_weights(ti, types[ti]);
    write_sidecar(path);
    ggml_backend_load_all();
    ggml_backend_t be = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, NULL);
    kurn = find_kurn();
    if (!kurn) { printf("no KURN buffer type\n"); return 1; }
    static const int64_t Ms[] = {1, 2, 3, 5, 8, 9, 12, 16, 31, 32, 33, 64, 100};
    for (int ti = 0; ti < ntypes; ti++) {
        for (size_t i = 0; i < sizeof Ms / sizeof Ms[0]; i++) case_oa(be, ti, Ms[i], 1 + (int)(i % 8));
        if (exact) case_inv(be, ti, 40);
        case_fused(be, ti, 1, 4);
        case_fused(be, ti, 7, 8);
        case_fused(be, ti, 64, 8);
    }
    unlink(path);
    printf("oa (%s): %d passed, %d failed\n", exact ? "exact" : "fast", npass, nfail);
    return nfail;
}
