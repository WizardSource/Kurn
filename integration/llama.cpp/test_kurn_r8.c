// R8 weights (rotated int8, kurn-r8.h) in the KURN buffer:
//   exact    MUL_MAT output == ((float) sum_k Q[n][k] q[m][k] * sw[n]) * sx[m] bit for bit, with q / sx from
//            kurn_r8_prep_row and the integer sum in int64, for M = 1..100 (VNNI GEMV, verify widths, AMX-INT8 tiles)
//            and 1..8 threads, in fast and exact mode alike
//   inv      every column of an M-column batch is bit-identical to that column alone (M = 2..40)
//   prep     kurn_r8_prep_row against a double-precision Walsh-Hadamard + per-token int8 (codes within 1, scale 1e-6)
//   float    end-to-end error against the unquantized W x (sanity: int8 x int8 after rotation, ~1e-2)
//   outlier  with an OA sidecar entry on the same weight: R8 main product on x with x[idx] = 0, plus side + bias
//   fused    gate + up + SwiGLU on two R8 weights (one shared rotation / quantization): bit-identical to unfused
//   test_kurn_r8                 exit code = failures
//   gcc -O2 -I$L/ggml/include test_kurn_r8.c -L$L/build/bin -lggml -lggml-base -lggml-cpu -lm -Wl,-rpath,$L/build/bin
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

void kurn_r8_prep_row(const float *x, int64_t K, int64_t hb, const int32_t *zero, int64_t nz, int8_t *q, uint8_t *qu, float *sx);

static uint64_t rng = 0x2545F4914F6CDD1Dull;
static float frand(void) {
    rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
    return (float)((rng >> 40) * (1.0 / (1ull << 23))) - 1.0f;
}
static int nfail, npass;
static ggml_backend_buffer_type_t kurn;
static void report(int ok, const char *what, double e) {
    ok ? npass++ : nfail++;
    if (!ok || getenv("VERBOSE")) printf("%s %s (%.3g)\n", ok ? "ok  " : "FAIL", what, e);
}
static ggml_backend_buffer_type_t find_kurn(void) {
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU));
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    typedef ggml_backend_buffer_type_t *(*fn_t)(ggml_backend_dev_t);
    fn_t fn = (fn_t)ggml_backend_reg_get_proc_address(reg, "ggml_backend_dev_get_extra_bufts");
    for (ggml_backend_buffer_type_t *b = fn ? fn(dev) : NULL; b && *b; b++)
        if (!strcmp(ggml_backend_buft_name(*b), "KURN")) return *b;
    return NULL;
}

#define NW 5
static const int64_t Ks[NW] = {2048, 6144, 4096, 2048, 2048}, Ns[NW] = {512, 256, 1024, 512, 512},
                     HBs[NW] = {2048, 2048, 256, 2048, 2048};
static int8_t *Q[NW];
static float *Wf[NW], *sw[NW];
#define KS 16
static int32_t oidx[KS];
static float oside[KS * 1024], obias[1024];

static void fwht_d(double *x, int64_t K, int64_t hb) {
    for (int64_t b0 = 0; b0 < K; b0 += hb)
        for (int64_t h = 1; h < hb; h *= 2)
            for (int64_t i = b0; i < b0 + hb; i += 2 * h)
                for (int64_t j = i; j < i + h; j++) { double a = x[j], b = x[j + h]; x[j] = a + b; x[j + h] = a - b; }
    for (int64_t k = 0; k < K; k++) x[k] /= sqrt((double)hb);
}

static void make(int w) {
    const int64_t K = Ks[w], N = Ns[w];
    Wf[w] = malloc(sizeof(float) * K * N); Q[w] = malloc(K * N); sw[w] = malloc(sizeof(float) * N);
    double *r = malloc(sizeof(double) * K);
    for (int64_t n = 0; n < N; n++) {
        for (int64_t k = 0; k < K; k++) r[k] = Wf[w][n * K + k] = 0.03f * (frand() + frand()) * (1 + (k / 64) % 4);
        fwht_d(r, K, HBs[w]);
        double amax = 0;
        for (int64_t k = 0; k < K; k++) amax = fmax(amax, fabs(r[k]));
        sw[w][n] = (float)(amax / 127);
        for (int64_t k = 0; k < K; k++) Q[w][n * K + k] = (int8_t)lrint(r[k] / sw[w][n]);
    }
    free(r);
}

static void write_sidecar(const char *path) {
    struct ggml_init_params ip = {64 * ggml_tensor_overhead() + (4 << 20), NULL, false};
    struct ggml_context *ctx = ggml_init(ip);
    struct gguf_context *g = gguf_init_empty();
    char name[64];
    for (int w = 0; w < NW; w++) {
        snprintf(name, sizeof name, "w%d.r8_scale", w);
        struct ggml_tensor *s = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, Ns[w]);
        memcpy(s->data, sw[w], sizeof(float) * Ns[w]); ggml_set_name(s, name); gguf_add_tensor(g, s);
        snprintf(name, sizeof name, "w%d.r8_rot", w);
        struct ggml_tensor *r = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
        ((int32_t *)r->data)[0] = (int32_t)HBs[w]; ggml_set_name(r, name); gguf_add_tensor(g, r);
    }
    // outlier-aware entry on w0 as well
    for (int j = 0; j < KS; j++) oidx[j] = 37 + 101 * j;
    for (int64_t i = 0; i < KS * Ns[0]; i++) oside[i] = 0.05f * frand();
    for (int64_t n = 0; n < Ns[0]; n++) obias[n] = 0.01f * frand();
    struct ggml_tensor *a = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, KS);
    memcpy(a->data, oidx, sizeof oidx); ggml_set_name(a, "w0.oa_idx"); gguf_add_tensor(g, a);
    struct ggml_tensor *s = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, KS, Ns[0]);
    for (int64_t i = 0; i < KS * Ns[0]; i++) ((ggml_fp16_t *)s->data)[i] = ggml_fp32_to_fp16(oside[i]);
    for (int64_t i = 0; i < KS * Ns[0]; i++) oside[i] = ggml_fp16_to_fp32(((ggml_fp16_t *)s->data)[i]);
    ggml_set_name(s, "w0.oa_side"); gguf_add_tensor(g, s);
    struct ggml_tensor *b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, Ns[0]);
    memcpy(b->data, obias, sizeof(float) * Ns[0]); ggml_set_name(b, "w0.oa_bias"); gguf_add_tensor(g, b);
    gguf_write_to_file(g, path, false);
    gguf_free(g); ggml_free(ctx);
}

static void matmul(ggml_backend_t be, int w, const float *x, int64_t M, int threads, float *y) {
    struct ggml_init_params ip = {8 * ggml_tensor_overhead() + ggml_graph_overhead(), NULL, true};
    struct ggml_context *cw = ggml_init(ip), *ca = ggml_init(ip);
    struct ggml_tensor *t = ggml_new_tensor_2d(cw, GGML_TYPE_I8, Ks[w], Ns[w]);
    char name[16];
    snprintf(name, sizeof name, "w%d", w);
    ggml_set_name(t, name);
    struct ggml_tensor *tx = ggml_new_tensor_2d(ca, GGML_TYPE_F32, Ks[w], M);
    struct ggml_tensor *out = ggml_mul_mat(ca, t, tx);
    ggml_backend_buffer_t bw = ggml_backend_alloc_ctx_tensors_from_buft(cw, kurn);
    ggml_backend_buffer_set_usage(bw, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_buffer_t ba = ggml_backend_alloc_ctx_tensors(ca, be);
    struct ggml_cgraph *gf = ggml_new_graph(ca);
    ggml_build_forward_expand(gf, out);
    ggml_backend_cpu_set_n_threads(be, threads);
    ggml_backend_tensor_set(t, Q[w], 0, ggml_nbytes(t));
    ggml_backend_tensor_set(tx, x, 0, ggml_nbytes(tx));
    ggml_backend_graph_compute(be, gf);
    ggml_backend_tensor_get(out, y, 0, sizeof(float) * Ns[w] * M);
    ggml_backend_buffer_free(ba); ggml_backend_buffer_free(bw); ggml_free(ca); ggml_free(cw);
}

static void make_x(int64_t K, int64_t M, float *x) {
    for (int64_t i = 0; i < K * M; i++) x[i] = frand() * (1 + (i % K) / 512);
    for (int64_t m = 0; m < M; m++) x[m * K + 7] = 80.0f * frand();  // one giant channel
}

static void case_exact(ggml_backend_t be, int w, int64_t M, int threads) {
    const int64_t K = Ks[w], N = Ns[w];
    float *x = malloc(sizeof(float) * K * M), *y = malloc(sizeof(float) * N * M), *sx = malloc(sizeof(float) * M);
    int8_t *q = malloc(K * M);
    make_x(K, M, x);
    matmul(be, w, x, M, threads, y);
    const int oa = w == 0;
    long bad = 0;
    double fe = 0, fm = 1e-30;
    for (int64_t m = 0; m < M; m++) {
        float *xz = malloc(sizeof(float) * K);
        memcpy(xz, x + m * K, sizeof(float) * K);
        if (oa) for (int j = 0; j < KS; j++) xz[oidx[j]] = 0;
        kurn_r8_prep_row(xz, K, HBs[w], NULL, 0, q + m * K, NULL, sx + m);
        for (int64_t n = 0; n < N; n++) {
            int64_t acc = 0;
            for (int64_t k = 0; k < K; k++) acc += (int64_t)Q[w][n * K + k] * q[m * K + k];
            float r = ((float)acc * sw[w][n]) * sx[m];
            if (oa) {  // the OA epilogue's arithmetic order: bias, then fp32 FMA over j, added to the main result
                float s = obias[n];
                for (int j = 0; j < KS; j++) s = fmaf(oside[n * KS + j], x[m * K + oidx[j]], s);
                r = r + s;
            }
            if (memcmp(&r, &y[m * N + n], 4)) bad++;
            double ref = 0;
            for (int64_t k = 0; k < K; k++) ref += (double)Wf[w][n * K + k] * xz[k];
            if (oa) for (int j = 0; j < KS; j++) ref += (double)oside[n * KS + j] * x[m * K + oidx[j]];
            if (oa) ref += obias[n];
            fe = fmax(fe, fabs(ref - y[m * N + n])); fm = fmax(fm, fabs(ref));
        }
        free(xz);
    }
    char what[160];
    snprintf(what, sizeof what, "exact K=%ld N=%ld hb=%ld M=%ld T=%d%s: %ld of %ld outputs differ from the integer reference",
             (long)K, (long)N, (long)HBs[w], (long)M, threads, oa ? " +OA" : "", bad, (long)(N * M));
    report(bad == 0, what, (double)bad);
    snprintf(what, sizeof what, "float K=%ld N=%ld M=%ld vs unquantized W x", (long)K, (long)N, (long)M);
    report(fe / fm < 3e-2, what, fe / fm);
    free(x); free(y); free(sx); free(q);
}

static void case_inv(ggml_backend_t be, int w, int maxm) {
    const int64_t K = Ks[w], N = Ns[w];
    float *x = malloc(sizeof(float) * K * maxm), *y1 = malloc(sizeof(float) * N * maxm), *ym = malloc(sizeof(float) * N * maxm);
    make_x(K, maxm, x);
    for (int64_t m = 0; m < maxm; m++) matmul(be, w, x + m * K, 1, 1 + m % 8, y1 + m * N);
    int bad = 0, badm = 0;
    for (int M = 2; M <= maxm; M++) {
        matmul(be, w, x, M, 1 + M % 8, ym);
        if (memcmp(y1, ym, sizeof(float) * N * M)) { bad++; if (!badm) badm = M; }
    }
    char what[160];
    snprintf(what, sizeof what, "batch invariance K=%ld N=%ld M=2..%d (first differing M=%d)", (long)K, (long)N, maxm, badm);
    report(bad == 0, what, bad);
    free(x); free(y1); free(ym);
}

static void case_prep(int64_t K, int64_t hb) {
    float *x = malloc(sizeof(float) * K), sx;
    double *r = malloc(sizeof(double) * K);
    int8_t *q = malloc(K);
    for (int64_t k = 0; k < K; k++) r[k] = x[k] = frand() * (k == 3 ? 50 : 1);
    kurn_r8_prep_row(x, K, hb, NULL, 0, q, NULL, &sx);
    fwht_d(r, K, hb);
    double amax = 0;
    for (int64_t k = 0; k < K; k++) amax = fmax(amax, fabs(r[k]));
    int worst = 0;
    for (int64_t k = 0; k < K; k++) { int d = abs(q[k] - (int)lrint(r[k] / (amax / 127))); if (d > worst) worst = d; }
    char what[128];
    snprintf(what, sizeof what, "prep K=%ld hb=%ld: codes within %d, scale relerr", (long)K, (long)hb, worst);
    report(worst <= 1 && fabs(sx - amax / 127) / (amax / 127) < 1e-5, what, fabs(sx - amax / 127) / (amax / 127));
    free(x); free(r); free(q);
}

static void case_fused(ggml_backend_t be, int64_t M, int threads) {
    const int64_t K = Ks[3], N = Ns[3];
    float *x = malloc(sizeof(float) * K * M), *s[2];
    make_x(K, M, x);
    for (int v = 0; v < 2; v++) {
        struct ggml_init_params ip = {16 * ggml_tensor_overhead() + ggml_graph_overhead(), NULL, true};
        struct ggml_context *cw = ggml_init(ip), *ca = ggml_init(ip);
        struct ggml_tensor *tg = ggml_new_tensor_2d(cw, GGML_TYPE_I8, K, N), *tu = ggml_new_tensor_2d(cw, GGML_TYPE_I8, K, N);
        ggml_set_name(tg, "w3"); ggml_set_name(tu, "w4");
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
        ggml_backend_tensor_set(tg, Q[3], 0, ggml_nbytes(tg));
        ggml_backend_tensor_set(tu, Q[4], 0, ggml_nbytes(tu));
        ggml_backend_tensor_set(tx, x, 0, ggml_nbytes(tx));
        ggml_backend_graph_compute(be, gf);
        s[v] = malloc(sizeof(float) * N * M);
        ggml_backend_tensor_get(out, s[v], 0, sizeof(float) * N * M);
        ggml_backend_buffer_free(ba); ggml_backend_buffer_free(bw); ggml_free(ca); ggml_free(cw);
    }
    char what[96];
    snprintf(what, sizeof what, "fused gate+up+swiglu R8 M=%ld T=%d", (long)M, threads);
    report(!memcmp(s[0], s[1], sizeof(float) * N * M), what, 0);
    free(x); free(s[0]); free(s[1]);
}

int main(void) {
    char path[128];
    snprintf(path, sizeof path, "/tmp/test_kurn_r8_%d.oa.gguf", (int)getpid());
    setenv("GGML_KURN_OA", path, 1);
    setenv("GGML_KURN_OA_PARTIAL", "1", 1);
    for (int w = 0; w < NW; w++) make(w);
    write_sidecar(path);
    ggml_backend_load_all();
    ggml_backend_t be = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, NULL);
    kurn = find_kurn();
    if (!kurn) { printf("no KURN buffer type\n"); return 1; }
    case_prep(2048, 2048); case_prep(6144, 2048); case_prep(4096, 16); case_prep(12288, 4096);
    static const int64_t Ms[] = {1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64, 100};
    for (int w = 0; w < NW; w++) {
        for (size_t i = 0; i < sizeof Ms / sizeof Ms[0]; i++) case_exact(be, w, Ms[i], 1 + (int)(i % 8));
        case_inv(be, w, 40);
    }
    case_fused(be, 1, 4); case_fused(be, 5, 8); case_fused(be, 16, 3); case_fused(be, 64, 8); case_fused(be, 100, 8);
    unlink(path);
    printf("r8: %d passed, %d failed\n", npass, nfail);
    return nfail;
}
