// test-backend-ops-style check of ggml's CPU MUL_MAT for Q8_0 weights, with or
// without the kurn vnni16 decode hook (set GGML_KURN_VNNI16=1 to enable it; the
// hook reads the variable once per process). Every case is compared against a
// reference built from ggml's own vec_dot on the same quantized activations.
//
//   test_mul_mat_hook [quick|full]    exit code = number of failing cases
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

typedef struct {
    int64_t K, N, M, B;   // weight [K, N(, B3)], activation [K, M, B]
    int threads;
    int view;             // 1: src0 is a strided view (every other row) -> non-contiguous
    int w3d;              // 1: 3D weight [K, N, B] (batched weights)
    int reuse;            // 1: overwrite the same weight buffer with new data and recompute
} tcase;

static int nfail, npass;

static void ref_matmul(const void *wq, size_t wrow, int64_t wstride_rows, const float *x, int64_t K, int64_t N, int64_t M,
                       float *out) {
    const struct ggml_type_traits_cpu *tt = ggml_get_type_traits_cpu(GGML_TYPE_Q8_0);
    size_t xrow = ggml_row_size(GGML_TYPE_Q8_0, K);
    void *xq = malloc(xrow * M);
    for (int64_t m = 0; m < M; m++) tt->from_float(x + m * K, (char *)xq + m * xrow, K);
    for (int64_t m = 0; m < M; m++)
        for (int64_t n = 0; n < N; n++)
            tt->vec_dot(K, &out[m * N + n], 0, (const char *)wq + n * wstride_rows * wrow, 0, (char *)xq + m * xrow, 0, 1);
    free(xq);
}

static double cmp(const float *got, const float *ref, int64_t n) {
    double err = 0, mag = 1e-30;
    for (int64_t i = 0; i < n; i++) {
        if (!isfinite(got[i])) return INFINITY;
        err = fmax(err, fabs((double)got[i] - ref[i]));
        mag = fmax(mag, fabs(ref[i]));
    }
    return err / mag;
}

static void run_case(ggml_backend_t be, tcase c) {
    const int64_t NW = c.view ? 2 * c.N : c.N;       // allocated rows
    const int64_t B3 = c.w3d ? c.B : 1;
    const size_t wrow = ggml_row_size(GGML_TYPE_Q8_0, c.K);
    float *wf = malloc(sizeof(float) * c.K * NW * B3);
    float *x = malloc(sizeof(float) * c.K * c.M * c.B);
    void *wq = malloc(wrow * NW * B3), *wq2 = malloc(wrow * NW * B3);
    for (int64_t i = 0; i < c.K * NW * B3; i++) wf[i] = 0.05f * (frand() + frand() + frand()) * (1 + (i / 32) % 5);
    ggml_quantize_chunk(GGML_TYPE_Q8_0, wf, wq, 0, NW * B3, c.K, NULL);
    for (int64_t i = 0; i < c.K * NW * B3; i++) wf[i] = 0.05f * frand();
    ggml_quantize_chunk(GGML_TYPE_Q8_0, wf, wq2, 0, NW * B3, c.K, NULL);
    for (int64_t i = 0; i < c.K * c.M * c.B; i++) x[i] = frand() * (1 + (i / 64) % 3);

    struct ggml_init_params ip = {64 * ggml_tensor_overhead() + ggml_graph_overhead(), NULL, true};
    struct ggml_context *cw = ggml_init(ip), *ca = ggml_init(ip);
    struct ggml_tensor *w = c.w3d ? ggml_new_tensor_3d(cw, GGML_TYPE_Q8_0, c.K, NW, B3) : ggml_new_tensor_2d(cw, GGML_TYPE_Q8_0, c.K, NW);
    struct ggml_tensor *a = ggml_new_tensor_3d(ca, GGML_TYPE_F32, c.K, c.M, c.B);
    struct ggml_tensor *src0 = c.view ? ggml_view_2d(ca, w, c.K, c.N, 2 * w->nb[1], 0) : w;
    struct ggml_tensor *d = ggml_mul_mat(ca, src0, a);
    ggml_backend_buffer_t bw = ggml_backend_alloc_ctx_tensors_from_buft(cw, ggml_backend_cpu_buffer_type());
    ggml_backend_buffer_set_usage(bw, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_buffer_t ba = ggml_backend_alloc_ctx_tensors(ca, be);
    ggml_backend_tensor_set(w, wq, 0, ggml_nbytes(w));
    ggml_backend_tensor_set(a, x, 0, ggml_nbytes(a));
    struct ggml_cgraph *gf = ggml_new_graph(ca);
    ggml_build_forward_expand(gf, d);
    ggml_backend_cpu_set_n_threads(be, c.threads);

    const int64_t nout = c.N * c.M * c.B;
    float *got = malloc(sizeof(float) * nout), *ref = malloc(sizeof(float) * nout);
    for (int pass = 0; pass < (c.reuse ? 2 : 1); pass++) {
        const void *wsrc = pass ? wq2 : wq;
        if (pass) ggml_backend_tensor_set(w, wq2, 0, ggml_nbytes(w)); // same buffer, new weights
        ggml_backend_graph_compute(be, gf);
        ggml_backend_tensor_get(d, got, 0, sizeof(float) * nout);
        for (int64_t b = 0; b < c.B; b++) {
            const char *wb = (const char *)wsrc + (c.w3d ? b * wrow * NW : 0);
            ref_matmul(wb, wrow, c.view ? 2 : 1, x + b * c.K * c.M, c.K, c.N, c.M, ref + b * c.N * c.M);
        }
        double e = cmp(got, ref, nout);
        int ok = e < 1e-4;
        ok ? npass++ : nfail++;
        if (!ok || getenv("VERBOSE"))
            printf("%s K=%ld N=%ld M=%ld B=%ld T=%d view=%d w3d=%d reuse=%d pass=%d relerr=%.2e\n", ok ? "ok  " : "FAIL",
                   (long)c.K, (long)c.N, (long)c.M, (long)c.B, c.threads, c.view, c.w3d, c.reuse, pass, e);
    }
    free(got); free(ref);
    ggml_backend_buffer_free(ba); ggml_backend_buffer_free(bw);
    ggml_free(ca); ggml_free(cw);
    free(wf); free(x); free(wq); free(wq2);
}

int main(int argc, char **argv) {
    const int full = argc > 1 && !strcmp(argv[1], "full");
    ggml_backend_t be = ggml_backend_cpu_init();
    const int64_t Ks[] = {32, 64, 96, 128, 160, 1024, 1152, 2048, 4096, 4160};
    const int64_t Ns[] = {1, 7, 15, 16, 17, 33, 100, 255, 1000, 4097};
    const int64_t Ms[] = {1, 2, 3, 16};
    const int Ts[] = {1, 4, 8};
    for (size_t ki = 0; ki < sizeof Ks / sizeof *Ks; ki++)
        for (size_t ni = 0; ni < sizeof Ns / sizeof *Ns; ni++)
            for (size_t mi = 0; mi < (full ? 4 : 2); mi++)
                for (size_t ti = 0; ti < (full ? 3 : 1); ti++)
                    run_case(be, (tcase){Ks[ki], Ns[ni], Ms[mi], 1, full ? Ts[ti] : 8, 0, 0, 0});
    for (int t = 1; t <= 8; t++)                  // every thread count, odd N
        run_case(be, (tcase){2048, 1037, 1, 1, t, 0, 0, 0});
    for (int64_t m = 1; m <= 16; m++)             // every batch size 1..16
        run_case(be, (tcase){1024, 513, m, 1, 8, 0, 0, 0});
    run_case(be, (tcase){32768, 4097, 1, 1, 8, 0, 0, 0});   // largest K the hook accepts
    run_case(be, (tcase){32832, 129, 1, 1, 8, 0, 0, 0});    // K above the limit -> fallback
    run_case(be, (tcase){1024, 151936, 1, 1, 8, 0, 0, 0});  // vocabulary-sized output layer
    run_case(be, (tcase){2048, 300, 1, 1, 6, 1, 0, 0});     // non-contiguous src0 (view)
    run_case(be, (tcase){1024, 64, 1, 2, 4, 0, 1, 0});      // 3D weights, batched activations
    run_case(be, (tcase){1024, 64, 1, 3, 4, 0, 0, 0});      // 2D weight broadcast over ne12=3
    run_case(be, (tcase){2048, 512, 1, 1, 8, 0, 0, 1});     // same buffer reused for new weights
    run_case(be, (tcase){2048, 512, 1, 1, 1, 0, 0, 1});
    printf("%s: %d passed, %d failed (hook %s)\n", full ? "full" : "quick", npass, nfail, getenv("GGML_KURN_VNNI16") ? "ON" : "off");
    ggml_backend_free(be);
    return nfail;
}
