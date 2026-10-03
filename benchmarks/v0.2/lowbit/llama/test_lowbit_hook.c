// ggml CPU MUL_MAT with the kurn lowbit hook (GGML_KURN_LOWBIT=1, variant via GGML_KURN_<FMT>,
// GGML_KURN_LUT_SHARED) against a reference from ggml's own vec_dot, for Q1_0, Q2_0, TQ2_0, TQ1_0
// and Q2_K weights. The kernels are exact, so the tolerance only covers float summation order.
//   test_lowbit_hook        exit code = number of failing cases
// Build against the patched llama.cpp copy (see build_llama_lowbit.sh):
//   gcc -O2 -I$L/ggml/include test_lowbit_hook.c -L$L/build/bin -lggml -lggml-base -lggml-cpu -lm -Wl,-rpath,$L/build/bin
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

static void run_case(ggml_backend_t be, enum ggml_type wt, int64_t K, int64_t N, int64_t M, int threads, int reuse) {
    const struct ggml_type_traits_cpu *tt = ggml_get_type_traits_cpu(wt);
    const enum ggml_type xt = tt->vec_dot_type;
    const size_t wrow = ggml_row_size(wt, K), xrow = ggml_row_size(xt, K);
    float *wf = malloc(sizeof(float) * K * N), *x = malloc(sizeof(float) * K * M);
    void *wq = malloc(wrow * N), *wq2 = malloc(wrow * N), *xq = malloc(xrow * M);
    for (int64_t i = 0; i < K * N; i++) wf[i] = 0.05f * (frand() + frand() + frand()) * (1 + (i / 32) % 5);
    ggml_quantize_chunk(wt, wf, wq, 0, N, K, NULL);
    for (int64_t i = 0; i < K * N; i++) wf[i] = 0.05f * frand();
    ggml_quantize_chunk(wt, wf, wq2, 0, N, K, NULL);
    for (int64_t i = 0; i < K * M; i++) x[i] = frand() * (1 + (i / 64) % 3);

    struct ggml_init_params ip = {16 * ggml_tensor_overhead() + ggml_graph_overhead(), NULL, true};
    struct ggml_context *cw = ggml_init(ip), *ca = ggml_init(ip);
    struct ggml_tensor *w = ggml_new_tensor_2d(cw, wt, K, N);
    struct ggml_tensor *a = ggml_new_tensor_2d(ca, GGML_TYPE_F32, K, M);
    struct ggml_tensor *d = ggml_mul_mat(ca, w, a);
    ggml_backend_buffer_t bw = ggml_backend_alloc_ctx_tensors_from_buft(cw, ggml_backend_cpu_buffer_type());
    ggml_backend_buffer_set_usage(bw, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_buffer_t ba = ggml_backend_alloc_ctx_tensors(ca, be);
    ggml_backend_tensor_set(w, wq, 0, ggml_nbytes(w));
    ggml_backend_tensor_set(a, x, 0, ggml_nbytes(a));
    struct ggml_cgraph *gf = ggml_new_graph(ca);
    ggml_build_forward_expand(gf, d);
    ggml_backend_cpu_set_n_threads(be, threads);
    float *got = malloc(sizeof(float) * N * M), *ref = malloc(sizeof(float) * N * M);
    for (int64_t m = 0; m < M; m++) ggml_get_type_traits_cpu(xt)->from_float(x + m * K, (char *)xq + m * xrow, K);
    for (int pass = 0; pass < (reuse ? 2 : 1); pass++) {
        const char *ws = pass ? wq2 : wq;
        if (pass) ggml_backend_tensor_set(w, wq2, 0, ggml_nbytes(w));
        ggml_backend_graph_compute(be, gf);
        ggml_backend_tensor_get(d, got, 0, sizeof(float) * N * M);
        for (int64_t m = 0; m < M; m++)
            for (int64_t n = 0; n < N; n++) tt->vec_dot(K, &ref[m * N + n], 0, ws + n * wrow, 0, (char *)xq + m * xrow, 0, 1);
        double err = 0, mag = 1e-30;
        for (int64_t i = 0; i < N * M; i++) {
            if (!isfinite(got[i])) { err = INFINITY; break; }
            err = fmax(err, fabs((double)got[i] - ref[i]));
            mag = fmax(mag, fabs(ref[i]));
        }
        const int ok = err / mag < 1e-4;
        ok ? npass++ : nfail++;
        if (!ok || getenv("VERBOSE"))
            printf("%s %s K=%ld N=%ld M=%ld T=%d reuse=%d pass=%d relerr=%.2e\n", ok ? "ok  " : "FAIL", ggml_type_name(wt), (long)K,
                   (long)N, (long)M, threads, reuse, pass, err / mag);
    }
    free(got); free(ref);
    ggml_backend_buffer_free(ba); ggml_backend_buffer_free(bw);
    ggml_free(ca); ggml_free(cw);
    free(wf); free(x); free(wq); free(wq2); free(xq);
}

int main(void) {
    ggml_backend_t be = ggml_backend_cpu_init();
    const enum ggml_type types[] = {GGML_TYPE_Q1_0, GGML_TYPE_Q2_0, GGML_TYPE_TQ2_0, GGML_TYPE_TQ1_0, GGML_TYPE_Q2_K};
    const int64_t Ks[] = {256, 768, 2560, 6912};
    const int64_t Ns[] = {1, 15, 33, 100, 1037, 4097};
    const int Ts[] = {1, 3, 8};
    for (size_t t = 0; t < sizeof types / sizeof *types; t++) {
        for (size_t ki = 0; ki < sizeof Ks / sizeof *Ks; ki++)
            for (size_t ni = 0; ni < sizeof Ns / sizeof *Ns; ni++)
                for (size_t ti = 0; ti < sizeof Ts / sizeof *Ts; ti++) run_case(be, types[t], Ks[ki], Ns[ni], 1, Ts[ti], 0);
        for (int th = 1; th <= 8; th++) run_case(be, types[t], 2048, 1037, 1, th, 0);  // every thread count
        run_case(be, types[t], 2048, 300, 2, 8, 0);   // M = 2: falls back to ggml
        run_case(be, types[t], 32768, 257, 1, 8, 0);  // largest K the hook accepts
        run_case(be, types[t], 2048, 512, 1, 8, 1);   // same buffer reused for new weights
    }
    printf("%d passed, %d failed (hook %s)\n", npass, nfail, getenv("GGML_KURN_LOWBIT") ? "ON" : "off");
    ggml_backend_free(be);
    return nfail;
}
