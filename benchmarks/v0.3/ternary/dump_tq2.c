// Raw MUL_MAT outputs of TQ2_0 weights in the KURN buffer (GGML_KURN_TBM=0: TQ2_0 records) over shapes, widths
// 1..129 and thread counts, written to stdout as float32. Build against two llama.cpp checkouts (e.g. rows=2 and
// rows=4 + line prefetch TQ2_0 kernels) and `cmp` the outputs: a schedule change must not change a single bit.
//   gcc -O2 -I$L/ggml/include dump_tq2.c -L$L/build/bin -lggml -lggml-base -lggml-cpu -lm -Wl,-rpath,$L/build/bin -o dump
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint64_t rng = 0x9E3779B97F4A7C15ull;
static float frand(void) {
    rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
    return (float)((rng >> 40) * (1.0 / (1ull << 23))) - 1.0f;
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

int main(void) {
    setenv("GGML_KURN_TBM", "0", 1);
    ggml_backend_buffer_type_t kurn = find_kurn();
    if (!kurn) { fprintf(stderr, "no KURN buffer type\n"); return 1; }
    ggml_backend_t be = ggml_backend_cpu_init();
    const int64_t Ks[] = {256, 2560, 6912}, Ns[] = {17, 1000, 4097}, Ms[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 33, 64, 129};
    const int Ts[] = {1, 3, 8};
    long cases = 0;
    for (int ki = 0; ki < 3; ki++)
        for (int ni = 0; ni < 3; ni++) {
            const int64_t K = Ks[ki], N = Ns[ni];
            float *f = malloc(sizeof(float) * K * N);
            for (int64_t i = 0; i < K * N; i++) {  // ternary at z ~ 0.4, plus random-float rows quantized by ggml
                const float sc = 0.01f * (1 + (i / K) % 7);
                f[i] = (i / K) % 3 == 2 ? 0.05f * frand() : (0.5f * (frand() + 1) < 0.4f ? 0 : (frand() < 0 ? -sc : sc));
            }
            void *wq = malloc(ggml_row_size(GGML_TYPE_TQ2_0, K) * N);
            ggml_quantize_chunk(GGML_TYPE_TQ2_0, f, wq, 0, N, K, NULL);
            for (int mi = 0; mi < 13; mi++)
                for (int ti = 0; ti < 3; ti++) {
                    const int64_t M = Ms[mi];
                    struct ggml_init_params ip = {16 * ggml_tensor_overhead() + ggml_graph_overhead(), NULL, true};
                    struct ggml_context *cw = ggml_init(ip), *ca = ggml_init(ip);
                    struct ggml_tensor *w = ggml_new_tensor_2d(cw, GGML_TYPE_TQ2_0, K, N);
                    struct ggml_tensor *x = ggml_new_tensor_2d(ca, GGML_TYPE_F32, K, M);
                    struct ggml_tensor *out = ggml_mul_mat(ca, w, x);
                    ggml_backend_buffer_t bw = ggml_backend_alloc_ctx_tensors_from_buft(cw, kurn);
                    ggml_backend_buffer_set_usage(bw, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
                    ggml_backend_buffer_t ba = ggml_backend_alloc_ctx_tensors(ca, be);
                    struct ggml_cgraph *gf = ggml_new_graph(ca);
                    ggml_build_forward_expand(gf, out);
                    float *xs = malloc(sizeof(float) * K * M), *y = malloc(sizeof(float) * N * M);
                    for (int64_t i = 0; i < K * M; i++) xs[i] = frand();
                    ggml_backend_tensor_set(w, wq, 0, ggml_nbytes(w));
                    ggml_backend_tensor_set(x, xs, 0, ggml_nbytes(x));
                    ggml_backend_cpu_set_n_threads(be, Ts[ti]);
                    ggml_backend_graph_compute(be, gf);
                    ggml_backend_tensor_get(out, y, 0, sizeof(float) * N * M);
                    fwrite(y, sizeof(float), N * M, stdout);
                    cases++;
                    free(xs); free(y);
                    ggml_backend_buffer_free(ba); ggml_backend_buffer_free(bw); ggml_free(ca); ggml_free(cw);
                }
            free(f); free(wq);
        }
    fprintf(stderr, "%ld cases\n", cases);
    ggml_backend_free(be);
    return 0;
}
