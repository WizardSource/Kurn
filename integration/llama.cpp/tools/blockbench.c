// Per-op decode benchmark for ggml-cpu extra buffer types: one graph holding every MUL_MAT of a
// model's decoder layers (real shapes, M activation columns), weights in the chosen buffer type,
// computed by the CPU backend on a pinned threadpool. Isolates the matmul path (kernel + dispatch
// + activation quantization + partitioning) from the rest of llama.cpp.
//
//   opbench BUFT TYPE [M] [THREADS] [REPS] [LAYERS] [SHAPES]
//     BUFT    KURN | AMX | CPU_REPACK | CPU (plain ggml vec_dot)
//     TYPE    q4_0 | q4_K | q8_0 | ...
//     SHAPES  "K:N,K:N,..." (default: Qwen3-1.7B q,k,v,o,gate,up,down = 2048:2048,2048:1024,
//             2048:1024,2048:2048,2048:6144,2048:6144,6144:2048)
//   env OPBENCH_SPLIT=1   also time one graph per shape (all layers of that shape)
//   env OPBENCH_CHECK=1   compare the output of the first op with ggml vec_dot
// Prints: buft type M threads ms/graph (median, min, spread) GB/s (native bytes) per-shape ms.
//
//   gcc -O2 -I$L/ggml/include opbench.c -L$L/build/bin -lggml -lggml-base -lggml-cpu -lm -lpthread
//       -Wl,-rpath,$L/build/bin -o opbench      (one command line)
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static uint64_t rng = 88172645463325252ull;
static float frand(void) {
    rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
    return (float)((rng >> 40) * (1.0 / (1ull << 23))) - 1.0f;
}

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

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

static int cmpd(const void *a, const void *b) {
    const double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : x > y;
}

#define MAXS 16

typedef struct {
    int64_t K, N;
} shape;

// time `reps` computes of gf; returns median, writes min and (max-min)/median spread
static double time_graph(ggml_backend_t be, struct ggml_cgraph *gf, int reps, double *mn, double *spread) {
    double *t = malloc(sizeof(double) * reps);
    for (int i = 0; i < 3; i++) ggml_backend_graph_compute(be, gf);
    for (int i = 0; i < reps; i++) {
        const double t0 = now();
        ggml_backend_graph_compute(be, gf);
        t[i] = now() - t0;
    }
    qsort(t, reps, sizeof(double), cmpd);
    const double med = t[reps / 2];
    *mn = t[0];
    *spread = (t[reps - 1] - t[0]) / med;
    free(t);
    return med;
}


// blockbench TYPE M [THREADS] [REPS] [LAYERS] [ORDER]
// Qwen3-8B-shaped transformer block matmul graph on KURN weights, L layers chained:
//   a = rms(h); q,k,v = Wq a, Wk a, Wv a; o = Wo q; h2 = h + o; b = rms(h2); s = swiglu(Wg b, Wu b); h = h2 + Wd s
// (k and v are summed into a side output so they stay live). ORDER=0: q, v, k as llama.cpp builds Qwen3 with q-norm
// work after q (q's rms between); ORDER=1: q, k, v adjacent. Prints median ms per graph.
int main(int argc, char **argv) {
    enum ggml_type t = GGML_TYPE_COUNT;
    for (int i = 0; i < GGML_TYPE_COUNT; i++)
        if (ggml_get_type_traits(i)->type_name && !strcmp(ggml_get_type_traits(i)->type_name, argv[1])) t = i;
    const int64_t M = argc > 2 ? atoll(argv[2]) : 1;
    const int threads = argc > 3 ? atoi(argv[3]) : 8;
    const int reps = argc > 4 ? atoi(argv[4]) : 15;
    const int layers = argc > 5 ? atoi(argv[5]) : 8;
    const int order = argc > 6 ? atoi(argv[6]) : 0;
    const int64_t D = 4096, F = 12288, KV = 1024;
    ggml_backend_buffer_type_t bt = find_buft("KURN");
    if (!bt) { fprintf(stderr, "no KURN\n"); return 3; }
    ggml_backend_t be = ggml_backend_cpu_init();
    struct ggml_threadpool_params tpp = ggml_threadpool_params_default(threads);
    for (int i = 0; i < threads && i < GGML_MAX_N_THREADS; i++) tpp.cpumask[i] = true;
    tpp.strict_cpu = true;
    struct ggml_threadpool *tp = ggml_threadpool_new(&tpp);
    ggml_backend_cpu_set_threadpool(be, tp);
    ggml_backend_cpu_set_n_threads(be, threads);
    const int nw = 7 * layers;
    struct ggml_init_params ipw = {(size_t)(nw + 8) * ggml_tensor_overhead(), NULL, true};
    struct ggml_init_params ipa = {(size_t)(40 * layers + 64) * ggml_tensor_overhead() + ggml_graph_overhead_custom(64 * layers + 64, false), NULL, true};
    struct ggml_context *cw = ggml_init(ipw), *ca = ggml_init(ipa);
    struct ggml_tensor *W[7 * 64];
    const int64_t KN[7][2] = {{D, D}, {D, KV}, {D, KV}, {D, D}, {D, F}, {D, F}, {F, D}};
    for (int l = 0; l < layers; l++)
        for (int j = 0; j < 7; j++) W[l * 7 + j] = ggml_new_tensor_2d(cw, t, KN[j][0], KN[j][1]);
    struct ggml_tensor *h = ggml_new_tensor_2d(ca, GGML_TYPE_F32, D, M), *x0 = h;
    struct ggml_cgraph *gf = ggml_new_graph_custom(ca, 64 * layers + 64, false);
    struct ggml_tensor *side = NULL;
    for (int l = 0; l < layers; l++) {
        struct ggml_tensor **w = W + 7 * l;
        struct ggml_tensor *a = ggml_rms_norm(ca, h, 1e-6f);
        struct ggml_tensor *q, *k, *v;
        if (order == 0) {
            q = ggml_mul_mat(ca, w[0], a); q = ggml_rms_norm(ca, q, 1e-6f); ggml_build_forward_expand(gf, q);
            v = ggml_mul_mat(ca, w[2], a); ggml_build_forward_expand(gf, v);
            k = ggml_mul_mat(ca, w[1], a); k = ggml_rms_norm(ca, k, 1e-6f); ggml_build_forward_expand(gf, k);
        } else {
            q = ggml_mul_mat(ca, w[0], a); k = ggml_mul_mat(ca, w[1], a); v = ggml_mul_mat(ca, w[2], a);
            ggml_build_forward_expand(gf, q); ggml_build_forward_expand(gf, k); ggml_build_forward_expand(gf, v);
            q = ggml_rms_norm(ca, q, 1e-6f); k = ggml_rms_norm(ca, k, 1e-6f);
        }
        struct ggml_tensor *kv = ggml_add(ca, k, v);
        side = side ? ggml_add(ca, side, kv) : kv;
        struct ggml_tensor *h2 = ggml_add(ca, h, ggml_mul_mat(ca, w[3], q));
        struct ggml_tensor *b = ggml_rms_norm(ca, h2, 1e-6f);
        struct ggml_tensor *s = ggml_swiglu_split(ca, ggml_mul_mat(ca, w[4], b), ggml_mul_mat(ca, w[5], b));
        h = ggml_add(ca, h2, ggml_mul_mat(ca, w[6], s));
        h = ggml_scale(ca, h, 0.5f);
    }
    ggml_build_forward_expand(gf, h);
    ggml_build_forward_expand(gf, side);
    ggml_backend_buffer_t bw = ggml_backend_alloc_ctx_tensors_from_buft(cw, bt);
    ggml_backend_buffer_set_usage(bw, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_buffer_t ba = ggml_backend_alloc_ctx_tensors(ca, be);
    for (int j = 0; j < 7; j++) {
        const size_t wrow = ggml_row_size(t, KN[j][0]);
        void *q = malloc(wrow * KN[j][1]);
        float *f = malloc(sizeof(float) * KN[j][0] * KN[j][1]);
        for (int64_t i = 0; i < KN[j][0] * KN[j][1]; i++) f[i] = 0.02f * (frand() + frand() + frand()) * (1 + (i / 32) % 5);
        ggml_quantize_chunk(t, f, q, 0, KN[j][1], KN[j][0], NULL);
        free(f);
        for (int l = 0; l < layers; l++) ggml_backend_tensor_set(W[l * 7 + j], q, 0, wrow * KN[j][1]);
        free(q);
    }
    float *x = malloc(sizeof(float) * D * M);
    for (int64_t i = 0; i < D * M; i++) x[i] = frand();
    ggml_backend_tensor_set(x0, x, 0, sizeof(float) * D * M);
    double mn, spread;
    const double med = time_graph(be, gf, reps, &mn, &spread);
    if (getenv("BLOCKBENCH_OUT")) {
        FILE *fo = fopen(getenv("BLOCKBENCH_OUT"), "wb");
        float *y = malloc(sizeof(float) * D * M);
        ggml_backend_tensor_get(h, y, 0, sizeof(float) * D * M); fwrite(y, 4, D * M, fo);
        ggml_backend_tensor_get(side, y, 0, sizeof(float) * KV * M); fwrite(y, 4, KV * M, fo);
        fclose(fo); free(y);
    }
    printf("blockbench %s M=%lld T=%d L=%d order=%d: %.3f ms (min %.3f, spread %.1f%%)\n", argv[1], (long long)M, threads, layers, order, 1e3 * med, 1e3 * mn, 100 * spread);
    ggml_backend_buffer_free(ba); ggml_backend_buffer_free(bw);
    ggml_free(ca); ggml_free(cw); ggml_backend_free(be); ggml_threadpool_free(tp);
    return 0;
}
