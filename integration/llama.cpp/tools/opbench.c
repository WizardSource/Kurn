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
//   env OPBENCH_OA=k      outlier-aware weights: every weight gets k side channels (KURN sidecar, written to /tmp)
//   TYPE i8               R8 weights (rotated int8, kurn-r8.h): random codes and scales through a KURN sidecar
// Prints: buft type M threads ms/graph (median, min, spread) GB/s (native bytes) per-shape ms.
//
//   gcc -O2 -I$L/ggml/include opbench.c -L$L/build/bin -lggml -lggml-base -lggml-cpu -lm -lpthread
//       -Wl,-rpath,$L/build/bin -o opbench      (one command line)
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"
#include <unistd.h>
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

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s BUFT TYPE [M] [THREADS] [REPS] [LAYERS] [SHAPES]\n", argv[0]);
        return 2;
    }
    const char *bname = argv[1];
    enum ggml_type t = GGML_TYPE_COUNT;
    for (int i = 0; i < GGML_TYPE_COUNT; i++)
        if (ggml_get_type_traits(i)->type_name && !strcmp(ggml_get_type_traits(i)->type_name, argv[2])) t = i;
    if (t == GGML_TYPE_COUNT) { fprintf(stderr, "unknown type %s\n", argv[2]); return 2; }
    const int64_t M = argc > 3 ? atoll(argv[3]) : 1;
    const int threads = argc > 4 ? atoi(argv[4]) : 8;
    const int reps = argc > 5 ? atoi(argv[5]) : 21;
    const int layers = argc > 6 ? atoi(argv[6]) : 28;
    const char *sh = argc > 7 ? argv[7] : "2048:2048,2048:1024,2048:1024,2048:2048,2048:6144,2048:6144,6144:2048";
    shape S[MAXS];
    int ns = 0;
    for (const char *p = sh; *p && ns < MAXS;) {
        long long k, n;
        if (sscanf(p, "%lld:%lld", &k, &n) != 2) break;
        S[ns].K = k; S[ns].N = n; ns++;
        p = strchr(p, ',');
        if (!p) break;
        p++;
    }
    ggml_backend_buffer_type_t bt = find_buft(bname);
    if (!bt) { fprintf(stderr, "buffer type %s not available\n", bname); return 3; }

    ggml_backend_t be = ggml_backend_cpu_init();
    struct ggml_threadpool_params tpp = ggml_threadpool_params_default(threads);
    for (int i = 0; i < threads && i < GGML_MAX_N_THREADS; i++) tpp.cpumask[i] = true;
    tpp.strict_cpu = true;
    tpp.prio = GGML_SCHED_PRIO_NORMAL;
    struct ggml_threadpool *tp = ggml_threadpool_new(&tpp);
    ggml_backend_cpu_set_threadpool(be, tp);
    ggml_backend_cpu_set_n_threads(be, threads);

    const int nw = layers * ns;
    struct ggml_init_params ipw = {(size_t)(nw + 8) * ggml_tensor_overhead(), NULL, true};
    struct ggml_init_params ipa = {(size_t)(3 * nw + 64) * ggml_tensor_overhead() + 2 * ggml_graph_overhead_custom(4 * nw + 64, false), NULL, true};
    struct ggml_context *cw = ggml_init(ipw), *ca = ggml_init(ipa);
    struct ggml_tensor **w = malloc(sizeof(*w) * nw);
    struct ggml_tensor *xs[MAXS] = {0};
    for (int s = 0; s < ns; s++) {
        for (int j = 0; j < s && !xs[s]; j++) if (S[j].K == S[s].K) xs[s] = xs[j];
        if (!xs[s]) xs[s] = ggml_new_tensor_2d(ca, GGML_TYPE_F32, S[s].K, M);
    }
    struct ggml_cgraph *gf = ggml_new_graph_custom(ca, 4 * nw + 64, false);
    struct ggml_cgraph *gs[MAXS];
    struct ggml_tensor **outs = malloc(sizeof(*outs) * nw);
    for (int l = 0; l < layers; l++)
        for (int s = 0; s < ns; s++) {
            const int i = l * ns + s;
            w[i] = ggml_new_tensor_2d(cw, t, S[s].K, S[s].N);
            ggml_format_name(w[i], "w%d", i);
            outs[i] = ggml_mul_mat(ca, w[i], xs[s]);
            ggml_build_forward_expand(gf, outs[i]);
        }
    const int split = getenv("OPBENCH_SPLIT") && atoi(getenv("OPBENCH_SPLIT"));
    if (split)
        for (int s = 0; s < ns; s++) {
            gs[s] = ggml_new_graph_custom(ca, 4 * layers + 8, false);
            for (int l = 0; l < layers; l++) ggml_build_forward_expand(gs[s], outs[l * ns + s]);
        }
    const int oak = getenv("OPBENCH_OA") ? atoi(getenv("OPBENCH_OA")) : 0;
    if (t == GGML_TYPE_I8 || oak > 0) {  // KURN sidecar for every weight, before the buffer reads it
        char path[96];
        snprintf(path, sizeof path, "/tmp/opbench_%d.oa.gguf", (int)getpid());
        struct ggml_init_params ip = {(size_t)(4 * nw + 8) * ggml_tensor_overhead() + ((size_t)256 << 20), NULL, false};
        struct ggml_context *sc = ggml_init(ip);
        struct gguf_context *g = gguf_init_empty();
        for (int i = 0; i < nw; i++) {
            const shape *ss = &S[i % ns];
            char nm[64];
            if (t == GGML_TYPE_I8) {
                int64_t hb = 1;
                while (ss->K % (hb * 2) == 0 && hb < 4096) hb *= 2;
                struct ggml_tensor *a = ggml_new_tensor_1d(sc, GGML_TYPE_F32, ss->N);
                for (int64_t n = 0; n < ss->N; n++) ((float *)a->data)[n] = 1e-3f * (1.5f + frand());
                snprintf(nm, sizeof nm, "w%d.r8_scale", i); ggml_set_name(a, nm); gguf_add_tensor(g, a);
                struct ggml_tensor *r = ggml_new_tensor_1d(sc, GGML_TYPE_I32, 1);
                ((int32_t *)r->data)[0] = (int32_t)hb;
                snprintf(nm, sizeof nm, "w%d.r8_rot", i); ggml_set_name(r, nm); gguf_add_tensor(g, r);
            }
            if (oak > 0) {
                struct ggml_tensor *a = ggml_new_tensor_1d(sc, GGML_TYPE_I32, oak);
                for (int j = 0; j < oak; j++) ((int32_t *)a->data)[j] = (int32_t)(j * (ss->K / oak));
                snprintf(nm, sizeof nm, "w%d.oa_idx", i); ggml_set_name(a, nm); gguf_add_tensor(g, a);
                struct ggml_tensor *sd = ggml_new_tensor_2d(sc, GGML_TYPE_F16, oak, ss->N);
                for (int64_t j = 0; j < oak * ss->N; j++) ((ggml_fp16_t *)sd->data)[j] = ggml_fp32_to_fp16(0.01f * frand());
                snprintf(nm, sizeof nm, "w%d.oa_side", i); ggml_set_name(sd, nm); gguf_add_tensor(g, sd);
                struct ggml_tensor *b = ggml_new_tensor_1d(sc, GGML_TYPE_F32, ss->N);
                for (int64_t n = 0; n < ss->N; n++) ((float *)b->data)[n] = 0.01f * frand();
                snprintf(nm, sizeof nm, "w%d.oa_bias", i); ggml_set_name(b, nm); gguf_add_tensor(g, b);
            }
        }
        gguf_write_to_file(g, path, false);
        gguf_free(g);
        ggml_free(sc);
        setenv("GGML_KURN_OA", path, 1);
    }
    ggml_backend_buffer_t bw = ggml_backend_alloc_ctx_tensors_from_buft(cw, bt);
    if (!bw) { fprintf(stderr, "weight alloc failed\n"); return 3; }
    ggml_backend_buffer_set_usage(bw, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_buffer_t ba = ggml_backend_alloc_ctx_tensors(ca, be);

    double bytes = 0;
    const char *zenv = getenv("OPBENCH_ZERO");
    const int ternary = zenv && (t == GGML_TYPE_TQ1_0 || t == GGML_TYPE_TQ2_0);
    for (int s = 0; s < ns; s++) {
        const size_t wrow = ggml_row_size(t, S[s].K);
        void *q = malloc(wrow * S[s].N);
        if (t == GGML_TYPE_I8) {
            for (int64_t i = 0; i < S[s].K * S[s].N; i++) ((int8_t *)q)[i] = (int8_t)(127 * frand());
        } else {
            float *f = malloc(sizeof(float) * S[s].K * S[s].N);
            for (int64_t i = 0; i < S[s].K * S[s].N; i++) f[i] = 0.05f * (frand() + frand() + frand()) * (1 + (i / 32) % 5);
            if (ternary) {  // exact ternary weights, zero fraction OPBENCH_ZERO: d * {-1, 0, +1}, one +-d per 256 values
                const float z = (float)atof(zenv);
                for (int64_t i = 0; i < S[s].K * S[s].N; i++) {
                    const float u = 0.5f * (frand() + 1.0f), sc = 0.05f * (float)(1 + (i / S[s].K) % 5);
                    f[i] = u < z ? 0.0f : (frand() < 0 ? -sc : sc);
                    if (i % 256 == 0) f[i] = frand() < 0 ? -sc : sc;
                }
            }
            ggml_quantize_chunk(t, f, q, 0, S[s].N, S[s].K, NULL);
            free(f);
        }
        for (int l = 0; l < layers; l++) ggml_backend_tensor_set(w[l * ns + s], q, 0, wrow * S[s].N);
        bytes += (double)layers * wrow * S[s].N;
        if (s == 0 && t != GGML_TYPE_I8 && getenv("OPBENCH_CHECK") && atoi(getenv("OPBENCH_CHECK"))) {
            float *x = malloc(sizeof(float) * S[0].K * M);
            for (int64_t i = 0; i < S[0].K * M; i++) x[i] = frand();
            ggml_backend_tensor_set(xs[0], x, 0, sizeof(float) * S[0].K * M);
            ggml_backend_graph_compute(be, gf);
            float *got = malloc(sizeof(float) * S[0].N * M);
            ggml_backend_tensor_get(outs[0], got, 0, sizeof(float) * S[0].N * M);
            const struct ggml_type_traits_cpu *tt = ggml_get_type_traits_cpu(t);
            const size_t xrow = ggml_row_size(tt->vec_dot_type, S[0].K);
            void *xq = malloc(xrow);
            double err = 0, mag = 1e-30;
            for (int64_t m = 0; m < M; m++) {
                ggml_get_type_traits_cpu(tt->vec_dot_type)->from_float(x + m * S[0].K, xq, S[0].K);
                for (int64_t n = 0; n < S[0].N; n++) {
                    float r;
                    tt->vec_dot(S[0].K, &r, 0, (const char *)q + n * wrow, 0, xq, 0, 1);
                    err = fmax(err, fabs((double)got[m * S[0].N + n] - r));
                    mag = fmax(mag, fabs(r));
                }
            }
            printf("check %s %s K=%lld N=%lld M=%lld relerr=%.2e\n", bname, argv[2], (long long)S[0].K, (long long)S[0].N,
                   (long long)M, err / mag);
            free(x); free(got); free(xq);
        }
        free(q);
    }
    for (int s = 0; s < ns; s++) {
        float *x = malloc(sizeof(float) * S[s].K * M);
        for (int64_t i = 0; i < S[s].K * M; i++) x[i] = frand();
        ggml_backend_tensor_set(xs[s], x, 0, sizeof(float) * S[s].K * M);
        free(x);
    }
    double mn, spread;
    const double med = time_graph(be, gf, reps, &mn, &spread);
    printf("%s %s M=%lld T=%d L=%d ops=%d: %.3f ms (min %.3f, spread %.1f%%) %.1f GB/s  %.2f us/op\n", bname, argv[2],
           (long long)M, threads, layers, nw, 1e3 * med, 1e3 * mn, 100 * spread, bytes / med / 1e9, 1e6 * med / nw);
    if (split)
        for (int s = 0; s < ns; s++) {
            double m2, s2;
            const double md = time_graph(be, gs[s], reps, &m2, &s2);
            const double b = (double)layers * ggml_row_size(t, S[s].K) * S[s].N;
            printf("  shape K=%lld N=%lld: %.3f ms (%.2f us/op, spread %.1f%%) %.1f GB/s\n", (long long)S[s].K,
                   (long long)S[s].N, 1e3 * md, 1e6 * md / layers, 100 * s2, b / md / 1e9);
        }
    if (getenv("GGML_KURN_OA") && strstr(getenv("GGML_KURN_OA"), "/tmp/opbench_")) unlink(getenv("GGML_KURN_OA"));
    ggml_backend_buffer_free(ba); ggml_backend_buffer_free(bw);
    ggml_free(ca); ggml_free(cw);
    ggml_backend_free(be);
    ggml_threadpool_free(tp);
    return 0;
}
