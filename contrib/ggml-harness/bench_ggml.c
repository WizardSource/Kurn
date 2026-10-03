// Kernel benchmark harness. One thread pool, pinning, timing and correctness
// check for every implementation; implementations are dlopen'ed shared
// libraries exporting the ABI in kurn.h, or ggml built-ins.
//
//   bench --impl <lib.so|ggml|ggml-graph-cpu|ggml-graph-amx|ggml-graph-repack>
//         --kernel q8gemv|q4kgemv|q8gemm --regime hot|cold --threads T
//         [--K 4096] [--N n] [--M 128] [--secs 2] [--csv out.csv] [--label name]
//         [--wait spin|sleep|<spins>]   barrier wait: spin forever (ggml-like) or spin then futex-sleep
//   bench --bw [--threads T]                 memory read bandwidth (roofline)
//   bench --dump DIR --kernel ... --regime   write W/x/yref for out-of-process runners
#define _GNU_SOURCE
#include "kurn.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include <dlfcn.h>
#include <immintrin.h>
#include <math.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#include <linux/futex.h>
#include <sys/syscall.h>
#include <limits.h>

#define PROXY_W_PER_CORE 5.47 // Project convention: 350 W / 64-core server, per busy core-second

typedef void (*gemv_fn)(const void *, const void *, float *, int64_t, int64_t, int64_t);
typedef void (*gemm_fn)(const void *, const void *, float *, int64_t, int64_t, int64_t, int64_t, int64_t);
typedef void *(*prep_fn)(const void *, int64_t, int64_t);
typedef void (*tinit_fn)(void);

enum { K_Q8GEMV, K_Q4KGEMV, K_Q8GEMM, K_GEMV };
// decode GEMV kernels by weight type (activation type = ggml's vec_dot_type for it)
static const struct { const char *arg, *name, *entry; enum ggml_type wt; } GEMVS[] = {
    {"q8gemv", "q8_0_gemv", "kq8_gemv", GGML_TYPE_Q8_0},     {"q4kgemv", "q4_K_gemv", "kq4k_gemv", GGML_TYPE_Q4_K},
    {"q40gemv", "q4_0_gemv", "kq40_gemv", GGML_TYPE_Q4_0},   {"iq4nlgemv", "iq4_nl_gemv", "kiq4nl_gemv", GGML_TYPE_IQ4_NL},
    {"q20gemv", "q2_0_gemv", "kq20_gemv", GGML_TYPE_Q2_0},   {"tq20gemv", "tq2_0_gemv", "ktq20_gemv", GGML_TYPE_TQ2_0},
    {"q10gemv", "q1_0_gemv", "kq10_gemv", GGML_TYPE_Q1_0},
};
static int gemv_idx = 0;
#define WTYPE() (G.kernel == K_Q8GEMM ? GGML_TYPE_Q8_0 : GEMVS[gemv_idx].wt)
#define XTYPE() (ggml_get_type_traits_cpu(WTYPE())->vec_dot_type)

void report(const char *impl, long calls, double wall, double cpu, double rel, double drift, FILE *csv);

static struct {
    int kernel, threads, cold, M;
    int64_t K, N, count;
    double secs;
    const char *impl, *csv, *label, *dump;
    size_t row_bytes, mat_bytes, x_row_bytes;
    uint8_t *W;   // count matrices, contiguous
    uint8_t *X;   // quantized activation(s): 1 row (gemv) or M rows (gemm)
    float *Xf;    // float activations (for ggml graph paths)
    float *Y, *Yref;
    void **packed;
    gemv_fn gemv;
    gemm_fn gemm, gemm_packed;
    gemv_fn gemv_packed;
    tinit_fn tinit;
    int builtin_ggml;
} G;

static double now(clockid_t c) {
    struct timespec ts;
    clock_gettime(c, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}
static double cpu_time(void) {
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    return ru.ru_utime.tv_sec + ru.ru_utime.tv_usec * 1e-6 + ru.ru_stime.tv_sec + ru.ru_stime.tv_usec * 1e-6;
}

static uint64_t rng = 0x9E3779B97F4A7C15ull;
static float frand(void) { // xorshift, uniform [-1, 1)
    rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
    return (float)((rng >> 40) * (1.0 / (1ull << 23))) - 1.0f;
}

static void *xalloc(size_t n) {
    void *p = NULL;
    if (posix_memalign(&p, 2 << 20, (n + 4095) & ~(size_t)4095)) { perror("alloc"); exit(1); }
    return p;
}

// ---------------------------------------------------------------- data
static void generate_data(void);

// Generated data is deterministic; BENCH_CACHE=dir stores it so the 1.2 GB
// cold-regime weight sets are quantized once instead of once per process.
static int cache_io(int save) {
    const char *dir = getenv("BENCH_CACHE");
    if (!dir) return 0;
    const int xrows = G.kernel == K_Q8GEMM ? G.M : 1;
    char path[1024];
    snprintf(path, sizeof path, "%s/k%d_K%ld_N%ld_M%d_c%ld.bin", dir, G.kernel, (long)G.K, (long)G.N, xrows, (long)G.count);
    FILE *f = fopen(path, save ? "wb" : "rb");
    if (!f) return 0;
    struct { void *p; size_t n; } parts[] = {
        {G.W, G.mat_bytes * G.count}, {G.Xf, sizeof(float) * G.K * xrows},
        {G.X, G.x_row_bytes * xrows}, {G.Yref, sizeof(float) * G.N * xrows}};
    int ok = 1;
    for (int i = 0; i < 4 && ok; i++)
        ok = (save ? fwrite(parts[i].p, 1, parts[i].n, f) : fread(parts[i].p, 1, parts[i].n, f)) == parts[i].n;
    fclose(f);
    return ok;
}

static void make_data(void) {
    const enum ggml_type wt = WTYPE();
    const enum ggml_type xt = XTYPE();
    const int xrows = G.kernel == K_Q8GEMM ? G.M : 1;
    G.row_bytes = ggml_row_size(wt, G.K);
    G.mat_bytes = G.row_bytes * G.N;
    G.x_row_bytes = ggml_row_size(xt, G.K);
    G.W = xalloc(G.mat_bytes * G.count);
    G.Xf = xalloc(sizeof(float) * G.K * xrows);
    G.X = xalloc(G.x_row_bytes * xrows);
    G.Y = xalloc(sizeof(float) * G.N * xrows);
    G.Yref = xalloc(sizeof(float) * G.N * xrows);
    if (cache_io(0)) return;
    generate_data();
    if (getenv("BENCH_CACHE")) {
        mkdir(getenv("BENCH_CACHE"), 0755);
        cache_io(1);
    }
}

static void generate_data(void) {
    const enum ggml_type wt = WTYPE();
    const enum ggml_type xt = XTYPE();
    const int xrows = G.kernel == K_Q8GEMM ? G.M : 1;
    float *tmp = malloc(sizeof(float) * G.K * 256);
    // Weights: roughly Gaussian rows with per-row magnitude variation, so block
    // scales differ (exercises the fp16 scale path, not just the int dot).
    for (int64_t m = 0; m < G.count; m++) {
        for (int64_t r0 = 0; r0 < G.N; r0 += 256) {
            int64_t nr = G.N - r0 < 256 ? G.N - r0 : 256;
            for (int64_t i = 0; i < nr * G.K; i++) {
                float g = frand() + frand() + frand();
                tmp[i] = 0.02f * g * (1.0f + 0.5f * (float)(((r0 * G.K + i) / 32) % 7));
            }
            ggml_quantize_chunk(wt, tmp, G.W + m * G.mat_bytes + r0 * G.row_bytes, 0, nr, G.K, NULL);
        }
    }
    free(tmp);
    for (int64_t i = 0; i < G.K * xrows; i++) G.Xf[i] = frand() * (1.0f + (float)((i / 64) % 5));
    const struct ggml_type_traits_cpu *tt = ggml_get_type_traits_cpu(wt);
    if (tt->vec_dot_type != xt) { fprintf(stderr, "unexpected vec_dot_type\n"); exit(1); }
    ggml_from_float_t q = ggml_get_type_traits_cpu(xt)->from_float;
    for (int r = 0; r < xrows; r++) q(G.Xf + r * G.K, G.X + r * G.x_row_bytes, G.K);
    // Reference: ggml's own vec_dot (the kernel llama.cpp runs with --no-repack).
    for (int r = 0; r < xrows; r++)
        for (int64_t n = 0; n < G.N; n++)
            tt->vec_dot(G.K, &G.Yref[r * G.N + n], 0, G.W + n * G.row_bytes, 0, G.X + r * G.x_row_bytes, 0, 1);
}

// ---------------------------------------------------------------- pool
static atomic_int barrier_count;
static atomic_int barrier_gen;
static atomic_int stop_flag;

static atomic_int barrier_sleepers;
static long wait_spins = -1;
static double serial_us = 0; // --serial-us: single-threaded gap after each call (models router/top-k/small ops) // --wait spin (default): spin forever like ggml; sleep: spin this many pauses, then futex

static void barrier(int nth) {
    int gen = atomic_load_explicit(&barrier_gen, memory_order_acquire);
    if (atomic_fetch_add_explicit(&barrier_count, 1, memory_order_acq_rel) == nth - 1) {
        atomic_store_explicit(&barrier_count, 0, memory_order_relaxed);
        atomic_fetch_add_explicit(&barrier_gen, 1, memory_order_release);
        if (atomic_load_explicit(&barrier_sleepers, memory_order_acquire))
            syscall(SYS_futex, &barrier_gen, FUTEX_WAKE_PRIVATE, INT32_MAX, NULL, NULL, 0);
    } else {
        for (long i = 0; atomic_load_explicit(&barrier_gen, memory_order_acquire) == gen; i++) {
            if (wait_spins >= 0 && i >= wait_spins) {
                atomic_fetch_add(&barrier_sleepers, 1);
                while (atomic_load_explicit(&barrier_gen, memory_order_acquire) == gen)
                    syscall(SYS_futex, &barrier_gen, FUTEX_WAIT_PRIVATE, gen, NULL, NULL, 0);
                atomic_fetch_sub(&barrier_sleepers, 1);
                break;
            }
            _mm_pause();
        }
    }
}

static void pin(int cpu) {
    cpu_set_t s;
    CPU_ZERO(&s);
    CPU_SET(cpu, &s);
    sched_setaffinity(0, sizeof(s), &s);
}

static void slice(int64_t n, int ith, int nth, int64_t *a, int64_t *b) {
    // contiguous chunks aligned to 16 rows (kernels may process row groups)
    int64_t per = ((n + nth - 1) / nth + 15) & ~(int64_t)15;
    *a = ith * per < n ? ith * per : n;
    *b = (ith + 1) * per < n ? (ith + 1) * per : n;
}

static void run_one(int64_t it, int ith, int nth) {
    const int64_t m = G.cold ? it % G.count : 0;
    const uint8_t *W = G.W + m * G.mat_bytes;
    int64_t a, b;
    slice(G.N, ith, nth, &a, &b);
    if (a >= b) return;
    if (G.kernel == K_Q8GEMM) {
        if (G.gemm_packed) G.gemm_packed(G.packed[m], G.X, G.Y, G.K, G.N, G.M, a, b);
        else G.gemm(W, G.X, G.Y, G.K, G.N, G.M, a, b);
    } else if (G.gemv_packed) {
        G.gemv_packed(G.packed[m], G.X, G.Y, G.K, a, b);
    } else if (G.builtin_ggml) {
        const enum ggml_type wt = WTYPE();
        ggml_vec_dot_t vd = ggml_get_type_traits_cpu(wt)->vec_dot;
        for (int64_t r = a; r < b; r++) vd(G.K, &G.Y[r], 0, W + r * G.row_bytes, 0, G.X, 0, 1);
    } else {
        G.gemv(W, G.X, G.Y, G.K, a, b);
    }
}

// Each iteration: start barrier, every thread computes its row slice, end barrier.
static void *worker(void *arg) {
    int ith = (int)(intptr_t)arg;
    pin(ith);
    if (G.tinit) G.tinit();
    for (long it = 0;; it++) {
        barrier(G.threads);
        if (atomic_load(&stop_flag)) break;
        run_one(it, ith, G.threads);
        barrier(G.threads);
    }
    return NULL;
}

// Main thread is thread 0. Runs >= 1 iteration; returns iterations done.
static long timed_loop(double secs, double *wall, double *cpu, double *drift) {
    static int started = 0;
    static pthread_t th[256];
    static long it = 0;
    if (!started) {
        pin(0);
        if (G.tinit) G.tinit();
        for (int t = 1; t < G.threads; t++) pthread_create(&th[t], NULL, worker, (void *)(intptr_t)t);
        started = 1;
    }
    double c0 = cpu_time(), w0 = now(CLOCK_MONOTONIC), r0 = now(CLOCK_REALTIME), w;
    long n = 0;
    do {
        barrier(G.threads);
        run_one(it++, 0, G.threads);
        barrier(G.threads);
        if (serial_us > 0) {
            double t_end = now(CLOCK_MONOTONIC) + serial_us * 1e-6;
            while (now(CLOCK_MONOTONIC) < t_end) _mm_pause();
        }
        n++;
        w = now(CLOCK_MONOTONIC);
    } while (w - w0 < secs);
    *wall = w - w0;
    *cpu = cpu_time() - c0;
    *drift = (now(CLOCK_REALTIME) - r0) - *wall;
    return n;
}

static void stop_pool(void) {
    atomic_store(&stop_flag, 1);
    barrier(G.threads);
}

// ---------------------------------------------------------------- check
static double max_rel_err(void) {
    const int64_t n = G.N * (G.kernel == K_Q8GEMM ? G.M : 1);
    double err = 0, mag = 0;
    for (int64_t i = 0; i < n; i++) {
        double d = fabs((double)G.Y[i] - (double)G.Yref[i]);
        if (d > err) err = d;
        if (fabs(G.Yref[i]) > mag) mag = fabs(G.Yref[i]);
        if (isnan(G.Y[i])) return INFINITY;
    }
    return err / (mag > 0 ? mag : 1);
}

// ---------------------------------------------------------------- ggml graph paths
static ggml_backend_buffer_type_t find_buft(const char *name) {
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (!strcmp(name, "CPU")) return ggml_backend_cpu_buffer_type();
    ggml_backend_dev_get_extra_bufts_t f = (ggml_backend_dev_get_extra_bufts_t)ggml_backend_reg_get_proc_address(
        ggml_backend_dev_backend_reg(dev), "ggml_backend_dev_get_extra_bufts");
    for (ggml_backend_buffer_type_t *b = f ? f(dev) : NULL; b && *b; b++)
        if (!strcmp(ggml_backend_buft_name(*b), name)) return *b;
    return NULL;
}

// Full ggml MUL_MAT op (activation quantization + ggml's own threading and
// kernel selection). One graph holds `ops` mul_mats, like consecutive layers.
static void run_ggml_graph(const char *buftname, FILE *csv) {
    ggml_backend_buffer_type_t wbuft = find_buft(buftname);
    if (!wbuft) { fprintf(stderr, "no buffer type %s\n", buftname); exit(2); }
    const enum ggml_type wt = WTYPE();
    const int M = G.kernel == K_Q8GEMM ? G.M : 1;
    const int nmat = (int)G.count;
    const int ops = G.cold ? nmat : 16; // hot: 16 ops on the same weight
    struct ggml_init_params ip = {(size_t)(4 * ops + 16) * ggml_tensor_overhead() + ggml_graph_overhead_custom(4 * ops + 16, false), NULL, true};
    struct ggml_context *cw = ggml_init(ip), *ca = ggml_init(ip);
    struct ggml_tensor **w = calloc(nmat, sizeof(*w));
    for (int i = 0; i < nmat; i++) w[i] = ggml_new_tensor_2d(cw, wt, G.K, G.N);
    struct ggml_tensor *a = ggml_new_tensor_2d(ca, GGML_TYPE_F32, G.K, M);
    struct ggml_cgraph *gf = ggml_new_graph_custom(ca, 4 * ops + 16, false);
    struct ggml_tensor *first = NULL;
    for (int i = 0; i < ops; i++) {
        struct ggml_tensor *d = ggml_mul_mat(ca, w[G.cold ? i : 0], a);
        if (!first) first = d;
        ggml_build_forward_expand(gf, d);
    }
    ggml_backend_buffer_t bw = ggml_backend_alloc_ctx_tensors_from_buft(cw, wbuft);
    ggml_backend_buffer_set_usage(bw, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_t be = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(be, G.threads);
    ggml_backend_buffer_t ba = ggml_backend_alloc_ctx_tensors(ca, be);
    for (int i = 0; i < nmat; i++) ggml_backend_tensor_set(w[i], G.W + i * G.mat_bytes, 0, G.mat_bytes);
    ggml_backend_tensor_set(a, G.Xf, 0, sizeof(float) * G.K * M);
    ggml_backend_graph_compute(be, gf); // warm-up / check
    ggml_backend_tensor_get(first, G.Y, 0, sizeof(float) * G.N * M);
    double rel = max_rel_err();
    double t_end = now(CLOCK_MONOTONIC) + 0.3;
    while (now(CLOCK_MONOTONIC) < t_end) ggml_backend_graph_compute(be, gf);
    double c0 = cpu_time(), w0 = now(CLOCK_MONOTONIC), r0 = now(CLOCK_REALTIME), wl;
    long n = 0;
    do { ggml_backend_graph_compute(be, gf); n++; wl = now(CLOCK_MONOTONIC); } while (wl - w0 < G.secs);
    double wall = wl - w0, cpu = cpu_time() - c0, drift = (now(CLOCK_REALTIME) - r0) - wall;
    long calls = n * ops;
    report(G.label ? G.label : buftname, calls, wall, cpu, rel, drift, csv);
    ggml_backend_buffer_free(ba); ggml_backend_buffer_free(bw);
    ggml_backend_free(be); ggml_free(ca); ggml_free(cw); free(w);
}

// ---------------------------------------------------------------- report
static const char *kname(void) { return G.kernel == K_Q8GEMM ? "q8_0_gemm" : GEMVS[gemv_idx].name; }

void report(const char *impl, long calls, double wall, double cpu, double rel, double drift, FILE *csv) {
    const int M = G.kernel == K_Q8GEMM ? G.M : 1;
    double per = wall / calls;
    double bytes = (double)G.mat_bytes + (double)G.x_row_bytes * M + 4.0 * G.N * M;
    double ops = 2.0 * G.N * G.K * M;
    double gbs = bytes / per / 1e9, gops = ops / per / 1e9;
    double proxy_uj = cpu / calls * PROXY_W_PER_CORE * 1e6; // CPU-time energy proxy per call
    double pj_per_mac = proxy_uj * 1e6 / (ops / 2);
    const char *ok = rel < 1e-4 ? "ok" : (rel < 1e-2 ? "approx" : "FAIL");
    printf("%-22s %-10s %-4s T=%d K=%ld N=%ld M=%d | %9.1f us/call %8.1f GB/s %8.1f GOP/s | cpu/wall %.2f | proxy %.1f uJ/call %.2f pJ/MAC | relerr %.1e %s%s\n",
           impl, kname(), G.cold ? "cold" : "hot", G.threads, (long)G.K, (long)G.N, M, per * 1e6, gbs, gops,
           cpu / wall, proxy_uj, pj_per_mac, rel, ok, fabs(drift) > 0.05 ? " DRIFT!" : "");
    if (csv)
        fprintf(csv, "%s,%s,%s,%d,%ld,%ld,%d,%ld,%.6f,%.6f,%.3f,%.3f,%.3f,%.3f,%.4f,%.3e,%s,%.4f\n", impl, kname(),
                G.cold ? "cold" : "hot", G.threads, (long)G.K, (long)G.N, M, calls, wall, cpu, per * 1e6, gbs, gops,
                proxy_uj, pj_per_mac, rel, ok, drift);
}

// ---------------------------------------------------------------- bandwidth
static uint8_t *bw_buf;
static size_t bw_per;
static int bw_streams = 1;
static atomic_long bw_sink;
// Each thread reads its slice as `bw_streams` interleaved sequential streams
// (kernels that walk several row groups at once generate several streams).
static void *bw_worker(void *arg) {
    int ith = (int)(intptr_t)arg;
    pin(ith);
    const int S = bw_streams;
    const size_t per_stream = bw_per / S / 256 * 256;
    __m512i acc[4] = {_mm512_setzero_si512(), _mm512_setzero_si512(), _mm512_setzero_si512(), _mm512_setzero_si512()};
    for (int rep = 0; rep < (int)G.count; rep++) {
        barrier(G.threads);
        for (size_t off = 0; off < per_stream; off += 256)
            for (int st = 0; st < S; st++) {
                const __m512i *p = (const __m512i *)(bw_buf + ith * bw_per + st * per_stream + off);
                acc[0] = _mm512_xor_si512(acc[0], _mm512_load_si512(p));
                acc[1] = _mm512_xor_si512(acc[1], _mm512_load_si512(p + 1));
                acc[2] = _mm512_xor_si512(acc[2], _mm512_load_si512(p + 2));
                acc[3] = _mm512_xor_si512(acc[3], _mm512_load_si512(p + 3));
            }
    }
    __m512i a = _mm512_xor_si512(_mm512_xor_si512(acc[0], acc[1]), _mm512_xor_si512(acc[2], acc[3]));
    atomic_fetch_add(&bw_sink, _mm512_reduce_add_epi64(a));
    return NULL;
}
static void run_bw(size_t total, int reps, const char *what, FILE *csv) {
    bw_per = (total / G.threads) & ~(size_t)4095;
    bw_buf = xalloc(bw_per * G.threads);
    memset(bw_buf, 1, bw_per * G.threads);
    G.count = reps;
    atomic_store(&barrier_count, 0);
    pthread_t th[256];
    double c0 = cpu_time(), w0 = now(CLOCK_MONOTONIC);
    for (int t = 0; t < G.threads; t++) pthread_create(&th[t], NULL, bw_worker, (void *)(intptr_t)t);
    for (int t = 0; t < G.threads; t++) pthread_join(th[t], NULL);
    double wall = now(CLOCK_MONOTONIC) - w0, cpu = cpu_time() - c0;
    double gbs = (double)(bw_per / bw_streams / 256 * 256) * bw_streams * G.threads * reps / wall / 1e9;
    printf("bandwidth %-5s T=%d streams=%d bytes=%zu reps=%d : %.1f GB/s (cpu/wall %.2f)\n", what, G.threads, bw_streams, bw_per * G.threads, reps, gbs, cpu / wall);
    if (csv) fprintf(csv, "bandwidth_%s,read,%s,%d,0,0,%d,%d,%.6f,%.6f,0,%.3f,0,0,0,0,ok,0\n", what, what, G.threads, bw_streams, reps, wall, cpu, gbs);
    free(bw_buf);
}

// ---------------------------------------------------------------- dump
static void write_file(const char *dir, const char *name, const void *p, size_t n) {
    char path[1024];
    snprintf(path, sizeof path, "%s/%s", dir, name);
    FILE *f = fopen(path, "wb");
    if (!f || fwrite(p, 1, n, f) != n) { perror(path); exit(1); }
    fclose(f);
}
static void dump(void) {
    mkdir(G.dump, 0755);
    const int M = G.kernel == K_Q8GEMM ? G.M : 1;
    write_file(G.dump, "W.bin", G.W, G.mat_bytes * G.count);
    write_file(G.dump, "x.bin", G.X, G.x_row_bytes * M);
    write_file(G.dump, "yref.bin", G.Yref, sizeof(float) * G.N * M);
    char path[1024];
    snprintf(path, sizeof path, "%s/meta.txt", G.dump);
    FILE *f = fopen(path, "w");
    fprintf(f, "kernel %s\nK %ld\nN %ld\nM %d\ncount %ld\nrow_bytes %zu\nx_row_bytes %zu\nmat_bytes %zu\n", kname(),
            (long)G.K, (long)G.N, M, (long)G.count, G.row_bytes, G.x_row_bytes, G.mat_bytes);
    fclose(f);
    printf("dumped %s (%.1f MB weights)\n", G.dump, G.mat_bytes * G.count / 1e6);
}

// ---------------------------------------------------------------- main
int main(int argc, char **argv) {
    G.threads = 1; G.K = 4096; G.N = 0; G.M = 128; G.secs = 2.0; G.impl = "ggml";
    int bw = 0;
    const char *kern = "q8gemv", *regime = "hot";
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i], *v = i + 1 < argc ? argv[i + 1] : "";
        if (!strcmp(a, "--impl")) G.impl = v, i++;
        else if (!strcmp(a, "--kernel")) kern = v, i++;
        else if (!strcmp(a, "--regime")) regime = v, i++;
        else if (!strcmp(a, "--threads")) G.threads = atoi(v), i++;
        else if (!strcmp(a, "--K")) G.K = atol(v), i++;
        else if (!strcmp(a, "--N")) G.N = atol(v), i++;
        else if (!strcmp(a, "--M")) G.M = atoi(v), i++;
        else if (!strcmp(a, "--secs")) G.secs = atof(v), i++;
        else if (!strcmp(a, "--csv")) G.csv = v, i++;
        else if (!strcmp(a, "--label")) G.label = v, i++;
        else if (!strcmp(a, "--dump")) G.dump = v, i++;
        else if (!strcmp(a, "--bw")) bw = 1;
        else if (!strcmp(a, "--streams")) bw_streams = atoi(v), i++;
        else if (!strcmp(a, "--wait")) { wait_spins = !strcmp(v, "spin") ? -1 : !strcmp(v, "sleep") ? 100 : atol(v); i++; }
        else if (!strcmp(a, "--serial-us")) serial_us = atof(v), i++;
        else { fprintf(stderr, "unknown arg %s\n", a); return 2; }
    }
    ggml_cpu_init();
    FILE *csv = G.csv ? fopen(G.csv, "a") : NULL;
    if (bw) {
        run_bw((size_t)2 << 30, 6, "dram", csv);
        run_bw((size_t)G.threads << 20, 4000, "l2", csv);
        return 0;
    }
    G.kernel = K_Q8GEMM;
    for (size_t i = 0; i < sizeof GEMVS / sizeof GEMVS[0]; i++)
        if (!strcmp(kern, GEMVS[i].arg)) { G.kernel = i == 0 ? K_Q8GEMV : i == 1 ? K_Q4KGEMV : K_GEMV; gemv_idx = (int)i; }
    if (G.kernel == K_Q8GEMM && strcmp(kern, "q8gemm")) { fprintf(stderr, "unknown --kernel %s\n", kern); return 2; }
    G.cold = !strcmp(regime, "cold");
    if (!G.N) {
        // hot: per-thread slice ~1 MB or less (L2 = 2 MB/core); cold: matrices
        // rotate through >1 GB so every call streams from DRAM (L3 = 320 MB).
        if (G.kernel == K_Q8GEMM) G.N = 4096;
        else G.N = G.cold ? 4096 : 256 * G.threads;
    }
    G.count = 1;
    if (G.cold && G.kernel != K_Q8GEMM) { // GEMM is compute-bound; its weights stay L3-resident
        size_t mb = ggml_row_size(WTYPE(), G.K) * G.N;
        G.count = (int64_t)((1.2e9 + mb - 1) / mb);
    }
    make_data();
    if (G.dump) { dump(); return 0; }

    if (!strncmp(G.impl, "ggml-graph-", 11)) {
        const char *b = G.impl + 11;
        run_ggml_graph(!strcmp(b, "cpu") ? "CPU" : !strcmp(b, "amx") ? "AMX" : "CPU_REPACK", csv);
        return 0;
    }
    if (!strcmp(G.impl, "ggml")) {
        G.builtin_ggml = 1;
        if (G.kernel == K_Q8GEMM) { fprintf(stderr, "use ggml-graph-* for gemm\n"); return 2; }
    } else {
        void *h = dlopen(G.impl, RTLD_NOW | RTLD_LOCAL);
        if (!h) { fprintf(stderr, "dlopen: %s\n", dlerror()); return 2; }
        G.gemv = (gemv_fn)dlsym(h, GEMVS[gemv_idx].entry);
        G.gemm = (gemm_fn)dlsym(h, "kq8_gemm");
        G.tinit = (tinit_fn)dlsym(h, "kern_thread_init");
        prep_fn prep = (prep_fn)dlsym(h, "kq8_gemm_prepare");
        if (G.kernel == K_Q8GEMM && prep) {
            G.gemm_packed = (gemm_fn)dlsym(h, "kq8_gemm_packed");
            G.packed = calloc(G.count, sizeof(void *));
            for (int64_t m = 0; m < G.count; m++) G.packed[m] = prep(G.W + m * G.mat_bytes, G.K, G.N);
        }
        // optional weight repacking for GEMV (layout chosen by the implementation)
        char vps[64], vpk[64];
        snprintf(vps, sizeof vps, "%s_prepare", GEMVS[gemv_idx].entry);
        snprintf(vpk, sizeof vpk, "%s_packed", GEMVS[gemv_idx].entry);
        prep_fn vprep = (prep_fn)dlsym(h, vps);
        if (G.kernel != K_Q8GEMM && vprep) {
            G.gemv_packed = (gemv_fn)dlsym(h, vpk);
            G.packed = calloc(G.count, sizeof(void *));
            for (int64_t m = 0; m < G.count; m++) G.packed[m] = vprep(G.W + m * G.mat_bytes, G.K, G.N);
            if (!G.gemv) G.gemv = G.gemv_packed;
        }
        if ((G.kernel != K_Q8GEMM && !G.gemv) || (G.kernel == K_Q8GEMM && !G.gemm && !G.gemm_packed)) {
            printf("%-22s %-10s : not implemented\n", G.label ? G.label : G.impl, kname());
            return 3;
        }
    }
    double wall, cpu, drift;
    // correctness: one call on matrix 0 with all threads, then warm-up
    int saved = G.cold;
    G.cold = 0;
    memset(G.Y, 0, sizeof(float) * G.N * (G.kernel == K_Q8GEMM ? G.M : 1));
    timed_loop(0.0, &wall, &cpu, &drift);
    G.cold = saved;
    double rel = max_rel_err();
    if (getenv("BENCH_DEBUG")) for (int i = 0; i < 4; i++) printf("y[%d]=%g ref=%g\n", i, G.Y[i], G.Yref[i]);
    timed_loop(0.3, &wall, &cpu, &drift);
    long n = timed_loop(G.secs, &wall, &cpu, &drift);
    const char *name = G.label ? G.label : G.impl;
    report(name, n, wall, cpu, rel, drift, csv);
    stop_pool();
    if (csv) fclose(csv);
    return rel < 1e-2 ? 0 : 1;
}
