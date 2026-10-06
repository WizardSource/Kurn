// kurn benchmark harness: thread pool, pinning, timing, energy proxy and a
// correctness check for kernels implementing kurn.h. Kernels are dlopen'ed
// shared libraries. No dependency on ggml: weights and activations are random
// but valid ggml blocks, and the reference is an exact scalar implementation of
// each format's semantics, accumulated in double precision.
//
//   bench --impl lib.so --kernel q8gemv|q4kgemv|q8gemm [--regime hot|cold] [--threads T]
//         [--K 4096] [--N n] [--M 128] [--secs 2] [--csv out.csv] [--label name]
//         [--wait spin|sleep|<spins>] [--serial-us U] [--seed S] [--footprint MB]
//   bench --bw [--threads T] [--streams S]     peak memory read bandwidth (roofline); no --streams: best of 1, 2, 4, 8
//
// Linux only (futex, sched_setaffinity). Builds on x86-64 and AArch64.
#define _GNU_SOURCE
#include "kurn.h"
#include <dlfcn.h>
#include <limits.h>
#include <linux/futex.h>
#include <math.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#define PROXY_W_PER_CORE 5.47 // energy proxy: watts per busy core (350 W / 64-core server)
#define MAX_THREADS 256

typedef void (*gemv_fn)(const void *, const void *, float *, int64_t, int64_t, int64_t);
typedef void (*gemm_fn)(const void *, const void *, float *, int64_t, int64_t, int64_t, int64_t, int64_t);
typedef void *(*prep_fn)(const void *, int64_t, int64_t);
typedef void (*tinit_fn)(void);

// Formats and kernels the harness knows. A new format adds a format_desc (random
// valid blocks); a new kernel adds a kernel_desc row (and its exact reference).
typedef struct {
    const char *name;
    int block;                                  // values per block
    size_t bytes;                               // bytes per block
    void (*gen)(void *blocks, int64_t nblocks); // random but valid blocks
} format_desc;
typedef struct {
    const char *arg, *name;                     // --kernel value, report name
    const format_desc *w, *x;                   // weight and activation formats
    double (*ref)(const void *w, const void *x, int64_t nblocks); // exact dot of one weight row, double precision
    int gemm;                                   // 1: M activation rows (prefill); 0: one row (decode)
    const char *entry;                          // kurn.h entry point; <entry>_prepare / <entry>_packed are optional
} kernel_desc;

static struct {
    const kernel_desc *k;
    int threads, cold, M, ncpu;
    int64_t K, N, count;
    double secs;
    const char *impl, *csv, *label;
    size_t row_bytes, mat_bytes, x_row_bytes;
    uint8_t *W; // count matrices, contiguous
    uint8_t *X; // quantized activation(s): 1 row (gemv) or M rows (gemm)
    float *Y, *Yref;
    void **packed;
    gemv_fn gemv, gemv_packed;
    gemm_fn gemm, gemm_packed;
    tinit_fn tinit;
} G;

static inline void cpu_relax(void) {
#if defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#elif defined(__aarch64__)
    __asm__ volatile("yield");
#endif
}

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

static void *xalloc(size_t n) {
    void *p = NULL;
    if (posix_memalign(&p, 2 << 20, (n + 4095) & ~(size_t)4095)) { perror("alloc"); exit(1); }
    return p;
}

// ---------------------------------------------------------------- formats
static float h2f(uint16_t h) {
    uint32_t s = (uint32_t)(h & 0x8000) << 16, e = (h >> 10) & 0x1f, m = h & 0x3ff, b;
    if (e == 0) {
        if (!m) b = s;
        else { e = 113; while (!(m & 0x400)) { m <<= 1; e--; } b = s | (e << 23) | ((m & 0x3ff) << 13); }
    } else if (e == 31) b = s | 0x7f800000 | (m << 13);
    else b = s | ((e + 112) << 23) | (m << 13);
    float f;
    memcpy(&f, &b, 4);
    return f;
}

static uint64_t rng;
static inline uint64_t next(void) { // xorshift64*
    rng ^= rng >> 12; rng ^= rng << 25; rng ^= rng >> 27;
    return rng * 0x2545F4914F6CDD1Dull;
}
static void fill_i8(int8_t *q, int n) { // uniform in [-127, 127], like ggml's quantizers
    for (int i = 0; i < n; i += 8) {
        uint64_t r = next();
        for (int j = 0; j < 8 && i + j < n; j++, r >>= 8) q[i + j] = (int8_t)(r & 0xff) == -128 ? -127 : (int8_t)(r & 0xff);
    }
}
static void fill_u8(uint8_t *q, int n) {
    for (int i = 0; i < n; i += 8) {
        uint64_t r = next();
        for (int j = 0; j < 8 && i + j < n; j++, r >>= 8) q[i + j] = (uint8_t)r;
    }
}
static uint16_t rand_scale(void) { // positive fp16 in [2^-6, 2^-1), random mantissa
    uint64_t r = next();
    return (uint16_t)(((9 + r % 5) << 10) | ((r >> 8) & 0x3ff));
}

static void gen_q8_0(void *p, int64_t n) {
    block_q8_0 *b = p;
    for (int64_t i = 0; i < n; i++) { b[i].d = rand_scale(); fill_i8(b[i].qs, QK8_0); }
}
static void gen_q4_K(void *p, int64_t n) {
    block_q4_K *b = p;
    for (int64_t i = 0; i < n; i++) {
        b[i].d = rand_scale(); b[i].dmin = rand_scale();
        fill_u8(b[i].scales, 12); fill_u8(b[i].qs, QK_K / 2);
    }
}
static void gen_q8_K(void *p, int64_t n) {
    block_q8_K *b = p;
    for (int64_t i = 0; i < n; i++) {
        b[i].d = 0.001f + (float)(next() % 1000) * 2e-5f;
        fill_i8(b[i].qs, QK_K);
        for (int t = 0; t < QK_K / 16; t++) {
            int s = 0;
            for (int l = 0; l < 16; l++) s += b[i].qs[16 * t + l];
            b[i].bsums[t] = (int16_t)s;
        }
    }
}

// v0.2 formats (byte layouts identical to ggml's block_q4_0 / block_iq4_nl / block_q2_0 / block_tq2_0 / block_q1_0)
typedef struct { uint16_t d; uint8_t qs[16]; } blk_d16;   // q4_0, iq4_nl (32 values), q2_0 (64), q1_0 (128)
typedef struct { uint8_t qs[64]; uint16_t d; } blk_tq2_0; // 256 values
static const int8_t KV_IQ4NL[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
static void gen_d16(void *p, int64_t n) {
    blk_d16 *b = p;
    for (int64_t i = 0; i < n; i++) { b[i].d = rand_scale(); fill_u8(b[i].qs, 16); }
}
static void gen_tq2_0(void *p, int64_t n) {
    blk_tq2_0 *b = p;
    for (int64_t i = 0; i < n; i++) { fill_u8(b[i].qs, 64); b[i].d = rand_scale(); }
}
static double ref_q4_0(const void *wv, const void *xv, int64_t nb) {
    const blk_d16 *w = wv; const block_q8_0 *x = xv;
    double acc = 0;
    for (int64_t b = 0; b < nb; b++) {
        int32_t s = 0;
        for (int l = 0; l < 16; l++) s += ((w[b].qs[l] & 15) - 8) * x[b].qs[l] + ((w[b].qs[l] >> 4) - 8) * x[b].qs[l + 16];
        acc += (double)h2f(w[b].d) * (double)h2f(x[b].d) * s;
    }
    return acc;
}
static double ref_iq4_nl(const void *wv, const void *xv, int64_t nb) {
    const blk_d16 *w = wv; const block_q8_0 *x = xv;
    double acc = 0;
    for (int64_t b = 0; b < nb; b++) {
        int32_t s = 0;
        for (int l = 0; l < 16; l++) s += KV_IQ4NL[w[b].qs[l] & 15] * x[b].qs[l] + KV_IQ4NL[w[b].qs[l] >> 4] * x[b].qs[l + 16];
        acc += (double)h2f(w[b].d) * (double)h2f(x[b].d) * s;
    }
    return acc;
}
static double ref_q2_0(const void *wv, const void *xv, int64_t nb) {  // 64 values = 2 Q8_0 blocks
    const blk_d16 *w = wv; const block_q8_0 *x = xv;
    double acc = 0;
    for (int64_t b = 0; b < nb; b++)
        for (int h = 0; h < 2; h++) {
            int32_t s = 0;
            for (int l = 0; l < 32; l++) { const int v = 32 * h + l; s += (((w[b].qs[v / 4] >> (2 * (v % 4))) & 3) - 1) * x[2 * b + h].qs[l]; }
            acc += (double)h2f(w[b].d) * (double)h2f(x[2 * b + h].d) * s;
        }
    return acc;
}
static double ref_q1_0(const void *wv, const void *xv, int64_t nb) {  // 128 values = 4 Q8_0 blocks
    const blk_d16 *w = wv; const block_q8_0 *x = xv;
    double acc = 0;
    for (int64_t b = 0; b < nb; b++)
        for (int h = 0; h < 4; h++) {
            int32_t s = 0;
            for (int l = 0; l < 32; l++) { const int v = 32 * h + l; s += ((w[b].qs[v / 8] >> (v % 8)) & 1 ? 1 : -1) * x[4 * b + h].qs[l]; }
            acc += (double)h2f(w[b].d) * (double)h2f(x[4 * b + h].d) * s;
        }
    return acc;
}
static double ref_tq2_0(const void *wv, const void *xv, int64_t nb) {
    const blk_tq2_0 *w = wv; const block_q8_K *x = xv;
    double acc = 0;
    for (int64_t b = 0; b < nb; b++) {
        int32_t s = 0;
        for (int v = 0; v < 256; v++) s += (((w[b].qs[(v / 128) * 32 + v % 32] >> (2 * ((v % 128) / 32))) & 3) - 1) * x[b].qs[v];
        acc += (double)x[b].d * (double)h2f(w[b].d) * s;
    }
    return acc;
}

static double ref_q8_0(const void *wv, const void *xv, int64_t nb) {
    const block_q8_0 *w = wv, *x = xv;
    double acc = 0;
    for (int64_t b = 0; b < nb; b++) {
        int32_t s = 0;
        for (int l = 0; l < QK8_0; l++) s += (int32_t)w[b].qs[l] * x[b].qs[l];
        acc += (double)h2f(w[b].d) * (double)h2f(x[b].d) * s;
    }
    return acc;
}
static double ref_q4_K(const void *wv, const void *xv, int64_t nb) {
    const block_q4_K *w = wv;
    const block_q8_K *x = xv;
    double acc = 0;
    for (int64_t i = 0; i < nb; i++) {
        const uint8_t *q = w[i].scales;
        int64_t isum = 0, msum = 0;
        for (int s = 0; s < 8; s++) {
            int sc = s < 4 ? q[s] & 63 : (q[s + 4] & 0xF) | ((q[s - 4] >> 6) << 4);
            int mn = s < 4 ? q[s + 4] & 63 : (q[s + 4] >> 4) | ((q[s] >> 6) << 4);
            const uint8_t *q4 = w[i].qs + (s / 2) * 32;
            int32_t t = 0;
            for (int l = 0; l < 32; l++) t += ((q4[l] >> (4 * (s & 1))) & 0xF) * x[i].qs[32 * s + l];
            isum += (int64_t)sc * t;
            msum += (int64_t)mn * (x[i].bsums[2 * s] + x[i].bsums[2 * s + 1]);
        }
        acc += (double)x[i].d * ((double)h2f(w[i].d) * (double)isum - (double)h2f(w[i].dmin) * (double)msum);
    }
    return acc;
}

static const format_desc F_Q8_0 = {"q8_0", QK8_0, sizeof(block_q8_0), gen_q8_0};
static const format_desc F_Q4_K = {"q4_K", QK_K, sizeof(block_q4_K), gen_q4_K};
static const format_desc F_Q8_K = {"q8_K", QK_K, sizeof(block_q8_K), gen_q8_K};
static const format_desc F_Q4_0 = {"q4_0", 32, sizeof(blk_d16), gen_d16};
static const format_desc F_IQ4_NL = {"iq4_nl", 32, sizeof(blk_d16), gen_d16};
static const format_desc F_Q2_0 = {"q2_0", 64, sizeof(blk_d16), gen_d16};
static const format_desc F_Q1_0 = {"q1_0", 128, sizeof(blk_d16), gen_d16};
static const format_desc F_TQ2_0 = {"tq2_0", QK_K, sizeof(blk_tq2_0), gen_tq2_0};
/* --- lowbit --- */
typedef struct { uint8_t qs[48]; uint8_t qh[4]; uint16_t d; } blk_tq1_0;                       // ggml block_tq1_0
typedef struct { uint8_t scales[16]; uint8_t qs[64]; uint16_t d, dmin; } blk_q2_K;              // ggml block_q2_K
static void gen_tq1_0(void *p, int64_t n) {
    blk_tq1_0 *b = p;
    for (int64_t i = 0; i < n; i++) { fill_u8(b[i].qs, 48); fill_u8(b[i].qh, 4); b[i].d = rand_scale(); }
}
static void gen_q2_K(void *p, int64_t n) {
    blk_q2_K *b = p;
    for (int64_t i = 0; i < n; i++) { fill_u8(b[i].scales, 16); fill_u8(b[i].qs, 64); b[i].d = rand_scale(); b[i].dmin = rand_scale(); }
}
static int tq1_0_trit(const blk_tq1_0 *b, int v) {  // ggml's base-3 fixed point: trit l of q = ((uint8_t)(q * 3^l) * 3) >> 8
    static const uint8_t pow3[5] = {1, 3, 9, 27, 81};
    const uint8_t q = v < 160 ? b->qs[v % 32] : v < 240 ? b->qs[32 + (v - 160) % 16] : b->qh[(v - 240) % 4];
    const int l = v < 160 ? v / 32 : v < 240 ? (v - 160) / 16 : (v - 240) / 4;
    return ((uint8_t)(q * pow3[l]) * 3) >> 8;
}
static double ref_tq1_0(const void *wv, const void *xv, int64_t nb) {
    const blk_tq1_0 *w = wv; const block_q8_K *x = xv;
    double acc = 0;
    for (int64_t b = 0; b < nb; b++) {
        int32_t s = 0;
        for (int v = 0; v < 256; v++) s += (tq1_0_trit(w + b, v) - 1) * x[b].qs[v];
        acc += (double)x[b].d * (double)h2f(w[b].d) * s;
    }
    return acc;
}
static double ref_q2_K(const void *wv, const void *xv, int64_t nb) {
    const blk_q2_K *w = wv; const block_q8_K *x = xv;
    double acc = 0;
    for (int64_t b = 0; b < nb; b++) {
        int64_t isum = 0, msum = 0;
        for (int s = 0; s < 16; s++) {
            int32_t t = 0;
            for (int v = 16 * s; v < 16 * s + 16; v++) t += ((w[b].qs[(v / 128) * 32 + v % 32] >> (2 * ((v % 128) / 32))) & 3) * x[b].qs[v];
            isum += (int64_t)(w[b].scales[s] & 15) * t;
            msum += (int64_t)(w[b].scales[s] >> 4) * x[b].bsums[s];
        }
        acc += (double)x[b].d * ((double)h2f(w[b].d) * (double)isum - (double)h2f(w[b].dmin) * (double)msum);
    }
    return acc;
}
static const format_desc F_TQ1_0 = {"tq1_0", QK_K, sizeof(blk_tq1_0), gen_tq1_0};
static const format_desc F_Q2_K = {"q2_K", QK_K, sizeof(blk_q2_K), gen_q2_K};
/* --- end lowbit --- */
/* --- fourbit --- */
typedef struct { uint8_t e; uint8_t qs[16]; } blk_mxfp4;     // ggml block_mxfp4
typedef struct { uint8_t d[4]; uint8_t qs[32]; } blk_nvfp4;  // ggml block_nvfp4
static const int8_t KV_FP4[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};  // kvalues_mxfp4
static double e8m0_half(uint8_t e) { return ldexp(1.0, e - 128); }  // ggml_e8m0_to_fp32_half
static double ue4m3_half(uint8_t u) {                                // ggml_ue4m3_to_fp32
    if (u == 0 || u == 0x7F) return 0.0;
    const int e = (u >> 3) & 15, m = u & 7;
    return e ? ldexp(1.0 + m / 8.0, e - 8) : ldexp((double)m, -10);
}
static void gen_mxfp4(void *p, int64_t n) {  // e in [118, 131]: e == 0 (scale 2^-128) is read as 0 by the kernels
    blk_mxfp4 *b = p;
    for (int64_t i = 0; i < n; i++) { b[i].e = (uint8_t)(118 + next() % 14); fill_u8(b[i].qs, 16); }
}
static void gen_nvfp4(void *p, int64_t n) {  // mostly normal UE4M3 scales, some subnormal / zero / 0x7F
    blk_nvfp4 *b = p;
    for (int64_t i = 0; i < n; i++) {
        for (int s = 0; s < 4; s++) {
            const uint64_t r = next() % 100;
            b[i].d[s] = (uint8_t)(r < 3 ? 0x7F : r < 5 ? 0 : r < 15 ? 1 + r % 7 : 0x28 + next() % 0x30);
        }
        fill_u8(b[i].qs, 32);
    }
}
static double ref_mxfp4(const void *wv, const void *xv, int64_t nb) {
    const blk_mxfp4 *w = wv; const block_q8_0 *x = xv;
    double acc = 0;
    for (int64_t b = 0; b < nb; b++) {
        int32_t s = 0;
        for (int l = 0; l < 16; l++) s += KV_FP4[w[b].qs[l] & 15] * x[b].qs[l] + KV_FP4[w[b].qs[l] >> 4] * x[b].qs[l + 16];
        acc += e8m0_half(w[b].e) * (double)h2f(x[b].d) * s;
    }
    return acc;
}
static double ref_nvfp4(const void *wv, const void *xv, int64_t nb) {  // 64 values = 2 Q8_0 blocks, 4 scales
    const blk_nvfp4 *w = wv; const block_q8_0 *x = xv;
    double acc = 0;
    for (int64_t b = 0; b < nb; b++)
        for (int s = 0; s < 4; s++) {
            const block_q8_0 *xb = x + 2 * b + s / 2;
            int32_t t = 0;
            for (int j = 0; j < 8; j++)
                t += KV_FP4[w[b].qs[8 * s + j] & 15] * xb->qs[16 * (s & 1) + j] + KV_FP4[w[b].qs[8 * s + j] >> 4] * xb->qs[16 * (s & 1) + j + 8];
            acc += ue4m3_half(w[b].d[s]) * (double)h2f(xb->d) * t;
        }
    return acc;
}
static const format_desc F_MXFP4 = {"mxfp4", 32, sizeof(blk_mxfp4), gen_mxfp4};
static const format_desc F_NVFP4 = {"nvfp4", 64, sizeof(blk_nvfp4), gen_nvfp4};
/* --- end fourbit --- */
/* --- compress --- */
// E8P lattice codebook (kurn.codebook), planar: lo[g] = idx low 7 bits | t << 7, hi[g] = signs;
// table index bit 7 = parity of the sign byte; q = (sign ? -4a : 4a) + (t ? 1 : -1).
typedef struct { uint16_t d; uint8_t lo[32]; uint8_t hi[32]; } blk_e8p;
static const uint64_t E8P_ABS4[256] = {
    0x0202020202020202ULL, 0x0606020202020202ULL, 0x0602060202020202ULL, 0x0206060202020202ULL,
    0x0602020602020202ULL, 0x0206020602020202ULL, 0x0202060602020202ULL, 0x0602020206020202ULL,
    0x0206020206020202ULL, 0x0202060206020202ULL, 0x0202020606020202ULL, 0x0602020202060202ULL,
    0x0206020202060202ULL, 0x0202060202060202ULL, 0x0202020602060202ULL, 0x0202020206060202ULL,
    0x0602020202020602ULL, 0x0206020202020602ULL, 0x0202060202020602ULL, 0x0202020602020602ULL,
    0x0202020206020602ULL, 0x0202020202060602ULL, 0x0602020202020206ULL, 0x0206020202020206ULL,
    0x0202060202020206ULL, 0x0202020602020206ULL, 0x0202020206020206ULL, 0x0202020202060206ULL,
    0x0202020202020606ULL, 0x0a02020202020202ULL, 0x020a020202020202ULL, 0x02020a0202020202ULL,
    0x0202020a02020202ULL, 0x020202020a020202ULL, 0x02020202020a0202ULL, 0x0202020202020a02ULL,
    0x020202020202020aULL, 0x0606060602020202ULL, 0x0606060206020202ULL, 0x0606020606020202ULL,
    0x0602060606020202ULL, 0x0206060606020202ULL, 0x0606060202060202ULL, 0x0606020602060202ULL,
    0x0602060602060202ULL, 0x0206060602060202ULL, 0x0606020206060202ULL, 0x0602060206060202ULL,
    0x0206060206060202ULL, 0x0602020606060202ULL, 0x0206020606060202ULL, 0x0202060606060202ULL,
    0x0606060202020602ULL, 0x0606020602020602ULL, 0x0602060602020602ULL, 0x0206060602020602ULL,
    0x0606020206020602ULL, 0x0602060206020602ULL, 0x0206060206020602ULL, 0x0602020606020602ULL,
    0x0206020606020602ULL, 0x0202060606020602ULL, 0x0606020202060602ULL, 0x0602060202060602ULL,
    0x0206060202060602ULL, 0x0602020602060602ULL, 0x0206020602060602ULL, 0x0202060602060602ULL,
    0x0602020206060602ULL, 0x0206020206060602ULL, 0x0202060206060602ULL, 0x0202020606060602ULL,
    0x0606060202020206ULL, 0x0606020602020206ULL, 0x0602060602020206ULL, 0x0206060602020206ULL,
    0x0606020206020206ULL, 0x0602060206020206ULL, 0x0206060206020206ULL, 0x0602020606020206ULL,
    0x0206020606020206ULL, 0x0202060606020206ULL, 0x0606020202060206ULL, 0x0602060202060206ULL,
    0x0206060202060206ULL, 0x0602020602060206ULL, 0x0206020602060206ULL, 0x0202060602060206ULL,
    0x0602020206060206ULL, 0x0206020206060206ULL, 0x0202060206060206ULL, 0x0202020606060206ULL,
    0x0606020202020606ULL, 0x0602060202020606ULL, 0x0206060202020606ULL, 0x0602020602020606ULL,
    0x0206020602020606ULL, 0x0202060602020606ULL, 0x0602020206020606ULL, 0x0206020206020606ULL,
    0x0202060206020606ULL, 0x0202020606020606ULL, 0x0602020202060606ULL, 0x0206020202060606ULL,
    0x0202060202060606ULL, 0x0202020602060606ULL, 0x0202020206060606ULL, 0x0a06060202020202ULL,
    0x060a060202020202ULL, 0x06060a0202020202ULL, 0x0a06020602020202ULL, 0x060a020602020202ULL,
    0x0a02060602020202ULL, 0x020a060602020202ULL, 0x06020a0602020202ULL, 0x02060a0602020202ULL,
    0x0606020a02020202ULL, 0x0602060a02020202ULL, 0x0206060a02020202ULL, 0x0a06020206020202ULL,
    0x060a020206020202ULL, 0x0a02060206020202ULL, 0x020a060206020202ULL, 0x06020a0206020202ULL,
    0x02060a0206020202ULL, 0x0a02020606020202ULL, 0x020a020606020202ULL, 0x02020a0606020202ULL,
    0x0602020202020202ULL, 0x0206020202020202ULL, 0x0202060202020202ULL, 0x0202020602020202ULL,
    0x0202020206020202ULL, 0x0202020202060202ULL, 0x0202020202020602ULL, 0x0202020202020206ULL,
    0x0606060202020202ULL, 0x0606020602020202ULL, 0x0602060602020202ULL, 0x0206060602020202ULL,
    0x0606020206020202ULL, 0x0602060206020202ULL, 0x0206060206020202ULL, 0x0602020606020202ULL,
    0x0206020606020202ULL, 0x0202060606020202ULL, 0x0606020202060202ULL, 0x0602060202060202ULL,
    0x0206060202060202ULL, 0x0602020602060202ULL, 0x0206020602060202ULL, 0x0202060602060202ULL,
    0x0602020206060202ULL, 0x0206020206060202ULL, 0x0202060206060202ULL, 0x0202020606060202ULL,
    0x0606020202020602ULL, 0x0602060202020602ULL, 0x0206060202020602ULL, 0x0602020602020602ULL,
    0x0206020602020602ULL, 0x0202060602020602ULL, 0x0602020206020602ULL, 0x0206020206020602ULL,
    0x0202060206020602ULL, 0x0202020606020602ULL, 0x0602020202060602ULL, 0x0206020202060602ULL,
    0x0202060202060602ULL, 0x0202020602060602ULL, 0x0202020206060602ULL, 0x0606020202020206ULL,
    0x0602060202020206ULL, 0x0206060202020206ULL, 0x0602020602020206ULL, 0x0206020602020206ULL,
    0x0202060602020206ULL, 0x0602020206020206ULL, 0x0206020206020206ULL, 0x0202060206020206ULL,
    0x0202020606020206ULL, 0x0602020202060206ULL, 0x0206020202060206ULL, 0x0202060202060206ULL,
    0x0202020602060206ULL, 0x0202020206060206ULL, 0x0602020202020606ULL, 0x0206020202020606ULL,
    0x0202060202020606ULL, 0x0202020602020606ULL, 0x0202020206020606ULL, 0x0202020202060606ULL,
    0x0a06020202020202ULL, 0x060a020202020202ULL, 0x0a02060202020202ULL, 0x020a060202020202ULL,
    0x06020a0202020202ULL, 0x02060a0202020202ULL, 0x0a02020602020202ULL, 0x020a020602020202ULL,
    0x02020a0602020202ULL, 0x0602020a02020202ULL, 0x0206020a02020202ULL, 0x0202060a02020202ULL,
    0x0a02020206020202ULL, 0x020a020206020202ULL, 0x02020a0206020202ULL, 0x0202020a06020202ULL,
    0x060202020a020202ULL, 0x020602020a020202ULL, 0x020206020a020202ULL, 0x020202060a020202ULL,
    0x0a02020202060202ULL, 0x020a020202060202ULL, 0x02020a0202060202ULL, 0x0202020a02060202ULL,
    0x020202020a060202ULL, 0x06020202020a0202ULL, 0x02060202020a0202ULL, 0x02020602020a0202ULL,
    0x02020206020a0202ULL, 0x02020202060a0202ULL, 0x0a02020202020602ULL, 0x020a020202020602ULL,
    0x02020a0202020602ULL, 0x0202020a02020602ULL, 0x020202020a020602ULL, 0x02020202020a0602ULL,
    0x0602020202020a02ULL, 0x0206020202020a02ULL, 0x0202060202020a02ULL, 0x0202020602020a02ULL,
    0x0202020206020a02ULL, 0x0202020202060a02ULL, 0x0a02020202020206ULL, 0x020a020202020206ULL,
    0x02020a0202020206ULL, 0x0202020a02020206ULL, 0x020202020a020206ULL, 0x02020202020a0206ULL,
    0x0202020202020a06ULL, 0x060202020202020aULL, 0x020602020202020aULL, 0x020206020202020aULL,
    0x020202060202020aULL, 0x020202020602020aULL, 0x020202020206020aULL, 0x020202020202060aULL,
    0x0606060606020202ULL, 0x0606060602060202ULL, 0x0606060206060202ULL, 0x0606020606060202ULL,
    0x0602060606060202ULL, 0x0206060606060202ULL, 0x0606060602020602ULL, 0x0606060206020602ULL,
};
static void gen_e8p(void *p, int64_t n) {
    blk_e8p *b = p;
    for (int64_t i = 0; i < n; i++) {
        b[i].d = rand_scale();
        fill_u8(b[i].lo, 32);
        fill_u8(b[i].hi, 32);
    }
}
static double ref_e8p(const void *wv, const void *xv, int64_t nb) {
    const blk_e8p *w = wv; const block_q8_K *x = xv;
    double acc = 0;
    for (int64_t b = 0; b < nb; b++) {
        int32_t s = 0;
        for (int g = 0; g < 32; g++) {
            unsigned c = w[b].lo[g], sg = w[b].hi[g], p = sg ^ (sg >> 4);
            p ^= p >> 2; p ^= p >> 1;
            uint64_t a = E8P_ABS4[(c & 0x7F) | ((p & 1) << 7)];
            for (int l = 0; l < 8; l++) {
                int q = (int)((a >> (8 * l)) & 0xFF);
                q = ((sg >> l) & 1 ? -q : q) + ((c & 0x80) ? 1 : -1);
                s += q * x[b].qs[8 * g + l];
            }
        }
        acc += (double)x[b].d * (double)h2f(w[b].d) * s;
    }
    return acc;
}
static const format_desc F_E8P = {"e8p", QK_K, sizeof(blk_e8p), gen_e8p};
/* --- end compress --- */
// gemm = 1: several activation rows. *vfy kernels (multi-token verify) take a small --M
// (default 4) and support the hot/cold regimes like the GEMVs.
static const kernel_desc KERNELS[] = {
    {"q8gemv", "q8_0_gemv", &F_Q8_0, &F_Q8_0, ref_q8_0, 0, "kq8_gemv"},
    {"q4kgemv", "q4_K_gemv", &F_Q4_K, &F_Q8_K, ref_q4_K, 0, "kq4k_gemv"},
    {"q8gemm", "q8_0_gemm", &F_Q8_0, &F_Q8_0, ref_q8_0, 1, "kq8_gemm"},
    {"q40gemv", "q4_0_gemv", &F_Q4_0, &F_Q8_0, ref_q4_0, 0, "kq40_gemv"},
    {"iq4nlgemv", "iq4_nl_gemv", &F_IQ4_NL, &F_Q8_0, ref_iq4_nl, 0, "kiq4nl_gemv"},
    {"q20gemv", "q2_0_gemv", &F_Q2_0, &F_Q8_0, ref_q2_0, 0, "kq20_gemv"},
    {"tq20gemv", "tq2_0_gemv", &F_TQ2_0, &F_Q8_K, ref_tq2_0, 0, "ktq20_gemv"},
    {"q10gemv", "q1_0_gemv", &F_Q1_0, &F_Q8_0, ref_q1_0, 0, "kq10_gemv"},
    {"q8vfy", "q8_0_verify", &F_Q8_0, &F_Q8_0, ref_q8_0, 1, "kq8_vfy"},
    {"q4kvfy", "q4_K_verify", &F_Q4_K, &F_Q8_K, ref_q4_K, 1, "kq4k_vfy"},
    {"q40vfy", "q4_0_verify", &F_Q4_0, &F_Q8_0, ref_q4_0, 1, "kq40_vfy"},
    {"iq4nlvfy", "iq4_nl_verify", &F_IQ4_NL, &F_Q8_0, ref_iq4_nl, 1, "kiq4nl_vfy"},
    {"q20vfy", "q2_0_verify", &F_Q2_0, &F_Q8_0, ref_q2_0, 1, "kq20_vfy"},
    {"tq20vfy", "tq2_0_verify", &F_TQ2_0, &F_Q8_K, ref_tq2_0, 1, "ktq20_vfy"},
    {"q10vfy", "q1_0_verify", &F_Q1_0, &F_Q8_0, ref_q1_0, 1, "kq10_vfy"},
    /* --- lowbit --- */
    {"tq10gemv", "tq1_0_gemv", &F_TQ1_0, &F_Q8_K, ref_tq1_0, 0, "ktq10_gemv"},
    {"q2kgemv", "q2_K_gemv", &F_Q2_K, &F_Q8_K, ref_q2_K, 0, "kq2k_gemv"},
    /* --- end lowbit --- */
    /* --- fourbit --- */
    {"mxfp4gemv", "mxfp4_gemv", &F_MXFP4, &F_Q8_0, ref_mxfp4, 0, "kmxfp4_gemv"},
    {"nvfp4gemv", "nvfp4_gemv", &F_NVFP4, &F_Q8_0, ref_nvfp4, 0, "knvfp4_gemv"},
    {"mxfp4vfy", "mxfp4_verify", &F_MXFP4, &F_Q8_0, ref_mxfp4, 1, "kmxfp4_vfy"},
    {"nvfp4vfy", "nvfp4_verify", &F_NVFP4, &F_Q8_0, ref_nvfp4, 1, "knvfp4_vfy"},
    /* --- end fourbit --- */
    /* --- compress --- */
    {"e8pgemv", "e8p_gemv", &F_E8P, &F_Q8_K, ref_e8p, 0, "ke8p_gemv"},
    /* --- end compress --- */
};
static int is_verify(void) { return strstr(G.k->arg, "vfy") != NULL; }

static void make_data(uint64_t seed) {
    const kernel_desc *k = G.k;
    const int xrows = k->gemm ? G.M : 1;
    const int64_t nbw = G.K / k->w->block, nbx = G.K / k->x->block;
    G.row_bytes = nbw * k->w->bytes;
    G.mat_bytes = G.row_bytes * G.N;
    G.x_row_bytes = nbx * k->x->bytes;
    G.W = xalloc(G.mat_bytes * G.count);
    G.X = xalloc(G.x_row_bytes * xrows);
    G.Y = xalloc(sizeof(float) * G.N * xrows);
    G.Yref = xalloc(sizeof(float) * G.N * xrows);
    rng = seed ? seed : 0x9E3779B97F4A7C15ull;
    k->w->gen(G.W, nbw * G.N * G.count);
    k->x->gen(G.X, nbx * xrows);
    for (int r = 0; r < xrows; r++)
        for (int64_t n = 0; n < G.N; n++)
            G.Yref[r * G.N + n] = (float)k->ref(G.W + n * G.row_bytes, G.X + r * G.x_row_bytes, nbw);
}

// ---------------------------------------------------------------- pool
static atomic_int barrier_count, barrier_gen, barrier_sleepers, stop_flag;
static long wait_spins = -1; // -1: spin forever (ggml-like); >= 0: spin this many pauses, then futex-sleep
static double footprint_mb = 0; // --footprint: total weight bytes rotated through (cache-cliff sweeps); implies --regime cold
static double serial_us = 0; // single-threaded gap after each call (models router/top-k/small ops)

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
            cpu_relax();
        }
    }
}

static void pin(int t) {
    cpu_set_t s;
    CPU_ZERO(&s);
    CPU_SET(t % G.ncpu, &s);
    sched_setaffinity(0, sizeof(s), &s);
}

static void slice(int64_t n, int ith, int nth, int64_t *a, int64_t *b) {
    // contiguous chunks aligned to 16 rows (kernels may process 16-row groups)
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
    if (G.k->gemm) {
        if (G.gemm_packed) G.gemm_packed(G.packed[m], G.X, G.Y, G.K, G.N, G.M, a, b);
        else G.gemm(W, G.X, G.Y, G.K, G.N, G.M, a, b);
    } else if (G.gemv_packed) {
        G.gemv_packed(G.packed[m], G.X, G.Y, G.K, a, b);
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
    static pthread_t th[MAX_THREADS];
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
            while (now(CLOCK_MONOTONIC) < t_end) cpu_relax();
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

// ---------------------------------------------------------------- check + report
static double max_rel_err(void) {
    const int64_t n = G.N * (G.k->gemm ? G.M : 1);
    double err = 0, mag = 0;
    for (int64_t i = 0; i < n; i++) {
        if (!isfinite(G.Y[i])) return INFINITY;
        double d = fabs((double)G.Y[i] - (double)G.Yref[i]);
        if (d > err) err = d;
        if (fabs(G.Yref[i]) > mag) mag = fabs(G.Yref[i]);
    }
    return err / (mag > 0 ? mag : 1);
}

static const char *kname(void) { return G.k->name; }

static void report(const char *impl, long calls, double wall, double cpu, double rel, double drift, FILE *csv) {
    const int M = G.k->gemm ? G.M : 1;
    double per = wall / calls;
    double bytes = (double)G.mat_bytes + (double)G.x_row_bytes * M + 4.0 * G.N * M;
    double ops = 2.0 * G.N * G.K * M;
    double gbs = bytes / per / 1e9, gops = ops / per / 1e9;
    double proxy_uj = cpu / calls * PROXY_W_PER_CORE * 1e6;
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

// ---------------------------------------------------------------- bandwidth (roofline)
// Peak read bandwidth. Each pinned thread reads its slice as `S` interleaved sequential streams with the widest vector loads
// this build has (AVX-512, AVX2, NEON, else 64-bit), 256 bytes per stream step into 4 independent accumulators. Every timed
// group of passes is bracketed by barriers; the result is the best group (the peak), with the median alongside. Without
// --streams, S = 1, 2, 4, 8 are all measured and the best is reported (one stream per thread under-fills some memory systems).
#if defined(__AVX512F__)
#include <immintrin.h>
#define BW_ISA "avx512"
typedef __m512i bw_acc;
static inline void bw_chunk(const uint8_t *p, bw_acc *a) {
    a[0] = _mm512_xor_si512(a[0], _mm512_load_si512((const void *)p));
    a[1] = _mm512_xor_si512(a[1], _mm512_load_si512((const void *)(p + 64)));
    a[2] = _mm512_xor_si512(a[2], _mm512_load_si512((const void *)(p + 128)));
    a[3] = _mm512_xor_si512(a[3], _mm512_load_si512((const void *)(p + 192)));
}
static inline uint64_t bw_fold(const bw_acc *a) {
    return (uint64_t)_mm512_reduce_add_epi64(_mm512_xor_si512(_mm512_xor_si512(a[0], a[1]), _mm512_xor_si512(a[2], a[3])));
}
#elif defined(__AVX2__)
#include <immintrin.h>
#define BW_ISA "avx2"
typedef __m256i bw_acc;
static inline void bw_chunk(const uint8_t *p, bw_acc *a) {
    for (int i = 0; i < 8; i++) a[i & 3] = _mm256_xor_si256(a[i & 3], _mm256_load_si256((const __m256i *)(p + 32 * i)));
}
static inline uint64_t bw_fold(const bw_acc *a) {
    uint64_t t[4];
    _mm256_storeu_si256((__m256i *)t, _mm256_xor_si256(_mm256_xor_si256(a[0], a[1]), _mm256_xor_si256(a[2], a[3])));
    return t[0] ^ t[1] ^ t[2] ^ t[3];
}
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#define BW_ISA "neon"
typedef uint8x16_t bw_acc;
static inline void bw_chunk(const uint8_t *p, bw_acc *a) {
    for (int i = 0; i < 16; i++) a[i & 3] = veorq_u8(a[i & 3], vld1q_u8(p + 16 * i));
}
static inline uint64_t bw_fold(const bw_acc *a) {
    uint64x2_t x = vreinterpretq_u64_u8(veorq_u8(veorq_u8(a[0], a[1]), veorq_u8(a[2], a[3])));
    return vgetq_lane_u64(x, 0) ^ vgetq_lane_u64(x, 1);
}
#else
#define BW_ISA "scalar"
typedef uint64_t bw_acc;
static inline void bw_chunk(const uint8_t *p, bw_acc *a) {
    const uint64_t *q = (const uint64_t *)p;
    for (int i = 0; i < 32; i++) a[i & 3] ^= q[i];
}
static inline uint64_t bw_fold(const bw_acc *a) { return a[0] ^ a[1] ^ a[2] ^ a[3]; }
#endif

#define BW_MAX_GROUPS 64
static uint8_t *bw_buf;
static size_t bw_per, bw_stream_bytes;
static int bw_streams = 0, bw_cur_streams, bw_groups, bw_group;
static double bw_t0[BW_MAX_GROUPS], bw_t1[BW_MAX_GROUPS];
static atomic_ulong bw_sink;
static void *bw_worker(void *arg) {
    int ith = (int)(intptr_t)arg;
    pin(ith);
    const int S = bw_cur_streams;
    const size_t ps = bw_stream_bytes;
    const uint8_t *base = bw_buf + (size_t)ith * bw_per;
    bw_acc a[4];
    memset(a, 0, sizeof a);
    for (int g = 0; g < bw_groups; g++) {
        barrier(G.threads);
        if (ith == 0) bw_t0[g] = now(CLOCK_MONOTONIC);
        for (int r = 0; r < bw_group; r++)
            for (size_t off = 0; off < ps; off += 256)
                for (int st = 0; st < S; st++) bw_chunk(base + st * ps + off, a);
        barrier(G.threads);
        if (ith == 0) bw_t1[g] = now(CLOCK_MONOTONIC);
    }
    atomic_fetch_add(&bw_sink, bw_fold(a));
    return NULL;
}
static int cmp_double(const void *x, const void *y) {
    double a = *(const double *)x, b = *(const double *)y;
    return (a > b) - (a < b);
}
// one stream count: returns the best group's GB/s, *median gets the median group
static double bw_measure(int S, double *median) {
    bw_cur_streams = S;
    bw_stream_bytes = bw_per / S / 256 * 256;
    atomic_store(&barrier_count, 0);
    pthread_t th[MAX_THREADS];
    for (int t = 0; t < G.threads; t++) pthread_create(&th[t], NULL, bw_worker, (void *)(intptr_t)t);
    for (int t = 0; t < G.threads; t++) pthread_join(th[t], NULL);
    double bytes = (double)bw_stream_bytes * S * G.threads * bw_group, gbs[BW_MAX_GROUPS];
    for (int g = 0; g < bw_groups; g++) gbs[g] = bytes / (bw_t1[g] - bw_t0[g]) / 1e9;
    qsort(gbs, bw_groups, sizeof gbs[0], cmp_double);
    *median = gbs[bw_groups / 2];
    return gbs[bw_groups - 1];
}
static void run_bw(size_t total, int groups, int group, const char *what, FILE *csv) {
    bw_per = (total / G.threads) & ~(size_t)4095;
    bw_buf = xalloc(bw_per * G.threads);
    memset(bw_buf, 1, bw_per * G.threads);
    bw_groups = groups < BW_MAX_GROUPS ? groups : BW_MAX_GROUPS;
    bw_group = group;
    const int sweep[] = {1, 2, 4, 8};
    int n = bw_streams > 0 ? 1 : 4, best_s = 0;
    double best = 0, best_med = 0, w0 = now(CLOCK_MONOTONIC), c0 = cpu_time();
    for (int i = 0; i < n; i++) {
        int S = bw_streams > 0 ? bw_streams : sweep[i];
        double med, peak = bw_measure(S, &med);
        printf("bw-sweep  %-5s T=%d streams=%d : best %.1f GB/s, median %.1f GB/s over %d groups\n", what, G.threads, S, peak, med,
               bw_groups);
        if (peak > best) best = peak, best_med = med, best_s = S;
    }
    double wall = now(CLOCK_MONOTONIC) - w0, cpu = cpu_time() - c0;
    printf("bandwidth %-5s T=%d streams=%d bytes=%zu groups=%d : %.1f GB/s (best; median %.1f GB/s, %s loads, cpu/wall %.2f)\n",
           what, G.threads, best_s, bw_per * G.threads, bw_groups, best, best_med, BW_ISA, cpu / wall);
    if (csv)
        fprintf(csv, "bandwidth_%s,read,%s,%d,0,0,%d,%d,%.6f,%.6f,0,%.3f,0,0,0,0,ok,0\n", what, what, G.threads, best_s, bw_groups,
                wall, cpu, best);
    free(bw_buf);
}

/* --- runtime --- */
// --wait values from kurn.ext.runtime: "futex" (sleep at once), "hybrid:N" (spin N pauses, then
// futex), or a plain pause count.
static long runtime_wait_spins(const char *v) {
    if (!strcmp(v, "futex")) return 0;
    if (!strncmp(v, "hybrid:", 7)) return atol(v + 7);
    return atol(v);
}
/* --- end runtime --- */

// ---------------------------------------------------------------- main
static int usage(const char *msg) {
    fprintf(stderr, "bench: %s\nusage: bench --impl lib.so --kernel q8gemv|q4kgemv|q8gemm [--regime hot|cold] [--threads T]\n"
                    "             [--K K] [--N N] [--M M] [--secs S] [--csv out.csv] [--label L] [--wait spin|sleep|N]\n"
                    "             [--serial-us U] [--seed S]\n       bench --bw [--threads T] [--streams S]\n", msg);
    return 2;
}

int main(int argc, char **argv) {
    G.threads = 1; G.K = 4096; G.N = 0; G.M = 0; G.secs = 2.0; G.impl = NULL;
    G.ncpu = (int)sysconf(_SC_NPROCESSORS_ONLN);
    if (G.ncpu < 1) G.ncpu = 1;
    int bw = 0;
    uint64_t seed = 0;
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
        else if (!strcmp(a, "--seed")) seed = strtoull(v, NULL, 0), i++;
        else if (!strcmp(a, "--bw")) bw = 1;
        else if (!strcmp(a, "--streams")) bw_streams = atoi(v), i++;
        else if (!strcmp(a, "--wait")) { wait_spins = !strcmp(v, "spin") ? -1 : !strcmp(v, "sleep") ? 100 : runtime_wait_spins(v); i++; }
        else if (!strcmp(a, "--serial-us")) serial_us = atof(v), i++;
        else if (!strcmp(a, "--footprint")) footprint_mb = atof(v), i++;
        else { fprintf(stderr, "unknown arg %s\n", a); return usage("bad arguments"); }
    }
    if (G.threads < 1 || G.threads > MAX_THREADS) return usage("--threads must be in 1..256");
    if (bw_streams < 0) return usage("--streams must be >= 0 (0: measure 1, 2, 4 and 8 and report the best)");
    FILE *csv = G.csv ? fopen(G.csv, "a") : NULL;
    if (bw) {
        run_bw((size_t)2 << 30, 8, 1, "dram", csv);
        run_bw((size_t)G.threads << 20, 20, 200, "l2", csv);
        if (csv) fclose(csv);
        return 0;
    }
    if (!G.impl) return usage("--impl is required");
    for (size_t i = 0; i < sizeof KERNELS / sizeof KERNELS[0]; i++)
        if (!strcmp(kern, KERNELS[i].arg)) G.k = &KERNELS[i];
    if (!G.k) return usage("unknown --kernel");
    G.cold = !strcmp(regime, "cold") || footprint_mb > 0;
    if (!G.M) G.M = is_verify() ? 4 : 128;
    const int blk = G.k->w->block > G.k->x->block ? G.k->w->block : G.k->x->block;
    if (G.K <= 0 || G.K % blk || G.K > 32768) return usage("--K must be a positive multiple of the block size and <= 32768");
    if (!G.N) {
        // hot: per-thread slice of 256 rows (fits a 2 MB L2); cold: matrices rotate
        // through >1.2 GB so every call streams from DRAM.
        if (G.k->gemm && !is_verify()) G.N = 4096;
        else G.N = G.cold ? 4096 : 256 * G.threads;
    }
    G.count = 1;
    if (G.cold && (!G.k->gemm || is_verify())) { // GEMM is compute-bound; its weights stay cache-resident
        size_t mb = (size_t)(G.K / G.k->w->block) * G.k->w->bytes * G.N;
        const double total = footprint_mb > 0 ? footprint_mb * 1e6 : 1.2e9;
        G.count = (int64_t)((total + mb - 1) / mb);
        if (G.count < 1) G.count = 1;
    }
    make_data(seed);

    void *h = dlopen(G.impl, RTLD_NOW | RTLD_LOCAL);
    if (!h) { fprintf(stderr, "dlopen: %s\n", dlerror()); return 2; }
    char prep_sym[128], packed_sym[128];
    snprintf(prep_sym, sizeof prep_sym, "%s_prepare", G.k->entry);
    snprintf(packed_sym, sizeof packed_sym, "%s_packed", G.k->entry);
    G.tinit = (tinit_fn)dlsym(h, "kern_thread_init");
    prep_fn prep = (prep_fn)dlsym(h, prep_sym);
    if (G.k->gemm) {
        G.gemm = (gemm_fn)dlsym(h, G.k->entry);
        if (prep) G.gemm_packed = (gemm_fn)dlsym(h, packed_sym);
    } else {
        G.gemv = (gemv_fn)dlsym(h, G.k->entry);
        if (prep) G.gemv_packed = (gemv_fn)dlsym(h, packed_sym);
    }
    if (G.gemm_packed || G.gemv_packed) { // optional weight repacking (layout chosen by the kernel)
        G.packed = calloc(G.count, sizeof(void *));
        for (int64_t m = 0; m < G.count; m++) G.packed[m] = prep(G.W + m * G.mat_bytes, G.K, G.N);
    }
    if (G.k->gemm ? !G.gemm && !G.gemm_packed : !G.gemv && !G.gemv_packed) {
        printf("%-22s %-10s : not implemented\n", G.label ? G.label : G.impl, kname());
        return 3;
    }
    double wall, cpu, drift;
    // correctness: one call on matrix 0 with all threads, then warm-up and timing
    int saved = G.cold;
    G.cold = 0;
    memset(G.Y, 0xff, sizeof(float) * G.N * (G.k->gemm ? G.M : 1)); // NaN: unwritten outputs fail
    timed_loop(0.0, &wall, &cpu, &drift);
    G.cold = saved;
    double rel = max_rel_err();
    if (getenv("KURN_BENCH_DEBUG"))
        for (int i = 0; i < 4; i++) printf("y[%d]=%g ref=%g\n", i, G.Y[i], G.Yref[i]);
    if (G.secs > 0) timed_loop(0.3 < G.secs ? 0.3 : G.secs, &wall, &cpu, &drift);
    long n = timed_loop(G.secs, &wall, &cpu, &drift);
    report(G.label ? G.label : G.impl, n, wall, cpu, rel, drift, csv);
    stop_pool();
    if (csv) fclose(csv);
    return rel < 1e-2 ? 0 : 1;
}
