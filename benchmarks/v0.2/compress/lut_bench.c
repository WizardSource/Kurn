// AQLM-style additive 2x8 codebook GEMV (2.06 bpw): decode-to-int8 + vpdpbusd vs per-chunk LUT
// (kurn compress, item 1).   lut_bench N K THREADS SECS [hot|cold]
// Block vq2x8: fp16 d; u8 idx[64] (i, j per 8 weights) per 256 weights; w8 = C0[i] + C1[j] (int8,
// |C0|,|C1| <= 63). Activations Q8_K-like (f32 d + int8 qs[256]).
//   decode: per 8 codes, gather the two 8-byte codewords (vpgatherdq x 2), add, sign trick, vpdpbusd.
//   lut:    per token, L[c][e] = C0[e] . x[8c..8c+7] and C1[e] . x[...] (int32, 2 x 256 per chunk of 8
//           activations, built once and shared by all rows); per 8 weights two dword gathers.
// Both are checked against a scalar reference (exact int32 sums -> rel err <= 1e-6).
#define _GNU_SOURCE
#include <immintrin.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct { uint16_t d; uint8_t idx[64]; } blk_vq;
typedef struct { float d; int8_t qs[256]; } blk_x;
static int8_t C0[256][8] __attribute__((aligned(64))), C1[256][8] __attribute__((aligned(64)));
static int N, K, NSET;
static blk_vq **W;
static blk_x *X;
static int32_t *LUT;  // [K/8][512]

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
static double cpu_now(void) { struct timespec t; clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }

static void gemv_decode(const blk_vq *w, float *y, int r0, int r1) {
    const int nb = K / 256;
    const __m512i Z = _mm512_setzero_si512();
    for (int r = r0; r < r1; r++) {
        __m512 acc = _mm512_setzero_ps();
        for (int b = 0; b < nb; b++) {
            const blk_vq *bk = w + (size_t)r * nb + b;
            __m512i ai = Z;
            for (int g = 0; g < 4; g++) {  // 8 codes = 64 weights
                const __m128i ij = _mm_loadu_si128((const __m128i *)(bk->idx + 16 * g));
                const __m256i i = _mm256_cvtepu8_epi32(_mm_shuffle_epi8(ij, _mm_setr_epi8(0, 2, 4, 6, 8, 10, 12, 14, -1, -1, -1, -1, -1, -1, -1, -1)));
                const __m256i j = _mm256_cvtepu8_epi32(_mm_shuffle_epi8(ij, _mm_setr_epi8(1, 3, 5, 7, 9, 11, 13, 15, -1, -1, -1, -1, -1, -1, -1, -1)));
                const __m512i wv = _mm512_add_epi8(_mm512_i32gather_epi64(i, (const void *)C0, 8), _mm512_i32gather_epi64(j, (const void *)C1, 8));
                const __m512i xv = _mm512_loadu_si512(X[b].qs + 64 * g);
                const __mmask64 neg = _mm512_movepi8_mask(wv);
                ai = _mm512_dpbusd_epi32(ai, _mm512_abs_epi8(wv), _mm512_mask_sub_epi8(xv, neg, Z, xv));
            }
            acc = _mm512_fmadd_ps(_mm512_cvtepi32_ps(ai), _mm512_set1_ps(_cvtsh_ss(bk->d) * X[b].d), acc);
        }
        y[r] = _mm512_reduce_add_ps(acc);
    }
}

// codebooks as biased-unsigned dword planes (bytes 0-3 / 4-7 of each entry, +64): 4 x 256 u32
static uint32_t PL[4][256] __attribute__((aligned(64)));
static void init_planes(void) {
    for (int e = 0; e < 256; e++)
        for (int h = 0; h < 2; h++) {
            uint32_t a = 0, b = 0;
            for (int i = 0; i < 4; i++) { a |= (uint32_t)(uint8_t)(C0[e][4 * h + i] + 64) << (8 * i); b |= (uint32_t)(uint8_t)(C1[e][4 * h + i] + 64) << (8 * i); }
            PL[h][e] = a; PL[2 + h][e] = b;
        }
}
// L[c][e] = C0[e] . x_c, L[c][256 + e] = C1[e] . x_c via vpdpbusd on the planes, minus 64 * sum(x_c)
static void build_lut(int c0, int c1) {  // chunks [c0, c1)
    for (int c = c0; c < c1; c++) {
        const int8_t *x = X[c / 32].qs + 8 * (c % 32);
        int32_t xl, xh;
        memcpy(&xl, x, 4); memcpy(&xh, x + 4, 4);
        int sx = 0;
        for (int i = 0; i < 8; i++) sx += x[i];
        const __m512i bl = _mm512_set1_epi32(xl), bh = _mm512_set1_epi32(xh), corr = _mm512_set1_epi32(-64 * sx);
        int32_t *L = LUT + (size_t)c * 512;
        for (int cb = 0; cb < 2; cb++)
            for (int e = 0; e < 256; e += 16) {
                __m512i a = _mm512_dpbusd_epi32(corr, _mm512_load_si512(&PL[2 * cb][e]), bl);
                a = _mm512_dpbusd_epi32(a, _mm512_load_si512(&PL[2 * cb + 1][e]), bh);
                _mm512_store_si512(L + 256 * cb + e, a);
            }
    }
}

static void gemv_lut(const blk_vq *w, float *y, int r0, int r1) {
    const int nb = K / 256;
    const __m512i OFF = _mm512_set1_epi32(256), STEP = _mm512_set_epi32(15 * 512, 14 * 512, 13 * 512, 12 * 512, 11 * 512, 10 * 512, 9 * 512, 8 * 512,
                                                                         7 * 512, 6 * 512, 5 * 512, 4 * 512, 3 * 512, 2 * 512, 512, 0);
    for (int r = r0; r < r1; r++) {
        __m512 acc = _mm512_setzero_ps();
        for (int b = 0; b < nb; b++) {
            const blk_vq *bk = w + (size_t)r * nb + b;
            __m512i s = _mm512_setzero_si512();
            for (int h = 0; h < 2; h++) {  // 16 codes
                const __m256i ij = _mm256_loadu_si256((const __m256i *)(bk->idx + 32 * h));
                const __m512i ij16 = _mm512_cvtepu8_epi16(ij);
                const __m512i i = _mm512_and_si512(ij16, _mm512_set1_epi32(0xFFFF)), j = _mm512_srli_epi32(ij16, 16);
                const int32_t *L = LUT + (size_t)(32 * b + 16 * h) * 512;
                s = _mm512_add_epi32(s, _mm512_i32gather_epi32(_mm512_add_epi32(i, STEP), L, 4));
                s = _mm512_add_epi32(s, _mm512_i32gather_epi32(_mm512_add_epi32(_mm512_add_epi32(j, OFF), STEP), L, 4));
            }
            acc = _mm512_fmadd_ps(_mm512_cvtepi32_ps(s), _mm512_set1_ps(_cvtsh_ss(bk->d) * X[b].d), acc);
        }
        y[r] = _mm512_reduce_add_ps(acc);
    }
}

typedef struct { int tid, nth, mode; long iters; float *y; pthread_barrier_t *bar; } job_t;
static void *worker(void *arg) {
    job_t *j = arg;
    cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(j->tid, &cs); pthread_setaffinity_np(pthread_self(), sizeof cs, &cs);
    const int per = (N + j->nth - 1) / j->nth, r0 = j->tid * per, r1 = r0 + per < N ? r0 + per : N;
    const int nc = K / 8, cp = (nc + j->nth - 1) / j->nth, c0 = j->tid * cp, c1 = c0 + cp < nc ? c0 + cp : nc;
    for (long it = 0; it < j->iters; it++) {
        const blk_vq *w = W[it % NSET];
        if (j->mode) {  // LUT build split over threads, then rows
            build_lut(c0, c1);
            pthread_barrier_wait(j->bar);
            gemv_lut(w, j->y, r0, r1);
        } else gemv_decode(w, j->y, r0, r1);
        pthread_barrier_wait(j->bar);
    }
    return NULL;
}

static double run(int nth, int mode, long iters, float *y, double *cpu) {
    pthread_t th[64]; job_t jb[64]; pthread_barrier_t bar;
    pthread_barrier_init(&bar, NULL, nth);
    const double t0 = now(), c0 = cpu_now();
    for (int i = 0; i < nth; i++) { jb[i] = (job_t){i, nth, mode, iters, y, &bar}; pthread_create(&th[i], NULL, worker, &jb[i]); }
    for (int i = 0; i < nth; i++) pthread_join(th[i], NULL);
    *cpu = cpu_now() - c0;
    pthread_barrier_destroy(&bar);
    return now() - t0;
}

int main(int argc, char **argv) {
    if (argc < 5) { fprintf(stderr, "usage: %s N K THREADS SECS [hot|cold]\n", argv[0]); return 2; }
    N = atoi(argv[1]); K = atoi(argv[2]);
    const int nth = atoi(argv[3]);
    const double secs = atof(argv[4]);
    const int cold = argc > 5 && !strcmp(argv[5], "cold");
    const double wbytes = (double)N * K / 256 * 66;
    NSET = cold ? (int)(1.2e9 / wbytes) + 1 : 1;
    srand(1);
    for (int e = 0; e < 256; e++) for (int i = 0; i < 8; i++) { C0[e][i] = (int8_t)(rand() % 127 - 63); C1[e][i] = (int8_t)(rand() % 127 - 63); }
    init_planes();
    W = calloc(NSET, sizeof *W);
    for (int s = 0; s < NSET; s++) {
        W[s] = malloc(sizeof(blk_vq) * (size_t)N * K / 256);
        for (size_t b = 0; b < (size_t)N * K / 256; b++) {
            W[s][b].d = _cvtss_sh(1e-3f * (1 + rand() % 9), 0);
            for (int i = 0; i < 64; i++) W[s][b].idx[i] = (uint8_t)rand();
        }
    }
    X = malloc(sizeof(blk_x) * K / 256);
    for (int b = 0; b < K / 256; b++) { X[b].d = 0.01f; for (int i = 0; i < 256; i++) X[b].qs[i] = (int8_t)(rand() % 255 - 127); }
    LUT = aligned_alloc(64, sizeof(int32_t) * (size_t)K / 8 * 512);
    float *y = calloc(N, 4), *y2 = calloc(N, 4);
    gemv_decode(W[0], y, 0, N);
    build_lut(0, K / 8);
    gemv_lut(W[0], y2, 0, N);
    for (int r = 0; r < N; r += 61) {
        double ref = 0;
        for (int b = 0; b < K / 256; b++) {
            int64_t s = 0;
            for (int v = 0; v < 32; v++)
                for (int i = 0; i < 8; i++) s += (int64_t)(C0[W[0][(size_t)r * K / 256 + b].idx[2 * v]][i] + C1[W[0][(size_t)r * K / 256 + b].idx[2 * v + 1]][i]) * X[b].qs[8 * v + i];
            ref += (double)_cvtsh_ss(W[0][(size_t)r * K / 256 + b].d) * X[b].d * s;
        }
        if (fabs(ref - y[r]) > 1e-5 * fabs(ref) + 1e-6 || fabs(ref - y2[r]) > 1e-5 * fabs(ref) + 1e-6) {
            fprintf(stderr, "MISMATCH row %d: ref %g decode %g lut %g\n", r, ref, y[r], y2[r]); return 3;
        }
    }
    printf("N=%d K=%d threads=%d regime=%s sets=%d bytes=%.0f exact=yes\n", N, K, nth, cold ? "cold" : "hot", NSET, wbytes);
    const char *names[2] = {"vq2x8_decode", "vq2x8_lut"};
    long iters[2];
    for (int m = 0; m < 2; m++) { double cpu, w = run(nth, m, 8, y, &cpu); iters[m] = (long)(secs / (w / 8)) + 1; }
    for (int rep = 0; rep < 3; rep++)
        for (int m = 0; m < 2; m++) {
            double cpu, w = run(nth, m, iters[m], y, &cpu);
            printf("%s rep=%d iters=%ld WALL_S %.6f CPU_S %.6f per_call_us %.2f Gw_s %.2f GB_s %.2f\n", names[m], rep, iters[m], w, cpu,
                   1e6 * w / iters[m], (double)N * K / (w / iters[m]) / 1e9, wbytes / (w / iters[m]) / 1e9);
        }
    return 0;
}
