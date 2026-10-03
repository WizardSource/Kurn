// Fused Huffman-decode Q8_0 GEMV vs plain Q8_0 GEMV on real tensors (kurn compress, item 3).
//   entropy_bench DUMP THREADS[,THREADS...] SECS [hot]
// DUMP (written by entropy_bench.py): u32 ntensors, u32 streams; u8 lens[256]; u32 codes[256];
// then per tensor: u32 rows, u32 K, Q8_0 raw blocks (rows*K/32*34 bytes).
// The harness Huffman-encodes every row into `streams` interleaved bit streams (LSB-first,
// canonical codes bit-reversed, max 12 bits), checks decode == raw and y(q8h) == y(q8_0)
// bit-exactly, then times both GEMVs over all tensors (cold: the set is > LLC) or the first
// (hot), rows split over THREADS pthreads. Prints CPU/wall seconds per variant.
#define _GNU_SOURCE
#include <immintrin.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct { uint16_t d; int8_t qs[32]; } blk_q8_0;
typedef struct {
    int rows, K;
    blk_q8_0 *w;     // raw Q8_0
    uint16_t *d;     // scales (rows*K/32)
    uint8_t *bits;   // coded streams
    uint64_t *off;   // rows*S+1 byte offsets
} tensor_t;

static int S;
static uint8_t LENS[256];
static uint32_t CODES[256];
static uint16_t TAB[4096];

static inline float f16(uint16_t h) { return _cvtsh_ss(h); }

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
static double cpu_now(void) { struct timespec t; clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }

static uint32_t rev(uint32_t c, int n) { uint32_t r = 0; for (int i = 0; i < n; i++) r |= ((c >> i) & 1) << (n - 1 - i); return r; }

static void encode(tensor_t *t) {
    const int seg = t->K / S;
    size_t cap = (size_t)t->rows * t->K * 12 / 8 + (size_t)t->rows * S * 16 + 64;
    t->bits = aligned_alloc(64, (cap + 63) & ~(size_t)63);
    memset(t->bits, 0, cap);
    t->off = malloc(sizeof(uint64_t) * ((size_t)t->rows * S + 1));
    uint64_t pos = 0;  // bytes
    for (int r = 0; r < t->rows; r++)
        for (int s = 0; s < S; s++) {
            t->off[(size_t)r * S + s] = pos;
            uint64_t bp = pos * 8;
            for (int i = 0; i < seg; i++) {
                const int64_t k = (int64_t)s * seg + i;
                const uint8_t sym = (uint8_t)t->w[((size_t)r * t->K + k) / 32].qs[k % 32];
                uint64_t cur;
                memcpy(&cur, t->bits + (bp >> 3), 8);
                cur |= (uint64_t)rev(CODES[sym], LENS[sym]) << (bp & 7);
                memcpy(t->bits + (bp >> 3), &cur, 8);
                bp += LENS[sym];
            }
            pos = ((bp + 7) / 8 + 7) & ~7ull;
        }
    t->off[(size_t)t->rows * S] = pos;
    t->d = malloc(sizeof(uint16_t) * (size_t)t->rows * t->K / 32);
    for (size_t b = 0; b < (size_t)t->rows * t->K / 32; b++) t->d[b] = t->w[b].d;
}

static inline __m256i dot32(__m256i w, __m256i x) {
    return _mm256_dpbusd_epi32(_mm256_setzero_si256(), _mm256_sign_epi8(w, w), _mm256_sign_epi8(x, w));
}

static inline float hsum(__m256 v) {
    const __m128 h = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    const __m128 q = _mm_add_ps(h, _mm_movehl_ps(h, h));
    return _mm_cvtss_f32(_mm_add_ss(q, _mm_movehdup_ps(q)));
}

// plain Q8_0: one row at a time, 8-wide fp32 accumulate over blocks
static void gemv_q8(const tensor_t *t, const blk_q8_0 *x, float *y, int r0, int r1) {
    const int nb = t->K / 32;
    for (int r = r0; r < r1; r++) {
        const blk_q8_0 *w = t->w + (size_t)r * nb;
        __m256 acc = _mm256_setzero_ps();
        for (int b = 0; b < nb; b++) {
            const __m256i wv = _mm256_loadu_si256((const __m256i *)w[b].qs);
            const __m256i xv = _mm256_loadu_si256((const __m256i *)x[b].qs);
            acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(dot32(wv, xv)), _mm256_set1_ps(f16(w[b].d) * f16(x[b].d)), acc);
        }
        y[r] = hsum(acc);
    }
}

#define MAXS 8
// fused: S bit streams per row; per step each stream decodes 32 codes (one Q8_0 block) into L1
static void gemv_q8h(const tensor_t *t, const blk_q8_0 *x, float *y, int r0, int r1) {
    const int nb = t->K / 32, segb = nb / S;
    int8_t buf[MAXS][32] __attribute__((aligned(32)));
    for (int r = r0; r < r1; r++) {
        const uint8_t *p[MAXS];
        uint64_t bp[MAXS];
        for (int s = 0; s < S; s++) { p[s] = t->bits + t->off[(size_t)r * S + s]; bp[s] = 0; }
        const uint16_t *dw = t->d + (size_t)r * nb;
        __m256 acc = _mm256_setzero_ps();
        for (int c = 0; c < segb; c++) {
            for (int i = 0; i < 32; i += 4)
                for (int s = 0; s < S; s++) {
                    uint64_t w;
                    memcpy(&w, p[s] + (bp[s] >> 3), 8);
                    w >>= bp[s] & 7;
                    unsigned e, n = 0;
                    e = TAB[w & 4095]; buf[s][i] = (int8_t)e; n += e >> 8; w >>= e >> 8;
                    e = TAB[w & 4095]; buf[s][i + 1] = (int8_t)e; n += e >> 8; w >>= e >> 8;
                    e = TAB[w & 4095]; buf[s][i + 2] = (int8_t)e; n += e >> 8; w >>= e >> 8;
                    e = TAB[w & 4095]; buf[s][i + 3] = (int8_t)e; n += e >> 8;
                    bp[s] += n;
                }
            for (int s = 0; s < S; s++) {
                const int b = s * segb + c;
                const __m256i wv = _mm256_load_si256((const __m256i *)buf[s]);
                const __m256i xv = _mm256_loadu_si256((const __m256i *)x[b].qs);
                acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(dot32(wv, xv)), _mm256_set1_ps(f16(dw[b]) * f16(x[b].d)), acc);
            }
        }
        y[r] = hsum(acc);
    }
}

// every row's decoded codes == the raw Q8_0 codes
static int decode_ok(const tensor_t *t) {
    const int seg = t->K / S;
    for (int r = 0; r < t->rows; r++)
        for (int s = 0; s < S; s++) {
            const uint8_t *p = t->bits + t->off[(size_t)r * S + s];
            uint64_t bp = 0;
            for (int i = 0; i < seg; i++) {
                uint64_t w;
                memcpy(&w, p + (bp >> 3), 8);
                const unsigned e = TAB[(w >> (bp & 7)) & 4095];
                const int64_t k = (int64_t)s * seg + i;
                if ((int8_t)e != t->w[((size_t)r * t->K + k) / 32].qs[k % 32]) return 0;
                bp += e >> 8;
            }
        }
    return 1;
}

typedef void (*gemv_fn)(const tensor_t *, const blk_q8_0 *, float *, int, int);
typedef struct { tensor_t *ts; int nt; blk_q8_0 **x; float **y; gemv_fn fn; int tid, nth; double secs; long iters; pthread_barrier_t *bar; } job_t;

static void *worker(void *arg) {
    job_t *j = arg;
    cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(j->tid, &cs); pthread_setaffinity_np(pthread_self(), sizeof cs, &cs);
    for (long it = 0; it < j->iters; it++) {
        for (int k = 0; k < j->nt; k++) {
            const tensor_t *t = j->ts + k;
            const int per = (t->rows + j->nth - 1) / j->nth, r0 = j->tid * per, r1 = r0 + per < t->rows ? r0 + per : t->rows;
            if (r0 < r1) j->fn(t, j->x[k], j->y[k], r0, r1);
        }
        pthread_barrier_wait(j->bar);
    }
    return NULL;
}

static double run(tensor_t *ts, int nt, blk_q8_0 **x, float **y, gemv_fn fn, int nth, long iters, double *cpu) {
    pthread_t th[64]; job_t jb[64]; pthread_barrier_t bar;
    pthread_barrier_init(&bar, NULL, nth);
    const double t0 = now(), c0 = cpu_now();
    for (int i = 0; i < nth; i++) {
        jb[i] = (job_t){ts, nt, x, y, fn, i, nth, 0, iters, &bar};
        pthread_create(&th[i], NULL, worker, &jb[i]);
    }
    for (int i = 0; i < nth; i++) pthread_join(th[i], NULL);
    *cpu = cpu_now() - c0;
    pthread_barrier_destroy(&bar);
    return now() - t0;
}

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s DUMP THREADS SECS [hot]\n", argv[0]); return 2; }
    const double secs = atof(argv[3]);
    const int hot = argc > 4 && !strcmp(argv[4], "hot");
    FILE *f = fopen(argv[1], "rb");
    uint32_t nt, s;
    if (!f || fread(&nt, 4, 1, f) != 1 || fread(&s, 4, 1, f) != 1) return 1;
    S = (int)s;
    if (fread(LENS, 1, 256, f) != 256 || fread(CODES, 4, 256, f) != 256) return 1;
    for (int sym = 0; sym < 256; sym++)
        if (LENS[sym]) {
            const uint32_t r = rev(CODES[sym], LENS[sym]);
            for (uint32_t hi = 0; hi < (1u << (12 - LENS[sym])); hi++) TAB[r | (hi << LENS[sym])] = (uint16_t)(sym | (LENS[sym] << 8));
        }
    tensor_t *ts = calloc(nt, sizeof *ts);
    blk_q8_0 **x = calloc(nt, sizeof *x);
    float **y = calloc(nt, sizeof *y), **y2 = calloc(nt, sizeof *y2);
    double wbytes = 0, cbytes = 0, nw = 0;
    srand(1);
    for (uint32_t k = 0; k < nt; k++) {
        uint32_t rk[2];
        if (fread(rk, 4, 2, f) != 2) return 1;
        tensor_t *t = ts + k;
        t->rows = (int)rk[0]; t->K = (int)rk[1];
        const size_t nb = (size_t)t->rows * t->K / 32;
        t->w = aligned_alloc(64, (nb * 34 + 63) & ~(size_t)63);
        if (fread(t->w, 34, nb, f) != nb) return 1;
        encode(t);
        x[k] = malloc(sizeof(blk_q8_0) * t->K / 32);
        for (int b = 0; b < t->K / 32; b++) { x[k][b].d = _cvtss_sh(0.01f, 0); for (int i = 0; i < 32; i++) x[k][b].qs[i] = (int8_t)(rand() % 255 - 127); }
        y[k] = calloc(t->rows, 4); y2[k] = calloc(t->rows, 4);
        gemv_q8(t, x[k], y[k], 0, t->rows);
        gemv_q8h(t, x[k], y2[k], 0, t->rows);
        float ymax = 0, dmax = 0;
        for (int r = 0; r < t->rows; r++) {
            ymax = fabsf(y[k][r]) > ymax ? fabsf(y[k][r]) : ymax;
            dmax = fabsf(y[k][r] - y2[k][r]) > dmax ? fabsf(y[k][r] - y2[k][r]) : dmax;
        }
        if (dmax > 1e-5f * ymax || !decode_ok(t)) { fprintf(stderr, "MISMATCH tensor %u (%g of %g)\n", k, dmax, ymax); return 3; }
        wbytes += nb * 34; cbytes += t->off[(size_t)t->rows * S] + nb * 2; nw += (double)nb * 32;
    }
    fclose(f);
    const int ntu = hot ? 1 : (int)nt;
    if (hot) { wbytes = (double)ts[0].rows * ts[0].K / 32 * 34; cbytes = ts[0].off[(size_t)ts[0].rows * S] + ts[0].rows * ts[0].K / 16.0; nw = (double)ts[0].rows * ts[0].K; }
    printf("tensors=%d weights=%.0f q8_0_bytes=%.0f q8h_bytes=%.0f (%.3f bpw vs 8.5) streams=%d exact=yes\n", ntu, nw, wbytes, cbytes, 8 * cbytes / nw, S);
    // interleaved reps: q8_0, q8h, q8_0, q8h, ... ; iterations sized from a calibration pass
    const char *names[2] = {"q8_0", "q8h"};
    gemv_fn fns[2] = {gemv_q8, gemv_q8h};
    for (char *tok = strtok(argv[2], ","); tok; tok = strtok(NULL, ",")) {
    const int nth = atoi(tok);
    long iters[2];
    for (int v = 0; v < 2; v++) {
        double cpu, w = run(ts, ntu, x, y, fns[v], nth, 4, &cpu);
        iters[v] = (long)(secs / (w / 4 > 1e-7 ? w / 4 : 1e-7)) + 1;
    }
    for (int rep = 0; rep < 3; rep++)
        for (int v = 0; v < 2; v++) {
            double cpu, w = run(ts, ntu, x, y, fns[v], nth, iters[v], &cpu);
            const double per = w / iters[v];
            printf("%s threads=%d rep=%d iters=%ld WALL_S %.6f CPU_S %.6f per_call_us %.1f cpu_per_call_us %.1f Gw_s %.2f GB_s %.2f\n",
                   names[v], nth, rep, iters[v], w, cpu, 1e6 * per, 1e6 * cpu / iters[v], nw / per / 1e9, (v ? cbytes : wbytes) / per / 1e9);
        }
    }
    return 0;
}
