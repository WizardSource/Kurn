// Fused low-rank-corrected GEMV y = Q(W) x + U (V x) vs the plain base GEMV (kurn compress, item 4).
//   lowrank_bench N K RANKS THREADS SECS [hot|cold]
// Base Q(W): Q4_0 blocks (18 B / 32 weights, 4.5 bpw) against Q8_0 activations; U (N x r) and
// V (r x K) int8 with one f32 scale per row. Each thread first computes the r-vector t = V x
// for its own use (r*K MACs, no barrier), then streams its rows: y_r = base_r + s_r * (U_r . t).
// cold: NSET copies of (W, U, V) cycled so the working set exceeds the LLC. Synthetic data;
// the fused result is checked against a scalar reference (rel. err <= 1e-5).
#define _GNU_SOURCE
#include <immintrin.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct { uint16_t d; uint8_t qs[16]; } blk_q4_0;
typedef struct { uint16_t d; int8_t qs[32]; } blk_q8_0;

typedef struct { blk_q4_0 *w; int8_t *u, *v; float *us, *vs; } set_t;
static int N, K, R, NSET;
static set_t *sets;
static blk_q8_0 *xq;
static float *xf;

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
static double cpu_now(void) { struct timespec t; clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }

static inline float hsum8(__m256 v) {
    __m128 h = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    h = _mm_add_ps(h, _mm_movehl_ps(h, h));
    return _mm_cvtss_f32(_mm_add_ss(h, _mm_movehdup_ps(h)));
}

static inline float base_row(const blk_q4_0 *w, int nb) {
    const __m256i m4 = _mm256_set1_epi8(0x0F), b8 = _mm256_set1_epi8(8);
    __m256 acc = _mm256_setzero_ps();
    for (int b = 0; b < nb; b++) {
        const __m128i q = _mm_loadu_si128((const __m128i *)w[b].qs);
        __m256i wv = _mm256_and_si256(_mm256_set_m128i(_mm_srli_epi16(q, 4), q), m4);
        wv = _mm256_sub_epi8(wv, b8);
        const __m256i xv = _mm256_loadu_si256((const __m256i *)xq[b].qs);
        const __m256i p = _mm256_dpbusd_epi32(_mm256_setzero_si256(), _mm256_sign_epi8(wv, wv), _mm256_sign_epi8(xv, wv));
        acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p), _mm256_set1_ps(_cvtsh_ss(w[b].d) * _cvtsh_ss(xq[b].d)), acc);
    }
    return hsum8(acc);
}

// int8 row . float vector
static inline float dot_i8f(const int8_t *a, const float *t, int n) {
    __m512 acc = _mm512_setzero_ps();
    int i = 0;
    for (; i + 16 <= n; i += 16)
        acc = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i *)(a + i)))), _mm512_loadu_ps(t + i), acc);
    float s = _mm512_reduce_add_ps(acc);
    for (; i < n; i++) s += a[i] * t[i];
    return s;
}

static float tshared[1024];
// lowrank: 0 off, 1 every thread computes all of t = V x, 2 t computed from the shared slices
static void gemv(const set_t *S, float *y, int r0, int r1, int lowrank) {
    const int nb = K / 32;
    float tl[1024];
    const float *t = lowrank == 2 ? tshared : tl;
    if (lowrank == 1)
        for (int j = 0; j < R; j++) tl[j] = S->vs[j] * dot_i8f(S->v + (size_t)j * K, xf, K);
    for (int r = r0; r < r1; r++) {
        float acc = base_row(S->w + (size_t)r * nb, nb);
        if (lowrank) acc += S->us[r] * dot_i8f(S->u + (size_t)r * R, t, R);
        y[r] = acc;
    }
}

typedef struct { int tid, nth, lowrank; long iters; float *y; pthread_barrier_t *bar; } job_t;
static void *worker(void *arg) {
    job_t *j = arg;
    cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(j->tid, &cs); pthread_setaffinity_np(pthread_self(), sizeof cs, &cs);
    const int per = (N + j->nth - 1) / j->nth, r0 = j->tid * per, r1 = r0 + per < N ? r0 + per : N;
    const int tp = (R + j->nth - 1) / j->nth, t0 = j->tid * tp, t1 = t0 + tp < R ? t0 + tp : R;
    for (long it = 0; it < j->iters; it++) {
        const set_t *S = &sets[it % NSET];
        if (j->lowrank == 2) {
            for (int q = t0; q < t1; q++) tshared[q] = S->vs[q] * dot_i8f(S->v + (size_t)q * K, xf, K);
            pthread_barrier_wait(j->bar);
        }
        gemv(S, j->y, r0, r1, j->lowrank);
        pthread_barrier_wait(j->bar);
    }
    return NULL;
}

static double run(int nth, int lowrank, long iters, float *y, double *cpu) {
    pthread_t th[64]; job_t jb[64]; pthread_barrier_t bar;
    pthread_barrier_init(&bar, NULL, nth);
    const double t0 = now(), c0 = cpu_now();
    for (int i = 0; i < nth; i++) { jb[i] = (job_t){i, nth, lowrank, iters, y, &bar}; pthread_create(&th[i], NULL, worker, &jb[i]); }
    for (int i = 0; i < nth; i++) pthread_join(th[i], NULL);
    *cpu = cpu_now() - c0;
    pthread_barrier_destroy(&bar);
    return now() - t0;
}


int main(int argc, char **argv) {
    if (argc < 6) { fprintf(stderr, "usage: %s N K RANKS THREADS SECS [hot|cold]\n", argv[0]); return 2; }
    N = atoi(argv[1]); K = atoi(argv[2]);
    const int nth = atoi(argv[4]);
    const double secs = atof(argv[5]);
    const int cold = argc > 6 && !strcmp(argv[6], "cold");
    int ranks[16], nr = 0;
    for (char *tok = strtok(argv[3], ","); tok && nr < 16; tok = strtok(NULL, ",")) ranks[nr++] = atoi(tok);
    int rmax = 0;
    for (int i = 0; i < nr; i++) rmax = ranks[i] > rmax ? ranks[i] : rmax;
    const double wbytes = (double)N * K / 32 * 18;
    NSET = cold ? (int)(1.2e9 / wbytes) + 1 : 1;
    srand(1);
    sets = calloc(NSET, sizeof *sets);
    for (int s = 0; s < NSET; s++) {
        set_t *S = sets + s;
        S->w = malloc(sizeof(blk_q4_0) * (size_t)N * K / 32);
        for (size_t b = 0; b < (size_t)N * K / 32; b++) {
            S->w[b].d = _cvtss_sh(0.01f + 0.01f * (rand() % 7), 0);
            for (int i = 0; i < 16; i++) S->w[b].qs[i] = (uint8_t)rand();
        }
        S->u = malloc((size_t)N * (rmax ? rmax : 1)); S->v = malloc((size_t)(rmax ? rmax : 1) * K);
        S->us = malloc(4 * (size_t)N); S->vs = malloc(4 * (size_t)(rmax ? rmax : 1));
        for (size_t i = 0; i < (size_t)N * rmax; i++) S->u[i] = (int8_t)(rand() % 255 - 127);
        for (size_t i = 0; i < (size_t)rmax * K; i++) S->v[i] = (int8_t)(rand() % 255 - 127);
        for (int i = 0; i < N; i++) S->us[i] = 1e-3f * (1 + rand() % 5);
        for (int i = 0; i < rmax; i++) S->vs[i] = 1e-3f * (1 + rand() % 5);
    }
    xq = malloc(sizeof(blk_q8_0) * K / 32); xf = malloc(4 * (size_t)K);
    for (int b = 0; b < K / 32; b++) { xq[b].d = _cvtss_sh(0.02f, 0); for (int i = 0; i < 32; i++) { xq[b].qs[i] = (int8_t)(rand() % 255 - 127); xf[32 * b + i] = 0.02f * xq[b].qs[i]; } }
    float *y = calloc(N, 4);
    printf("N=%d K=%d threads=%d regime=%s sets=%d base_bytes=%.0f\n", N, K, nth, cold ? "cold" : "hot", NSET, wbytes);
    for (int ri = 0; ri < nr; ri++) {
        R = ranks[ri];
        // reference check (set 0)
        gemv(&sets[0], y, 0, N, R > 0);
        double maxe = 0, maxy = 0;
        for (int r = 0; r < N; r += 97) {
            double ref = 0;
            for (int b = 0; b < K / 32; b++) {
                int32_t s = 0;
                for (int i = 0; i < 32; i++) {
                    const uint8_t q = sets[0].w[(size_t)r * K / 32 + b].qs[i % 16];
                    s += ((i < 16 ? (q & 15) : (q >> 4)) - 8) * xq[b].qs[i];
                }
                ref += (double)_cvtsh_ss(sets[0].w[(size_t)r * K / 32 + b].d) * _cvtsh_ss(xq[b].d) * s;
            }
            if (R) {
                double c = 0;
                for (int j = 0; j < R; j++) {
                    double tj = 0;
                    for (int k = 0; k < K; k++) tj += (double)sets[0].v[(size_t)j * K + k] * xf[k];
                    c += sets[0].u[(size_t)r * R + j] * tj * sets[0].vs[j];
                }
                ref += sets[0].us[r] * c;
            }
            maxe = fmax(maxe, fabs(ref - y[r])); maxy = fmax(maxy, fabs(ref));
        }
        if (maxe > 1e-5 * maxy) { fprintf(stderr, "MISMATCH r=%d %g of %g\n", R, maxe, maxy); return 3; }
        const double extra = R ? ((double)R * (N + K) + 4.0 * (N + R)) : 0;
        for (int mode = R ? 1 : 0; mode <= (R ? 2 : 0); mode++) {
        double cpu, w = run(nth, mode, 8, y, &cpu);
        const long iters = (long)(secs / (w / 8)) + 1;
        for (int rep = 0; rep < 3; rep++) {
            w = run(nth, mode, iters, y, &cpu);
            printf("rank=%d mode=%s rep=%d iters=%ld WALL_S %.6f CPU_S %.6f per_call_us %.2f extra_bytes=%.0f extra_bpw=%.4f GB_s %.2f\n", R,
                   mode == 0 ? "base" : mode == 1 ? "redundant" : "split", rep, iters, w, cpu,
                   1e6 * w / iters, extra, 8 * extra / ((double)N * K), (wbytes + extra) / (w / iters) / 1e9);
        }
        }
    }
    return 0;
}
