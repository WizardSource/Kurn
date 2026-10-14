#define _GNU_SOURCE
// f32 -> bf16 activation conversion as in mul_mat_bf16: thread t converts rows [64t, 64t+64) of a 512 x K source
// (written by the same thread before, as rms_norm does) into a shared A buffer that every thread then reads.
#include <immintrin.h>
#include <x86intrin.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#define NT 8
static int64_t M = 512, K = 4096; static float *src; static uint16_t *A, *AR[4]; static pthread_barrier_t bar; static uint64_t tcv[NT], tsum[NT];
static int mode;
static void *th(void *p) {
    int t = (int)(intptr_t)p; cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(t, &cs); pthread_setaffinity_np(pthread_self(), sizeof cs, &cs);
    for (int it = 0; it < 30; it++) {
        // producer: own rows of src written (rms_norm)
        for (int64_t m = M * t / NT; m < M * (t + 1) / NT; m++) for (int64_t k = 0; k < K; k += 16) _mm512_storeu_ps(src + m * K + k, _mm512_set1_ps(1.0f + it + k));
        // consumers of the previous A: every thread reads all of A (tile loads)
        uint16_t *Ar = mode == 3 ? AR[(it + 3) % 4] : A, *Aw = mode == 3 ? AR[it % 4] : A;
        if (mode >= 1) { __m512i acc = _mm512_setzero_si512(); for (int64_t i = 0; i < M * K / 32; i++) acc = _mm512_add_epi32(acc, _mm512_loadu_si512(Ar + i * 32)); tsum[t] += _mm512_reduce_add_epi32(acc); if (mode == 4) for (int64_t i = 0; i < M * K * 2; i += 64) _cldemote((const char *)Ar + i); }
        pthread_barrier_wait(&bar);
        uint64_t t0 = __rdtsc();
        for (int64_t m = M * t / NT; m < M * (t + 1) / NT; m++) {
            const float *s = src + m * K; uint16_t *d = Aw + m * K;
            for (int64_t k = 0; k < K; k += 32) { __m512i b = (__m512i)_mm512_cvtne2ps_pbh(_mm512_loadu_ps(s + k + 16), _mm512_loadu_ps(s + k)); if (mode == 2) _mm512_stream_si512((__m512i *)(d + k), b); else _mm512_storeu_si512((__m512i *)(d + k), b); }
        }
        if (it >= 10) tcv[t] += __rdtsc() - t0;
        pthread_barrier_wait(&bar);
    }
    return 0;
}
int main(int c, char **v) {
    mode = c > 1 ? atoi(v[1]) : 0; if (c > 2) K = atoll(v[2]);
    src = aligned_alloc(64, M * K * 4); A = aligned_alloc(64, M * K * 2); memset(A, 0, M * K * 2); for (int r = 0; r < 4; r++) { AR[r] = aligned_alloc(64, M * K * 2); memset(AR[r], 0, M * K * 2); }
    pthread_barrier_init(&bar, 0, NT); pthread_t ts[NT];
    for (int t = 0; t < NT; t++) pthread_create(&ts[t], 0, th, (void *)(intptr_t)t);
    uint64_t s = 0; for (int t = 0; t < NT; t++) { pthread_join(ts[t], 0); s += tcv[t]; }
    printf("mode=%d K=%ld: convert %.1f kcycles per thread per op (%.2f bytes read/cycle)\n", mode, K, s / NT / 20 / 1e3, (double)M / NT * K * 4 / (s / NT / 20.0));
}
