// thread-level AMX-BF16 blocking: panel-outer (current kurn order) vs K-chunk-outer with partial sums in Y.
// Each thread: NP panels of P rows (B: NP x P x K bf16, pre-packed), M tokens of A (shared), K.
#include <immintrin.h>
#include <x86intrin.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <sys/syscall.h>
#include <unistd.h>
static int64_t M = 512, P = 128, NP = 12, K = 4096, KC = 1024; static int kouter = 0;
static uint16_t *A, *Ap; static int packed_a = 0; typedef struct { uint16_t *B; float *Y; double mpc; } arg_t;
static void cfg(void){ struct __attribute__((aligned(64))) {uint8_t p,s,r[14];uint16_t c[16];uint8_t rows[16];} c; memset(&c,0,64); c.p=1; for(int t=0;t<8;t++){c.rows[t]=16;c.c[t]=64;} _tile_loadconfig(&c);}
// one 32 x 32 output block (tokens mg*32.., rows of pair j of panel p) over k-steps [t0, t1)
static inline void block(const uint16_t *B, float *Y, int64_t p, int64_t mg, int64_t j, int64_t t0, int64_t t1, int load) {
    const int64_t nt = K / 32;
    const uint16_t *a0 = A + mg * 32 * K, *a1 = a0 + 16 * K;  // row-major A like kurn
    long as = K * 2;
    if (packed_a) { a0 = Ap + mg * nt * 1024; a1 = a0 + 512; as = 64; }  // [mg][t][32 rows][32]: tile t at a0 + t * 1024
    const uint16_t *b0 = B + ((p * (P / 16)) + 2 * j) * nt * 512, *b1 = b0 + nt * 512;
    float *y = Y + (mg * 32) * (NP * P) + p * P + j * 32;  // Y row stride NP*P
    long ys = NP * P * 4;
    int64_t o1 = 16, o2 = 16 * NP * P, o3 = 16 * NP * P + 16;
    if (kouter == 3) { y = Y + ((p * (M / 32) + mg) * (P / 32) + j) * 1024; ys = 64; o1 = 256; o2 = 512; o3 = 768; }
    if (load) { _tile_loadd(0, y, ys); _tile_loadd(1, y + o1, ys); _tile_loadd(2, y + o2, ys); _tile_loadd(3, y + o3, ys); }
    else { _tile_zero(0); _tile_zero(1); _tile_zero(2); _tile_zero(3); }
    for (int64_t t = t0; t < t1; t++) {
        const int64_t ao = packed_a ? t * 1024 : t * 32;
        _tile_loadd(4, a0 + ao, as); _tile_loadd(6, b0 + t * 512, 64); _tile_loadd(5, a1 + ao, as); _tile_loadd(7, b1 + t * 512, 64);
        _tile_dpbf16ps(0, 4, 6); _tile_dpbf16ps(1, 4, 7); _tile_dpbf16ps(2, 5, 6); _tile_dpbf16ps(3, 5, 7);
    }
    _tile_stored(0, y, ys); _tile_stored(1, y + o1, ys); _tile_stored(2, y + o2, ys); _tile_stored(3, y + o3, ys);
}
static void run(uint16_t *B, float *Y) {
    const int64_t nt = K / 32, ntc = KC / 32, nmg = M / 32, npair = P / 32;
    if (kouter >= 2) {  // panel, K chunk, pair, token group: the B pair of a short K chunk stays in L1 across token groups
        for (int64_t p = 0; p < NP; p++) for (int64_t c = 0; c < nt; c += ntc) for (int64_t j = 0; j < npair; j++) for (int64_t mg = 0; mg < nmg; mg++)
            block(B, Y, p, mg, j, c, c + ntc, c > 0);
        return;
    }
    if (!kouter) {
        for (int64_t p = 0; p < NP; p++) for (int64_t c = 0; c < nt; c += ntc) for (int64_t mg = 0; mg < nmg; mg++) for (int64_t j = 0; j < npair; j++)
            block(B, Y, p, mg, j, c, c + ntc, c > 0);
    } else {
        for (int64_t c = 0; c < nt; c += ntc) for (int64_t p = 0; p < NP; p++) for (int64_t mg = 0; mg < nmg; mg++) for (int64_t j = 0; j < npair; j++)
            block(B, Y, p, mg, j, c, c + ntc, c > 0);
    }
}
static void *th(void *v) { arg_t *a = v; syscall(SYS_arch_prctl, 0x1023, 18); cfg(); double best = 1e30;
    for (int r = 0; r < 4; r++) { uint64_t t0 = __rdtsc(); run(a->B, a->Y); uint64_t t1 = __rdtsc(); if (t1 - t0 < best) best = t1 - t0; }
    a->mpc = (double)M * NP * P * K / best; return 0; }
int main(int c, char **v) { int nth = 8; kouter = c > 1 ? atoi(v[1]) : 0; if (c > 2) KC = atoll(v[2]); if (c > 3) P = atoll(v[3]); if (c > 4) NP = atoll(v[4]);
    A = aligned_alloc(64, M * K * 2); for (int64_t i = 0; i < M * K; i++) A[i] = 0x3f80;
    Ap = aligned_alloc(64, M * K * 2); for (int64_t i = 0; i < M * K; i++) Ap[i] = 0x3f80; packed_a = getenv("PACKED_A") ? atoi(getenv("PACKED_A")) : 0;
    pthread_t t[8]; arg_t a[8];
    for (int i = 0; i < nth; i++) { a[i].B = aligned_alloc(64, NP * P * K * 2); for (int64_t x = 0; x < NP * P * K; x++) a[i].B[x] = 0x3f80; a[i].Y = aligned_alloc(64, M * NP * P * 4); }
    for (int i = 0; i < nth; i++) pthread_create(&t[i], 0, th, &a[i]); double s = 0; for (int i = 0; i < nth; i++) { pthread_join(t[i], 0); s += a[i].mpc; }
    printf("kouter=%d KC=%ld P=%ld NP=%ld: %.0f MAC/TSC-cycle/core\n", kouter, KC, P, NP, s / nth);
}
