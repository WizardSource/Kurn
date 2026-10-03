// AMX 4-bit experiment (WS-B fourbit, item 4): Q4_0 weights x M Q8_0 activation rows
// (M = 1..16: decode / multi-token verify / small batch), L2-hot, one thread.
//
// Weights stay 4-bit in memory in the i16 interleaved order of kurn's mask layout (one 64-byte
// vector = 16 rows x 8 values: low nibbles k 0..3, high nibbles k 4..7). Per 32-value block,
// 4 vector loads are unpacked (AND, shift+AND) into an 8 x 64-byte u8 B tile (VNNI order:
// row k/4, column n*4 + k%4), multiplied by the M x 32 int8 activation tile with tdpbsud
// (signed A x unsigned B) into an M x 16 int32 C tile, and the M useful C rows are scaled:
// y[m][n] += d_w[n] * d_x[m] * (C[m][n] - 8 * sum(x[m])).
// Checked against a scalar reference; prints us/call and effective GMAC/s.
//   gcc -O3 -march=native amx4.c -o amx4 && ./amx4 [M] [secs]
#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#define K 4096
#define N 256
#define NB (K / 32)

typedef struct __attribute__((packed, aligned(64))) {
    uint8_t palette_id, start_row, reserved[14];
    uint16_t colsb[16];
    uint8_t rows[16];
} tilecfg;

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + 1e-9 * t.tv_nsec;
}
static uint64_t rng = 88172645463325252ull;
static uint64_t nxt(void) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return rng; }

static uint8_t W[N][NB][16];  // q4_0 codes (ggml order: low nibble = value j, high = value j + 16)
static float DW[N][NB];
static int8_t X[16][K];
static float DX[16][NB];
static int32_t SX[16][NB];
// repacked: per 16-row group g and block b, 4 x 64 bytes of nibbles, and 16 fp32 scales
static uint8_t RW[N / 16][NB][256] __attribute__((aligned(64)));
static float RD[N / 16][NB][16] __attribute__((aligned(64)));
static int8_t XA[NB][16][32] __attribute__((aligned(64)));  // activation tile per block (M rows used)

static int code(int n, int b, int v) { return v < 16 ? W[n][b][v] & 15 : W[n][b][v - 16] >> 4; }

static void repack(void) {
    for (int g = 0; g < N / 16; g++)
        for (int b = 0; b < NB; b++) {
            memset(RW[g][b], 0, 256);
            for (int r = 0; r < 16; r++) {
                RD[g][b][r] = DW[16 * g + r][b];
                for (int v = 0; v < 32; v++) {  // value v -> k-group kk = v / 4, vector kk / 2, nibble kk & 1
                    const int kk = v / 4, j = v % 4;
                    RW[g][b][(kk / 2) * 64 + r * 4 + j] |= (uint8_t)(code(16 * g + r, b, v) << (4 * (kk & 1)));
                }
            }
        }
}

static void gemm_amx(int M, float *Y) {
    tilecfg c;
    memset(&c, 0, sizeof c);
    c.palette_id = 1;
    for (int t = 0; t < 2; t++) { c.rows[t] = M; c.colsb[t] = 64; }    // C0, C1: M x 16 int32 (rows must match A)
    for (int t = 2; t < 4; t++) { c.rows[t] = M; c.colsb[t] = 32; }    // A0, A1: M x 32 int8
    for (int t = 4; t < 6; t++) { c.rows[t] = 8; c.colsb[t] = 64; }    // B0, B1: 8 x 64 u8 (K = 32)
    _tile_loadconfig(&c);
    const __m512i m4 = _mm512_set1_epi8(15);
    uint8_t bt[2][512] __attribute__((aligned(64)));
    int32_t cb[2][16][16] __attribute__((aligned(64)));
    for (int g = 0; g < N / 16; g++) {
        __m512 acc[16];
        for (int m = 0; m < M; m++) acc[m] = _mm512_setzero_ps();
        for (int b = 0; b < NB; b += 2) {
            for (int h = 0; h < 2; h++) {
                const uint8_t *src = RW[g][b + h];
                for (int t = 0; t < 4; t++) {
                    const __m512i v = _mm512_load_si512(src + 64 * t);
                    _mm512_store_si512(bt[h] + 128 * t, _mm512_and_si512(v, m4));
                    _mm512_store_si512(bt[h] + 128 * t + 64, _mm512_and_si512(_mm512_srli_epi16(v, 4), m4));
                }
            }
            _tile_zero(0);
            _tile_zero(1);
            _tile_loadd(2, XA[b], 32);
            _tile_loadd(3, XA[b + 1], 32);
            _tile_loadd(4, bt[0], 64);
            _tile_loadd(5, bt[1], 64);
            _tile_dpbsud(0, 2, 4);
            _tile_dpbsud(1, 3, 5);
            _tile_stored(0, cb[0], 64);
            _tile_stored(1, cb[1], 64);
            for (int h = 0; h < 2; h++) {
                const __m512 dw = _mm512_load_ps(RD[g][b + h]);
                for (int m = 0; m < M; m++) {
                    const __m512i s = _mm512_sub_epi32(_mm512_load_si512(cb[h][m]), _mm512_set1_epi32(8 * SX[m][b + h]));
                    acc[m] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(s), _mm512_mul_ps(dw, _mm512_set1_ps(DX[m][b + h])), acc[m]);
                }
            }
        }
        for (int m = 0; m < M; m++) _mm512_storeu_ps(Y + m * N + 16 * g, acc[m]);
    }
    _tile_release();
}

// same data, AVX-512 VNNI: per vector 3 ALU (and, shift, and) + 2 vpdpbusd per activation row
static void gemm_vnni(int M, float *Y) {
    const __m512i m4 = _mm512_set1_epi8(15);
    for (int g = 0; g < N / 16; g++) {
        __m512 acc[16];
        for (int m = 0; m < M; m++) acc[m] = _mm512_setzero_ps();
        for (int b = 0; b < NB; b++) {
            __m512i s[16];
            for (int m = 0; m < M; m++) s[m] = _mm512_set1_epi32(-8 * SX[m][b]);
            for (int t = 0; t < 4; t++) {
                const __m512i v = _mm512_load_si512(RW[g][b] + 64 * t);
                const __m512i lo = _mm512_and_si512(v, m4), hi = _mm512_and_si512(_mm512_srli_epi16(v, 4), m4);
                for (int m = 0; m < M; m++) {
                    int32_t a0, a1;
                    memcpy(&a0, X[m] + 32 * b + 8 * t, 4);
                    memcpy(&a1, X[m] + 32 * b + 8 * t + 4, 4);
                    s[m] = _mm512_dpbusd_epi32(s[m], lo, _mm512_set1_epi32(a0));
                    s[m] = _mm512_dpbusd_epi32(s[m], hi, _mm512_set1_epi32(a1));
                }
            }
            const __m512 dw = _mm512_load_ps(RD[g][b]);
            for (int m = 0; m < M; m++)
                acc[m] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(s[m]), _mm512_mul_ps(dw, _mm512_set1_ps(DX[m][b])), acc[m]);
        }
        for (int m = 0; m < M; m++) _mm512_storeu_ps(Y + m * N + 16 * g, acc[m]);
    }
}

static double check(int M, const float *Y) {
    double emax = 0, mag = 0;
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            double r = 0;
            for (int b = 0; b < NB; b++) {
                int32_t s = 0;
                for (int v = 0; v < 32; v++) s += (code(n, b, v) - 8) * X[m][32 * b + v];
                r += (double)DW[n][b] * DX[m][b] * s;
            }
            emax = fmax(emax, fabs(r - Y[m * N + n]));
            mag = fmax(mag, fabs(r));
        }
    return emax / mag;
}

int main(int argc, char **argv) {
    const int M = argc > 1 ? atoi(argv[1]) : 4;
    const double secs = argc > 2 ? atof(argv[2]) : 1.0;
    if (M < 1 || M > 16) return 2;
    if (syscall(SYS_arch_prctl, 0x1023, 18)) { perror("AMX permission"); return 1; }
    for (int n = 0; n < N; n++)
        for (int b = 0; b < NB; b++) {
            for (int i = 0; i < 16; i++) W[n][b][i] = (uint8_t)nxt();
            DW[n][b] = 0.01f + (nxt() % 1000) * 1e-5f;
        }
    for (int m = 0; m < 16; m++) {
        for (int k = 0; k < K; k++) X[m][k] = (int8_t)(nxt() % 255 - 127);
        for (int b = 0; b < NB; b++) {
            DX[m][b] = 0.001f + (nxt() % 1000) * 1e-6f;
            SX[m][b] = 0;
            for (int v = 0; v < 32; v++) SX[m][b] += X[m][32 * b + v];
        }
    }
    memset(XA, 0, sizeof XA);
    for (int b = 0; b < NB; b++)
        for (int m = 0; m < M; m++) memcpy(XA[b][m], X[m] + 32 * b, 32);
    repack();
    float *Y = aligned_alloc(64, sizeof(float) * 16 * N);
    void (*fns[2])(int, float *) = {gemm_amx, gemm_vnni};
    const char *names[2] = {"amx", "vnni"};
    for (int f = 0; f < 2; f++) {
        fns[f](M, Y);
        const double err = check(M, Y);
        long calls = 0;
        const double t0 = now();
        double t;
        do { fns[f](M, Y); calls++; } while ((t = now() - t0) < secs);
        const double us = 1e6 * t / calls;
        printf("%-5s M=%2d  %8.2f us/call  %7.1f GMAC/s (useful)  %6.2f GB/s  relerr %.1e %s\n", names[f], M, us,
               (double)N * K * M / us * 1e-3, (double)N * K * 0.5625 / us * 1e-3, err, err < 1e-5 ? "ok" : "FAIL");
    }
    return 0;
}
