// Kernel-level gain of the fused GEMV epilogues (kurn.epilogue) on one core, at the shapes one
// engine thread runs per layer, from DRAM (768 MB of copies), L3 (LAYERS copies) and L2 (one copy);
// "fused +x%" = the fused kernel is x% faster:
//   swiglu: kq8e_swiglu_q8 (GEMV + SiLU*up + Q8_0 in registers)  vs  kq8e_store -> kq8e_swiglu -> kq8e_quantize
//   axpy:   kq8e_axpy (GEMV + scale + accumulate)                vs  kq8e_store -> y += s * tmp
// Build (see measure.sh epi): gcc -O3 -march=native epibench.c epilogue_kernels.c -I model/ -I data/
//   epibench NBE FF_ROWS LAYERS REPS     e.g. Qwen3-1.7B, 8 threads: 64 768 28 30
#define _GNU_SOURCE
#include "kq8e.h"
#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec + ts.tv_nsec * 1e-9; }
static int cmpd(const void *a, const void *b) { double x = *(const double *)a, y = *(const double *)b; return (x > y) - (x < y); }

int main(int argc, char **argv) {
    const int nbe = argc > 1 ? atoi(argv[1]) : 64, ff = argc > 2 ? atoi(argv[2]) : 768;
    const int layers = argc > 3 ? atoi(argv[3]) : 28, reps = argc > 4 ? atoi(argv[4]) : 30;
    const size_t BLK = kq8e_blk_bytes();
    const int ng_gu = 2 * ff / 16, nfb = ff / 32, n_out = nbe * 32, ng_dn = n_out / 16;
    const size_t gu_sz = (size_t)ng_gu * nbe * BLK, dn_sz = (size_t)ng_dn * nfb * BLK;
    block_q8_0 *tmp = malloc(sizeof(block_q8_0) * (size_t)(2 * ff > n_out ? 2 * ff : n_out) * (nbe > nfb ? nbe : nfb));
    const int total = (int)((768ull << 20) / (gu_sz + dn_sz)) + 1;  // > L3 (320 MB here)
    uint8_t *GU = aligned_alloc(64, gu_sz * total), *DN = aligned_alloc(64, dn_sz * total);
    srand(1);
    for (int l = 0; l < total; l++) {
        for (size_t i = 0; i < (size_t)2 * ff * nbe; i++) {
            tmp[i].d = _cvtss_sh(0.01f + 0.001f * (rand() % 7), 0);
            for (int j = 0; j < 32; j++) tmp[i].qs[j] = (int8_t)(rand() % 255 - 127);
        }
        kq8e_pack(GU + l * gu_sz, tmp, nbe, 2 * ff);
        for (size_t i = 0; i < (size_t)n_out * nfb; i++) {
            tmp[i].d = _cvtss_sh(0.01f + 0.001f * (rand() % 7), 0);
            for (int j = 0; j < 32; j++) tmp[i].qs[j] = (int8_t)(rand() % 255 - 127);
        }
        kq8e_pack(DN + l * dn_sz, tmp, nfb, n_out);
    }
    float *x = malloc(sizeof(float) * n_out), *gu = aligned_alloc(64, sizeof(float) * 2 * ff), *act = aligned_alloc(64, sizeof(float) * ff);
    float *y = aligned_alloc(64, sizeof(float) * n_out), *t2 = aligned_alloc(64, sizeof(float) * n_out);
    for (int i = 0; i < n_out; i++) x[i] = (float)(rand() % 2001 - 1000) / 500.0f;
    block_q8_0 *xq = aligned_alloc(64, sizeof(block_q8_0) * nbe), *fq = aligned_alloc(64, sizeof(block_q8_0) * nfb);
    block_q8_0 *fq2 = aligned_alloc(64, sizeof(block_q8_0) * nfb);
    static kq8e_act ax, af;
    kq8e_quantize(x, xq, n_out);
    kq8e_prep(xq, nbe, &ax);

    // regimes: DRAM (all copies, > L3), L3 (model's layer count of copies), L2 (one copy, warmed per variant)
    const char *rname[3] = {"dram", "l3  ", "l2  "};
    const int rlay[3] = {total, layers < total ? layers : total, 1};
    double t[4][256];
    const int R = reps > 256 ? 256 : reps;
    for (int reg = 0; reg < 3; reg++) {
        const int L = rlay[reg];
        for (int r = 0; r < R; r++) {
            for (int v = 0; v < 4; v++) {  // interleaved variants; in the L2 regime each runs once untimed first
                for (int pass = reg == 2 ? 0 : 1; pass < 2; pass++) {
                    double t0 = now();
                    for (int l = 0; l < L; l++) {
                        const uint8_t *gw = GU + (size_t)l * gu_sz, *dw = DN + (size_t)l * dn_sz;
                        if (v == 0) kq8e_swiglu_q8(gw, nbe, &ax, 0, nbe, 0, ng_gu, fq, NULL);
                        else if (v == 1) {
                            kq8e_store(gw, nbe, &ax, 0, nbe, 0, ng_gu, gu);
                            for (int j = 0; j < 2 * nfb; j++) kq8e_swiglu(gu + 32 * j, gu + 32 * j + 16, act + 16 * j, 16);
                            kq8e_quantize(act, fq2, ff);
                        } else if (v == 2) kq8e_axpy(dw, nfb, &af, 0, nfb, 0, ng_dn, y, 0.37f);
                        else {
                            kq8e_store(dw, nfb, &af, 0, nfb, 0, ng_dn, t2);
                            for (int i = 0; i < n_out; i++) y[i] += 0.37f * t2[i];
                        }
                    }
                    t[v][r] = (now() - t0) / L;
                }
                if (v == 1 && memcmp(fq, fq2, sizeof(block_q8_0) * nfb)) { fprintf(stderr, "fused != unfused\n"); return 1; }
            }
        }
        for (int k = 0; k < 4; k++) qsort(t[k], R, sizeof(double), cmpd);
        const double m0 = t[0][R / 2], m1 = t[1][R / 2], m2 = t[2][R / 2], m3 = t[3][R / 2];
        printf("%s nbe=%d ff=%d copies=%d: swiglu fused %.2f us (%.1f GB/s) unfused %.2f us -> fused %+.1f%% | "
               "axpy fused %.2f us (%.1f GB/s) store+add %.2f us -> fused %+.1f%%\n",
               rname[reg], nbe, ff, L, m0 * 1e6, gu_sz / m0 / 1e9, m1 * 1e6, 100 * (m1 / m0 - 1), m2 * 1e6,
               dn_sz / m2 / 1e9, m3 * 1e6, 100 * (m3 / m2 - 1));
    }
    return 0;
}
