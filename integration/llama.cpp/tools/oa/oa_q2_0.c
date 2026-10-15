// Weighted Q2_0 quantizer for oaquant.py (ctypes): block of 64, fp16 d, codes {-1, 0, 1, 2} * d.
// Per block: candidate scales s * amax / 2 for both signs of d, each refit by weighted least squares for its
// codes; the lowest weighted error wins. imp: per-column importance (E[x^2], 0 on salient columns).
//   gcc -O3 -march=native -fopenmp -shared -fPIC oa_q2_0.c -o liboa_q2_0.so
#include <math.h>
#include <stdint.h>
#include <string.h>
#include <immintrin.h>

static inline int code(float w, float d) {
    int q = (int)rintf(w / d);
    return q < -1 ? -1 : q > 2 ? 2 : q;
}

void oa_quant_q2_0(const float *W, const float *imp, uint8_t *out, int64_t N, int64_t K) {
    const int64_t nb = K / 64;
#pragma omp parallel for schedule(dynamic, 16)
    for (int64_t n = 0; n < N; n++) {
        for (int64_t b = 0; b < nb; b++) {
            const float *w = W + n * K + b * 64, *im = imp + b * 64;
            float amax = 0;
            for (int j = 0; j < 64; j++) amax = fmaxf(amax, fabsf(w[j]));
            float best_e = INFINITY, best_d = 0;
            if (amax > 0) {
                for (int si = 0; si < 40; si++) {
                    const float s = 0.2f + si * 0.035f;
                    for (int sg = -1; sg <= 1; sg += 2) {
                        float d = sg * amax * s / 2;
                        for (int it = 0; it < 2; it++) {
                            double num = 0, den = 0;
                            for (int j = 0; j < 64; j++) {
                                const int q = code(w[j], d);
                                num += (double)im[j] * w[j] * q;
                                den += (double)im[j] * q * q;
                            }
                            if (den <= 0) break;
                            const float d2 = (float)(num / den);
                            if (d2 == 0 || (d2 > 0) != (sg > 0)) break;
                            d = d2;
                        }
                        const float dh = _cvtsh_ss(_cvtss_sh(d, 0));
                        if (dh == 0) continue;
                        double e = 0;
                        for (int j = 0; j < 64; j++) {
                            const float r = w[j] - code(w[j], dh) * dh;
                            e += (double)im[j] * r * r;
                        }
                        if (e < best_e) { best_e = (float)e; best_d = dh; }
                    }
                }
            }
            uint8_t *o = out + (n * nb + b) * 18;
            const uint16_t h = _cvtss_sh(best_d, 0);
            memcpy(o, &h, 2);
            memset(o + 2, 0, 16);
            if (best_d != 0)
                for (int j = 0; j < 64; j++) o[2 + j / 4] |= (uint8_t)((code(w[j], best_d) + 1) << (2 * (j % 4)));
            else
                for (int j = 0; j < 64; j++) o[2 + j / 4] |= (uint8_t)(1 << (2 * (j % 4)));
        }
    }
}
