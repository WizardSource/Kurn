// R8 kernels (rotated int8 weights): see kurn-r8.h.
#include "kurn-r8.h"

#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)

#include <immintrin.h>
#include <x86intrin.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

std::atomic<uint64_t> g_redo{0};

// in-register butterfly stage of width h (1, 2, 4, 8): lanes with bit h clear get a + b, the others a - b, where a
// is the lower lane of the pair
template <int H>
inline __m512 fwht_lane(__m512 v) {
    __m512 sw;
    __mmask16 hi;
    if (H == 1) { sw = _mm512_permute_ps(v, 0xB1); hi = 0xAAAA; }
    else if (H == 2) { sw = _mm512_permute_ps(v, 0x4E); hi = 0xCCCC; }
    else if (H == 4) { sw = _mm512_shuffle_f32x4(v, v, 0xB1); hi = 0xF0F0; }
    else { sw = _mm512_shuffle_f32x4(v, v, 0x4E); hi = 0xFF00; }
    return _mm512_mask_sub_ps(_mm512_add_ps(v, sw), hi, sw, v);
}

// unnormalized Walsh-Hadamard transform of each hb-block of x (hb a power of two >= 16), in place
void fwht(float * x, int64_t K, int64_t hb) {
    for (int64_t b0 = 0; b0 < K; b0 += hb) {
        float * p = x + b0;
        for (int64_t i = 0; i < hb; i += 16) {
            __m512 v = _mm512_loadu_ps(p + i);
            v = fwht_lane<1>(v); v = fwht_lane<2>(v); v = fwht_lane<4>(v); v = fwht_lane<8>(v);
            _mm512_storeu_ps(p + i, v);
        }
        for (int64_t h = 16; h < hb; h *= 2) {
            for (int64_t i = 0; i < hb; i += 2 * h) {
                for (int64_t j = i; j < i + h; j += 16) {
                    const __m512 a = _mm512_loadu_ps(p + j), b = _mm512_loadu_ps(p + j + h);
                    _mm512_storeu_ps(p + j, _mm512_add_ps(a, b));
                    _mm512_storeu_ps(p + j + h, _mm512_sub_ps(a, b));
                }
            }
        }
    }
}

// G row groups (independent weight streams) x C columns of the GEMV; the last call of a range may have ng < G
template <int G, int C>
void gemv_blk(const uint8_t * W, const int32_t * rowsum, const float * sw, const uint8_t * X, const float * sx,
              float * Y, int64_t K, int64_t ldy, int64_t g, int ng) {
    static const int64_t pf = [] {
        const char * e = getenv("GGML_KURN_R8_PF");
        return (int64_t) (e ? atoi(e) : 4);
    }();
    const int64_t KB = K / 64;
    __m512i acc[G][C];
    for (int j = 0; j < G; j++)
        for (int c = 0; c < C; c++) acc[j][c] = _mm512_setzero_si512();
    const uint8_t * w[G];
    for (int j = 0; j < G; j++) w[j] = W + (size_t) (g + std::min(j, ng - 1)) * KB * 1024;
    for (int64_t b = 0; b < KB; b++) {
        for (int t = 0; t < 16; t++) {
            for (int j = 0; j < G; j++) {
                if (j < ng) {
                    _mm_prefetch((const char *) (w[j] + (b + pf) * 1024 + t * 64), _MM_HINT_T0);
                }
            }
            __m512i xb[C];
            for (int c = 0; c < C; c++) {
                int32_t xv;
                memcpy(&xv, X + c * K + 64 * b + 4 * t, 4);
                xb[c] = _mm512_set1_epi32(xv);
            }
            for (int j = 0; j < G; j++) {
                if (j >= ng) break;
                const __m512i a = _mm512_load_si512((const __m512i *) (w[j] + b * 1024 + t * 64));
                for (int c = 0; c < C; c++) acc[j][c] = _mm512_dpbusd_epi32(acc[j][c], xb[c], a);
            }
        }
    }
    for (int j = 0; j < ng; j++) {
        const int64_t n0 = (g + j) * 16;
        const __m512i corr = _mm512_slli_epi32(_mm512_loadu_si512((const __m512i *) (rowsum + n0)), 7);
        const __m512 s = _mm512_loadu_ps(sw + n0);
        for (int c = 0; c < C; c++) {
            const __m512 f = _mm512_cvtepi32_ps(_mm512_sub_epi32(acc[j][c], corr));
            _mm512_storeu_ps(Y + c * ldy + n0, _mm512_mul_ps(_mm512_mul_ps(f, s), _mm512_set1_ps(sx[c])));
        }
    }
}

template <int G, int C>
void gemv_range(const uint8_t * W, const int32_t * rowsum, const float * sw, const uint8_t * X, const float * sx, float * Y,
                int64_t K, int64_t ldy, int64_t r0, int64_t r1) {
    for (int64_t g = r0 / 16; g < r1 / 16; g += G) {
        gemv_blk<G, C>(W, rowsum, sw, X, sx, Y, K, ldy, g, (int) std::min<int64_t>(G, r1 / 16 - g));
    }
}

}  // namespace

extern "C" {

size_t kurn_r8_bytes(int64_t K, int64_t N) {
    return (size_t) N * (size_t) K;
}

void kurn_r8_pack(const int8_t * Q, int64_t K, int64_t N, uint8_t * out, int32_t * rowsum) {
    const int64_t KB = K / 64;
    for (int64_t g = 0; g < N / 16; g++) {
        for (int64_t b = 0; b < KB; b++) {
            uint8_t * t = out + ((size_t) g * KB + b) * 1024;
            for (int r = 0; r < 16; r++) {
                const int8_t * q = Q + (size_t) (16 * g + r) * K + 64 * b;
                for (int tt = 0; tt < 16; tt++) {
                    memcpy(t + tt * 64 + r * 4, q + 4 * tt, 4);
                }
            }
        }
    }
    for (int64_t n = 0; n < N; n++) {
        int32_t s = 0;
        for (int64_t k = 0; k < K; k++) {
            s += Q[(size_t) n * K + k];
        }
        rowsum[n] = s;
    }
}

void kurn_r8_prep_row(const float * x, int64_t K, int64_t hb, const int32_t * zero, int64_t nz, int8_t * q, uint8_t * qu,
                      float * sx) {
    static thread_local std::vector<float> buf;
    buf.assign(x, x + K);
    for (int64_t j = 0; j < nz; j++) {
        buf[zero[j]] = 0.0f;
    }
    fwht(buf.data(), K, hb);
    __m512 mx = _mm512_setzero_ps();
    for (int64_t k = 0; k < K; k += 16) {
        mx = _mm512_max_ps(mx, _mm512_abs_ps(_mm512_loadu_ps(buf.data() + k)));
    }
    const float amax = _mm512_reduce_max_ps(mx);
    const float id = amax > 0.0f ? 127.0f / amax : 0.0f;
    const __m512 vid = _mm512_set1_ps(id);
    for (int64_t k = 0; k < K; k += 16) {
        const __m512i v = _mm512_cvtps_epi32(_mm512_mul_ps(_mm512_loadu_ps(buf.data() + k), vid));
        const __m128i b = _mm512_cvtsepi32_epi8(v);
        _mm_storeu_si128((__m128i *) (q + k), b);
        if (qu) {
            _mm_storeu_si128((__m128i *) (qu + k), _mm_xor_si128(b, _mm_set1_epi8((char) 0x80)));
        }
    }
    *sx = amax / 127.0f / sqrtf((float) hb);
}

void kurn_r8_gemv(const uint8_t * W, const int32_t * rowsum, const float * sw, const uint8_t * X, const float * sx,
                  float * Y, int64_t K, int64_t ldy, int64_t M, int64_t r0, int64_t r1) {
    // up to 16 accumulators: 4 streams x 1-4 columns, 2 streams x 5-8 columns
    for (int64_t c0 = 0; c0 < M;) {
        const int64_t m = M - c0;
        const uint8_t * Xc = X + c0 * K;
        const float * sc = sx + c0;
        float * Yc = Y + c0 * ldy;
        switch (m) {
            case 1: gemv_range<4, 1>(W, rowsum, sw, Xc, sc, Yc, K, ldy, r0, r1); c0 += 1; break;
            case 2: gemv_range<4, 2>(W, rowsum, sw, Xc, sc, Yc, K, ldy, r0, r1); c0 += 2; break;
            case 3: gemv_range<4, 3>(W, rowsum, sw, Xc, sc, Yc, K, ldy, r0, r1); c0 += 3; break;
            case 4: gemv_range<4, 4>(W, rowsum, sw, Xc, sc, Yc, K, ldy, r0, r1); c0 += 4; break;
            case 5: gemv_range<2, 5>(W, rowsum, sw, Xc, sc, Yc, K, ldy, r0, r1); c0 += 5; break;
            case 6: gemv_range<2, 6>(W, rowsum, sw, Xc, sc, Yc, K, ldy, r0, r1); c0 += 6; break;
            case 7: gemv_range<2, 7>(W, rowsum, sw, Xc, sc, Yc, K, ldy, r0, r1); c0 += 7; break;
            default: gemv_range<2, 8>(W, rowsum, sw, Xc, sc, Yc, K, ldy, r0, r1); c0 += 8; break;
        }
    }
}

#if defined(__AMX_INT8__) && defined(__linux__)

// tmm0..3 C (16 x 16 int32), tmm4..5 A (16 tokens x 64 B), tmm6..7 B (16 x 64 B: 16 rows x 4 values per line)
static void amx_config_r8() {
    struct alignas(64) {
        uint8_t palette, start_row, reserved[14];
        uint16_t colsb[16];
        uint8_t rows[16];
    } c = {};
    c.palette = 1;
    for (int t = 0; t < 8; t++) { c.rows[t] = 16; c.colsb[t] = 64; }
    _tile_loadconfig(&c);
}

int kurn_r8_amx(const uint8_t * W, const float * sw, const int8_t * X, const float * sx, float * Y, int64_t K,
                int64_t ldy, int64_t M, int64_t r0, int64_t r1, uint64_t guard_base) {
    const int64_t KB = K / 64, Mp = (M + 31) / 32 * 32;
    const uint64_t guard = guard_base + (uint64_t) KB * 256;
    alignas(64) int32_t cb[4][256];
    amx_config_r8();
    for (int64_t n0 = r0; n0 < r1; n0 += 32) {
        const int two = n0 + 32 <= r1;
        const uint8_t * b0 = W + (size_t) (n0 / 16) * KB * 1024;
        const uint8_t * b1 = b0 + (size_t) KB * 1024;
        for (int64_t m0 = 0; m0 < Mp; m0 += 32) {
            const int8_t * a0 = X + m0 * K;
            const int8_t * a1 = a0 + 16 * K;
            for (;;) {
                const uint64_t t0 = __rdtsc();
                _tile_zero(0); _tile_zero(1); _tile_zero(2); _tile_zero(3);
                if (two) {
                    for (int64_t b = 0; b < KB; b++) {
                        _tile_loadd(4, a0 + 64 * b, K);
                        _tile_loadd(5, a1 + 64 * b, K);
                        _tile_loadd(6, b0 + b * 1024, 64);
                        _tile_loadd(7, b1 + b * 1024, 64);
                        _tile_dpbssd(0, 4, 6);
                        _tile_dpbssd(1, 4, 7);
                        _tile_dpbssd(2, 5, 6);
                        _tile_dpbssd(3, 5, 7);
                    }
                } else {
                    for (int64_t b = 0; b < KB; b++) {
                        _tile_loadd(4, a0 + 64 * b, K);
                        _tile_loadd(5, a1 + 64 * b, K);
                        _tile_loadd(6, b0 + b * 1024, 64);
                        _tile_dpbssd(0, 4, 6);
                        _tile_dpbssd(2, 5, 6);
                    }
                }
                _tile_stored(0, cb[0], 64); _tile_stored(1, cb[1], 64);
                _tile_stored(2, cb[2], 64); _tile_stored(3, cb[3], 64);
                unsigned aux;
                if (__rdtscp(&aux) - t0 > guard) {  // possibly descheduled with live tile data: redo from the inputs
                    g_redo.fetch_add(1, std::memory_order_relaxed);
                    continue;
                }
                break;
            }
            for (int tt = 0; tt < 4; tt++) {
                const int64_t mb = m0 + (tt >> 1) * 16, nb = n0 + (tt & 1) * 16;
                if ((tt & 1) && !two) continue;
                const __m512 s = _mm512_loadu_ps(sw + nb);
                for (int r = 0; r < 16 && mb + r < M; r++) {
                    const __m512 f = _mm512_cvtepi32_ps(_mm512_load_si512((const __m512i *) (cb[tt] + r * 16)));
                    _mm512_storeu_ps(Y + (mb + r) * ldy + nb, _mm512_mul_ps(_mm512_mul_ps(f, s), _mm512_set1_ps(sx[mb + r])));
                }
            }
        }
    }
    return 1;
}

#else

int kurn_r8_amx(const uint8_t *, const float *, const int8_t *, const float *, float *, int64_t, int64_t, int64_t,
                int64_t, int64_t, uint64_t) {
    return 0;
}

#endif

uint64_t kurn_r8_amx_redo(void) {
    return g_redo.load();
}

}  // extern "C"

#endif
