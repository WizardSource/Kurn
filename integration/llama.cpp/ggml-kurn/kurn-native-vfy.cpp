// Decode and verify kernels (1..8 activation columns, more in groups of 8) for Q6_K and Q5_K weights
// kept in ggml's own block layout in the KURN buffer (the AMX-BF16 prefill reads that layout, so they
// are not repacked). Each weight super-block is unpacked once and dotted with every column, instead
// of one ggml vec_dot per row and column.
//
// Arithmetic per (row, column), independent of how many columns or rows are computed together, so a
// token's result is the same in decode and in a verify batch:
//   per super-block, an exact int32 vector (16 lanes) of scale-weighted partial dots, the code offset
//   (Q6_K: -32) taken from the activation's block sums; then acc = fma(float(lanes), d_w * d_x, acc) in
//   super-block order (Q5_K: plus macc = fma(float(min lanes), -dmin_w * d_x, macc), 8 lanes), and one
//   fixed-order lane reduction per row at the end.
#include "kurn-native-vfy.h"

#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VL__) && defined(__AVX512VNNI__)

#include <immintrin.h>
#include <algorithm>
#include <cstring>

namespace {

struct blk_q8_K { float d; int8_t qs[256]; int16_t bsums[16]; };
struct blk_q6_K { uint8_t ql[128]; uint8_t qh[64]; int8_t scales[16]; uint16_t d; };
struct blk_q5_K { uint16_t d, dmin; uint8_t scales[12]; uint8_t qh[32]; uint8_t qs[128]; };
static_assert(sizeof(blk_q8_K) == 292 && sizeof(blk_q6_K) == 210 && sizeof(blk_q5_K) == 176, "ggml block sizes");

// software prefetch distance in bytes along a thread's row range (rows are contiguous)
constexpr int PF_BYTES = 2048;

inline float f16(uint16_t h) { return _cvtsh_ss(h); }

inline float reduce(__m512 v) {
    const __m256 a = _mm256_add_ps(_mm512_castps512_ps256(v), _mm512_extractf32x8_ps(v, 1));
    const __m128 b = _mm_add_ps(_mm256_castps256_ps128(a), _mm256_extractf128_ps(a, 1));
    const __m128 c = _mm_add_ps(b, _mm_movehl_ps(b, b));
    return _mm_cvtss_f32(_mm_add_ss(c, _mm_movehdup_ps(c)));
}

inline void prefetch_ahead(const void * p, int bytes) {
    for (int o = 0; o < bytes; o += 64) _mm_prefetch((const char *) p + PF_BYTES + o, _MM_HINT_T0);
}

// Q6_K: value = d * sc[j] * (q - 32), q = 4 low bits from ql | 2 high bits from qh, 16 values per scale.
// Half h (128 values): v0 = values 0..63 (ql[64h + 0..64) low nibbles, qh[32h + l] bits 0-1 for l < 32,
// bits 2-3 for the second 32), v1 = values 64..127 (high nibbles, qh bits 4-5 / 6-7).
template <int M>
void q6_K_rows(const char * W, size_t wrow, const char * X, size_t xrow, float * Y, int64_t N, int64_t nb, int64_t r0, int64_t r1) {
    const __m512i m4 = _mm512_set1_epi8(0x0F), m3 = _mm512_set1_epi8(3);
    const __mmask64 hi32 = 0xFFFFFFFF00000000ull;
    alignas(64) static const int16_t sidx[4][32] = {
        {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3},
        {4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7},
        {8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11},
        {12, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 15}};
    const __m512i i0 = _mm512_load_si512(sidx[0]), i1 = _mm512_load_si512(sidx[1]), i2 = _mm512_load_si512(sidx[2]),
                  i3 = _mm512_load_si512(sidx[3]);
    for (int64_t n = r0; n < r1; n++) {
        const blk_q6_K * w = (const blk_q6_K *) (W + n * wrow);
        __m512 acc[M];
        for (int m = 0; m < M; m++) acc[m] = _mm512_setzero_ps();
        for (int64_t b = 0; b < nb; b++) {
            prefetch_ahead(w + b, (int) sizeof(blk_q6_K));
            __m512i v[4];
            for (int h = 0; h < 2; h++) {
                const __m512i ql = _mm512_loadu_si512(w[b].ql + 64 * h);
                const __m512i qh = _mm512_broadcast_i64x4(_mm256_loadu_si256((const __m256i *) (w[b].qh + 32 * h)));
                const __m512i ha = _mm512_and_si512(_mm512_mask_blend_epi8(hi32, qh, _mm512_srli_epi16(qh, 2)), m3);
                const __m512i hb = _mm512_and_si512(_mm512_mask_blend_epi8(hi32, _mm512_srli_epi16(qh, 4), _mm512_srli_epi16(qh, 6)), m3);
                v[2 * h] = _mm512_or_si512(_mm512_and_si512(ql, m4), _mm512_slli_epi16(ha, 4));
                v[2 * h + 1] = _mm512_or_si512(_mm512_and_si512(_mm512_srli_epi16(ql, 4), m4), _mm512_slli_epi16(hb, 4));
            }
            const __m256i sc16 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *) w[b].scales));
            const __m512i sc = _mm512_castsi256_si512(sc16);
            const __m512i s0 = _mm512_permutexvar_epi16(i0, sc), s1 = _mm512_permutexvar_epi16(i1, sc),
                          s2 = _mm512_permutexvar_epi16(i2, sc), s3 = _mm512_permutexvar_epi16(i3, sc);
            const float dw = f16(w[b].d);
            for (int m = 0; m < M; m++) {
                const blk_q8_K * x = (const blk_q8_K *) (X + m * xrow) + b;
                __m512i p = _mm512_madd_epi16(_mm512_maddubs_epi16(v[0], _mm512_loadu_si512(x->qs)), s0);
                p = _mm512_dpwssd_epi32(p, _mm512_maddubs_epi16(v[1], _mm512_loadu_si512(x->qs + 64)), s1);
                p = _mm512_dpwssd_epi32(p, _mm512_maddubs_epi16(v[2], _mm512_loadu_si512(x->qs + 128)), s2);
                p = _mm512_dpwssd_epi32(p, _mm512_maddubs_epi16(v[3], _mm512_loadu_si512(x->qs + 192)), s3);
                // -32 * sum_j sc[j] * bsum[j]: 8 lanes of pair sums, subtracted in the low half
                const __m256i c = _mm256_slli_epi32(_mm256_madd_epi16(_mm256_loadu_si256((const __m256i *) x->bsums), sc16), 5);
                p = _mm512_sub_epi32(p, _mm512_zextsi256_si512(c));
                acc[m] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(p), _mm512_set1_ps(dw * x->d), acc[m]);
            }
        }
        for (int m = 0; m < M; m++) Y[m * N + n] = reduce(acc[m]);
    }
}

// Q5_K: value = d * sc[j] * q - dmin * mn[j], q = 4 bits from qs | 1 bit from qh, 32 values per scale.
// Chunk c (64 values): qs[32c + l] low nibbles = values 64c + l, high nibbles = 64c + 32 + l, qh bits
// 2c / 2c + 1. One vector per chunk: [low nibbles of the 32 bytes, high nibbles of the same bytes].
template <int M>
void q5_K_rows(const char * W, size_t wrow, const char * X, size_t xrow, float * Y, int64_t N, int64_t nb, int64_t r0, int64_t r1) {
    const __m512i m4 = _mm512_set1_epi8(0x0F), m1 = _mm512_set1_epi8(1);
    const __mmask64 hi32 = 0xFFFFFFFF00000000ull;
    alignas(64) static const int16_t sidx[4][32] = {
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
        {2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3},
        {4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5},
        {6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7}};
    alignas(16) static const int8_t midx[16] = {0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7};
    __m512i si[4];
    for (int c = 0; c < 4; c++) si[c] = _mm512_load_si512(sidx[c]);
    const __m128i mdup = _mm_load_si128((const __m128i *) midx);
    for (int64_t n = r0; n < r1; n++) {
        const blk_q5_K * w = (const blk_q5_K *) (W + n * wrow);
        __m512 acc[M];
        __m256 macc[M];  // mins term: 8 lanes
        for (int m = 0; m < M; m++) { acc[m] = _mm512_setzero_ps(); macc[m] = _mm256_setzero_ps(); }
        for (int64_t b = 0; b < nb; b++) {
            prefetch_ahead(w + b, (int) sizeof(blk_q5_K));
            // 6-bit scales and mins (ggml's get_scale_min_k4, all eight at once)
            uint32_t u[4];
            memcpy(u, w[b].scales, 12);
            u[3] = ((u[2] >> 4) & 0x0f0f0f0f) | (((u[1] >> 6) & 0x03030303) << 4);
            const uint32_t t = u[1] & 0x3f3f3f3f;
            u[1] = (u[2] & 0x0f0f0f0f) | (((u[0] >> 6) & 0x03030303) << 4);
            u[2] = t;
            u[0] &= 0x3f3f3f3f;
            const __m128i scmn = _mm_loadu_si128((const __m128i *) u);  // bytes 0..7 scales, 8..15 mins
            const __m512i sc = _mm512_castsi256_si512(_mm256_cvtepu8_epi16(scmn));
            const __m256i mn16 = _mm256_cvtepu8_epi16(_mm_shuffle_epi8(_mm_srli_si128(scmn, 8), mdup));
            const __m512i qh = _mm512_broadcast_i64x4(_mm256_loadu_si256((const __m256i *) w[b].qh));
            const __m512i q0 = _mm512_loadu_si512(w[b].qs), q1 = _mm512_loadu_si512(w[b].qs + 64);
            __m512i v[4];
            for (int c = 0; c < 4; c++) {
                const __m512i q = c < 2 ? q0 : q1;
                const __m512i lo = _mm512_and_si512(q, m4), hi = _mm512_and_si512(_mm512_srli_epi16(q, 4), m4);
                const __m512i nib = (c & 1) ? _mm512_shuffle_i64x2(lo, hi, 0xEE) : _mm512_shuffle_i64x2(lo, hi, 0x44);
                const __m512i hb = _mm512_and_si512(
                    _mm512_mask_blend_epi8(hi32, _mm512_srl_epi16(qh, _mm_cvtsi32_si128(2 * c)), _mm512_srl_epi16(qh, _mm_cvtsi32_si128(2 * c + 1))), m1);
                v[c] = _mm512_or_si512(nib, _mm512_slli_epi16(hb, 4));
            }
            __m512i s[4];
            for (int c = 0; c < 4; c++) s[c] = _mm512_permutexvar_epi16(si[c], sc);
            const float dw = f16(w[b].d), dmw = f16(w[b].dmin);
            for (int m = 0; m < M; m++) {
                const blk_q8_K * x = (const blk_q8_K *) (X + m * xrow) + b;
                __m512i p = _mm512_madd_epi16(_mm512_maddubs_epi16(v[0], _mm512_loadu_si512(x->qs)), s[0]);
                for (int c = 1; c < 4; c++)
                    p = _mm512_dpwssd_epi32(p, _mm512_maddubs_epi16(v[c], _mm512_loadu_si512(x->qs + 64 * c)), s[c]);
                const __m256i mins = _mm256_madd_epi16(_mm256_loadu_si256((const __m256i *) x->bsums), mn16);
                acc[m] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(p), _mm512_set1_ps(dw * x->d), acc[m]);
                macc[m] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(mins), _mm256_set1_ps(-(dmw * x->d)), macc[m]);
            }
        }
        for (int m = 0; m < M; m++) Y[m * N + n] = reduce(_mm512_add_ps(acc[m], _mm512_zextps256_ps512(macc[m])));
    }
}

typedef void (*rows_fn)(const char *, size_t, const char *, size_t, float *, int64_t, int64_t, int64_t, int64_t);

const rows_fn q6_fns[8] = {q6_K_rows<1>, q6_K_rows<2>, q6_K_rows<3>, q6_K_rows<4>, q6_K_rows<5>, q6_K_rows<6>, q6_K_rows<7>, q6_K_rows<8>};
const rows_fn q5_fns[8] = {q5_K_rows<1>, q5_K_rows<2>, q5_K_rows<3>, q5_K_rows<4>, q5_K_rows<5>, q5_K_rows<6>, q5_K_rows<7>, q5_K_rows<8>};

}  // namespace

bool kurn_native_vfy_supported(int type) {
    return type == KURN_NATIVE_Q6_K || type == KURN_NATIVE_Q5_K;
}

bool kurn_native_vfy(int type, const char * W, size_t wrow, const char * X, size_t xrow, float * Y, int64_t K, int64_t N,
                     int64_t M, int64_t r0, int64_t r1) {
    if (!kurn_native_vfy_supported(type) || K % 256 != 0 || M < 1) {
        return false;
    }
    const rows_fn * fns = type == KURN_NATIVE_Q6_K ? q6_fns : q5_fns;
    const int64_t nb = K / 256;
    // row blocks small enough that a block's rows stay in L2 while every 8-column group passes over them
    const int64_t rb = M <= 8 ? r1 - r0 : std::max<int64_t>(1, (256 * 1024) / (int64_t) wrow);
    for (int64_t a = r0; a < r1; a += rb) {
        const int64_t e = std::min(r1, a + rb);
        for (int64_t c0 = 0; c0 < M; c0 += 8) {
            const int64_t m = std::min<int64_t>(8, M - c0);
            fns[m - 1](W, wrow, X + c0 * xrow, xrow, Y + c0 * N, N, nb, a, e);
        }
    }
    return true;
}

#else

bool kurn_native_vfy_supported(int) {
    return false;
}

bool kurn_native_vfy(int, const char *, size_t, const char *, size_t, float *, int64_t, int64_t, int64_t, int64_t, int64_t) {
    return false;
}

#endif
