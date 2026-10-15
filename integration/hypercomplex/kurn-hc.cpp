// kurn hypercomplex linear layers: see kurn-hc.h and README.md.
//
// KURN_HC_Q8 record format: per row group g (16 output numbers), input block kb (32 input numbers) and component p,
// one kurn i16 Q8_0 record (544 bytes, the same as kurn's real Q8_0 kernels use): 16 fp16 scales (one per row),
// then 8 lines of 64 bytes whose byte r * 4 + j holds value 4 kk + j of row r, biased by +128. The D records of a
// (g, kb) pair are contiguous, so one weight stream per row group carries all components.
//
// Product. For output component o and input component q, the term is s(o, q) * W_p * x_q with p = P(o, q):
//   quaternion (Hamilton, weight on the left)      complex
//   r = a x0 - b x1 - c x2 - d x3                  re = a x0 - b x1
//   i = b x0 + a x1 - d x2 + c x3                  im = b x0 + a x1
//   j = c x0 + d x1 + a x2 - b x3
//   k = d x0 - c x1 + b x2 + a x3
// Each weight line is loaded once and multiplied with all D input components (D^2 int32 block dots per row group
// and block, from D records instead of the D^2 the real expansion streams). Every output then takes its D terms in
// input-component order, acc = fma(float(s * isum), d_w * d_x, acc), block by block: exactly the per-block
// arithmetic and order of kurn's real Q8_0 GEMV / verify kernels on the block-planar expansion, so results are
// bit-identical to them.
#include "kurn-hc.h"
#include "../llama.cpp/ggml-kurn/kurn-amx-selftest.h"

#include <immintrin.h>
#include <x86intrin.h>
#include <omp.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#if defined(__linux__)
#include <sys/syscall.h>
#include <unistd.h>
#endif

struct kurn_hc {
    int D;
    int64_t Nq, Kq;
    kurn_hc_format fmt;
    uint8_t * buf;
    size_t bytes;
};

kurn_amx_state kurn_amx_check() __attribute__((weak));

namespace {

constexpr int REC = 544;  // kurn i16 Q8_0 record: 16 rows x 32 values

// output component o, input component q -> weight component, sign
constexpr int QP[4][4] = {{0, 1, 2, 3}, {1, 0, 3, 2}, {2, 3, 0, 1}, {3, 2, 1, 0}};
constexpr int QS[4][4] = {{1, -1, -1, -1}, {1, 1, -1, 1}, {1, 1, 1, -1}, {1, -1, 1, 1}};
constexpr int CP[2][2] = {{0, 1}, {1, 0}};
constexpr int CS[2][2] = {{1, -1}, {1, 1}};

template <int D> constexpr int tp(int o, int q) { return D == 4 ? QP[o][q] : CP[o][q]; }
template <int D> constexpr int ts(int o, int q) { return D == 4 ? QS[o][q] : CS[o][q]; }

int sign_of(int D, int o, int q) { return D == 4 ? QS[o][q] : CS[o][q]; }
int comp_of(int D, int o, int q) { return D == 4 ? QP[o][q] : CP[o][q]; }

inline uint16_t f2h(float f) { return _cvtss_sh(f, 0); }
inline float h2f(uint16_t h) { return _cvtsh_ss(h); }

int env_int(const char * n, int def) {
    const char * v = getenv(n);
    return v && *v ? atoi(v) : def;
}

// block-planar index of element (k, c) of a vector of D-component numbers
inline int64_t bp(int D, int64_t k, int c) { return (k / 32) * 32 * D + c * 32 + k % 32; }

// ggml's quantize_row_q8_0_ref for weights: d = amax / 127, q = roundf(x / d)
void quant_ref32(const float * x, uint16_t & dh, int8_t * q) {
    float amax = 0;
    for (int i = 0; i < 32; i++) amax = std::max(amax, std::fabs(x[i]));
    const float d = amax / 127.f, id = d ? 1.0f / d : 0.0f;
    dh = f2h(d);
    for (int i = 0; i < 32; i++) q[i] = (int8_t) roundf(x[i] * id);
}

// ggml's x86 quantize_row_q8_0 for activations: id = 127 / amax, round half to even
void quant_act32(const float * x, uint16_t & dh, int8_t * q) {
    float amax = 0;
    for (int i = 0; i < 32; i++) amax = std::max(amax, std::fabs(x[i]));
    const float d = amax / 127.f, id = amax != 0.0f ? 127.f / amax : 0.0f;
    dh = f2h(d);
    for (int i = 0; i < 32; i++) q[i] = (int8_t) std::max(-128, std::min(127, (int) nearbyintf(x[i] * id)));
}

// ------------------------------------------------------------------------------------------
// work-stealing schedule over units (row groups), as kurn's steal_sched in kurn-buft.cpp
struct alignas(64) slot { std::atomic<uint64_t> w; };

struct steal {
    slot * s;
    int nth;
    int64_t P;
    void init(int ith, int64_t U) {
        const int64_t np = (U + P - 1) / P;
        const int64_t u0 = std::min(U, np * ith / nth * P), u1 = std::min(U, np * (ith + 1) / nth * P);
        s[ith].w.store(((uint64_t) u1 << 32) | (uint64_t) u0, std::memory_order_relaxed);
    }
    bool next(int ith, int64_t & u0, int64_t & u1) {
        for (;;) {
            uint64_t w = s[ith].w.load(std::memory_order_acquire);
            for (;;) {
                const int64_t n = (int64_t) (w & 0xffffffffu), e = (int64_t) (w >> 32);
                if (n >= e) break;
                const int64_t rem = e - n, b = rem >= 2 * P ? std::max(P, rem / 4 / P * P) : rem >= P ? P : 1;
                if (s[ith].w.compare_exchange_weak(w, ((uint64_t) e << 32) | (uint64_t) (n + b), std::memory_order_acq_rel)) {
                    u0 = n; u1 = n + b;
                    return true;
                }
            }
            int v = -1;
            int64_t best = 0;
            for (int t = 0; t < nth; t++) {
                const uint64_t x = s[t].w.load(std::memory_order_relaxed);
                const int64_t r = (int64_t) (x >> 32) - (int64_t) (x & 0xffffffffu);
                if (t != ith && r > best) { best = r; v = t; }
            }
            if (v < 0) return false;
            uint64_t x = s[v].w.load(std::memory_order_acquire);
            const int64_t n = (int64_t) (x & 0xffffffffu), e = (int64_t) (x >> 32);
            if (n >= e) continue;
            const int64_t take = (e - n + 1) / 2;
            if (s[v].w.compare_exchange_strong(x, ((uint64_t) (e - take) << 32) | (uint64_t) n, std::memory_order_acq_rel))
                s[ith].w.store(((uint64_t) e << 32) | (uint64_t) (e - take), std::memory_order_release);
        }
    }
};

// ------------------------------------------------------------------------------------------
// Q8 activations: per column m and block b = kb * D + q: 32 int8 (xq), d_x as float, -128 * sum(x)
struct xprep {
    int64_t nb = 0;  // blocks per column (D * Kq / 32)
    const int8_t * xq = nullptr;
    const float * dx = nullptr;
    const int32_t * nc = nullptr;
};

template <int D, int M, int G>
void q8_rows(const kurn_hc * w, const xprep & xp, int64_t m0, float * Y, int64_t ldy, int64_t g0, int64_t g1) {
    constexpr int QC = D * M <= 16 ? D : 2 * M <= 16 ? 2 : 1;  // input components per pass over a record; divides D
    constexpr int PF = 4;  // prefetch distance in (g, kb) records
    const int64_t nkb = w->Kq / 32;
    const size_t qrec = (size_t) REC * D;
    const int8_t * xq = xp.xq + m0 * xp.nb * 32;
    const float * dx = xp.dx + m0 * xp.nb;
    const int32_t * nc = xp.nc + m0 * xp.nb;
    for (int64_t g = g0; g < g1; g += G) {
        const int ng = (int) std::min<int64_t>(G, g1 - g);
        __m512 acc[G][D][M];
        for (int j = 0; j < G; j++)
            for (int o = 0; o < D; o++)
                for (int m = 0; m < M; m++) acc[j][o][m] = _mm512_setzero_ps();
        for (int64_t kb = 0; kb < nkb; kb++) {
            for (int j = 0; j < G; j++) {
                const uint8_t * rec = w->buf + ((size_t) (g + (j < ng ? j : 0)) * nkb + kb) * qrec;
                for (size_t off = 0; off < qrec; off += 64) _mm_prefetch((const char *) rec + PF * qrec + off, _MM_HINT_T0);
                alignas(64) float F[D][D][M][16];
                for (int p = 0; p < D; p++) {
                    const uint8_t * rp = rec + p * REC;
                    for (int q0 = 0; q0 < D; q0 += QC) {
                        __m512i I[QC][M];
                        for (int qq = 0; qq < QC; qq++)
                            for (int m = 0; m < M; m++) I[qq][m] = _mm512_set1_epi32(nc[m * xp.nb + kb * D + q0 + qq]);
                        for (int kk = 0; kk < 8; kk++) {
                            const __m512i line = _mm512_loadu_si512((const void *) (rp + 32 + 64 * kk));
                            for (int qq = 0; qq < QC; qq++)
                                for (int m = 0; m < M; m++) {
                                    int32_t xw;
                                    memcpy(&xw, xq + (m * xp.nb + kb * D + q0 + qq) * 32 + 4 * kk, 4);
                                    I[qq][m] = _mm512_dpbusd_epi32(I[qq][m], line, _mm512_set1_epi32(xw));
                                }
                        }
                        for (int qq = 0; qq < QC; qq++)
                            for (int m = 0; m < M; m++) _mm512_store_ps(F[p][q0 + qq][m], _mm512_cvtepi32_ps(I[qq][m]));
                    }
                }
                __m512 dw[D];
                for (int p = 0; p < D; p++) dw[p] = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *) (rec + p * REC)));
                for (int o = 0; o < D; o++)
                    for (int q = 0; q < D; q++) {
                        const int p = tp<D>(o, q);
                        for (int m = 0; m < M; m++) {
                            const __m512 prod = _mm512_mul_ps(dw[p], _mm512_set1_ps(dx[m * xp.nb + kb * D + q]));
                            const __m512 f = _mm512_load_ps(F[p][q][m]);
                            acc[j][o][m] = ts<D>(o, q) > 0 ? _mm512_fmadd_ps(f, prod, acc[j][o][m]) : _mm512_fnmadd_ps(f, prod, acc[j][o][m]);
                        }
                    }
            }
        }
        for (int j = 0; j < ng; j++) {
            const int64_t n = (g + j) * 16;
            for (int o = 0; o < D; o++)
                for (int m = 0; m < M; m++) _mm512_storeu_ps(Y + m * ldy + bp(D, n, o), acc[j][o][m]);
        }
    }
}

// Input-component-outer variant: P(o, q) = o ^ q, so for a fixed input component q every weight component p feeds
// exactly one output (o = p ^ q), and that output's q term is the next in its fma chain. Each int32 block dot is
// scaled and accumulated as soon as it is complete (no scratch), activation broadcasts are shared by all planes,
// and weight lines are re-read from L1 once per input component. Same per-output operands and order as q8_rows.
template <int D, int M, int G>
void q8_rows_qo(const kurn_hc * w, const xprep & xp, int64_t m0, float * Y, int64_t ldy, int64_t g0, int64_t g1) {
    constexpr int PC = D * M <= 8 ? D : 2 * M <= 8 ? 2 : 1;  // weight components per pass
    constexpr int PF = 4;
    const int64_t nkb = w->Kq / 32;
    const size_t qrec = (size_t) REC * D;
    const int8_t * xq = xp.xq + m0 * xp.nb * 32;
    const float * dx = xp.dx + m0 * xp.nb;
    const int32_t * nc = xp.nc + m0 * xp.nb;
    for (int64_t g = g0; g < g1; g += G) {
        const int ng = (int) std::min<int64_t>(G, g1 - g);
        __m512 acc[G][D][M];
        for (int j = 0; j < G; j++)
            for (int o = 0; o < D; o++)
                for (int m = 0; m < M; m++) acc[j][o][m] = _mm512_setzero_ps();
        for (int64_t kb = 0; kb < nkb; kb++) {
            for (int j = 0; j < G; j++) {
                const uint8_t * rec = w->buf + ((size_t) (g + (j < ng ? j : 0)) * nkb + kb) * qrec;
                for (size_t off = 0; off < qrec; off += 64) _mm_prefetch((const char *) rec + PF * qrec + off, _MM_HINT_T0);
                __m512 dw[D];
                for (int p = 0; p < D; p++) dw[p] = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *) (rec + p * REC)));
                for (int q = 0; q < D; q++) {
                    const int64_t b = kb * D + q;
                    for (int p0 = 0; p0 < D; p0 += PC) {
                        __m512i I[PC][M];
                        for (int pp = 0; pp < PC; pp++)
                            for (int m = 0; m < M; m++) I[pp][m] = _mm512_set1_epi32(nc[m * xp.nb + b]);
                        for (int kk = 0; kk < 8; kk++) {
                            __m512i xb[M];
                            for (int m = 0; m < M; m++) {
                                int32_t xw;
                                memcpy(&xw, xq + (m * xp.nb + b) * 32 + 4 * kk, 4);
                                xb[m] = _mm512_set1_epi32(xw);
                            }
                            for (int pp = 0; pp < PC; pp++) {
                                const __m512i line = _mm512_loadu_si512((const void *) (rec + (p0 + pp) * REC + 32 + 64 * kk));
                                for (int m = 0; m < M; m++) I[pp][m] = _mm512_dpbusd_epi32(I[pp][m], line, xb[m]);
                            }
                        }
                        for (int pp = 0; pp < PC; pp++) {
                            const int p = p0 + pp, o = p ^ q;
                            for (int m = 0; m < M; m++) {
                                const __m512 prod = _mm512_mul_ps(dw[p], _mm512_set1_ps(dx[m * xp.nb + b]));
                                const __m512 f = _mm512_cvtepi32_ps(I[pp][m]);
                                acc[j][o][m] = ts<D>(o, q) > 0 ? _mm512_fmadd_ps(f, prod, acc[j][o][m]) : _mm512_fnmadd_ps(f, prod, acc[j][o][m]);
                            }
                        }
                    }
                }
            }
        }
        for (int j = 0; j < ng; j++) {
            const int64_t n = (g + j) * 16;
            for (int o = 0; o < D; o++)
                for (int m = 0; m < M; m++) _mm512_storeu_ps(Y + m * ldy + bp(D, n, o), acc[j][o][m]);
        }
    }
}

typedef void (*q8_fn)(const kurn_hc *, const xprep &, int64_t, float *, int64_t, int64_t, int64_t);

template <int D, template <int, int, int> class K>
struct kern {
    static q8_fn get(int M, int G) {
        switch (M) {
            case 1: return G >= 4 ? K<D, 1, 4>::f : G >= 2 ? K<D, 1, 2>::f : K<D, 1, 1>::f;
            case 2: return G >= 2 ? K<D, 2, 2>::f : K<D, 2, 1>::f;
            case 3: return G >= 2 ? K<D, 3, 2>::f : K<D, 3, 1>::f;
            case 4: return G >= 2 ? K<D, 4, 2>::f : K<D, 4, 1>::f;
            case 5: return K<D, 5, 1>::f;
            case 6: return K<D, 6, 1>::f;
            case 7: return K<D, 7, 1>::f;
            default: return K<D, 8, 1>::f;
        }
    }
};
template <int D, int M, int G> struct k_po { static constexpr q8_fn f = q8_rows<D, M, G>; };
template <int D, int M, int G> struct k_qo { static constexpr q8_fn f = q8_rows_qo<D, M, G>; };

// KURN_HC_KERNEL=po: weight-component-outer kernel (q8_rows); default: input-component-outer (q8_rows_qo)
template <int D>
q8_fn q8_kernel(int M, int G) {
    static const int po = env_int("KURN_HC_PO", 0);
    return po ? kern<D, k_po>::get(M, G) : kern<D, k_qo>::get(M, G);
}

// rows per pass at each width (row groups of 16 numbers); KURN_HC_G overrides
int q8_groups(int D, int M) {
    static const int g = env_int("KURN_HC_G", 0);
    if (g > 0) return g;
    return D == 4 ? (M == 1 ? 2 : 1) : (M <= 4 ? 2 : 1);
}

// ------------------------------------------------------------------------------------------
// f32 compact: per row group g and input k, D x 16 floats (component p, rows 0..15)
template <int D>
void f32_rows(const kurn_hc * w, const float * X, int64_t ldx, int64_t M, float * Y, int64_t ldy, int64_t g0, int64_t g1) {
    const int64_t Kq = w->Kq;
    const float * W = (const float *) w->buf;
    for (int64_t g = g0; g < g1; g++) {
        for (int64_t m = 0; m < M; m++) {
            const float * x = X + m * ldx;
            __m512 acc[D];
            for (int o = 0; o < D; o++) acc[o] = _mm512_setzero_ps();
            const float * wg = W + (size_t) g * Kq * D * 16;
            for (int64_t k = 0; k < Kq; k++) {
                __m512 wv[D];
                for (int p = 0; p < D; p++) wv[p] = _mm512_loadu_ps(wg + (k * D + p) * 16);
                if ((k & 7) == 0) _mm_prefetch((const char *) (wg + (k + 64) * D * 16), _MM_HINT_T0);
                for (int q = 0; q < D; q++) {
                    const __m512 xv = _mm512_set1_ps(x[bp(D, k, q)]);
                    for (int o = 0; o < D; o++) {
                        const int p = tp<D>(o, q);
                        acc[o] = ts<D>(o, q) > 0 ? _mm512_fmadd_ps(wv[p], xv, acc[o]) : _mm512_fnmadd_ps(wv[p], xv, acc[o]);
                    }
                }
            }
            for (int o = 0; o < D; o++) _mm512_storeu_ps(Y + m * ldy + bp(D, g * 16, o), acc[o]);
        }
    }
}

// ------------------------------------------------------------------------------------------
// AMX-BF16 prefill on the real expansion, built per panel from the compact weights (kurn-buft.cpp's
// amx_bf16_gemm, with the panel packer reading compact records). Tiles 0-3 C, 4-5 A, 6-7 B.
#if defined(__AMX_BF16__) && defined(__AVX512BF16__) && defined(__linux__)
#define KHC_AMX 1
// AMX only where kurn's KURN buffer type would use it: not with GGML_KURN_AMX=0 or KURN_HC_AMX=0, and not when kurn's
// AMX self-test turned AMX off (kurn_amx_check() in libggml-cpu, weak so that kurn-hc.o still links without it; then
// just the XTILEDATA permission). Otherwise compact and expanded layers would run different kernels.
bool amx_ok() {
    static const bool v = [] {
        if (env_int("KURN_HC_AMX", 1) == 0 || env_int("GGML_KURN_AMX", 1) == 0) {
            return false;
        }
        if (kurn_amx_check) {
            return kurn_amx_check() != KURN_AMX_UNAVAILABLE;
        }
        return syscall(SYS_arch_prctl, 0x1023, 18) == 0;
    }();
    return v;
}

void amx_config_bf16() {
    struct alignas(64) {
        uint8_t palette, start_row, reserved[14];
        uint16_t colsb[16];
        uint8_t rows[16];
    } c = {};
    c.palette = 1;
    for (int t = 0; t < 8; t++) { c.rows[t] = 16; c.colsb[t] = 64; }
    _tile_loadconfig(&c);
}

// one kurn i16 Q8_0 record (16 rows x 32 values) -> one B tile (16 k-pairs x 16 rows x 2 bf16) of value s * d * q,
// with kurn-buft.cpp pack_bf16_q8_0's arithmetic (float(q) * d, round to nearest even)
inline void rec_to_tile(const uint8_t * rec, float sgn, uint16_t * o) {
    const __m512i ev = _mm512_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30);
    const __m512i od = _mm512_setr_epi32(1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31);
    const __m512i b128 = _mm512_set1_epi32(128);
    __m512 d = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *) rec));
    if (sgn < 0) d = _mm512_sub_ps(_mm512_setzero_ps(), d);
    __m512 dx[4];
    for (int i = 0; i < 4; i++) {
        const __m512i rep = _mm512_setr_epi32(4 * i, 4 * i, 4 * i, 4 * i, 4 * i + 1, 4 * i + 1, 4 * i + 1, 4 * i + 1, 4 * i + 2,
                                              4 * i + 2, 4 * i + 2, 4 * i + 2, 4 * i + 3, 4 * i + 3, 4 * i + 3, 4 * i + 3);
        dx[i] = _mm512_permutexvar_ps(rep, d);
    }
    for (int kk = 0; kk < 8; kk++) {
        __m512 f[4];
        for (int i = 0; i < 4; i++) {
            const __m512i v = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *) (rec + 32 + kk * 64 + 16 * i)));
            f[i] = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_sub_epi32(v, b128)), dx[i]);
        }
        const __m512i b01 = (__m512i) _mm512_cvtne2ps_pbh(f[1], f[0]);
        const __m512i b23 = (__m512i) _mm512_cvtne2ps_pbh(f[3], f[2]);
        _mm512_store_si512((__m512i *) (o + (2 * kk) * 32), _mm512_permutex2var_epi32(b01, ev, b23));
        _mm512_store_si512((__m512i *) (o + (2 * kk + 1) * 32), _mm512_permutex2var_epi32(b01, od, b23));
    }
}

// expanded rows [R0, R0 + rows) x expanded values [k0, k0 + kc) -> B tiles, tile (h, t) at out + (h * nt + t) * 512
void pack_q8(const kurn_hc * w, int64_t R0, int64_t rows, int64_t k0, int64_t kc, uint16_t * out) {
    const int D = w->D;
    const int64_t nkb = w->Kq / 32, nt = kc / 32;
    for (int64_t h = 0; h < rows / 16; h++) {
        const int64_t R = R0 + 16 * h, nbk = R / (32 * D), o = (R % (32 * D)) / 32, n = nbk * 32 + R % 32;
        const int64_t g = n / 16;
        for (int64_t t = 0; t < nt; t++) {
            const int64_t b = k0 / 32 + t, kb = b / D, q = b % D;
            const uint8_t * rec = w->buf + (((size_t) g * nkb + kb) * D + comp_of(D, (int) o, (int) q)) * REC;
            rec_to_tile(rec, (float) sign_of(D, (int) o, (int) q), out + (h * nt + t) * 512);
        }
    }
}

// f32 compact -> B tiles of bf16(s * w)
void pack_f32(const kurn_hc * w, int64_t R0, int64_t rows, int64_t k0, int64_t kc, uint16_t * out) {
    const int D = w->D;
    const int64_t nt = kc / 32;
    const float * W = (const float *) w->buf;
    alignas(64) float row[16][32];
    for (int64_t h = 0; h < rows / 16; h++) {
        const int64_t R = R0 + 16 * h, nbk = R / (32 * D), o = (R % (32 * D)) / 32, n = nbk * 32 + R % 32;
        const int64_t g = n / 16;
        for (int64_t t = 0; t < nt; t++) {
            const int64_t b = k0 / 32 + t, kb = b / D, q = b % D;
            const int p = comp_of(D, (int) o, (int) q);
            const float s = (float) sign_of(D, (int) o, (int) q);
            for (int i = 0; i < 32; i++) {
                const float * src = W + (((size_t) g * w->Kq + kb * 32 + i) * D + p) * 16;
                for (int r = 0; r < 16; r++) row[r][i] = s * src[r];
            }
            uint16_t * ot = out + (h * nt + t) * 512;
            for (int r = 0; r < 16; r++) {
                alignas(64) uint16_t bf[32];
                _mm512_store_si512((__m512i *) bf, (__m512i) _mm512_cvtne2ps_pbh(_mm512_load_ps(row[r] + 16), _mm512_load_ps(row[r])));
                for (int kp = 0; kp < 16; kp++) {
                    ot[kp * 32 + r * 2] = bf[2 * kp];
                    ot[kp * 32 + r * 2 + 1] = bf[2 * kp + 1];
                }
            }
        }
    }
}

std::atomic<uint64_t> g_redo{0};

// Y[m * ldy + R] for R in [r0, r1) (multiples of 16), m < M; A: bf16 rows of K values, padded to a multiple of 32 rows
void amx_gemm(const kurn_hc * w, const uint16_t * A, int64_t M, float * Y, int64_t ldy, int64_t r0, int64_t r1) {
    const int64_t K = (int64_t) w->D * w->Kq;
    const int64_t P = 128, KC = std::min<int64_t>(K, 4096), kch = 32;
    const int64_t Mp = (M + 31) / 32 * 32, nmg = Mp / 32, npm = P / 32;
    static thread_local std::vector<uint8_t> bsv, csv;
    if (bsv.size() < (size_t) P * KC * 2 + 64) bsv.resize((size_t) P * KC * 2 + 64);
    if (csv.size() < (size_t) 2 * nmg * npm * 4096 + 64) csv.resize((size_t) 2 * nmg * npm * 4096 + 64);
    uint16_t * bs = (uint16_t *) (((uintptr_t) bsv.data() + 63) & ~(uintptr_t) 63);
    float * cst = (float *) (((uintptr_t) csv.data() + 63) & ~(uintptr_t) 63);
    alignas(64) float sub[2][1024];
    const uint64_t guard = 4096 + (uint64_t) kch * 128;
    for (int64_t n0 = r0; n0 < r1; n0 += P) {
        const int64_t rows = std::min(P, r1 - n0), ng = rows / 16, npair = (ng + 1) / 2;
        int64_t c = 0;
        for (int64_t kc0 = 0; kc0 < K; kc0 += KC, c++) {
            const int64_t kc = std::min(KC, K - kc0), nt = kc / 32;
            const bool last = kc0 + kc >= K;
            if (w->fmt == KURN_HC_Q8) pack_q8(w, n0, rows, kc0, kc, bs);
            else pack_f32(w, n0, rows, kc0, kc, bs);
            amx_config_bf16();
            for (int64_t mg = 0; mg < nmg; mg++) {
                const uint16_t * a0 = A + mg * 32 * K + kc0;
                const uint16_t * a1 = a0 + 16 * K;
                for (int64_t j = 0; j < npair; j++) {
                    const int64_t h = 2 * j;
                    const bool two = h + 1 < ng;
                    const uint16_t * b0 = bs + h * nt * 512;
                    const uint16_t * b1 = bs + (h + 1) * nt * 512;
                    const float * prev = cst + (((c + 1) & 1) * nmg * npm + mg * npm + j) * 1024;
                    float * cur = cst + ((c & 1) * nmg * npm + mg * npm + j) * 1024;
                    for (int64_t s0 = 0, si = 0; s0 < nt; s0 += kch, si++) {
                        const int64_t s1 = std::min(nt, s0 + kch);
                        const float * in = s0 == 0 ? (c == 0 ? nullptr : prev) : sub[(si + 1) & 1];
                        float * outp = s1 == nt ? cur : sub[si & 1];
                        for (;;) {
                            const uint64_t t0 = __rdtsc();
                            if (!in) {
                                _tile_zero(0); _tile_zero(1); _tile_zero(2); _tile_zero(3);
                            } else {
                                _tile_loadd(0, in, 64); _tile_loadd(2, in + 512, 64);
                                if (two) { _tile_loadd(1, in + 256, 64); _tile_loadd(3, in + 768, 64); }
                            }
                            if (two) {
                                for (int64_t t = s0; t < s1; t++) {
                                    _tile_loadd(4, a0 + t * 32, K * 2);
                                    _tile_loadd(6, b0 + t * 512, 64);
                                    _tile_loadd(5, a1 + t * 32, K * 2);
                                    _tile_loadd(7, b1 + t * 512, 64);
                                    _tile_dpbf16ps(0, 4, 6);
                                    _tile_dpbf16ps(1, 4, 7);
                                    _tile_dpbf16ps(2, 5, 6);
                                    _tile_dpbf16ps(3, 5, 7);
                                }
                            } else {
                                for (int64_t t = s0; t < s1; t++) {
                                    _tile_loadd(4, a0 + t * 32, K * 2);
                                    _tile_loadd(6, b0 + t * 512, 64);
                                    _tile_loadd(5, a1 + t * 32, K * 2);
                                    _tile_dpbf16ps(0, 4, 6);
                                    _tile_dpbf16ps(2, 5, 6);
                                }
                            }
                            _tile_stored(0, outp, 64);
                            _tile_stored(2, outp + 512, 64);
                            if (two) { _tile_stored(1, outp + 256, 64); _tile_stored(3, outp + 768, 64); }
                            unsigned aux;
                            if (__rdtscp(&aux) - t0 > guard) {
                                g_redo.fetch_add(1, std::memory_order_relaxed);
                                continue;
                            }
                            break;
                        }
                    }
                    if (!last) continue;
                    for (int qd = 0; qd < 4; qd++) {
                        if (!two && (qd & 1)) continue;
                        const int64_t mb = mg * 32 + (qd >> 1) * 16, nb = n0 + (h + (qd & 1)) * 16;
                        for (int m = 0; m < 16 && mb + m < M; m++)
                            _mm512_storeu_ps(Y + (mb + m) * ldy + nb, _mm512_load_ps(cur + qd * 256 + m * 16));
                    }
                }
            }
        }
    }
}
#else
#define KHC_AMX 0
#endif

}  // namespace

// ------------------------------------------------------------------------------------------
extern "C" {

void kurn_hc_quantize_q8_0(const float * x, void * out, int64_t n) {
    uint8_t * o = (uint8_t *) out;
    for (int64_t b = 0; b < n / 32; b++) {
        uint16_t dh;
        quant_act32(x + b * 32, dh, (int8_t *) (o + b * 34 + 2));
        memcpy(o + b * 34, &dh, 2);
    }
}

kurn_hc * kurn_hc_pack(int D, int64_t Nq, int64_t Kq, const float * W, kurn_hc_format fmt) {
    if ((D != 2 && D != 4) || Nq % 32 || Kq % 32 || Nq <= 0 || Kq <= 0) return nullptr;
    kurn_hc * w = new kurn_hc{D, Nq, Kq, fmt, nullptr, 0};
    const int64_t ng = Nq / 16, nkb = Kq / 32;
    if (fmt == KURN_HC_Q8) {
        w->bytes = (size_t) ng * nkb * D * REC;
        w->buf = (uint8_t *) aligned_alloc(64, (w->bytes + 63) & ~(size_t) 63);
        for (int64_t g = 0; g < ng; g++)
            for (int64_t kb = 0; kb < nkb; kb++)
                for (int p = 0; p < D; p++) {
                    uint8_t * rec = w->buf + ((g * nkb + kb) * D + p) * REC;
                    for (int r = 0; r < 16; r++) {
                        uint16_t dh;
                        int8_t q[32];
                        quant_ref32(W + ((size_t) p * Nq + g * 16 + r) * Kq + kb * 32, dh, q);
                        memcpy(rec + 2 * r, &dh, 2);
                        for (int kk = 0; kk < 8; kk++)
                            for (int j = 0; j < 4; j++) rec[32 + kk * 64 + r * 4 + j] = (uint8_t) (q[4 * kk + j] + 128);
                    }
                }
    } else {
        w->bytes = (size_t) Nq * Kq * D * 4;
        w->buf = (uint8_t *) aligned_alloc(64, (w->bytes + 63) & ~(size_t) 63);
        float * o = (float *) w->buf;
        for (int64_t g = 0; g < ng; g++)
            for (int64_t k = 0; k < Kq; k++)
                for (int p = 0; p < D; p++)
                    for (int r = 0; r < 16; r++) o[((g * Kq + k) * D + p) * 16 + r] = W[((size_t) p * Nq + g * 16 + r) * Kq + k];
    }
    return w;
}

void kurn_hc_free(kurn_hc * w) {
    if (w) {
        free(w->buf);
        delete w;
    }
}

size_t kurn_hc_bytes(const kurn_hc * w) { return w->bytes; }

size_t kurn_hc_expanded_bytes(const kurn_hc * w) {
    const int64_t N = (int64_t) w->D * w->Nq, K = (int64_t) w->D * w->Kq;
    return w->fmt == KURN_HC_Q8 ? (size_t) N * (K / 32) * 34 : (size_t) N * K * 4;
}

void kurn_hc_expand(const kurn_hc * w, void * out) {
    const int D = w->D;
    const int64_t N = (int64_t) D * w->Nq, K = (int64_t) D * w->Kq, nkb = w->Kq / 32;
    for (int64_t R = 0; R < N; R++) {
        const int64_t nbk = R / (32 * D), o = (R % (32 * D)) / 32, n = nbk * 32 + R % 32, g = n / 16, r = n % 16;
        for (int64_t b = 0; b < K / 32; b++) {
            const int64_t kb = b / D, q = b % D;
            const int p = comp_of(D, (int) o, (int) q), s = sign_of(D, (int) o, (int) q);
            if (w->fmt == KURN_HC_Q8) {
                const uint8_t * rec = w->buf + ((g * nkb + kb) * D + p) * REC;
                uint8_t * blk = (uint8_t *) out + (R * (K / 32) + b) * 34;
                memcpy(blk, rec + 2 * r, 2);
                for (int i = 0; i < 32; i++) {
                    const int v = (int) rec[32 + (i / 4) * 64 + r * 4 + i % 4] - 128;
                    blk[2 + i] = (uint8_t) (int8_t) (s * v);
                }
            } else {
                const float * W = (const float *) w->buf;
                float * row = (float *) out + R * K + b * 32;
                for (int i = 0; i < 32; i++) row[i] = s * W[((g * w->Kq + kb * 32 + i) * D + p) * 16 + r];
            }
        }
    }
}

void kurn_hc_matmul(const kurn_hc * w, const float * X, float * Y, int64_t M, int nth, int exact) {
    const int D = w->D;
    const int64_t K = (int64_t) D * w->Kq, N = (int64_t) D * w->Nq, NG = w->Nq / 16;
    if (nth < 1) nth = 1;
#if KHC_AMX
    static const int amx_min = std::max(16, env_int("KURN_HC_AMX_MIN", 32));
    if (!exact && M >= amx_min && amx_ok()) {
        const int64_t Mp = (M + 31) / 32 * 32;
        // 64-byte aligned: every A tile row must be one cache line (malloc's large blocks start 16 bytes into a page)
        std::vector<uint16_t> Av((size_t) Mp * K + 64);
        uint16_t * A = (uint16_t *) (((uintptr_t) Av.data() + 63) & ~(uintptr_t) 63);
        std::atomic<int64_t> next{0};
        const int64_t P = 64, npan = (N + P - 1) / P;  // scheduling unit: half an L2 panel, as kurn's row_chunks (64 rows)
#pragma omp parallel num_threads(nth)
        {
#pragma omp for schedule(static)
            for (int64_t c = 0; c < Mp; c++) {
                uint16_t * dst = A + c * K;
                if (c >= M) { memset(dst, 0, (size_t) K * 2); continue; }
                const float * src = X + c * K;
                for (int64_t k = 0; k < K; k += 32)
                    _mm512_storeu_si512((__m512i *) (dst + k), (__m512i) _mm512_cvtne2ps_pbh(_mm512_loadu_ps(src + k + 16), _mm512_loadu_ps(src + k)));
            }
            syscall(SYS_arch_prctl, 0x1023, 18);
            for (int64_t p; (p = next.fetch_add(1)) < npan;) amx_gemm(w, A, M, Y, N, p * P, std::min(N, (p + 1) * P));
        }
        return;
    }
#endif
    if (w->fmt == KURN_HC_F32) {
        std::vector<slot> slots(nth);
#pragma omp parallel num_threads(nth)
        {
            const int ith = omp_get_thread_num(), n = omp_get_num_threads();
            steal st{slots.data(), n, 1};
            st.init(ith, NG);
#pragma omp barrier
            for (int64_t u0, u1; st.next(ith, u0, u1);) {
                if (D == 4) f32_rows<4>(w, X, K, M, Y, N, u0, u1);
                else f32_rows<2>(w, X, K, M, Y, N, u0, u1);
            }
        }
        return;
    }
    // Q8: activations quantized once (ggml's quantize_row_q8_0), then GEMV / verify kernels over column groups of 8
    const int64_t nb = K / 32;
    std::vector<int8_t> xq((size_t) M * nb * 32);
    std::vector<float> dx((size_t) M * nb);
    std::vector<int32_t> nc((size_t) M * nb);
    std::vector<slot> slots(nth);
    const xprep xp{nb, xq.data(), dx.data(), nc.data()};
#pragma omp parallel num_threads(nth)
    {
        const int ith = omp_get_thread_num(), n = omp_get_num_threads();
#pragma omp for schedule(static)
        for (int64_t i = 0; i < M * nb; i++) {
            uint16_t dh;
            quant_act32(X + (i / nb) * K + (i % nb) * 32, dh, xq.data() + i * 32);
            dx[i] = h2f(dh);
            int32_t s = 0;
            for (int j = 0; j < 32; j++) s += xq[i * 32 + j];
            nc[i] = -128 * s;
        }
        if (M <= 8) {
            const int G = q8_groups(D, (int) M);
            const q8_fn fn = D == 4 ? q8_kernel<4>((int) M, G) : q8_kernel<2>((int) M, G);
            steal st{slots.data(), n, G};
            st.init(ith, NG);
#pragma omp barrier
            for (int64_t u0, u1; st.next(ith, u0, u1);) fn(w, xp, 0, Y, N, u0, u1);
        } else {
            // more columns: row chunks that stay in L2 while every 8-column group passes over them
            const int64_t chunk = std::max<int64_t>(1, (1 << 20) / ((int64_t) REC * D * (w->Kq / 32)));
            const int64_t nchunk = (NG + chunk - 1) / chunk;
#pragma omp barrier
#pragma omp for schedule(dynamic, 1)
            for (int64_t c = 0; c < nchunk; c++) {
                const int64_t g0 = c * chunk, g1 = std::min(NG, g0 + chunk);
                for (int64_t m0 = 0; m0 < M; m0 += 8) {
                    const int mm = (int) std::min<int64_t>(8, M - m0);
                    const q8_fn fn = D == 4 ? q8_kernel<4>(mm, 1) : q8_kernel<2>(mm, 1);
                    fn(w, xp, m0, Y + m0 * N, N, g0, g1);
                }
            }
        }
    }
}

}  // extern "C"
