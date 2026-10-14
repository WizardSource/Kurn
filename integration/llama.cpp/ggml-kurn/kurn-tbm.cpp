// Ternary weights as a presence bitmap plus packed signs ("tbm"; layout spec and reference encoder in
// kurn.ternary). TQ1_0 / TQ2_0 tensors whose zero fraction makes it worthwhile are repacked into this
// layout by the KURN buffer type at load time.
//
// A record is 16 rows x 256 values (one Q8_K activation block): 16 fp16 row scales, 64 bitmap words of
// 64 bits, a flag byte, then one sign bit per nonzero. Word i = kg * 8 + kk covers values kg*32 + kk*4 + j
// (j < 4) of the 16 rows, bit b <-> row b / 4, j = b % 4: exactly the bytes of the vpdpbusd operand
// (lane = row) that kurn's TQ2_0 kernels build from their 2-bit codes. Decoding a word:
//   P = bitmap word, S = pdep(next popcount(P) sign bits, P), u = P ? (S ? 2 : 0) : 1 = 1 + w
// u is TQ2_0's code, so the per-32-value integer dot (seeded with -sum(x)) and the float epilogue
//   acc = fma(float(isum), d_w * d_x, acc), in value order
// are those of the TQ2_0 GEMV and verify kernels: results are bit-identical to TQ2_0 at every width.
// Records are variable length (ceil(nnz / 8) sign bytes); a table gives each 16-row group's start.
// Signs are read as the 64 bits at byte offset / 8, shifted by offset % 8 (one load); a record where some
// word's signs do not fit that window (only when a word has > 57 nonzeros) is flagged and decoded with
// a second load.
#include "kurn-tbm.h"

#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__) && defined(__BMI2__)

#include <immintrin.h>
#include <x86intrin.h>
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#if defined(__AMX_INT8__) && defined(__linux__)
#define TBM_AMX 1
#else
#define TBM_AMX 0
#endif

namespace {

constexpr int HDR = 64, REC_FIXED = 32 + 512 + 1, SLACK = 64;
constexpr int QK = 256, XBLK = 292;  // Q8_K: float d, int8 qs[256], int16 bsums[16]

struct tbm_hdr {
    char magic[4];
    uint32_t version;
    int64_t K, N, ngroups, nrec_k, used, dense;
};
static_assert(sizeof(tbm_hdr) <= HDR, "header");

inline size_t table_bytes(int64_t ng) { return ((size_t) (ng + 1) * 8 + 63) & ~(size_t) 63; }

inline uint64_t ld64(const uint8_t * p) { uint64_t v; memcpy(&v, p, 8); return v; }

// ---------------------------------------------------------------------------------------------- packing
// value v (0..255) of a native block as a trit -1..1 (q = 3, which ternary TQ2_0 never holds, as 2):
// byte (v / 128) * 32 + v % 32, bits 2 * ((v % 128) / 32)
void trits_tq2_0(const uint8_t * b, int8_t * t) {
    const __m256i m3 = _mm256_set1_epi8(3), one = _mm256_set1_epi8(1);
    for (int h = 0; h < 2; h++) {
        const __m256i q = _mm256_loadu_si256((const __m256i *) (b + 32 * h));
        for (int s = 0; s < 4; s++)
            _mm256_storeu_si256((__m256i *) (t + 128 * h + 32 * s),
                                _mm256_sub_epi8(_mm256_and_si256(_mm256_srli_epi16(q, 2 * s), m3), one));
    }
}

inline int8_t tq1_digit(uint8_t byte, int n) {
    static const uint8_t pow3[6] = {1, 3, 9, 27, 81, 243};
    const uint8_t q = (uint8_t) (byte * pow3[n]);
    return (int8_t) (((uint16_t) q * 3) >> 8);
}

// ggml's dequantize_row_tq1_0 order: qs[0..32) x 5 digits, qs[32..48) x 5, qh[0..4) x 4
void trits_tq1_0(const uint8_t * b, int8_t * t) {
    int v = 0;
    for (int n = 0; n < 5; n++) for (int m = 0; m < 32; m++) t[v++] = (int8_t) (tq1_digit(b[m], n) - 1);
    for (int n = 0; n < 5; n++) for (int m = 0; m < 16; m++) t[v++] = (int8_t) (tq1_digit(b[32 + m], n) - 1);
    for (int n = 0; n < 4; n++) for (int j = 0; j < 4; j++) t[v++] = (int8_t) (tq1_digit(b[48 + j], n) - 1);
}

struct src_fmt {
    size_t bytes;
    int d_off;
    void (*trits)(const uint8_t *, int8_t *);
};

bool src_of(int type, src_fmt & f) {
    if (type == 35) { f = {66, 64, trits_tq2_0}; return true; }  // GGML_TYPE_TQ2_0
    if (type == 34) { f = {54, 52, trits_tq1_0}; return true; }  // GGML_TYPE_TQ1_0
    return false;
}

struct bit_writer {
    uint8_t * p;
    uint64_t acc = 0;
    int n = 0;
    explicit bit_writer(uint8_t * q) : p(q) {}
    void put(uint64_t v, int c) {  // low c bits of v, c <= 64
        if (!c) return;
        acc |= v << n;
        if (n + c >= 64) {
            memcpy(p, &acc, 8);
            p += 8;
            acc = n ? v >> (64 - n) : 0;
            n = n + c - 64;
        } else {
            n += c;
        }
    }
    uint8_t * flush() {
        const int nb = (n + 7) / 8;
        memcpy(p, &acc, (size_t) nb);
        return p + nb;
    }
};

// one record from 16 rows of trits t[16][256] and scales d[16] (fp16 bits); returns its end, or nullptr
// when a value is not ternary
uint8_t * pack_record(const int8_t (*t)[QK], const uint16_t * d, uint8_t * o, bool & dense) {
    memcpy(o, d, 32);
    uint8_t * bm = o + 32;
    bit_writer bw(o + REC_FIXED);
    int64_t off = 0;
    bool fl = false;
    alignas(64) uint32_t g[16];
    for (int i = 0; i < 64; i++) {
        const int kg = i / 8, kk = i % 8;
        for (int r = 0; r < 16; r++) memcpy(&g[r], &t[r][kg * 32 + kk * 4], 4);
        const __m512i v = _mm512_load_si512((const __m512i *) g);
        if (_mm512_cmpgt_epi8_mask(v, _mm512_set1_epi8(1))) return nullptr;
        const uint64_t P = _cvtmask64_u64(_mm512_test_epi8_mask(v, v));
        const uint64_t pos = _cvtmask64_u64(_mm512_cmpgt_epi8_mask(v, _mm512_setzero_si512()));
        const int n = (int) _mm_popcnt_u64(P);
        if (n > 64 - (int) (off & 7)) fl = true;
        memcpy(bm + 8 * i, &P, 8);
        bw.put(_pext_u64(pos, P), n);
        off += n;
    }
    o[REC_FIXED - 1] = fl ? 1 : 0;
    dense = fl;
    return bw.flush();
}

// ---------------------------------------------------------------------------------------------- kernels
const tbm_hdr * hdr_of(const void * buf) { return (const tbm_hdr *) buf; }
const uint64_t * table_of(const void * buf) { return (const uint64_t *) ((const uint8_t *) buf + HDR); }
const uint8_t * data_of(const void * buf) {
    return (const uint8_t *) buf + HDR + table_bytes(hdr_of(buf)->ngroups);
}

// software prefetch of each row group's record stream, PF_BYTES ahead (two lines per 32-value chunk, about one
// record); GGML_KURN_TBM_PF=0 turns it off. Weights streaming from DRAM (Ternary-Bonsai-8B shapes, 8 threads): 22.4 ms
// without, 17.3 at 1024 bytes, 17.6 at 2048, 18.9 at 4096; the TQ2_0 kernel with whole-record prefetch 20.3.
constexpr int PF_BYTES = 1024;
bool pf_on() {
    static const bool v = [] {
        const char * e = getenv("GGML_KURN_TBM_PF");
        return !(e && *e && atoi(e) == 0);
    }();
    return v;
}

template <bool SAFE>
inline __m512i decode_word(const uint8_t * bm, const uint8_t * sp, uint64_t & off, __m512i ones, __m512i twos) {
    const uint64_t P = ld64(bm);
    const unsigned sh = (unsigned) (off & 7);
    uint64_t w = ld64(sp + (off >> 3)) >> sh;
    if (SAFE) w |= (ld64(sp + (off >> 3) + 8) << 1) << (63 - sh);
    const uint64_t S = _pdep_u64(w, P);
    off += (uint64_t) _mm_popcnt_u64(P);
    return _mm512_mask_blend_epi8(_cvtu64_mask64(P), ones, _mm512_maskz_mov_epi8(_cvtu64_mask64(S), twos));
}

inline void storev(float * y, int64_t row0, int64_t r0, int64_t r1, __m512 v) {
    if (row0 >= r0 && row0 + 16 <= r1) { _mm512_storeu_ps(y + row0, v); return; }
    uint32_t m = 0xFFFF;
    if (row0 < r0) m &= 0xFFFFu << (r0 - row0);
    if (row0 + 16 > r1) m &= (1u << (r1 > row0 ? r1 - row0 : 0)) - 1;
    _mm512_mask_storeu_ps(y + row0, (__mmask16) m, v);
}

// R row groups x C columns, one record of each row group: 8 chunks of 32 values, per chunk and (row
// group, column) an int32 dot seeded with -sum(x) (two chains when there are few (row group, column)
// pairs), then acc = fma(float(isum), d_w * d_x, acc).
template <int R, int C, bool SAFE, bool PF>
inline void record(const uint8_t * const * rec, const uint8_t * const * xb, __m512 (*acc)[C], const uint8_t ** next) {
    constexpr bool TWO = R * C < 4;
    const __m512i ones = _mm512_set1_epi8(1), twos = _mm512_set1_epi8(2);
    __m512 d[R];
    uint64_t off[R];
    const uint8_t * sp[R];
    for (int r = 0; r < R; r++) {
        d[r] = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *) rec[r]));
        sp[r] = rec[r] + REC_FIXED;
        off[r] = 0;
    }
    float dx[C];
    for (int c = 0; c < C; c++) memcpy(&dx[c], xb[c], 4);
    for (int kg = 0; kg < 8; kg++) {
        if (PF) {
            for (int r = 0; r < R; r++) {
                _mm_prefetch((const char *) rec[r] + PF_BYTES + kg * 128, _MM_HINT_T0);
                _mm_prefetch((const char *) rec[r] + PF_BYTES + kg * 128 + 64, _MM_HINT_T0);
            }
        }
        __m512i e[R][C], o[R][C];
        for (int c = 0; c < C; c++) {
            int16_t b0, b1;
            memcpy(&b0, xb[c] + 4 + QK + 4 * kg, 2);
            memcpy(&b1, xb[c] + 4 + QK + 4 * kg + 2, 2);
            const __m512i seed = _mm512_set1_epi32(-(b0 + b1));
            for (int r = 0; r < R; r++) {
                e[r][c] = seed;
                if (TWO) o[r][c] = _mm512_setzero_si512();
            }
        }
#pragma GCC unroll 8
        for (int kk = 0; kk < 8; kk++) {
            __m512i xv[C];
            for (int c = 0; c < C; c++) {
                int32_t q;
                memcpy(&q, xb[c] + 4 + kg * 32 + kk * 4, 4);
                xv[c] = _mm512_set1_epi32(q);
            }
            for (int r = 0; r < R; r++) {
                const __m512i u = decode_word<SAFE>(rec[r] + 32 + 8 * (kg * 8 + kk), sp[r], off[r], ones, twos);
                for (int c = 0; c < C; c++) {
                    if (TWO && (kk & 1)) o[r][c] = _mm512_dpbusd_epi32(o[r][c], u, xv[c]);
                    else e[r][c] = _mm512_dpbusd_epi32(e[r][c], u, xv[c]);
                }
            }
        }
        for (int r = 0; r < R; r++)
            for (int c = 0; c < C; c++) {
                const __m512i s = TWO ? _mm512_add_epi32(e[r][c], o[r][c]) : e[r][c];
                acc[r][c] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(s), _mm512_mul_ps(d[r], _mm512_set1_ps(dx[c])), acc[r][c]);
            }
    }
    for (int r = 0; r < R; r++) next[r] = sp[r] + ((off[r] + 7) >> 3);
}

template <int R, int C, bool PF>
void groups(const void * buf, const uint8_t * X, size_t xrow, float * Y, int64_t N, int64_t g0, int64_t g1, int64_t r0,
            int64_t r1) {
    const tbm_hdr * h = hdr_of(buf);
    const uint64_t * tab = table_of(buf);
    const uint8_t * data = data_of(buf);
    for (int64_t g = g0; g < g1; g += R) {
        const uint8_t * rec[R];
        __m512 acc[R][C];
        for (int r = 0; r < R; r++) {
            rec[r] = data + tab[g + r];
            for (int c = 0; c < C; c++) acc[r][c] = _mm512_setzero_ps();
        }
        for (int64_t p = 0; p < h->nrec_k; p++) {
            const uint8_t * xb[C];
            for (int c = 0; c < C; c++) xb[c] = X + c * xrow + p * XBLK;
            bool dense = false;
            for (int r = 0; r < R; r++) dense |= rec[r][REC_FIXED - 1] != 0;
            const uint8_t * next[R];
            if (dense) record<R, C, true, PF>(rec, xb, acc, next);
            else record<R, C, false, PF>(rec, xb, acc, next);
            for (int r = 0; r < R; r++) rec[r] = next[r];
        }
        for (int r = 0; r < R; r++)
            for (int c = 0; c < C; c++) storev(Y + c * N, (g + r) * 16, r0, r1, acc[r][c]);
    }
}

// row groups per pass for C columns (register budget: R * C (row group, column) pairs); GEMV: GGML_KURN_TBM_ROWS
int gemv_rows() {
    static const int v = [] {
        const char * e = getenv("GGML_KURN_TBM_ROWS");
        return e && *e ? atoi(e) : 4;
    }();
    return v;
}

template <int R, int C>
void cols_r(const void * buf, const uint8_t * X, size_t xrow, float * Y, int64_t N, int64_t r0, int64_t r1) {
    const int64_t g0 = r0 / 16, g1 = (r1 + 15) / 16;
    const int64_t gm = g0 + (g1 - g0) / R * R;
    const bool pf = pf_on();
    if (gm > g0) (pf ? groups<R, C, true> : groups<R, C, false>)(buf, X, xrow, Y, N, g0, gm, r0, r1);
    if (g1 > gm) (pf ? groups<1, C, true> : groups<1, C, false>)(buf, X, xrow, Y, N, gm, g1, r0, r1);
}

template <int C>
void cols_c(const void * buf, const uint8_t * X, size_t xrow, float * Y, int64_t N, int64_t r0, int64_t r1) {
    if (C == 1) {
        switch (gemv_rows()) {
            case 2: cols_r<2, C>(buf, X, xrow, Y, N, r0, r1); return;
            case 8: cols_r<8, C>(buf, X, xrow, Y, N, r0, r1); return;
            case 6: cols_r<6, C>(buf, X, xrow, Y, N, r0, r1); return;
            default: cols_r<4, C>(buf, X, xrow, Y, N, r0, r1); return;
        }
    }
    // 2..4 columns: two row-group streams per pass (Ternary-Bonsai-8B shapes, 3 columns: 24.1-26.4 ms vs 28.4-31.4 with
    // one); 5..8 columns: one (two spill)
    cols_r<C <= 4 ? 2 : 1, C>(buf, X, xrow, Y, N, r0, r1);
}

#if TBM_AMX
// AMX-INT8 prefill. Panels of AG row groups x KC records: the panel's words are decoded once into s8 B
// tiles (per 32-value chunk: 8 lines of 64 bytes, line kk = word kg * 8 + kk as signed weights), then
// every 16-column group runs tdpbssd per chunk (exact isum = sum(w * x)) with the GEMV's epilogue
// acc = fma(float(isum), d_w * d_x, acc) in value order, accumulators in registers across the panel and
// in Y between panels. Same palette as ggml's AMX code (tmm0/1 B 8 x 64 B, tmm2/3 A 16 x 32 B, tmm4..7 C).
// Every step is timed and recomputed from Y when it spanned a preemption-length gap (KVM guests that drop
// tile data on preemption; see kurn-buft.cpp).
constexpr int AG = 4, KC_MAX = 8;  // KC_MAX records = 64 chunks of 32 values

void amx_config() {
    struct alignas(64) {
        uint8_t palette, start_row, reserved[14];
        uint16_t colsb[16];
        uint8_t rows[16];
    } c = {};
    c.palette = 1;
    c.rows[0] = 8;  c.colsb[0] = 64;
    c.rows[1] = 8;  c.colsb[1] = 64;
    c.rows[2] = 16; c.colsb[2] = 32;
    c.rows[3] = 16; c.colsb[3] = 32;
    for (int t = 4; t < 8; t++) { c.rows[t] = 16; c.colsb[t] = 64; }
    _tile_loadconfig(&c);
}

int env_i(const char * n, int def) {
    const char * v = getenv(n);
    return v && *v ? atoi(v) : def;
}

std::atomic<uint64_t> g_redo{0};
struct redo_report {
    ~redo_report() {
        if (env_i("GGML_KURN_VERBOSE", 0) && g_redo.load())
            fprintf(stderr, "kurn: tbm AMX steps recomputed after a preemption-length gap: %llu\n", (unsigned long long) g_redo.load());
    }
} g_redo_report;

inline void tiles(const uint8_t * xa, size_t xrow, const int8_t * bs, int set, int32_t * cbuf) {
    if (set == 0) {
        _tile_loadd(2, xa, (long) xrow); _tile_loadd(0, bs, 64); _tile_zero(4); _tile_dpbssd(4, 2, 0); _tile_stored(4, cbuf, 64);
    } else {
        _tile_loadd(3, xa, (long) xrow); _tile_loadd(1, bs, 64); _tile_zero(5); _tile_dpbssd(5, 3, 1); _tile_stored(5, cbuf, 64);
    }
}

void amx_rows(const void * buf, const uint8_t * X, size_t xrow, float * Y, int64_t N, int64_t M, int64_t r0, int64_t r1) {
    alignas(64) static thread_local int8_t bs[AG * KC_MAX * 8 * 512];
    alignas(64) static thread_local float dws[AG * KC_MAX * 16];
    alignas(64) int32_t cbuf[2][256];
    const tbm_hdr * h = hdr_of(buf);
    const uint64_t * tab = table_of(buf);
    const uint8_t * data = data_of(buf);
    const int64_t nrk = h->nrec_k;
    static const int64_t KC = std::max(1, std::min(KC_MAX, env_i("GGML_KURN_TBM_AMX_KC", 4)));
    static const uint64_t guard_base = (uint64_t) env_i("GGML_KURN_AMX_GUARD", 8192);
    const __m512i m1 = _mm512_set1_epi8(-1), p1 = _mm512_set1_epi8(1);
    amx_config();
    const int64_t g0 = r0 / 16, g1 = (r1 + 15) / 16;
    for (int64_t gp = g0; gp < g1; gp += AG) {
        const int ng = (int) std::min<int64_t>(AG, g1 - gp);
        const uint8_t * rec[AG];
        for (int j = 0; j < ng; j++) rec[j] = data + tab[gp + j];
        for (int64_t kc = 0; kc < nrk; kc += KC) {
            const int64_t kn = std::min(KC, nrk - kc);
            for (int j = 0; j < ng; j++) {
                for (int64_t k = 0; k < kn; k++) {
                    const uint8_t * rp = rec[j];
                    _mm512_store_ps(dws + (j * KC_MAX + k) * 16, _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *) rp)));
                    const uint8_t * sp = rp + REC_FIXED;
                    uint64_t off = 0;
                    int8_t * o = bs + (size_t) (j * KC_MAX + k) * 8 * 512;
                    const bool dense = rp[REC_FIXED - 1] != 0;
                    for (int i = 0; i < 64; i++) {
                        const uint64_t P = ld64(rp + 32 + 8 * i);
                        const unsigned sh = (unsigned) (off & 7);
                        uint64_t w = ld64(sp + (off >> 3)) >> sh;
                        if (dense) w |= (ld64(sp + (off >> 3) + 8) << 1) << (63 - sh);
                        const uint64_t S = _pdep_u64(w, P);
                        off += (uint64_t) _mm_popcnt_u64(P);
                        _mm512_store_si512((__m512i *) (o + 64 * i),
                                           _mm512_maskz_mov_epi8(_cvtu64_mask64(P), _mm512_mask_blend_epi8(_cvtu64_mask64(S), m1, p1)));
                    }
                    rec[j] = sp + ((off + 7) >> 3);
                }
            }
            const uint64_t guard = guard_base + (uint64_t) kn * 8 * 128;
            for (int64_t m0 = 0; m0 < M; m0 += 16) {
                const uint8_t * xm = X + m0 * xrow;
                for (int j = 0; j < ng; j++) {
                    const int64_t row0 = (gp + j) * 16;
                    uint32_t mask = 0xFFFF;
                    if (row0 + 16 > r1) mask = (1u << (r1 - row0)) - 1;
                    float * Yb = Y + m0 * N + row0;
                    const int8_t * bj = bs + (size_t) j * KC_MAX * 8 * 512;
                    const float * dj = dws + (size_t) j * KC_MAX * 16;
                    for (;;) {
                        const uint64_t t0 = __rdtsc();
                        __m512 acc[16];
                        for (int m = 0; m < 16; m++)
                            acc[m] = kc == 0 ? _mm512_setzero_ps() : _mm512_maskz_loadu_ps((__mmask16) mask, Yb + m * N);
                        const int64_t nch = kn * 8;
                        tiles(xm + (kc * XBLK + 4), xrow, bj, 0, cbuf[0]);
                        for (int64_t s = 0; s < nch; s++) {
                            if (s + 1 < nch) {
                                const int64_t s1 = s + 1;
                                tiles(xm + (kc + s1 / 8) * XBLK + 4 + (s1 % 8) * 32, xrow, bj + s1 * 512, (int) (s1 & 1), cbuf[s1 & 1]);
                            }
                            const __m512 dw = _mm512_load_ps(dj + (s / 8) * 16);
                            const int32_t * cb = cbuf[s & 1];
                            for (int m = 0; m < 16; m++) {
                                float dx;
                                memcpy(&dx, xm + m * xrow + (kc + s / 8) * XBLK, 4);
                                acc[m] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_load_si512(cb + m * 16)), _mm512_mul_ps(dw, _mm512_set1_ps(dx)),
                                                         acc[m]);
                            }
                        }
                        unsigned aux;
                        if (__rdtscp(&aux) - t0 > guard) {
                            g_redo.fetch_add(1, std::memory_order_relaxed);
                            continue;
                        }
                        for (int m = 0; m < 16; m++) _mm512_mask_storeu_ps(Yb + m * N, (__mmask16) mask, acc[m]);
                        break;
                    }
                }
            }
        }
    }
}
#endif

}  // namespace

extern "C" {

int kurn_tbm_available(void) { return 1; }

size_t kurn_tbm_bytes(int64_t K, int64_t N) {
    const int64_t ng = (N + 15) / 16, nrk = K / QK;
    return ((size_t) HDR + table_bytes(ng) + (size_t) ng * nrk * (REC_FIXED + 512) + SLACK + 63) & ~(size_t) 63;
}

int64_t kurn_tbm_zeros(int type, const void * W, int64_t K, int64_t N, int64_t step) {
    src_fmt f;
    if (!src_of(type, f) || K % QK) return -1;
    const int64_t nrk = K / QK;
    int64_t z = 0;
    alignas(64) int8_t t[QK];
    for (int64_t n = 0; n < N; n += std::max<int64_t>(1, step)) {
        for (int64_t p = 0; p < nrk; p++) {
            f.trits((const uint8_t *) W + (n * nrk + p) * f.bytes, t);
            for (int v = 0; v < QK; v++) z += t[v] == 0;
        }
    }
    return z;
}

size_t kurn_tbm_pack(int type, const void * W, int64_t K, int64_t N, void * dst, int64_t * dense_out) {
    src_fmt f;
    if (!src_of(type, f) || K % QK) return 0;
    const int64_t ng = (N + 15) / 16, nrk = K / QK;
    const size_t wrow = (size_t) nrk * f.bytes;
    uint8_t * base = (uint8_t *) dst;
    uint64_t * tab = (uint64_t *) (base + HDR);
    uint8_t * data = base + HDR + table_bytes(ng);
    uint8_t * o = data;
    int64_t dense = 0;
    alignas(64) static thread_local int8_t t[16][QK];
    uint16_t d[16];
    for (int64_t g = 0; g < ng; g++) {
        tab[g] = (uint64_t) (o - data);
        for (int64_t p = 0; p < nrk; p++) {
            for (int r = 0; r < 16; r++) {
                const int64_t n = g * 16 + r;
                if (n >= N) {
                    memset(t[r], 0, QK);
                    d[r] = 0;
                    continue;
                }
                const uint8_t * b = (const uint8_t *) W + n * wrow + p * f.bytes;
                f.trits(b, t[r]);
                memcpy(&d[r], b + f.d_off, 2);
            }
            bool dn;
            o = pack_record(t, d, o, dn);
            if (!o) return 0;
            dense += dn;
        }
    }
    tab[ng] = (uint64_t) (o - data);
    memset(o, 0, SLACK);
    const size_t used = (size_t) (o - base) + SLACK;
    tbm_hdr h;
    memcpy(h.magic, "KTBM", 4);
    h.version = 1;
    h.K = K; h.N = N; h.ngroups = ng; h.nrec_k = nrk; h.used = (int64_t) used; h.dense = dense;
    memset(base, 0, HDR);
    memcpy(base, &h, sizeof h);
    for (size_t i = (size_t) (ng + 1) * 8; i < table_bytes(ng); i++) base[HDR + i] = 0;
    if (dense_out) *dense_out = dense;
    return used;
}

void kurn_tbm_cols(const void * buf, const void * X, size_t xrow, float * Y, int64_t K, int64_t N, int64_t M, int64_t r0,
                   int64_t r1) {
    (void) K;
    const uint8_t * x = (const uint8_t *) X;
    switch (M) {
        case 1: cols_c<1>(buf, x, xrow, Y, N, r0, r1); break;
        case 2: cols_c<2>(buf, x, xrow, Y, N, r0, r1); break;
        case 3: cols_c<3>(buf, x, xrow, Y, N, r0, r1); break;
        case 4: cols_c<4>(buf, x, xrow, Y, N, r0, r1); break;
        case 5: cols_c<5>(buf, x, xrow, Y, N, r0, r1); break;
        case 6: cols_c<6>(buf, x, xrow, Y, N, r0, r1); break;
        case 7: cols_c<7>(buf, x, xrow, Y, N, r0, r1); break;
        case 8: cols_c<8>(buf, x, xrow, Y, N, r0, r1); break;
        default: abort();
    }
}

void kurn_tbm_tq1_to_tq2(const void * src, void * dst, int64_t nblocks) {
    alignas(64) int8_t t[QK];
    for (int64_t b = 0; b < nblocks; b++) {
        const uint8_t * s = (const uint8_t *) src + b * 54;
        uint8_t * o = (uint8_t *) dst + b * 66;
        trits_tq1_0(s, t);
        memset(o, 0, 64);
        for (int v = 0; v < QK; v++) o[(v / 128) * 32 + v % 32] |= (uint8_t) ((t[v] + 1) << (2 * ((v % 128) / 32)));
        memcpy(o + 64, s + 52, 2);
    }
}

int kurn_tbm_amx_available(void) { return TBM_AMX; }

void kurn_tbm_amx(const void * buf, const void * X, size_t xrow, float * Y, int64_t K, int64_t N, int64_t M, int64_t r0,
                  int64_t r1) {
    (void) K;
#if TBM_AMX
    amx_rows(buf, (const uint8_t *) X, xrow, Y, N, M, r0, r1);
#else
    (void) buf; (void) X; (void) xrow; (void) Y; (void) N; (void) M; (void) r0; (void) r1;
    abort();
#endif
}

}  // extern "C"

#else

extern "C" {
int kurn_tbm_available(void) { return 0; }
size_t kurn_tbm_bytes(int64_t, int64_t) { return 0; }
int64_t kurn_tbm_zeros(int, const void *, int64_t, int64_t, int64_t) { return -1; }
size_t kurn_tbm_pack(int, const void *, int64_t, int64_t, void *, int64_t *) { return 0; }
void kurn_tbm_cols(const void *, const void *, size_t, float *, int64_t, int64_t, int64_t, int64_t, int64_t) {}
void kurn_tbm_tq1_to_tq2(const void *, void *, int64_t) {}
int kurn_tbm_amx_available(void) { return 0; }
void kurn_tbm_amx(const void *, const void *, size_t, float *, int64_t, int64_t, int64_t, int64_t, int64_t) {}
}

#endif
