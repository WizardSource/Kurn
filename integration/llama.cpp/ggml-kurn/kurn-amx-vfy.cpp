// AMX verify kernel for Q8_0 on kurn's i16 layout: up to 16 activation columns in one pass over the weights (wide
// speculative verify, small batches; llama.cpp uses it for 9..16 columns, where the VNNI verify kernels need two
// 8-column passes). A record (16 rows x 32 values: 16 fp16 scales, then 8 lines of 64 bytes, line kk
// holding values 4kk..4kk+3 of the 16 rows biased by +128) is exactly an AMX B tile, so one
// tdpbsud per record and row group gives the exact int32 block dots of all M columns. They are
// scaled per block in block order, acc = fma(float(isum - 128 * sum(x)), d_w * d_x, acc), which is
// the GEMV's arithmetic: every column is bit-identical to the same column computed alone.
//
// The tile configuration has M-row A and C tiles. The caller's configuration (ggml's AMX buffer
// configures each thread once and relies on it) is saved and restored around every call.
//
// Some KVM guests drop AMX tile data when a vCPU is descheduled (the same caveat as kurn's AMX
// attention and ggml's AMX buffer; integration/llama.cpp/tools/amx_check.sh). Each block step
// (A load, B loads, tdpbsud, C stores) is timed, and recomputed after reloading the configuration
// when it took longer than KURN_AMX_GUARD cycles; C is stored before the check, so data that
// survived the check is in memory, and accumulation into the outputs happens only afterwards.
#include "kurn-amx-vfy.h"

#if defined(__AMX_INT8__) && defined(__AVX512F__) && defined(__AVX512BW__) && defined(__linux__)

#include <immintrin.h>
#include <x86intrin.h>
#include <algorithm>
#include <cstdint>
#include <cstring>

#ifndef KURN_AMX_GUARD
#define KURN_AMX_GUARD 4096
#endif

namespace {

constexpr int REC = 544;  // REC_BYTES of the generated Q8_0 kernels (layout=i16)
// software prefetch distance in records (every line of the record; the hardware prefetchers alone
// leave a few weight streams per core far below DRAM bandwidth)
#ifndef KURN_AMX_VFY_PF
#define KURN_AMX_VFY_PF 8
#endif
constexpr int PF = KURN_AMX_VFY_PF;

struct alignas(64) tilecfg {
    uint8_t palette, start_row, reserved[14];
    uint16_t colsb[16];
    uint8_t rows[16];
};

// tmm0..2: B (8 x 64 B: one record's code lines), tmm4: A (M x 32 B: one block of M columns),
// tmm5..7: C (M x 16 int32), one per row group of the pass
void config(int M) {
    tilecfg c = {};
    c.palette = 1;
    for (int t = 0; t < 4; t++) { c.rows[t] = 8; c.colsb[t] = 64; }   // B
    c.rows[4] = (uint8_t) M; c.colsb[4] = 32;                          // A
    for (int t = 5; t < 8; t++) { c.rows[t] = (uint8_t) M; c.colsb[t] = 64; }  // C (3 tiles)
    _tile_loadconfig(&c);
}

inline bool slow(uint64_t t0) {
    unsigned aux;
    return __rdtscp(&aux) - t0 > KURN_AMX_GUARD;
}

inline void store_rows(float * y, int64_t row0, int64_t r0, int64_t r1, __m512 v) {
    if (row0 >= r0 && row0 + 16 <= r1) {
        _mm512_storeu_ps(y + row0, v);
        return;
    }
    uint32_t m = 0xFFFF;
    if (row0 < r0) m &= 0xFFFFu << (r0 - row0);
    if (row0 + 16 > r1) m &= (1u << (r1 > row0 ? r1 - row0 : 0)) - 1;
    _mm512_mask_storeu_ps(y + row0, (__mmask16) m, v);
}

// NG row groups (16 rows each) per pass, M columns: NG * M float accumulators
template <int M, int NG>
void kernel(const uint8_t * buf, int64_t nrec_k, const char * X, size_t xrow, const float * dx, const int32_t * nc,
            float * Y, int64_t N, int64_t r0, int64_t r1) {
    static_assert(NG >= 1 && NG <= 3, "three C tiles");
    const int64_t nk = nrec_k;
    alignas(64) int32_t cbuf[NG][M * 16];
    const int64_t g0 = r0 / 16, g1 = (r1 + 15) / 16;
    for (int64_t g = g0; g < g1; g += NG) {
        const int ng = (int) std::min<int64_t>(NG, g1 - g);
        __m512 acc[NG][M];
        for (int j = 0; j < NG; j++)
            for (int m = 0; m < M; m++) acc[j][m] = _mm512_setzero_ps();
        const uint8_t * rec0 = buf + (size_t) g * nrec_k * REC;
        for (int64_t k = 0; k < nk; k++) {
            const uint8_t * rec[NG];
            for (int j = 0; j < NG; j++) rec[j] = rec0 + ((size_t) (j < ng ? j : 0) * nrec_k + k) * REC;
            if (PF) {
                for (int j = 0; j < NG; j++)
                    for (int off = 0; off < REC; off += 64) _mm_prefetch((const char *) rec[j] + PF * REC + off, _MM_HINT_T0);
            }
            uint64_t t0;
            for (;;) {
                t0 = __rdtsc();
                _tile_loadd(4, X + k * 34 + 2, (long) xrow);
                _tile_loadd(0, rec[0] + 32, 64);
                _tile_zero(5);
                _tile_dpbsud(5, 4, 0);
                if (NG > 1) { _tile_loadd(1, rec[1] + 32, 64); _tile_zero(6); _tile_dpbsud(6, 4, 1); }
                if (NG > 2) { _tile_loadd(2, rec[2] + 32, 64); _tile_zero(7); _tile_dpbsud(7, 4, 2); }
                _tile_stored(5, cbuf[0], 64);
                if (NG > 1) _tile_stored(6, cbuf[1], 64);
                if (NG > 2) _tile_stored(7, cbuf[2], 64);
                if (!slow(t0)) break;
                config(M);
            }
            for (int j = 0; j < NG; j++) {
                const __m512 dw = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *) rec[j]));
                for (int m = 0; m < M; m++) {
                    const __m512i isum = _mm512_add_epi32(_mm512_load_si512(cbuf[j] + m * 16), _mm512_set1_epi32(nc[m * nk + k]));
                    acc[j][m] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(isum), _mm512_mul_ps(dw, _mm512_set1_ps(dx[m * nk + k])), acc[j][m]);
                }
            }
        }
        for (int j = 0; j < ng; j++)
            for (int m = 0; m < M; m++) store_rows(Y + m * N, (g + j) * 16, r0, r1, acc[j][m]);
    }
}

template <int M>
void run_m(int ng, const uint8_t * buf, int64_t nrec_k, const char * X, size_t xrow, const float * dx, const int32_t * nc,
           float * Y, int64_t N, int64_t r0, int64_t r1) {
    // more than ~24 accumulators spill to L1, which costs less than giving up weight streams per core
    if (ng >= 3) kernel<M, 3>(buf, nrec_k, X, xrow, dx, nc, Y, N, r0, r1);
    else if (ng >= 2) kernel<M, 2>(buf, nrec_k, X, xrow, dx, nc, Y, N, r0, r1);
    else kernel<M, 1>(buf, nrec_k, X, xrow, dx, nc, Y, N, r0, r1);
}

}  // namespace

bool kurn_amx_vfy_q8_0(const uint8_t * buf, int64_t K, const char * X, size_t xrow, const float * dx, const int32_t * nc,
                       float * Y, int64_t N, int64_t M, int64_t r0, int64_t r1, int ng) {
    if (M < 2 || M > 16 || r0 >= r1) {
        return false;
    }
    tilecfg saved;
    _tile_storeconfig(&saved);
    config((int) M);
    const int64_t nrec_k = K / 32;
    switch (M) {
        case 2: run_m<2>(ng, buf, nrec_k, X, xrow, dx, nc, Y, N, r0, r1); break;
        case 3: run_m<3>(ng, buf, nrec_k, X, xrow, dx, nc, Y, N, r0, r1); break;
        case 4: run_m<4>(ng, buf, nrec_k, X, xrow, dx, nc, Y, N, r0, r1); break;
        case 5: run_m<5>(ng, buf, nrec_k, X, xrow, dx, nc, Y, N, r0, r1); break;
        case 6: run_m<6>(ng, buf, nrec_k, X, xrow, dx, nc, Y, N, r0, r1); break;
        case 7: run_m<7>(ng, buf, nrec_k, X, xrow, dx, nc, Y, N, r0, r1); break;
        case 8: run_m<8>(ng, buf, nrec_k, X, xrow, dx, nc, Y, N, r0, r1); break;
        case 9: run_m<9>(ng, buf, nrec_k, X, xrow, dx, nc, Y, N, r0, r1); break;
        case 10: run_m<10>(ng, buf, nrec_k, X, xrow, dx, nc, Y, N, r0, r1); break;
        case 11: run_m<11>(ng, buf, nrec_k, X, xrow, dx, nc, Y, N, r0, r1); break;
        case 12: run_m<12>(ng, buf, nrec_k, X, xrow, dx, nc, Y, N, r0, r1); break;
        case 13: run_m<13>(ng, buf, nrec_k, X, xrow, dx, nc, Y, N, r0, r1); break;
        case 14: run_m<14>(ng, buf, nrec_k, X, xrow, dx, nc, Y, N, r0, r1); break;
        case 15: run_m<15>(ng, buf, nrec_k, X, xrow, dx, nc, Y, N, r0, r1); break;
        default: run_m<16>(ng, buf, nrec_k, X, xrow, dx, nc, Y, N, r0, r1); break;
    }
    if (saved.palette) {
        _tile_loadconfig(&saved);
    } else {
        _tile_release();
    }
    return true;
}

#else

bool kurn_amx_vfy_q8_0(const uint8_t *, int64_t, const char *, size_t, const float *, const int32_t *, float *, int64_t, int64_t,
                       int64_t, int64_t, int) {
    return false;
}

#endif
