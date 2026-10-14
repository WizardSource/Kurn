// Ternary weights (TQ1_0 / TQ2_0) as a presence bitmap plus packed signs: 2.0625 - z bits per weight
// (z = zero fraction) instead of TQ2_0's 2.0625. Layout and arithmetic: kurn-tbm.cpp, kurn.ternary.
#pragma once
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

// 1 when this build has the kernels (AVX-512 BW + VNNI + BMI2)
int kurn_tbm_available(void);
// bytes to allocate for a [K x N] tensor (worst case: no zeros); K a multiple of 256
size_t kurn_tbm_bytes(int64_t K, int64_t N);
// zeros among the values of every step-th row of TQ1_0 (type 34) or TQ2_0 (type 35) rows; -1 for other types
int64_t kurn_tbm_zeros(int type, const void * W, int64_t K, int64_t N, int64_t step);
// repack TQ1_0 / TQ2_0 rows into dst (kurn_tbm_bytes(K, N) bytes); returns the bytes used (the rest of dst
// is not written; 0 when a value is not ternary: TQ2_0 code 3) and the number of records that need the
// decoder's two-load path in *dense
size_t kurn_tbm_pack(int type, const void * W, int64_t K, int64_t N, void * dst, int64_t * dense);
// Y[m * N + n] = dot(row n, column m) for n in [r0, r1) (r0 a multiple of 16), m < M <= 8; X: M Q8_K rows of
// xrow bytes. Same per-column arithmetic as kurn's TQ2_0 kernels (bit-identical), at every M.
void kurn_tbm_cols(const void * buf, const void * X, size_t xrow, float * Y, int64_t K, int64_t N, int64_t M, int64_t r0,
                   int64_t r1);
// TQ1_0 blocks -> TQ2_0 blocks with the same values (for TQ1_0 tensors that stay on the TQ2_0 kernels)
void kurn_tbm_tq1_to_tq2(const void * src, void * dst, int64_t nblocks);
// AMX-INT8 prefill (M >= 16 columns), same per-column arithmetic as kurn_tbm_cols; 0 when AMX is not built
int kurn_tbm_amx_available(void);
// set by the KURN buffer type after a failed AMX self-test (kurn-amx-selftest.cpp): GGML_KURN_AMX_GUARD above 8192
// is then ignored, as in kurn-buft.cpp. Call before the first kurn_tbm_amx.
void kurn_tbm_amx_set_lossy(int lossy);
void kurn_tbm_amx(const void * buf, const void * X, size_t xrow, float * Y, int64_t K, int64_t N, int64_t M, int64_t r0,
                  int64_t r1);

#ifdef __cplusplus
}
#endif
