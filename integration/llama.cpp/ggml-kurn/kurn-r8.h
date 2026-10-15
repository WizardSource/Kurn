#pragma once
// R8: rotated int8 weights for kurn's KURN buffer (kurn/integration/llama.cpp).
//
// A weight W [N x K] is stored as int8 codes of W H (H: orthonormal block Hadamard along K, block hb) with one f32
// scale per row: W H ~= diag(sw) Q. Activations get the same rotation online, x' = H x (H is its own inverse and
// orthogonal, so W x = (W H)(H x)), then one int8 scale per token: x' ~= sx q. Every output is one int32 dot
// product over all of K, scaled once:
//     y[m][n] = ((float) sum_k Q[n][k] q[m][k] * sw[n]) * sx[m]
// Integer sums are exact, so the GEMV (AVX-512 VNNI), the verify widths and the AMX-INT8 prefill (tdpbssd, int32
// accumulation across all of K) give bit-identical results for every column at every batch width: exact mode and
// batch invariance come for free, at full AMX-INT8 rate (no per-block rescaling as with Q8_0's 32-value scales).
//
// Packed layout: per 16-row group and 64-value K block, one 1 KB AMX B tile: byte [t * 64 + r * 4 + i] =
// Q[16 g + r][64 b + 4 t + i]. The same 64-byte lines feed vpdpbusd in the GEMV (16 rows x 4 values per line).
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// bytes of the packed codes for N rows (N % 32 == 0) and K values (K % 64 == 0)
size_t kurn_r8_bytes(int64_t K, int64_t N);
// Q: int8 [N][K] row-major -> out (kurn_r8_bytes); rowsum[n] = sum_k Q[n][k] (for the VNNI u8 bias)
void kurn_r8_pack(const int8_t * Q, int64_t K, int64_t N, uint8_t * out, int32_t * rowsum);
// one activation row: x' = H_hb x (x[zero[j]] read as 0 first, for nz outlier-aware channels), q = round(x' / sx),
// sx = max|x'| / 127; q is written as int8 (K bytes, AMX operand) and, when qu is given, as q + 128 in uint8 (VNNI operand)
void kurn_r8_prep_row(const float * x, int64_t K, int64_t hb, const int32_t * zero, int64_t nz, int8_t * q, uint8_t * qu,
                      float * sx);
// Y[m * ldy + n] for n in [r0, r1) (multiples of 16), m < M; Xu: M uint8 rows (q + 128) of K (stride K)
void kurn_r8_gemv(const uint8_t * W, const int32_t * rowsum, const float * sw, const uint8_t * Xu, const float * sx,
                  float * Y, int64_t K, int64_t ldy, int64_t M, int64_t r0, int64_t r1);
// same results through AMX-INT8 tiles; X padded with zero rows to a multiple of 32, r0 / r1 multiples of 32 (or r1 a
// multiple of 16 at the end). guard: the preemption-guard base in TSC cycles (kurn-buft.cpp's amx_guard_cycles(0));
// each 32 x 32 block over all of K is redone when it took longer than guard + 256 per 64-value step. Returns false when
// AMX is not usable (caller falls back to kurn_r8_gemv).
int kurn_r8_amx(const uint8_t * W, const float * sw, const int8_t * X, const float * sx, float * Y, int64_t K,
                int64_t ldy, int64_t M, int64_t r0, int64_t r1, uint64_t guard);
// AMX steps recomputed by the preemption guard (diagnostics)
uint64_t kurn_r8_amx_redo(void);

#ifdef __cplusplus
}
#endif
