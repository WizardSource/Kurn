// kurn_gpu.h: the ABI of every kernel generated for `target cuda` (see kurn/gpu/codegen.py).
// All pointers are device pointers (host pointers under the CPU emulator, -DKURN_EMU).
// Y[m * N + n] = dot(weight row n, activation row m); weights are ggml blocks, row-major.
#pragma once
#include <stddef.h>
#include <stdint.h>
#ifdef KURN_EMU
typedef void *cudaStream_t;
#else
#include <cuda_runtime.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif
const char *kg_config(void);
int kg_act(void);  // 0: kg_quant -> q8 activations (reference: kg_xblocks); 1: kg_run reads f32 X; 2: kg_quant -> f16 X
int kg_check_shape(int N, int K, int M);
size_t kg_prep_bytes(int N, int K);  // 0: kg_run reads the native weights directly
int kg_prepare(const void *W, void *Wp, int N, int K, cudaStream_t s);
size_t kg_xbytes(int K, int M);
int kg_quant(const float *X, void *Xq, int K, int M, cudaStream_t s);
size_t kg_xblock_bytes(int K, int M);  // ggml activation blocks (q8_0 or q8_K) for M rows
int kg_xblocks(const float *X, void *Xb, int K, int M, cudaStream_t s);
int kg_run(const void *W, const void *Xq, float *Y, int N, int K, int M, cudaStream_t s);
// Fused epilogues. 0: none. 1: swiglu - W holds N = 2F rows (gate rows 0..F-1, then up rows), Y[m * F + f] =
// silu(gate f . x_m) * (up f . x_m), N % 32 == 0 for op gemm. 2: fv - Y as usual, plus a Freivalds silent-data-corruption
// check (bigVerify's statistical float mode) fused into the epilogue: per block b of kg_fv_block() rows, column slot
// m % 16 (columns folded with secret signs) and probe p in {0, 1}, every kg_run compares sum_n r_p[n] Y[m][n] with
// sum_k F[b][k][p] x[m][k] against lambda * K * 2^-24 * sqrt(sum_k x[m][k]^2 G2[b][k]); a mismatch (or NaN) counts a fail.
int kg_epi(void);
int kg_fv_block(void);  // rows per Freivalds block (0: no fused check)
// F [NB][K][2] f32 = W_b^T r_p, G2 [NB][K] bf16 (rounded up) = sum over the block's rows of W^2, signs [2][(N + 31) / 32]
// (bit = 1: r = -1), NB = ceil(N / kg_fv_block()); key seeds the column signs. Device pointers; resets the counters.
int kg_fv_setup(void *Wp, int N, int K, const float *F, const uint16_t *G2, const uint32_t *signs, uint32_t key, float lambda,
                cudaStream_t s);
// host st = {checks, fails, worst passing |C - X| / tau as f32 bits} since kg_fv_setup (synchronizes)
int kg_fv_status(const void *Wp, int N, int K, unsigned st[3], cudaStream_t s);
#ifdef __cplusplus
}
#endif

typedef const char *(*kg_config_fn)(void);
typedef int (*kg_act_fn)(void);
typedef int (*kg_check_shape_fn)(int, int, int);
typedef size_t (*kg_prep_bytes_fn)(int, int);
typedef int (*kg_prepare_fn)(const void *, void *, int, int, cudaStream_t);
typedef size_t (*kg_xbytes_fn)(int, int);
typedef int (*kg_quant_fn)(const float *, void *, int, int, cudaStream_t);
typedef size_t (*kg_xblock_bytes_fn)(int, int);
typedef int (*kg_run_fn)(const void *, const void *, float *, int, int, int, cudaStream_t);
typedef int (*kg_epi_fn)(void);
typedef int (*kg_fv_block_fn)(void);
typedef int (*kg_fv_setup_fn)(void *, int, int, const float *, const uint16_t *, const uint32_t *, uint32_t, float, cudaStream_t);
typedef int (*kg_fv_status_fn)(const void *, int, int, unsigned *, cudaStream_t);
