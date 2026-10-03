// kurn_gpu.h: the ABI of every kernel generated for `target cuda` (see kurn/gpu/codegen.py).
// All pointers are device pointers (host pointers under the CPU emulator, -DKURN_EMU).
// Y[m * N + n] = dot(weight row n, activation row m); weights are ggml blocks, row-major.
#pragma once
#include <stddef.h>
#ifdef KURN_EMU
typedef void *cudaStream_t;
#else
#include <cuda_runtime.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif
const char *kg_config(void);
int kg_check_shape(int N, int K, int M);
size_t kg_prep_bytes(int N, int K);  // 0: kg_run reads the native weights directly
int kg_prepare(const void *W, void *Wp, int N, int K, cudaStream_t s);
size_t kg_xbytes(int K, int M);
int kg_quant(const float *X, void *Xq, int K, int M, cudaStream_t s);
size_t kg_xblock_bytes(int K, int M);  // ggml activation blocks (q8_0 or q8_K) for M rows
int kg_xblocks(const float *X, void *Xb, int K, int M, cudaStream_t s);
int kg_run(const void *W, const void *Xq, float *Y, int N, int K, int M, cudaStream_t s);
#ifdef __cplusplus
}
#endif

typedef const char *(*kg_config_fn)(void);
typedef int (*kg_check_shape_fn)(int, int, int);
typedef size_t (*kg_prep_bytes_fn)(int, int);
typedef int (*kg_prepare_fn)(const void *, void *, int, int, cudaStream_t);
typedef size_t (*kg_xbytes_fn)(int, int);
typedef int (*kg_quant_fn)(const float *, void *, int, int, cudaStream_t);
typedef size_t (*kg_xblock_bytes_fn)(int, int);
typedef int (*kg_run_fn)(const void *, const void *, float *, int, int, int, cudaStream_t);
