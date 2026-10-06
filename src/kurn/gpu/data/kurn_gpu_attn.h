// kurn_gpu_attn.h: the ABI of every attention kernel generated for `op attn target cuda` (see kurn/gpu/attn.py).
//
// Same semantics as the CPU op (kurn_attn.h), with device pointers (host pointers under the CPU emulator):
//
//   out[t][h][:] = softmax_j(scale * q[t][h] . k[j][h/G] + mask[t][j]) v[j][h/G]      G = n_head / n_head_kv
//
// causal: query row t sees kv j iff j <= q_pos0 + t. Rows that see nothing are written as 0. The field layout
// of kga_args matches kattn_args field for field, so a host that has one can fill the other by copying.
// Strides: q/out in floats, k/v in bytes; any layout (token-major ggml views, head-major caches) whose rows are
// contiguous. F16/BF16 rows must be 16-byte aligned, Q8_0 rows 4-byte aligned (kga_check).
// MLA (generated with mla=1): v must alias k with the same strides; v is the first dv values of each k row.
//
// kga_run launches the attention kernel and, when the KV range is split, a merge kernel on `s`. Both are
// graph-capturable; results are deterministic. `ws` must hold kga_workspace(a) bytes (any contents).
#pragma once
#include <stddef.h>
#include <stdint.h>
#ifdef KURN_EMU
typedef void *cudaStream_t;
#else
#include <cuda_runtime.h>
#endif

#define KGA_KV_F16 0
#define KGA_KV_BF16 1
#define KGA_KV_Q8_0 2

typedef struct {
  int64_t n_q, n_kv, q_pos0;
  int32_t n_head, n_head_kv;
  int32_t causal;
  float scale;
  const float *q;        int64_t q_s_tok, q_s_head;
  const void *k;         int64_t k_s_tok, k_s_head;
  const void *v;         int64_t v_s_tok, v_s_head;
  const uint16_t *mask;  int64_t mask_s_tok;
  float *out;            int64_t o_s_tok, o_s_head;
  /* kattn_args' KATTN_KV_K4C_* fields: layout only, the GPU op has no k4c formats and ignores them */
  const void *k_tail;    int64_t kt_s_tok, kt_s_head;
  const float *rope_freq;
  int64_t k_pos0;
  int32_t rope_dim, rope_mode;
} kga_args;

#ifdef __cplusplus
extern "C" {
#endif
// config string; head dims and KGA_KV_* format of this kernel
const char *kga_config(int *dk, int *dv, int *kv_format);
// 0 if the kernel supports these arguments, else a negative code (see kga_check in the template)
int kga_check(const kga_args *a);
// KV splits kga_run will use (depends on the device's SM count)
int kga_splits(const kga_args *a);
size_t kga_workspace(const kga_args *a);
int kga_run(const kga_args *a, void *ws, cudaStream_t s);
#ifdef __cplusplus
}
#endif

typedef const char *(*kga_config_fn)(int *, int *, int *);
typedef int (*kga_check_fn)(const kga_args *);
typedef int (*kga_splits_fn)(const kga_args *);
typedef size_t (*kga_workspace_fn)(const kga_args *);
typedef int (*kga_run_fn)(const kga_args *, void *, cudaStream_t);
