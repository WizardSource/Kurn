// kurn hypercomplex linear layers (CPU): quaternion (D = 4) and complex (D = 2) weights kept compact, with the
// Hamilton / complex product computed directly from them. See README.md in this directory.
//
// Shapes: a layer maps Kq hypercomplex inputs to Nq hypercomplex outputs (Nq, Kq multiples of 32). Its real
// equivalent is a (D Nq) x (D Kq) matrix; the compact layer stores D components per weight instead of D^2 reals.
//
// Vector layout ("block-planar"): a vector of Kq hypercomplex numbers is D * Kq floats, in groups of 32 numbers,
// each group stored component by component (32 real parts, then 32 i parts, ...): element (k, c) is at
// (k / 32) * 32 D + c * 32 + k % 32. Outputs use the same layout, so layers chain. With this order the real
// expansion of a quantized compact layer is an ordinary Q8_0 matrix (every 32-value block is one weight component
// of one row, possibly negated, with that component's scale), and the compact kernels reproduce kurn's real Q8_0
// kernels on it bit for bit.
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct kurn_hc kurn_hc;

enum kurn_hc_format {
    KURN_HC_F32 = 0,  // compact f32 components
    KURN_HC_Q8 = 1,   // compact Q8_0 per component: blocks of 32 inputs, one fp16 scale per (row, component, block)
};

// W: compact weights, component-major: W[(p * Nq + n) * Kq + k] is component p (0 = real, 1 = i, 2 = j, 3 = k)
// of weight (n, k). The layer computes y_n = sum_k W_nk * x_k (Hamilton / complex product, weight on the left).
kurn_hc * kurn_hc_pack(int D, int64_t Nq, int64_t Kq, const float * W, enum kurn_hc_format fmt);
void kurn_hc_free(kurn_hc * w);
size_t kurn_hc_bytes(const kurn_hc * w);  // bytes of packed weights (what decode streams)

// Real expansion, rows and columns in block-planar order (D Nq rows of D Kq values): KURN_HC_Q8 writes ggml
// block_q8_0 rows (D Kq / 32 blocks of 34 bytes per row), KURN_HC_F32 f32 rows.
size_t kurn_hc_expanded_bytes(const kurn_hc * w);
void kurn_hc_expand(const kurn_hc * w, void * out);

// Y = layer(X) for M vectors (X: column stride D Kq floats; Y: column stride D Nq floats), with nthreads threads.
// exact = 1: every column bit-identical to kurn's real Q8_0 GEMV on the expansion, at every M (KURN_HC_Q8).
// exact = 0: from 32 columns on AMX CPUs, AMX-BF16 on the expansion built per L2 panel (bit-identical to kurn's
// AMX-BF16 prefill on the expanded matrix); below 32 columns the exact kernels.
void kurn_hc_matmul(const kurn_hc * w, const float * X, float * Y, int64_t M, int nthreads, int exact);

// ggml's x86 quantize_row_q8_0 (block_q8_0 output), reproduced bit for bit; n a multiple of 32
void kurn_hc_quantize_q8_0(const float * x, void * out, int64_t n);

#ifdef __cplusplus
}
#endif
