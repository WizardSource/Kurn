/* kurn attention ABI: tiled (FlashAttention-style) CPU attention with online softmax.
 *
 * One generated library implements one configuration (head dims, KV format, engine,
 * tile sizes) fixed at generation time; everything else is per call.
 *
 *   out[t][h][:] = softmax_j(scale * q[t][h] . k[j][h/G] + mask[t][j]) v[j][h/G]      G = n_head / n_head_kv
 *
 * Threading follows ggml's model: every one of `nth` threads calls kattn(a, ws, ith, nth)
 * with the same arguments; the call returns when that thread's share is done. Callers
 * must not start the next call on the same workspace before all threads of the previous
 * one have returned (any barrier between ops does this). The workspace must be
 * zero-filled once when allocated; it resets itself after every call.
 */
#ifndef KURN_ATTN_H
#define KURN_ATTN_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define KATTN_KV_F16 0
#define KATTN_KV_BF16 1
#define KATTN_KV_Q8_0 2

typedef struct {
    int64_t n_q;      /* query rows (tokens) in this call */
    int64_t n_kv;     /* keys/values visible to the call */
    int64_t q_pos0;   /* causal: query row t sees kv j iff j <= q_pos0 + t */
    int32_t n_head, n_head_kv;
    int32_t causal;   /* 1: mask by position (fully masked KV tiles are skipped) */
    float scale;      /* usually 1/sqrt(dk) */
    const float *q;   int64_t q_s_tok, q_s_head;    /* strides in floats */
    const void *k;    int64_t k_s_tok, k_s_head;    /* strides in bytes (rows are contiguous) */
    const void *v;    int64_t v_s_tok, v_s_head;    /* v may alias k (MLA: v = first dv values of k) */
    const uint16_t *mask; int64_t mask_s_tok;       /* optional additive fp16 mask [n_q][n_kv] (ggml layout); NULL = none */
    float *out;       int64_t o_s_tok, o_s_head;    /* strides in floats */
} kattn_args;

/* Bytes of workspace for a call with these arguments on nth threads (zero-fill once). */
size_t kattn_workspace(const kattn_args *a, int nth);
void kattn(const kattn_args *a, void *ws, int ith, int nth);
/* The compiled configuration: head dims and KATTN_KV_* format. Returns the KV row size in bytes for dk. */
int64_t kattn_config(int *dk, int *dv, int *kv_format);

#ifdef __cplusplus
}
#endif
#endif
