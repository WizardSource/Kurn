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
/* Pre-RoPE K, 4-bit per channel (KIVI / KVQuant style); V Q4_0 or Q8_0 rows.
 *
 * K is stored before RoPE and rotated inside the kernel at position k_pos0 + j. Complete
 * groups of 32 tokens (j = 32 b .. 32 b + 31) are one block per kv head:
 *     f16 scale[dk], f16 min[dk], then 32 token rows of dk/2 bytes
 * with k[c] = q * scale[c] + min[c], q in 0..15; in each 16-byte run of a row the low nibbles
 * are channels c0..c0+15 and the high nibbles c0+16..c0+31 (Q4_0 order). Block b of head g
 * is at k + b * k_s_tok + g * k_s_head (k_s_tok is the stride of a 32-token group here).
 * Tokens from 32 * floor(n_kv / 32) on (the group still being filled) are f16 pre-RoPE rows
 * at k_tail + (j % 32) * kt_s_tok + g * kt_s_head. V rows are ggml block_q4_0 / block_q8_0. */
#define KATTN_KV_K4C_Q4 3
#define KATTN_KV_K4C_Q8 4
#define KATTN_K4C_GROUP 32
#define KATTN_K4C_BLOCK_BYTES(dk) ((int64_t)(dk) * 20)
#define KATTN_ROPE_NEOX 0 /* pairs (i, i + rope_dim / 2) */
#define KATTN_ROPE_NORM 1 /* pairs (2 i, 2 i + 1) */

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
    /* KATTN_KV_K4C_* only (other formats ignore these): partial-group f16 rows, and the RoPE
     * applied to K, angle(j, i) = (k_pos0 + j) * rope_freq[i] for i < rope_dim / 2; dims >=
     * rope_dim are not rotated. q is passed already rotated, as for the other formats. */
    const void *k_tail; int64_t kt_s_tok, kt_s_head; /* strides in bytes */
    const float *rope_freq;
    int64_t k_pos0;
    int32_t rope_dim, rope_mode;
} kattn_args;

/* Bytes of workspace for a call with these arguments on nth threads (zero-fill once). */
size_t kattn_workspace(const kattn_args *a, int nth);
void kattn(const kattn_args *a, void *ws, int ith, int nth);
/* The compiled configuration: head dims and KATTN_KV_* format. Returns the KV row size in bytes for dk
 * (KATTN_KV_K4C_*: the bytes of one 32-token K group block). */
int64_t kattn_config(int *dk, int *dv, int *kv_format);

/* Optional AMX-ready copy of K/V, written once per token at cache-write time instead of being packed
 * for every tile of every call (bf16 engines; others return 0 bytes and kattn_packed ignores it).
 * Allocate kattn_pack_bytes(a, cap) zero-filled bytes for up to cap tokens; after appending tokens
 * [j0, j1) to the cache, call kattn_pack on all threads (like kattn); then call kattn_packed instead
 * of kattn. Tokens are appended in order and must not change once packed. The decode row engine
 * (n_q * G <= dec_rows) reads the cache directly. */
size_t kattn_pack_bytes(const kattn_args *a, int64_t cap);
void kattn_pack(const kattn_args *a, void *kvp, int64_t cap, int64_t j0, int64_t j1, int ith, int nth);
void kattn_packed(const kattn_args *a, const void *kvp, int64_t cap, void *ws, int ith, int nth);

#ifdef __cplusplus
}
#endif
#endif
