#pragma once

// kurn expert streaming (KXS): MoE expert weights stay on SSD in kurn's repacked layout (.kxs
// file, written by llama-kurn-xs-convert) and are read on demand into per-tensor slab caches.
// See kurn-xs.cpp for the design. The public entry points are reached through
// ggml_backend_reg_get_proc_address(cpu_reg, "<function name>").

#include "ggml.h"

#include <stddef.h>
#include <stdint.h>

#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__) && defined(__linux__)
#define KURN_XS 1
#else
#define KURN_XS 0
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Open a .kxs file and size the slab caches: cache_bytes is split evenly over the MoE layers and,
// within a layer, gives every streamed tensor the same number of slots. Must be called before the
// model's tensors are allocated. Returns false (and logs why) when the file cannot be used.
typedef bool (*ggml_kurn_xs_open_t)(const char * kxs_path, const char * gguf_path, size_t cache_bytes);
bool ggml_kurn_xs_open(const char * kxs_path, const char * gguf_path, size_t cache_bytes);

// Is this weight (by name) streamed from the open .kxs file?
typedef bool (*ggml_kurn_xs_is_streamed_t)(const char * name);
bool ggml_kurn_xs_is_streamed(const char * name);

// GGML_OP_CUSTOM body (n_tasks = 1) placed right after a MoE layer's top-k: src[0] = selected experts
// (I32 [n_expert_used, n_tokens]), optional src[1] = the router input (F32 [n_embd, n_tokens]) for the
// next-layer prediction; userdata = one of the layer's streamed expert tensors.
void ggml_kurn_xs_plan_op(struct ggml_tensor * dst, int ith, int nth, void * userdata);

// Counters since the last reset as one JSON object; returns the length written (excluding NUL).
typedef size_t (*ggml_kurn_xs_stats_t)(char * buf, size_t n);
size_t ggml_kurn_xs_stats(char * buf, size_t n);
typedef void (*ggml_kurn_xs_stats_reset_t)(void);
void ggml_kurn_xs_stats_reset(void);

// Converter support: the kurn layout a 3D expert weight of this type and shape gets (the same rule
// the KURN buffer applies at load). Returns false when kurn has no kernel for it.
typedef bool (*ggml_kurn_xs_layout_t)(int type, int64_t K, int64_t N, size_t * expert_bytes, const char ** config);
bool ggml_kurn_xs_layout(int type, int64_t K, int64_t N, size_t * expert_bytes, const char ** config);
typedef bool (*ggml_kurn_xs_pack_t)(int type, int64_t K, int64_t N, const void * src, void * dst);
bool ggml_kurn_xs_pack(int type, int64_t K, int64_t N, const void * src, void * dst);

#ifdef __cplusplus
}

#if KURN_XS
namespace kxs {

struct tensor;

// KURN buffer hooks (kurn-buft.cpp)
tensor * find(const char * name);
// binds a streamed weight to the kernel the buffer chose; aborts on a layout mismatch
void attach(tensor * t, const char * config, size_t expert_bytes, int64_t K, int64_t N, int64_t E,
            void * (*view)(void *, int64_t, int64_t), void (*pack)(const void *, int64_t, int64_t, void *));

// One MUL_MAT_ID on a streamed weight. Thread 0 calls op_begin with the experts that have tokens
// (ascending ids) and the op's token count; it writes the compute order (hits first, then misses in read order) into order[].
// Every thread then takes work items whose expert is order[i]: wait(i) blocks until that expert's
// slab is readable and returns it; item_done(i, n) counts finished work items and unpins the slab
// (handing its slot to a pending miss) after the n-th. Thread 0 calls op_end after the last barrier.
void op_begin(tensor * t, const int64_t * act, int64_t n_act, int64_t n_tok, int64_t * order);
void wait(tensor * t, int64_t i, const void ** view, const uint8_t ** data);
void item_done(tensor * t, int64_t i, int64_t n_items);
void op_end(tensor * t);

}  // namespace kxs
#endif

#endif
