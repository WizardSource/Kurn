#pragma once

// kurn memory system for ggml's CPU buffers: huge pages and NUMA placement (kurn/integration/llama.cpp/memsys).
//
// Huge pages (GGML_KURN_HUGEPAGES, llama.cpp: --hugepages):
//   auto (default)  hugetlb 1 GB pages for whole GBs (single-node hosts), 2 MB hugetlb for the rest, as far as the
//                   free pools go; then transparent huge pages via madvise(MADV_HUGEPAGE); then 4 KB pages
//   1g | 2m         the same chain, starting at that hugetlb size
//   thp             madvise(MADV_HUGEPAGE) only
//   4k              anonymous mmap with MADV_NOHUGEPAGE (a guaranteed 4 KB baseline, even with THP=always)
//   off             ggml's allocator, untouched
//   Allocations below GGML_KURN_HUGEPAGES_MIN bytes (default 4 MiB) always use ggml's allocator.
//   Every allocation at or above it is logged with what backs it; GGML_KURN_MEM_VERBOSE=1 logs the small ones too.
//
// NUMA (GGML_KURN_NUMA, llama.cpp: --kurn-numa):
//   auto (default)  on when the process may run on CPUs of more than one node, else everything below is a no-op
//   0 | off         off;  1 | on: same as auto
//   GGML_KURN_NUMA_FAKE=N splits the allowed CPUs into N contiguous "nodes" (memory policies then target the real
//   node of each group), for testing the threading on single-node machines.
//   GGML_KURN_NUMA_PIN=0 keeps kurn from pinning threads (thread ith is still assumed to run on its node).
// Threads are split over nodes in proportion to each node's allowed CPU count: thread ith of nth runs on node n
// for ith in [ggml_kurn_numa_first_thread(n, nth), ggml_kurn_numa_first_thread(n + 1, nth)).

#include "ggml.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum ggml_kurn_mem_tag {
    GGML_KURN_MEM_BUFFER = 0,  // a CPU backend buffer without a more specific tag
    GGML_KURN_MEM_WEIGHTS,     // model weights (KURN repacked weights, CPU / CPU_REPACK weight buffers)
    GGML_KURN_MEM_KV,          // KV cache (all cache types, k4c included)
    GGML_KURN_MEM_COMPUTE,     // graph compute buffers (activations, scratch)
    GGML_KURN_MEM_WORK,        // the CPU backend's per-graph work buffer
    GGML_KURN_MEM_N_TAGS,
};

GGML_API void * ggml_kurn_mem_alloc(size_t size, enum ggml_kurn_mem_tag tag);  // 64-byte aligned at least
GGML_API void   ggml_kurn_mem_free(void * ptr, size_t size);                  // any pointer from ggml_kurn_mem_alloc

// tag that CPU backend buffer allocations made by this thread get; returns the previous one
GGML_API enum ggml_kurn_mem_tag ggml_kurn_mem_set_tag(enum ggml_kurn_mem_tag tag);
GGML_API enum ggml_kurn_mem_tag ggml_kurn_mem_get_tag(void);

// one line, per tag: bytes by backing ("weights 8.1G: 2M 100% | kv 1.1G: thp 97% | ..."); THP shares are measured
// from /proc/self/smaps at the time of the call
GGML_API const char * ggml_kurn_mem_summary(void);
GGML_API const char * ggml_kurn_mem_mode(void);  // the resolved GGML_KURN_HUGEPAGES mode

GGML_API int     ggml_kurn_numa_n_nodes(void);  // 1 when single-node or off
GGML_API int     ggml_kurn_numa_is_fake(void);
GGML_API int     ggml_kurn_numa_first_thread(int node, int nth);  // node in [0, n_nodes]
GGML_API int     ggml_kurn_numa_node_of_thread(int ith, int nth);
// start of node's share of n items, in whole units (node in [0, n_nodes]; node n_nodes gives n)
GGML_API int64_t ggml_kurn_numa_split(int64_t n, int node, int64_t unit);
GGML_API void    ggml_kurn_numa_pin_thread(int ith, int nth);  // once per (thread, ith, nth); no-op unless on
GGML_API void    ggml_kurn_numa_bind(void * ptr, size_t len, int node);  // prefer node for these pages
GGML_API void    ggml_kurn_numa_interleave(void * ptr, size_t len);     // spread these pages over all nodes
GGML_API void    ggml_kurn_numa_stats(uint64_t * n_bind, uint64_t * bytes_bind, uint64_t * n_failed);
// bytes placed per logical node (node_bytes[n_nodes]) and interleaved, as requested since start
GGML_API void    ggml_kurn_numa_placement(uint64_t * node_bytes, uint64_t * interleaved_bytes);

#ifdef __cplusplus
}

struct ggml_kurn_mem_tag_scope {  // CPU buffers allocated by this thread in this scope get `tag`
    enum ggml_kurn_mem_tag prev;
    explicit ggml_kurn_mem_tag_scope(enum ggml_kurn_mem_tag tag) : prev(ggml_kurn_mem_set_tag(tag)) {}
    ~ggml_kurn_mem_tag_scope() { ggml_kurn_mem_set_tag(prev); }
};
#endif
