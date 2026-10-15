// KURN extra buffer type for ggml-cpu (see kurn/integration/llama.cpp/README.md).
//
// Weights whose type has a kurn kernel (kurn_dispatch.h, generated from the kurn kernel
// registry) are repacked once at load time into kurn's interleaved i16 layout, inside this
// buffer: there is one copy of the weights, owned by ggml like the AMX / CPU_REPACK buffers.
// MUL_MAT and MUL_MAT_ID (MoE experts) on those weights are computed here:
//   1 activation column      -> decode GEMV
//   2..8 columns             -> verify kernel of that width (one weight pass for all columns)
//   more columns (prefill)   -> where AMX is usable (default): AMX-BF16 GEMM for dense MUL_MAT (weights
//                               dequantized per panel, bf16 activations, fp32 accumulation); in exact mode
//                               the AMX-INT8 kernel for Q8_0. Otherwise verify kernels over row chunks x
//                               groups of 8 columns.
// The GEMV, the verify kernels and the AMX-INT8 kernel do the same per-column arithmetic (exact
// int32 dot per 32-value block, then acc = fma(float(isum), d_w * d_x, acc) in block
// order), so a token's result does not depend on how many tokens are computed with it. The
// AMX-BF16 path does not have that property (bf16 activations); GGML_KURN_EXACT=1 (or kurn
// attention's GGML_KURN_FA_MODE=exact) keeps prefill batch invariant.
// Q6_K and Q5_K weights, which kurn has no repacked kernels for, are taken in ggml's own block
// layout when AMX-BF16 is usable: AMX-BF16 prefill, ggml's vec_dot for decode-sized calls.
//
// Environment: GGML_KURN=0 disables the buffer type; GGML_KURN_TYPES=q8_0,q4_0 restricts
// it to some formats; GGML_KURN_AMX=0 (or GGML_KURN_AMX_MM=0) turns the AMX prefill paths off,
// GGML_KURN_AMX_MIN=<cols> sets their threshold (default 32), GGML_KURN_Q8_BF16=0 keeps Q8_0 on
// AMX-INT8, GGML_KURN_NATIVE=0 leaves Q6_K/Q5_K to ggml, GGML_KURN_BF16_ROWS / _KC / _KCH set the
// AMX-BF16 panel rows, K chunk and guarded sub-step (128, 4096, 32 k-steps); GGML_KURN_CHUNK_KB sets
// the verify-kernel prefill row-chunk size; GGML_KURN_CHUNKS=<n> hands rows out in n equal chunks per
// thread (0 = one static range per thread; default: guided, see row_chunks) for prefill-sized ops, and for
// decode-sized ones with GGML_KURN_STEAL=0 (default there: work stealing, see steal_sched); GGML_KURN_FUSE=0 turns
// off fused MUL_MAT groups (fused_group); GGML_KURN_VERBOSE=1 logs repacks, prefill paths and AMX redo counts.
// AMX on VMs: some KVM guests drop AMX tile data when a thread is descheduled. Every AMX step here is
// short, timed, and recomputed from intact inputs when it took longer than a preemption-length gap
// (GGML_KURN_AMX_GUARD, default 4096 cycles plus a per-step allowance), as kurn attention does. A startup
// self-test (kurn-amx-selftest.cpp, GGML_KURN_AMX_SELFTEST) checks for the condition and warns; then the guard
// stays forced on, or AMX is turned off (GGML_KURN_AMX_SELFTEST_ACTION=off).
//
// Activations are quantized once per op, split across all threads by blocks, and the per-block
// tables the kernels need (scales, block sums) are built in the same pass (the generated
// `_xprep` / `_packed_x` entry points), so the number of kernel calls per op costs nothing extra.
//
// Memory (memsys/ggml-kurn-mem.h): the buffer comes from ggml_kurn_mem_alloc, so the repacked weights sit on
// 2 MB / 1 GB pages where the host has them (GGML_KURN_HUGEPAGES). On multi-node hosts (GGML_KURN_NUMA) every
// weight's rows are split into one contiguous shard per node (multiples of NUMA_SHARD rows, in proportion to the
// node's CPUs), each shard's pages are placed on its node before the repack touches them, and the threads of a
// node take rows of their own shard only (row_chunks within the shard, on a per-node chunk counter). A row is
// still computed by one thread over all of K, so results do not depend on the split; single-node: no change.
#include "kurn-buft.h"

#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)

#include "ggml-backend-impl.h"
#include "ggml-cpu.h"
#include "ggml-impl.h"
#include "ggml-cpu-impl.h"
#include "traits.h"
#include "kurn_dispatch.h"
#include "kurn-native-vfy.h"
#include "kurn-amx-vfy.h"
#include "kurn-amx-selftest.h"
#include "vec.h"
#include "kurn-tbm.h"
#include "ggml-kurn-mem.h"
#include "kurn-xs.h"

#include <immintrin.h>
#include <x86intrin.h>
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#if defined(__AMX_INT8__) && defined(__linux__)
#define KURN_HAVE_AMX 1
#else
#define KURN_HAVE_AMX 0
#endif

namespace {

struct block_q8_0 { uint16_t d; int8_t qs[32]; };  // ggml's block_q8_0
static_assert(sizeof(block_q8_0) == 34, "block_q8_0");

int env_int(const char * name, int def) {
    const char * v = getenv(name);
    return v && *v ? atoi(v) : def;
}

bool verbose() {
    static const bool v = env_int("GGML_KURN_VERBOSE", 0) != 0;
    return v;
}

bool type_enabled(const kurn_kernel & k) {
    static const char * list = getenv("GGML_KURN_TYPES");
    if (!list || !*list) {
        return true;
    }
    std::string s = std::string(",") + list + ",";
    return s.find(std::string(",") + k.name + ",") != std::string::npos;
}

const kurn_kernel * find_kernel(enum ggml_type t) {
    for (const kurn_kernel & k : kurn_kernels) {
        if (k.type == (int) t) {
            return type_enabled(k) ? &k : nullptr;
        }
    }
    return nullptr;
}

#if defined(__AMX_INT8__) && defined(__linux__)
bool amx_ok = false;     // AMX may be used (kurn's AMX paths, ggml's AMX buffer): XTILEDATA granted, not GGML_KURN_AMX=0,
                         // not turned off by the self-test
bool amx_lossy = false;  // the AMX self-test saw tile data lost across a context switch (KURN_AMX_GUARDED)
#endif

// AMX prefill matmuls: on by default where AMX is usable; GGML_KURN_AMX_MM=0 (or GGML_KURN_AMX=0) turns them off.
// GGML_KURN_AMX=1 additionally selects kurn attention's AMX engine (kurn-attn.cpp).
bool amx_mm_enabled() {
    static const bool v = env_int("GGML_KURN_AMX_MM", env_int("GGML_KURN_AMX", 1)) != 0;
    return v;
}

// Batch-invariant matmuls only: GGML_KURN_EXACT=1, or kurn attention's exact mode (GGML_KURN_FA_MODE=exact).
// Formats whose prefill path is AMX-BF16 then stay on the verify kernels.
bool exact_matmuls() {
    static const bool v = [] {
        const char * fa = getenv("GGML_KURN_FA_MODE");
        return env_int("GGML_KURN_EXACT", 0) != 0 || (fa && strcmp(fa, "exact") == 0);
    }();
    return v;
}

// Would ggml's own AMX extra buffer take this weight? (mirrors ggml-cpu/amx/amx.cpp supports_op)
bool ggml_amx_takes(const ggml_tensor * w) {
#if defined(__AMX_INT8__) && defined(__AVX512VNNI__) && defined(__linux__)
    if (!amx_ok || w->ne[2] != 1 || w->ne[3] != 1 || w->ne[1] % 32 != 0) {
        return false;
    }
    switch (w->type) {
        case GGML_TYPE_Q4_0: case GGML_TYPE_Q4_1: case GGML_TYPE_Q8_0:
            return w->ne[0] % 32 == 0;
        case GGML_TYPE_Q4_K: case GGML_TYPE_Q5_K: case GGML_TYPE_Q6_K: case GGML_TYPE_IQ4_XS:
            return w->ne[0] % 256 == 0;
        default:
            return false;
    }
#else
    GGML_UNUSED(w);
    return false;
#endif
}

// Per-tensor fallback: weights for which kurn is not measured to beat the ggml path that would get
// them stay with ggml (llama.cpp then places them in the next extra buffer). Measured on Qwen3
// decode shapes, 8 threads (benchmarks/v0.2/q4fix): kurn beats CPU_REPACK and plain vec_dot on
// every shape, and ggml's AMX buffer (whose M = 1 path is a VNNI GEMV) from N >= 1024, K >= 1024;
// at N <= 512 results are mixed (Q8_0 2048 x 512: 0.94x). So when the AMX buffer can take the
// weight and N <= GGML_KURN_FALLBACK_N (512) or K < GGML_KURN_FALLBACK_K (512), ggml keeps it.
// GGML_KURN_FALLBACK=0 turns the rule off (kurn takes every weight it has a kernel for).
// Not when the AMX self-test saw tile data lost: ggml's AMX buffer has no preemption guard.
bool keep_with_ggml(const ggml_tensor * w) {
#if defined(__AMX_INT8__) && defined(__linux__)
    if (amx_lossy) {
        return false;
    }
#endif
    static const bool on = env_int("GGML_KURN_FALLBACK", 1) != 0;
    static const int64_t max_n = env_int("GGML_KURN_FALLBACK_N", 512), min_k = env_int("GGML_KURN_FALLBACK_K", 512);
    return on && ggml_amx_takes(w) && (w->ne[1] <= max_n || w->ne[0] < min_k);
}

// Ternary tensors (TQ1_0, TQ2_0) may be repacked as presence bitmap + signs (kurn-tbm.cpp) instead of
// TQ2_0 records; TQ1_0 tensors use the TQ2_0 kernel entry (same activations, same values) either way.
// GGML_KURN_TBM=0: never; 1: always; default: when the tensor's zero fraction is at least
// GGML_KURN_TBM_Z / 1000 (default 300 = 0.30). Read at every set_tensor.
bool tbm_source(const ggml_tensor * w) {
    return (w->type == GGML_TYPE_TQ1_0 || w->type == GGML_TYPE_TQ2_0) && kurn_tbm_available();
}

// Shapes the kernels handle: 2D or 3D (expert-stacked) contiguous weights, K within the
// generated kernels' limits, minus the per-tensor fallback.
const kurn_kernel * kernel_for(const ggml_tensor * w) {
    const kurn_kernel * k = find_kernel(w->type == GGML_TYPE_TQ1_0 && tbm_source(w) ? GGML_TYPE_TQ2_0 : w->type);
    if (!k || w->ne[3] != 1 || w->ne[0] % k->k_multiple != 0 || w->ne[0] > KURN_MAX_K || w->ne[1] < 1) {
        return nullptr;
    }
    if (keep_with_ggml(w)) {
        if (verbose()) {
            GGML_LOG_INFO("kurn: %s [%lld x %lld] %s stays with ggml (per-tensor fallback)\n", w->name, (long long) w->ne[0],
                          (long long) w->ne[1], ggml_type_name(w->type));
        }
        return nullptr;
    }
    return k;
}

// Formats kurn has no repacked kernels for but takes in ggml's own block layout where AMX-BF16 runs:
// prefill through AMX-BF16 panels, decode-sized calls through kurn-native-vfy.cpp. In exact mode they
// are taken too, without AMX-BF16: kurn-native-vfy.cpp computes every width with the same per-column
// arithmetic, where ggml's AMX buffer (VNNI at 1 column, tiles at more) is not batch invariant.
// GGML_KURN_NATIVE=0 leaves them to ggml.
bool native_for(const ggml_tensor * w) {
#if defined(__AMX_INT8__) && defined(__AMX_BF16__) && defined(__AVX512BF16__) && defined(__linux__)
    static const bool on = env_int("GGML_KURN_NATIVE", 1) != 0;
    const bool bf16 = amx_ok && amx_mm_enabled() && !exact_matmuls();
    const bool exact = exact_matmuls() && kurn_native_vfy_supported(w->type);
    if (!on || !(bf16 || exact) || w->ne[2] != 1 || w->ne[3] != 1 || w->ne[0] % 256 != 0 || w->ne[1] % 16 != 0 ||
        !ggml_is_contiguous(w)) {
        return false;
    }
    if (w->type != GGML_TYPE_Q6_K && w->type != GGML_TYPE_Q5_K) {
        return false;
    }
    static const char * list = getenv("GGML_KURN_TYPES");
    if (list && *list) {
        std::string s = std::string(",") + list + ",";
        return s.find(std::string(",") + ggml_type_name(w->type) + ",") != std::string::npos;
    }
    return true;
#else
    GGML_UNUSED(w);
    return false;
#endif
}

size_t expert_bytes(const kurn_kernel * k, const ggml_tensor * w) {
    const size_t b = k->bytes(w->ne[0], w->ne[1]);  // multiple of 64
    // the bitmap layout's size depends on the data: room for its worst case, the unused tail is never touched
    return tbm_source(w) ? std::max(b, kurn_tbm_bytes(w->ne[0], w->ne[1])) : b;
}

// narrowest generated verify kernel with at least m columns (exact width when there is one)
int vfy_index(int64_t m) {
    int i = 0;
    while (i < KURN_VFY_N - 1 && kurn_vfy_cols[i] < m) {
        i++;
    }
    return i;
}

// Rows are handed out through ggml's shared chunk counter (thread ith starts at chunk ith).
// Default (guided): chunk ith is thread ith's own contiguous share of 3/4 of the rows, so each
// thread streams one long range (several short ranges per thread cost 15-30% of decode
// bandwidth); the last quarter goes out in small chunks so that a preempted or slow thread
// does not hold up the whole op (plain static ranges did, on a busy host).
// GGML_KURN_CHUNKS=0: static ranges only; GGML_KURN_CHUNKS=n > 0: n equal chunks per thread.
// Chunks are whole GEMV passes (row_pass rows: every row group streams) when there are at least as
// many passes as threads, otherwise single records (row_align rows) so that small N still uses
// every thread.
struct row_chunks {
    int64_t align = 1, big = 0, small = 1, nbig = 0, ng = 0, n = 0;

    row_chunks(int64_t N, int64_t row_align, int64_t row_pass, int nth) {
        static const int per_thread = env_int("GGML_KURN_CHUNKS", -1);
        align = N / row_pass >= nth ? row_pass : row_align;
        ng = (N + align - 1) / align;
        if (per_thread >= 0) {
            const int64_t want = per_thread == 0 ? nth : (int64_t) nth * per_thread;
            small = (ng + want - 1) / want;
            n = (ng + small - 1) / small;
            return;
        }
        if (ng < 4 * (int64_t) nth) {  // small op: chunk overhead outweighs balancing, one range per thread
            small = (ng + nth - 1) / nth;
            n = (ng + small - 1) / small;
            return;
        }
        big = ng * 3 / 4 / nth;
        nbig = nth;
        const int64_t tail = ng - nbig * big;
        small = std::max<int64_t>(1, (tail + 2 * nth - 1) / (2 * nth));
        n = nbig + (tail + small - 1) / small;
    }

    void range(int64_t N, int64_t c, int64_t & r0, int64_t & r1) const {
        int64_t g0, g1;
        if (c < nbig) {
            g0 = c * big;
            g1 = g0 + big;
        } else {
            g0 = nbig * big + (c - nbig) * small;
            g1 = g0 + small;
        }
        r0 = std::min(N, std::min(ng, g0) * align);
        r1 = std::min(N, std::min(ng, g1) * align);
    }
};

// Work-stealing row schedule for decode-sized ops (1..8 columns: memory-bound GEMV and verify kernels). Each thread
// starts on its own contiguous share of whole passes and claims work from the front of it in shrinking batches (a
// quarter of what is left, in whole passes; single row records once less than a pass is left), so its weight
// streams stay long. A thread whose share is done steals the back half of the unclaimed rest of the thread with
// the most left. Ranges only shrink from the front (owner) or the back (thieves), through one 64-bit CAS on
// (end << 32 | next). The guided row_chunks schedule hands out its last quarter in chunks of two passes, which
// left the last thread ~60 us behind the first on 250 us Qwen3-8B decode matmuls (each thread streams at its own
// ~15 GB/s, so the others cannot absorb the tail); finer static chunks cost bandwidth instead. Which thread
// computes a row does not change its result. GGML_KURN_STEAL=0 restores row_chunks.
bool steal_enabled() {
    static const bool v = env_int("GGML_KURN_STEAL", 1) != 0;
    return v;
}

constexpr size_t STEAL_RESERVE_SLOTS = 2;  // + n_threads slots of 64 bytes at the end of the work buffer

struct alignas(64) steal_slot {
    std::atomic<uint64_t> w;  // (end << 32) | next, in row units
};

struct steal_sched {
    steal_slot * s;
    int nth;
    int64_t P;  // units per pass
    int lo = 0, hi = 0;                    // threads this one shares rows with (and steals from): all, or its NUMA node's
    int64_t base = 0, end = INT64_MAX;     // rows of the shared range: unit u covers rows base + u * align ..

    steal_sched(const ggml_compute_params * p, int64_t units_per_pass)
        : s((steal_slot *) (((uintptr_t) p->wdata + p->wsize - 64 * ((size_t) p->nth + 1)) & ~(uintptr_t) 63)),
          nth(p->nth), P(std::max<int64_t>(1, units_per_pass)), hi(p->nth) {}

    // thread ith's initial share of U units, split over threads [t0, t0 + nt) (callers barrier before next())
    void init(int ith, int64_t U, int t0 = 0, int nt = -1) {
        lo = t0;
        hi = t0 + (nt < 0 ? nth : nt);
        const int li = ith - lo, ln = hi - lo;
        const int64_t np = (U + P - 1) / P;
        const int64_t u0 = std::min(U, np * li / ln * P), u1 = std::min(U, np * (li + 1) / ln * P);
        s[ith].w.store(((uint64_t) u1 << 32) | (uint64_t) u0, std::memory_order_relaxed);
    }

    bool next(int ith, int64_t & u0, int64_t & u1) {
        for (;;) {
            uint64_t w = s[ith].w.load(std::memory_order_acquire);
            for (;;) {
                const int64_t n = (int64_t) (w & 0xffffffffu), e = (int64_t) (w >> 32);
                if (n >= e) {
                    break;
                }
                const int64_t rem = e - n;
                const int64_t b = rem >= 2 * P ? std::max(P, rem / 4 / P * P) : rem >= P ? P : 1;
                if (s[ith].w.compare_exchange_weak(w, ((uint64_t) e << 32) | (uint64_t) (n + b), std::memory_order_acq_rel)) {
                    u0 = n;
                    u1 = n + b;
                    return true;
                }
            }
            // steal the back half of the unclaimed rest of the thread with the most left
            int v = -1;
            int64_t best = 0;
            for (int t = lo; t < hi; t++) {
                const uint64_t x = s[t].w.load(std::memory_order_relaxed);
                const int64_t r = (int64_t) (x >> 32) - (int64_t) (x & 0xffffffffu);
                if (t != ith && r > best) {
                    best = r;
                    v = t;
                }
            }
            if (v < 0) {
                return false;
            }
            uint64_t x = s[v].w.load(std::memory_order_acquire);
            const int64_t n = (int64_t) (x & 0xffffffffu), e = (int64_t) (x >> 32);
            if (n >= e) {
                continue;
            }
            const int64_t take = (e - n + 1) / 2;
            if (s[v].w.compare_exchange_strong(x, ((uint64_t) (e - take) << 32) | (uint64_t) n, std::memory_order_acq_rel)) {
                s[ith].w.store(((uint64_t) e << 32) | (uint64_t) (e - take), std::memory_order_release);
            }
        }
    }
};

// NUMA: shards are multiples of this many rows (a multiple of every kernel's row_align and row pass)
constexpr int64_t NUMA_SHARD = 128;
constexpr int KURN_MAX_NODES = 64;

// nodes the rows of an op are split over: 1 unless NUMA placement is on and every node has a thread
int numa_nodes(int nth) {
    const int nn = ggml_kurn_numa_n_nodes();
    if (nn <= 1 || nn > KURN_MAX_NODES) {
        return 1;
    }
    for (int n = 0; n < nn; n++) {
        if (ggml_kurn_numa_first_thread(n + 1, nth) == ggml_kurn_numa_first_thread(n, nth)) {
            return 1;
        }
    }
    return nn;
}

// per-node chunk counters of a threadpool (ggml's own counter is per threadpool too)
struct alignas(64) node_counter {
    std::atomic<int64_t> v{0};
};
struct numa_counters {
    node_counter c[KURN_MAX_NODES];
};

numa_counters * counters_for(const ggml_threadpool * tp) {
    thread_local const ggml_threadpool * last = nullptr;
    thread_local numa_counters * last_c = nullptr;
    if (tp == last && last_c) {
        return last_c;
    }
    static std::mutex mu;
    static std::unordered_map<const ggml_threadpool *, std::unique_ptr<numa_counters>> all;
    std::lock_guard<std::mutex> lock(mu);
    auto & c = all[tp];
    if (!c) {
        c = std::make_unique<numa_counters>();
    }
    last = tp;
    last_c = c.get();
    return last_c;
}

// Called where ith 0 sets ggml's chunk counter, before the barrier that precedes the row loop. With a steal
// schedule (decode-sized ops), every thread also publishes its initial share of the N rows (in units of align rows).
// Under NUMA placement the shares split the thread's node shard and threads steal only from their own node: rows of
// another node's shard would stream from remote memory.
void rows_begin(const ggml_compute_params * p, steal_sched * ss = nullptr, int64_t N = 0, int64_t align = 1) {
    if (ss) {
        const int nn = numa_nodes(p->nth);
        if (nn <= 1) {
            ss->base = 0;
            ss->end = N;
            ss->init(p->ith, (N + align - 1) / align);
        } else {
            const int node = ggml_kurn_numa_node_of_thread(p->ith, p->nth);
            const int t0 = ggml_kurn_numa_first_thread(node, p->nth), t1 = ggml_kurn_numa_first_thread(node + 1, p->nth);
            const int64_t s0 = ggml_kurn_numa_split(N, node, NUMA_SHARD), s1 = ggml_kurn_numa_split(N, node + 1, NUMA_SHARD);
            ss->base = s0;
            ss->end = s1;
            ss->init(p->ith, std::max<int64_t>(0, (s1 - s0 + align - 1) / align), t0, t1 - t0);
        }
    }
    if (p->ith != 0) {
        return;
    }
    ggml_threadpool_chunk_set(p->threadpool, p->nth);
    const int nn = numa_nodes(p->nth);
    if (nn > 1) {
        numa_counters * c = counters_for(p->threadpool);
        for (int n = 0; n < nn; n++) {
            c->c[n].v.store(ggml_kurn_numa_first_thread(n + 1, p->nth) - ggml_kurn_numa_first_thread(n, p->nth),
                            std::memory_order_relaxed);
        }
    }
}

// Rows [0, N) of an op in row_chunks: every thread over all rows on ggml's counter, or (NUMA) the threads of
// each node over that node's shard on the node's counter. f(r0, r1, item) for item in [0, n_items) per chunk
// (MUL_MAT_ID: active experts).
template <class F>
void for_rows(const ggml_compute_params * p, int64_t N, int64_t align, int64_t pass, int64_t n_items, F && f,
              steal_sched * ss = nullptr) {
    if (ss) {  // decode-sized op: the steal schedule rows_begin set up (one item)
        for (int64_t u0, u1; ss->next(p->ith, u0, u1);) {
            f(ss->base + u0 * align, std::min(ss->end, ss->base + u1 * align), 0);
        }
        return;
    }
    const int nn = numa_nodes(p->nth);
    if (nn <= 1) {
        const row_chunks rc(N, align, pass, p->nth);
        const int64_t nchunk = rc.n * n_items;
        for (int64_t c = p->ith; c < nchunk; c = nchunk > p->nth ? ggml_threadpool_chunk_add(p->threadpool, 1) : nchunk) {
            int64_t r0, r1;
            rc.range(N, c % rc.n, r0, r1);
            f(r0, r1, c / rc.n);
        }
        return;
    }
    const int node = ggml_kurn_numa_node_of_thread(p->ith, p->nth);
    const int t0 = ggml_kurn_numa_first_thread(node, p->nth), lnth = ggml_kurn_numa_first_thread(node + 1, p->nth) - t0;
    const int64_t s0 = ggml_kurn_numa_split(N, node, NUMA_SHARD), n = ggml_kurn_numa_split(N, node + 1, NUMA_SHARD) - s0;
    if (n <= 0) {
        return;
    }
    std::atomic<int64_t> & ctr = counters_for(p->threadpool)->c[node].v;
    const row_chunks rc(n, align, pass, lnth);
    const int64_t nchunk = rc.n * n_items;
    for (int64_t c = p->ith - t0; c < nchunk; c = nchunk > lnth ? ctr.fetch_add(1, std::memory_order_relaxed) : nchunk) {
        int64_t r0, r1;
        rc.range(n, c % rc.n, r0, r1);
        f(s0 + r0, s0 + r1, c / rc.n);
    }
}

// place each node's row shard of a weight ([N rows], row r at byte off(r) of data) on that node
template <class Off>
void numa_place(uint8_t * data, size_t bytes, int64_t N, Off && off) {
    const int nn = ggml_kurn_numa_n_nodes();
    if (nn <= 1) {
        return;
    }
    for (int n = 0; n < nn; n++) {
        const size_t a = std::min(bytes, off(ggml_kurn_numa_split(N, n, NUMA_SHARD)));
        const size_t b = n + 1 == nn ? bytes : std::min(bytes, off(ggml_kurn_numa_split(N, n + 1, NUMA_SHARD)));
        if (b > a) {
            ggml_kurn_numa_bind(data + a, b - a, n);
        }
    }
}

// ------------------------------------------------------------------------------------------
// AMX prefill for Q8_0 on the i16 layout. A record (16 rows x 32 values) is 16 fp16 scales
// followed by 8 lines of 64 bytes; line kk holds values 4kk..4kk+3 of the 16 rows, biased by
// +128 (u8). That is the B-tile format (K/4 rows x 16 columns x 4 bytes): the AMX-INT8 kernel
// unbiases it to s8 per panel and applies the scaling per block, as in the GEMV, which keeps the
// result identical; the AMX-BF16 path dequantizes it.
#if KURN_HAVE_AMX
constexpr int KURN_REC = 544;  // must match REC_BYTES of the generated Q8_0 kernels

// Same palette as ggml's AMX code (ggml_tile_config_init), which configures each thread once
// and never again: tmm0/1 B (8 x 64 B), tmm2/3 A (16 x 32 B), tmm4..7 C (16 x 16 int32).
void amx_config() {
    struct alignas(64) {
        uint8_t palette, start_row, reserved[14];
        uint16_t colsb[16];
        uint8_t rows[16];
    } c = {};
    c.palette = 1;
    c.rows[0] = 8;  c.colsb[0] = 64;
    c.rows[1] = 8;  c.colsb[1] = 64;
    c.rows[2] = 16; c.colsb[2] = 32;
    c.rows[3] = 16; c.colsb[3] = 32;
    for (int t = 4; t < 8; t++) { c.rows[t] = 16; c.colsb[t] = 64; }
    _tile_loadconfig(&c);
}

// Per activation row and block: d_x as float and -128 * sum(x) (the u8 weight bias).
void amx_prep_rows(const char * X, size_t xrow, int64_t nk, int64_t row0, int64_t row1, float * dx, int32_t * nc) {
    const __m256i bias = _mm256_set1_epi8((char) 0x80);
    for (int64_t m = row0; m < row1; m++) {
        const block_q8_0 * xb = (const block_q8_0 *) (X + m * xrow);
        for (int64_t k = 0; k < nk; k++) {
            const __m256i v = _mm256_xor_si256(_mm256_loadu_si256((const __m256i *) xb[k].qs), bias);
            const __m256i s = _mm256_sad_epu8(v, _mm256_setzero_si256());
            const int64_t t = _mm256_extract_epi64(s, 0) + _mm256_extract_epi64(s, 1) + _mm256_extract_epi64(s, 2) +
                              _mm256_extract_epi64(s, 3);
            dx[m * nk + k] = _cvtsh_ss(xb[k].d);
            nc[m * nk + k] = -128 * (int32_t) (t - 32 * 128);
        }
    }
}

// Large-M AMX-INT8 GEMM for Q8_0 on the same i16 records, same per-block arithmetic as the GEMV:
// acc = fma(float(isum), d_w * d_x, acc) in block order, so results stay batch invariant.
// Blocking: panels of AMX_G row groups (16 rows each) x AMX_KC blocks; the panel's codes are unbiased
// to s8 once (tdpbssd, no -128 * sum(x) correction) and reused by every 16-token group. Per 16 tokens x
// 16 rows the fp32 accumulators stay in registers across the panel's blocks and live in Y between
// panels. Tile ops for block k + 1 are issued before the epilogue of block k.
// Some KVM guests drop AMX tile data when a thread is descheduled (see kurn attention): every step
// (16 x 16 x AMX_KC blocks) is timed and recomputed from Y when it took longer than the guard.
constexpr int AMX_G = 4;
constexpr int AMX_KC_MAX = 64;

int64_t amx_kc() {
    static const int64_t v = std::min<int64_t>(AMX_KC_MAX, std::max(4, env_int("GGML_KURN_AMX_KC", 32)));
    return v;
}

// Once the self-test saw tile data lost, GGML_KURN_AMX_GUARD above 8192 (weaker than the previous default) is not
// honoured. Up to 8192 is allowed: the 4096 default puts an AMX-BF16 step at about twice its normal duration (~2% of
// steps redone on Emerald Rapids), and a machine with slower memory may need the margin.
uint64_t amx_guard_cycles(int64_t kn) {
    static const uint64_t base = [] {
        const int g = env_int("GGML_KURN_AMX_GUARD", 4096);
        if (amx_lossy && (g <= 0 || g > 8192)) {
            GGML_LOG_WARN("kurn: GGML_KURN_AMX_GUARD=%d ignored after the failed AMX self-test, using 4096\n", g);
            return (uint64_t) 4096;
        }
        return (uint64_t) g;
    }();
    return base + (uint64_t) kn * 128;
}

std::atomic<uint64_t> g_amx_redo{0};

// one 16 x 16 x 32 block into cbuf; set 0 uses tiles A2 B0 C4, set 1 A3 B1 C5 (alternated so block k + 1 can start
// before block k's epilogue)
inline void amx_tiles_q8(const char * xa, size_t xrow, const uint8_t * bs, int set, int32_t * cbuf) {
    if (set == 0) {
        _tile_loadd(2, xa, (long) xrow); _tile_loadd(0, bs, 64); _tile_zero(4); _tile_dpbssd(4, 2, 0); _tile_stored(4, cbuf, 64);
    } else {
        _tile_loadd(3, xa, (long) xrow); _tile_loadd(1, bs, 64); _tile_zero(5); _tile_dpbssd(5, 3, 1); _tile_stored(5, cbuf, 64);
    }
}

// Y[m*N + n] for n in [r0, r1) (r0 a multiple of 16), m < M (a multiple of 16).
void amx2_q8_0(const uint8_t * buf, int64_t nrec_k, const char * X, size_t xrow, const float * dx, float * Y, int64_t K,
               int64_t N, int64_t M, int64_t r0, int64_t r1) {
    alignas(64) static thread_local uint8_t bs[AMX_G * AMX_KC_MAX * 512];
    alignas(64) static thread_local float dws[AMX_G * AMX_KC_MAX * 16];
    alignas(64) int32_t cbuf[2][16 * 16];
    const int64_t nk = K / 32, KC = amx_kc();
    const __m512i bias = _mm512_set1_epi8((char) 0x80);
    amx_config();
    const int64_t g0 = r0 / 16, g1 = (r1 + 15) / 16;
    for (int64_t gp = g0; gp < g1; gp += AMX_G) {
        const int ng = (int) std::min<int64_t>(AMX_G, g1 - gp);
        for (int64_t kc = 0; kc < nk; kc += KC) {
            const int64_t kn = std::min(KC, nk - kc);
            for (int j = 0; j < ng; j++) {
                for (int64_t k = 0; k < kn; k++) {
                    const uint8_t * rec = buf + ((size_t) (gp + j) * nrec_k + kc + k) * KURN_REC;
                    uint8_t * d = bs + (j * AMX_KC_MAX + k) * 512;
                    for (int l = 0; l < 8; l++) {
                        _mm512_store_si512((__m512i *) (d + 64 * l),
                                           _mm512_xor_si512(_mm512_loadu_si512((const __m512i *) (rec + 32 + 64 * l)), bias));
                    }
                    _mm512_store_ps(dws + (j * AMX_KC_MAX + k) * 16, _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *) rec)));
                }
            }
            const uint64_t guard = amx_guard_cycles(kn);
            for (int64_t m0 = 0; m0 < M; m0 += 16) {
                const char * xa = X + m0 * xrow + kc * sizeof(block_q8_0) + 2;
                const float * dxm = dx + m0 * nk + kc;
                for (int j = 0; j < ng; j++) {
                    const int64_t row0 = (gp + j) * 16;
                    uint32_t mask = 0xFFFF;
                    if (row0 + 16 > r1) mask = (1u << (r1 - row0)) - 1;
                    float * Yb = Y + m0 * N + row0;
                    const uint8_t * bj = bs + (size_t) j * AMX_KC_MAX * 512;
                    const float * dj = dws + (size_t) j * AMX_KC_MAX * 16;
                    for (;;) {
                        const uint64_t t0 = __rdtsc();
                        __m512 acc[16];
                        for (int m = 0; m < 16; m++) {
                            acc[m] = kc == 0 ? _mm512_setzero_ps() : _mm512_maskz_loadu_ps((__mmask16) mask, Yb + m * N);
                        }
                        amx_tiles_q8(xa, xrow, bj, 0, cbuf[0]);
                        for (int64_t k = 0; k < kn; k++) {
                            if (k + 1 < kn) {
                                amx_tiles_q8(xa + (k + 1) * sizeof(block_q8_0), xrow, bj + (k + 1) * 512, (int) ((k + 1) & 1),
                                             cbuf[(k + 1) & 1]);
                            }
                            const __m512 dw = _mm512_load_ps(dj + k * 16);
                            const int32_t * cb = cbuf[k & 1];
                            for (int m = 0; m < 16; m++) {
                                acc[m] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_load_si512(cb + m * 16)),
                                                         _mm512_mul_ps(dw, _mm512_set1_ps(dxm[m * nk + k])), acc[m]);
                            }
                        }
                        unsigned aux;
                        if (__rdtscp(&aux) - t0 > guard) {
                            g_amx_redo.fetch_add(1, std::memory_order_relaxed);
                            continue;
                        }
                        for (int m = 0; m < 16; m++) {
                            _mm512_mask_storeu_ps(Yb + m * N, (__mmask16) mask, acc[m]);
                        }
                        break;
                    }
                }
            }
        }
    }
}

// ------------------------------------------------------------------------------------------
// AMX-BF16 prefill for formats without an exact AMX-INT8 kernel: the weights of a row panel are
// dequantized once per op into bf16 B tiles (16 k-pairs x 16 rows x 2) and every 32-token group of
// bf16 activations runs over them with fp32 accumulation in tiles across all of K. Not batch
// invariant (bf16 activations instead of the GEMV's Q8 activations); GGML_KURN_EXACT=1 keeps such
// formats on the verify kernels.
// Tiles: 0-3 C (16 x 16 fp32), 4-5 A (16 tokens x 32 bf16), 6-7 B.
void amx_config_bf16() {
    struct alignas(64) {
        uint8_t palette, start_row, reserved[14];
        uint16_t colsb[16];
        uint8_t rows[16];
    } c = {};
    c.palette = 1;
    for (int t = 0; t < 8; t++) { c.rows[t] = 16; c.colsb[t] = 64; }
    _tile_loadconfig(&c);
}

// out: (nrows / 16) x (kc / 32) B tiles of 512 bf16, tile (h, t) at out + (h * (kc / 32) + t) * 512, for weight
// rows [n0, n0 + nrows) and values [k0, k0 + kc) (multiples of the format's record rows and record values).
typedef void (*bf16_pack_fn)(const uint8_t * buf, int64_t K, int64_t n0, int64_t nrows, int64_t k0, int64_t kc, uint16_t * out);

// Q4_K on kurn's i16 records (32 rows x 256 values: fp16 d[32], fp16 dmin[32], 6-bit scales [8][32],
// mins at 384 + (s / 4) * 128 + row * 4 + s % 4, then per 32-value sub-block 8 lines of 64 bytes whose
// byte (row % 16) * 4 + j holds value 4 * line + j of row `row` (low nibble) and of row + 16 (high nibble)).
// w = d * sc * q - dmin * m, as ggml's dequantize_row_q4_K.
void pack_bf16_q4_K(const uint8_t * buf, int64_t K, int64_t n0, int64_t nrows, int64_t k0, int64_t kc, uint16_t * out) {
    constexpr int REC = 4736, HDR = 640;
    const int64_t nrec_k = K / 256, nt = kc / 32;
    const __m512i ev = _mm512_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30);
    const __m512i od = _mm512_setr_epi32(1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31);
    const __m512i lo4 = _mm512_set1_epi32(15);
    __m512i rep[4];
    for (int i = 0; i < 4; i++) {
        rep[i] = _mm512_setr_epi32(4 * i, 4 * i, 4 * i, 4 * i, 4 * i + 1, 4 * i + 1, 4 * i + 1, 4 * i + 1, 4 * i + 2, 4 * i + 2,
                                   4 * i + 2, 4 * i + 2, 4 * i + 3, 4 * i + 3, 4 * i + 3, 4 * i + 3);
    }
    for (int64_t g = n0 / 32; g < (n0 + nrows) / 32; g++) {
        const int64_t hl = 2 * (g - n0 / 32);
        for (int64_t p = k0 / 256; p < (k0 + kc) / 256; p++) {
            const uint8_t * hp = buf + ((size_t) g * nrec_k + p) * REC;
            __m512 d[2], dm[2];
            for (int h = 0; h < 2; h++) {
                d[h] = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *) (hp + 32 * h)));
                dm[h] = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *) (hp + 64 + 32 * h)));
            }
            for (int s = 0; s < 8; s++) {
                const uint8_t * codes = hp + HDR + s * 512;
                __m512 sx[2][4], mx[2][4];
                for (int h = 0; h < 2; h++) {
                    const __m512 sc = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *) (hp + 128 + s * 32 + 16 * h))));
                    const __m512i mr = _mm512_loadu_si512((const __m512i *) (hp + 384 + (s / 4) * 128 + 64 * h));
                    const __m512 mn = _mm512_cvtepi32_ps(_mm512_and_si512(_mm512_srli_epi32(mr, 8 * (s % 4)), _mm512_set1_epi32(255)));
                    const __m512 vs = _mm512_mul_ps(d[h], sc), vm = _mm512_mul_ps(dm[h], mn);
                    for (int i = 0; i < 4; i++) {
                        sx[h][i] = _mm512_permutexvar_ps(rep[i], vs);
                        mx[h][i] = _mm512_permutexvar_ps(rep[i], vm);
                    }
                }
                const int64_t t = (p - k0 / 256) * 8 + s;
                uint16_t * o0 = out + ((hl + 0) * nt + t) * 512;
                uint16_t * o1 = out + ((hl + 1) * nt + t) * 512;
                for (int kk = 0; kk < 8; kk++) {
                    __m512 f[2][4];
                    for (int i = 0; i < 4; i++) {
                        const __m512i v = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *) (codes + kk * 64 + 16 * i)));
                        f[0][i] = _mm512_fmsub_ps(_mm512_cvtepi32_ps(_mm512_and_si512(v, lo4)), sx[0][i], mx[0][i]);
                        f[1][i] = _mm512_fmsub_ps(_mm512_cvtepi32_ps(_mm512_srli_epi32(v, 4)), sx[1][i], mx[1][i]);
                    }
                    for (int h = 0; h < 2; h++) {
                        const __m512i b01 = (__m512i) _mm512_cvtne2ps_pbh(f[h][1], f[h][0]);
                        const __m512i b23 = (__m512i) _mm512_cvtne2ps_pbh(f[h][3], f[h][2]);
                        uint16_t * o = h ? o1 : o0;
                        _mm512_store_si512((__m512i *) (o + (2 * kk) * 32), _mm512_permutex2var_epi32(b01, ev, b23));
                        _mm512_store_si512((__m512i *) (o + (2 * kk + 1) * 32), _mm512_permutex2var_epi32(b01, od, b23));
                    }
                }
            }
        }
    }
}

// Q8_0 on kurn's i16 records (16 rows x 32 values: fp16 d[16], then 8 lines of 64 bytes whose byte row * 4 + j
// holds value 4 * line + j of the row, biased by +128). w = d * q.
void pack_bf16_q8_0(const uint8_t * buf, int64_t K, int64_t n0, int64_t nrows, int64_t k0, int64_t kc, uint16_t * out) {
    const int64_t nk = K / 32, nt = kc / 32;
    const __m512i ev = _mm512_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30);
    const __m512i od = _mm512_setr_epi32(1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31);
    const __m512i b128 = _mm512_set1_epi32(128);
    __m512i rep[4];
    for (int i = 0; i < 4; i++) {
        rep[i] = _mm512_setr_epi32(4 * i, 4 * i, 4 * i, 4 * i, 4 * i + 1, 4 * i + 1, 4 * i + 1, 4 * i + 1, 4 * i + 2, 4 * i + 2,
                                   4 * i + 2, 4 * i + 2, 4 * i + 3, 4 * i + 3, 4 * i + 3, 4 * i + 3);
    }
    for (int64_t g = n0 / 16; g < (n0 + nrows) / 16; g++) {
        const int64_t h = g - n0 / 16;
        for (int64_t k = k0 / 32; k < (k0 + kc) / 32; k++) {
            const uint8_t * rec = buf + ((size_t) g * nk + k) * KURN_REC;
            const __m512 d = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *) rec));
            __m512 dx[4];
            for (int i = 0; i < 4; i++) {
                dx[i] = _mm512_permutexvar_ps(rep[i], d);
            }
            uint16_t * o = out + (h * nt + k - k0 / 32) * 512;
            for (int kk = 0; kk < 8; kk++) {
                __m512 f[4];
                for (int i = 0; i < 4; i++) {
                    const __m512i v = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *) (rec + 32 + kk * 64 + 16 * i)));
                    f[i] = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_sub_epi32(v, b128)), dx[i]);
                }
                const __m512i b01 = (__m512i) _mm512_cvtne2ps_pbh(f[1], f[0]);
                const __m512i b23 = (__m512i) _mm512_cvtne2ps_pbh(f[3], f[2]);
                _mm512_store_si512((__m512i *) (o + (2 * kk) * 32), _mm512_permutex2var_epi32(b01, ev, b23));
                _mm512_store_si512((__m512i *) (o + (2 * kk + 1) * 32), _mm512_permutex2var_epi32(b01, od, b23));
            }
        }
    }
}

// Native-layout formats (ggml blocks, not repacked): one 256-value block of one row -> 256 bf16.
struct block_q6_K_n { uint8_t ql[128]; uint8_t qh[64]; int8_t scales[16]; uint16_t d; };
struct block_q5_K_n { uint16_t d, dmin; uint8_t scales[12]; uint8_t qh[32]; uint8_t qs[128]; };
static_assert(sizeof(block_q6_K_n) == 210, "q6_K");
static_assert(sizeof(block_q5_K_n) == 176, "q5_K");

inline void store_bf16x32(uint16_t * o, __m512 a, __m512 b) {
    _mm512_storeu_si512((__m512i *) o, (__m512i) _mm512_cvtne2ps_pbh(b, a));
}

// 32 int8 -> 32 bf16 of s * q (two halves with their own scales)
inline void deq32_i8(uint16_t * o, __m256i q, float s0, float s1) {
    const __m512 a = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm256_castsi256_si128(q))), _mm512_set1_ps(s0));
    const __m512 b = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm256_extracti128_si256(q, 1))), _mm512_set1_ps(s1));
    store_bf16x32(o, a, b);
}

// as ggml's dequantize_row_q6_K: y = d * sc[l / 16] * (q - 32)
void deq_block_q6_K(const uint8_t * p, uint16_t * o) {
    const block_q6_K_n * b = (const block_q6_K_n *) p;
    const float d = _cvtsh_ss(b->d);
    const __m256i m4 = _mm256_set1_epi8(0x0F), m3 = _mm256_set1_epi8(3), c32 = _mm256_set1_epi8(32);
    for (int h = 0; h < 2; h++) {
        const uint8_t * ql = b->ql + 64 * h;
        const uint8_t * qh = b->qh + 32 * h;
        const int8_t * sc = b->scales + 8 * h;
        const __m256i l0 = _mm256_loadu_si256((const __m256i *) ql), l1 = _mm256_loadu_si256((const __m256i *) (ql + 32));
        const __m256i hb = _mm256_loadu_si256((const __m256i *) qh);
        const __m256i q1 = _mm256_sub_epi8(_mm256_or_si256(_mm256_and_si256(l0, m4), _mm256_slli_epi16(_mm256_and_si256(hb, m3), 4)), c32);
        const __m256i q2 = _mm256_sub_epi8(_mm256_or_si256(_mm256_and_si256(l1, m4),
                                                           _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(hb, 2), m3), 4)), c32);
        const __m256i q3 = _mm256_sub_epi8(_mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(l0, 4), m4),
                                                           _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(hb, 4), m3), 4)), c32);
        const __m256i q4 = _mm256_sub_epi8(_mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(l1, 4), m4),
                                                           _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(hb, 6), m3), 4)), c32);
        uint16_t * y = o + 128 * h;
        deq32_i8(y + 0, q1, d * sc[0], d * sc[1]);
        deq32_i8(y + 32, q2, d * sc[2], d * sc[3]);
        deq32_i8(y + 64, q3, d * sc[4], d * sc[5]);
        deq32_i8(y + 96, q4, d * sc[6], d * sc[7]);
    }
}

inline void scale_min_k4(int j, const uint8_t * q, uint8_t & d, uint8_t & m) {
    if (j < 4) {
        d = q[j] & 63; m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

// as ggml's dequantize_row_q5_K: y = d * sc * (q4 + 16 * hbit) - dmin * m
void deq_block_q5_K(const uint8_t * p, uint16_t * o) {
    const block_q5_K_n * b = (const block_q5_K_n *) p;
    const float d = _cvtsh_ss(b->d), dmin = _cvtsh_ss(b->dmin);
    const __m256i m4 = _mm256_set1_epi8(0x0F), one = _mm256_set1_epi8(1);
    const __m256i hb = _mm256_loadu_si256((const __m256i *) b->qh);
    for (int j = 0; j < 4; j++) {
        uint8_t sc0, m0, sc1, m1;
        scale_min_k4(2 * j, b->scales, sc0, m0);
        scale_min_k4(2 * j + 1, b->scales, sc1, m1);
        const __m256i ql = _mm256_loadu_si256((const __m256i *) (b->qs + 32 * j));
        const __m256i h0 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(hb, 2 * j), one), 4);
        const __m256i h1 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(hb, 2 * j + 1), one), 4);
        const __m256i qa = _mm256_or_si256(_mm256_and_si256(ql, m4), h0);
        const __m256i qb = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(ql, 4), m4), h1);
        const __m512 sa = _mm512_set1_ps(d * sc0), ma = _mm512_set1_ps(dmin * m0);
        const __m512 sb = _mm512_set1_ps(d * sc1), mb = _mm512_set1_ps(dmin * m1);
        const __m512 a0 = _mm512_fmsub_ps(_mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm256_castsi256_si128(qa))), sa, ma);
        const __m512 a1 = _mm512_fmsub_ps(_mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm256_extracti128_si256(qa, 1))), sa, ma);
        const __m512 b0 = _mm512_fmsub_ps(_mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm256_castsi256_si128(qb))), sb, mb);
        const __m512 b1 = _mm512_fmsub_ps(_mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm256_extracti128_si256(qb, 1))), sb, mb);
        store_bf16x32(o + 64 * j, a0, a1);
        store_bf16x32(o + 64 * j + 32, b0, b1);
    }
}

// r[i] = row i of a 16 x 16 dword matrix -> r[c] = column c
inline void transpose16_epi32(__m512i r[16]) {
    __m512i t[16];
    for (int i = 0; i < 8; i++) {
        t[2 * i] = _mm512_unpacklo_epi32(r[2 * i], r[2 * i + 1]);
        t[2 * i + 1] = _mm512_unpackhi_epi32(r[2 * i], r[2 * i + 1]);
    }
    for (int i = 0; i < 4; i++) {
        r[4 * i + 0] = _mm512_unpacklo_epi64(t[4 * i + 0], t[4 * i + 2]);
        r[4 * i + 1] = _mm512_unpackhi_epi64(t[4 * i + 0], t[4 * i + 2]);
        r[4 * i + 2] = _mm512_unpacklo_epi64(t[4 * i + 1], t[4 * i + 3]);
        r[4 * i + 3] = _mm512_unpackhi_epi64(t[4 * i + 1], t[4 * i + 3]);
    }
    // r[4i + j], 128-bit lane L: rows 4i..4i+3 of column 4L + j
    for (int j = 0; j < 4; j++) {
        const __m512i p = _mm512_shuffle_i32x4(r[j], r[4 + j], 0x44), q = _mm512_shuffle_i32x4(r[8 + j], r[12 + j], 0x44);
        const __m512i p2 = _mm512_shuffle_i32x4(r[j], r[4 + j], 0xEE), q2 = _mm512_shuffle_i32x4(r[8 + j], r[12 + j], 0xEE);
        t[0 + j] = _mm512_shuffle_i32x4(p, q, 0x88);
        t[4 + j] = _mm512_shuffle_i32x4(p, q, 0xDD);
        t[8 + j] = _mm512_shuffle_i32x4(p2, q2, 0x88);
        t[12 + j] = _mm512_shuffle_i32x4(p2, q2, 0xDD);
    }
    for (int c = 0; c < 16; c++) {
        r[c] = t[c];
    }
}

typedef void (*deq_block_fn)(const uint8_t *, uint16_t *);

template <deq_block_fn DEQ, size_t BLK>
void pack_bf16_native(const uint8_t * buf, int64_t K, int64_t n0, int64_t nrows, int64_t k0, int64_t kc, uint16_t * out) {
    const int64_t nb = K / 256, nt = kc / 32;
    const size_t wrow = (size_t) nb * BLK;
    alignas(64) uint16_t rowbuf[16 * 256];
    for (int64_t h = 0; h < nrows / 16; h++) {
        for (int64_t b = k0 / 256; b < (k0 + kc) / 256; b++) {
            for (int r = 0; r < 16; r++) {
                DEQ(buf + (size_t) (n0 + h * 16 + r) * wrow + (size_t) b * BLK, rowbuf + r * 256);
            }
            for (int t = 0; t < 8; t++) {
                __m512i v[16];
                for (int r = 0; r < 16; r++) {
                    v[r] = _mm512_load_si512((const __m512i *) (rowbuf + r * 256 + t * 32));
                }
                transpose16_epi32(v);
                uint16_t * o = out + (h * nt + (b - k0 / 256) * 8 + t) * 512;
                for (int kp = 0; kp < 16; kp++) {
                    _mm512_store_si512((__m512i *) (o + kp * 32), v[kp]);
                }
            }
        }
    }
}

// Y[m*N + n] for n in [r0, r1), m < M. A: bf16 activations, row m at A + m * K, rows padded with zeros to a
// multiple of 32. r0 and r1 multiples of `align` (the format's record rows), or r1 == N; K a multiple of kalign
// (the format's record values). Blocking: panels of GGML_KURN_BF16_ROWS weight rows x GGML_KURN_BF16_KC values
// are dequantized into bs (L2); each 32-token chunk of A (L1) runs over every 32-row pair of the panel with the
// four C tiles; partial sums live in cst between K chunks, double-buffered by chunk parity so that a step hit by
// a preemption (see amx2_q8_0) is redone from intact sums.
void amx_bf16_gemm(const uint8_t * buf, bf16_pack_fn pack, int64_t align, int64_t kalign, const uint16_t * A, int64_t M,
                   float * Y, int64_t K, int64_t N, int64_t r0, int64_t r1) {
    static const int64_t rows_env = env_int("GGML_KURN_BF16_ROWS", 128), kc_env = env_int("GGML_KURN_BF16_KC", 4096);
    static const int64_t kch = std::max(4, env_int("GGML_KURN_BF16_KCH", 32));
    alignas(64) float sub[2][1024];
    const int64_t Mp = (M + 31) / 32 * 32, nmg = Mp / 32;
    const int64_t P = std::max(align, rows_env / align * align);
    const int64_t KC = std::max(kalign, kc_env / kalign * kalign);
    const int64_t npm = (P / 16 + 1) / 2;
    static thread_local std::vector<uint8_t> bsv, csv;
    const size_t bs_need = (size_t) P * KC * 2 + 64, cs_need = (size_t) 2 * nmg * npm * 4096 + 64;
    if (bsv.size() < bs_need) bsv.resize(bs_need);
    if (csv.size() < cs_need) csv.resize(cs_need);
    uint16_t * bs = (uint16_t *) (((uintptr_t) bsv.data() + 63) & ~(uintptr_t) 63);
    float * cst = (float *) (((uintptr_t) csv.data() + 63) & ~(uintptr_t) 63);
    for (int64_t n0 = r0; n0 < r1; n0 += P) {
        const int64_t rows = std::min(P, (r1 - n0 + align - 1) / align * align);
        const int64_t ng = rows / 16, npair = (ng + 1) / 2;
        int64_t c = 0;
        for (int64_t kc0 = 0; kc0 < K; kc0 += KC, c++) {
            const int64_t kc = std::min(KC, K - kc0), nt = kc / 32;
            const bool last = kc0 + kc >= K;
            const uint64_t guard = amx_guard_cycles(kch);
            pack(buf, K, n0, rows, kc0, kc, bs);
            amx_config_bf16();
            for (int64_t mg = 0; mg < nmg; mg++) {
                const uint16_t * a0 = A + mg * 32 * K + kc0;
                const uint16_t * a1 = a0 + 16 * K;
                for (int64_t j = 0; j < npair; j++) {
                    const int64_t h = 2 * j;
                    const bool two = h + 1 < ng;
                    const uint16_t * b0 = bs + h * nt * 512;
                    const uint16_t * b1 = bs + (h + 1) * nt * 512;
                    const float * prev = cst + (((c + 1) & 1) * nmg * npm + mg * npm + j) * 1024;
                    float * cur = cst + ((c & 1) * nmg * npm + mg * npm + j) * 1024;
                    // sub-steps of kch k-steps keep every guarded window short; their partial sums alternate
                    // between two L1 buffers so that a redo starts from intact sums
                    for (int64_t s0 = 0, si = 0; s0 < nt; s0 += kch, si++) {
                        const int64_t s1 = std::min(nt, s0 + kch);
                        const float * in = s0 == 0 ? (c == 0 ? nullptr : prev) : sub[(si + 1) & 1];
                        float * outp = s1 == nt ? cur : sub[si & 1];
                        for (;;) {
                            const uint64_t t0 = __rdtsc();
                            if (!in) {
                                _tile_zero(0); _tile_zero(1); _tile_zero(2); _tile_zero(3);
                            } else {
                                _tile_loadd(0, in, 64); _tile_loadd(2, in + 512, 64);
                                if (two) { _tile_loadd(1, in + 256, 64); _tile_loadd(3, in + 768, 64); }
                            }
                            if (two) {
                                for (int64_t t = s0; t < s1; t++) {
                                    _tile_loadd(4, a0 + t * 32, K * 2);
                                    _tile_loadd(6, b0 + t * 512, 64);
                                    _tile_loadd(5, a1 + t * 32, K * 2);
                                    _tile_loadd(7, b1 + t * 512, 64);
                                    _tile_dpbf16ps(0, 4, 6);
                                    _tile_dpbf16ps(1, 4, 7);
                                    _tile_dpbf16ps(2, 5, 6);
                                    _tile_dpbf16ps(3, 5, 7);
                                }
                            } else {
                                for (int64_t t = s0; t < s1; t++) {
                                    _tile_loadd(4, a0 + t * 32, K * 2);
                                    _tile_loadd(6, b0 + t * 512, 64);
                                    _tile_loadd(5, a1 + t * 32, K * 2);
                                    _tile_dpbf16ps(0, 4, 6);
                                    _tile_dpbf16ps(2, 5, 6);
                                }
                            }
                            _tile_stored(0, outp, 64);
                            _tile_stored(2, outp + 512, 64);
                            if (two) {
                                _tile_stored(1, outp + 256, 64);
                                _tile_stored(3, outp + 768, 64);
                            }
                            unsigned aux;
                            if (__rdtscp(&aux) - t0 > guard) {
                                g_amx_redo.fetch_add(1, std::memory_order_relaxed);
                                continue;
                            }
                            break;
                        }
                    }
                    if (!last) {
                        continue;
                    }
                    for (int q = 0; q < 4; q++) {
                        if (!two && (q & 1)) {
                            continue;
                        }
                        const int64_t mb = mg * 32 + (q >> 1) * 16, nb = n0 + (h + (q & 1)) * 16;
                        if (nb >= r1) {
                            continue;
                        }
                        const __mmask16 mask = (__mmask16) (nb + 16 <= r1 ? 0xFFFF : (1u << (r1 - nb)) - 1);
                        for (int m = 0; m < 16 && mb + m < M; m++) {
                            _mm512_mask_storeu_ps(Y + (mb + m) * N + nb, mask, _mm512_load_ps(cur + q * 256 + m * 16));
                        }
                    }
                }
            }
        }
    }
}
#endif

// ------------------------------------------------------------------------------------------
// GGML_KURN_PROFILE=1: per-phase cycle counts of MUL_MAT (decode-sized ops, M <= 8), printed at exit.
// Phases per thread: quantize + activation prep, wait at the post-quantize barrier, kernels; plus,
// per op, the spread between the first and the last thread to finish its kernels (load imbalance,
// paid as wait time at ggml's barrier after the node).
struct kurn_profile {
    static constexpr int MAXT = 64;
    struct alignas(64) slot { uint64_t quant = 0, wait = 0, kern = 0, ops = 0, done = 0; };
    slot t[MAXT];
    uint64_t spread = 0, nops = 0;
    bool on = env_int("GGML_KURN_PROFILE", 0) != 0;
    ~kurn_profile() {
#if KURN_HAVE_AMX
        if (verbose()) {
            fprintf(stderr, "kurn: AMX steps recomputed after a preemption-length gap: %llu\n",
                    (unsigned long long) g_amx_redo.load());
        }
#endif
        if (!on || !nops) {
            return;
        }
        uint64_t q = 0, w = 0, k = 0, n = 0;
        for (const slot & s : t) { q += s.quant; w += s.wait; k += s.kern; n += s.ops; }
        fprintf(stderr, "kurn profile: %llu ops; per op and thread (kcycles): quantize+prep %.2f, barrier wait %.2f, kernels %.2f; "
                "finish spread %.2f kcycles/op\n", (unsigned long long) nops, 1e-3 * q / n, 1e-3 * w / n, 1e-3 * k / n,
                1e-3 * spread / nops);
    }
};
kurn_profile g_prof;

struct tensor_traits : ggml::cpu::tensor_traits {
    const kurn_kernel * k = nullptr;
    std::vector<void *> views;  // packed_t header per expert, pointing into the tensor data
    std::vector<uint8_t *> data;
    size_t stride = 0;
    bool amx = false;
#if KURN_HAVE_AMX
    bf16_pack_fn bpack = nullptr;  // AMX-BF16 prefill
    int64_t balign = 0, bkalign = 256;  // record rows and record values of bpack's format
#endif
    ggml_type ntype = GGML_TYPE_COUNT;  // native-layout weight (k == nullptr): ggml blocks, ggml vec_dot below amx_min
    int64_t ralign = 16, rpass = 64;    // row granularity for row_chunks
    int64_t nrows = 1;                  // weight rows (ne[1]); Y's column stride can differ in fused ops
    bool tbm = false;                   // TQ1_0 / TQ2_0 repacked as bitmap + signs (kurn-tbm.cpp), set at set_tensor
#if KURN_XS
    kxs::tensor * xs = nullptr;         // experts streamed from SSD (kurn-xs.cpp): no views / data here
#endif

    bool use_bf16(int64_t M) const {
#if KURN_HAVE_AMX
        // native formats switch at 16 columns: ggml's vec_dot, their only other path, is 2x slower than AMX-BF16
        // at 16-24 columns (Q6_K 12288 x 4096)
        return bpack && M >= (k ? amx_min() : std::min<int64_t>(16, amx_min()));
#else
        GGML_UNUSED(M);
        return false;
#endif
    }

    ~tensor_traits() override {
        for (void * v : views) {
            free(v);
        }
    }

    static int64_t amx_min() {
        // below 32 columns the verify kernels beat both AMX paths (Qwen3-8B 4096 x 12288, 8 threads: Q8_0 at 16
        // columns 1.46 vs 0.90 (BF16) / 1.35 (INT8) TMAC/s; at 32 1.75 vs 2.10 / 1.91)
        static const int64_t v = std::max(16, env_int("GGML_KURN_AMX_MIN", 32));
        return v;
    }

    bool use_amx(int64_t M) const {
        return amx && M >= amx_min();
    }

    // 9..16 columns of Q8_0: kurn-amx-vfy.cpp (M-row tiles, one pass over the weights, bit-identical to the GEMV per
    // column) instead of two 8-column VNNI passes. GGML_KURN_AMX_VFY=0 turns it off.
    bool use_amx_vfy(int64_t M) const {
        static const bool on = env_int("GGML_KURN_AMX_VFY", 1) != 0;
        return amx && on && M > KURN_VFY_MAX && M <= 16 && M < amx_min();
    }

    // xw: shared activation prep of these columns (k->xprep layout for xc columns, starting at column xc0)
    struct xprep_ref {
        const void * ws = nullptr;
        int64_t C = 0, c0 = 0;
    };

    // one activation column, rows [r0, r1) (r0 a multiple of row_align): full-width passes of `gemv`
    // over the bulk, the one-record `gemv1` over the rest
    void gemv_rows(const void * pk, const char * X, float * Y, int64_t K, int64_t r0, int64_t r1, const xprep_ref & xr,
                   int64_t col) const {
        const int64_t mid = std::min(r1, r0 + (r1 - r0) / k->row_pass * k->row_pass);
        if (xr.ws) {
            if (mid > r0) k->gemv_x(pk, X, xr.ws, xr.C, xr.c0 + col, Y, K, r0, mid);
            if (r1 > mid) k->gemv1_x(pk, X, xr.ws, xr.C, xr.c0 + col, Y, K, mid, r1);
        } else {
            if (mid > r0) k->gemv(pk, X, Y, K, r0, mid);
            if (r1 > mid) k->gemv1(pk, X, Y, K, mid, r1);
        }
    }

    void cols(const void * pk, const char * X, size_t xrow, float * Y, int64_t K, int64_t N, int64_t M, int64_t r0,
              int64_t r1, const xprep_ref & xr) const {
        for (int64_t c0 = 0; c0 < M; c0 += KURN_VFY_MAX) {
            const int64_t m = std::min<int64_t>(KURN_VFY_MAX, M - c0);
            if (m == 1) {
                gemv_rows(pk, X + c0 * xrow, Y + c0 * N, K, r0, r1, xr, c0);
            } else if (xr.ws) {
                k->vfy_x[vfy_index(m)](pk, X + c0 * xrow, xr.ws, xr.C, xr.c0 + c0, Y + c0 * N, K, N, m, r0, r1);
            } else {
                k->vfy[vfy_index(m)](pk, X + c0 * xrow, Y + c0 * N, K, N, m, r0, r1);
            }
        }
    }

    // M columns of quantized activations X (row stride xrow) -> Y (column stride N), rows [r0, r1).
    // dx / nc: AMX activation prep for the same rows (only when use_amx(M)).
    void run(int e, const char * X, size_t xrow, float * Y, int64_t K, int64_t N, int64_t M, int64_t r0, int64_t r1,
             const xprep_ref & xr, const float * dx = nullptr, const int32_t * nc = nullptr) const {
        if (tbm) {
            if (r0 < r1) {
                run_tbm(e, X, xrow, Y, K, N, M, r0, r1);
            }
            return;
        }
        run_pk(views[e], data[e], X, xrow, Y, K, N, M, r0, r1, xr, dx, nc);
    }

    // the same with expert e's packed view and data given explicitly
    void run_pk(const void * pk, const uint8_t * w, const char * X, size_t xrow, float * Y, int64_t K, int64_t N, int64_t M,
                int64_t r0, int64_t r1, const xprep_ref & xr, const float * dx = nullptr, const int32_t * nc = nullptr) const {
        if (r0 >= r1) {
            return;
        }
        if (M <= KURN_VFY_MAX) {
            cols(pk, X, xrow, Y, K, N, M, r0, r1, xr);
            return;
        }
#if KURN_HAVE_AMX
        if (dx && use_amx_vfy(M)) {
            static const int ng = std::min(3, std::max(1, env_int("GGML_KURN_AMX_VFY_NG", 3)));
            if (kurn_amx_vfy_q8_0(w, K, X, xrow, dx, nc, Y, N, M, r0, r1, ng)) {
                return;
            }
        }
        if (dx && use_amx(M)) {
            const int64_t full = M / 16 * 16;
            amx2_q8_0(w, K / 32, X, xrow, dx, Y, K, N, full, r0, r1);
            if (full < M) {
                xprep_ref xt = xr;
                xt.c0 += full;
                cols(pk, X + full * xrow, xrow, Y + full * N, K, N, M - full, r0, r1, xt);
            }
            return;
        }
#endif
        GGML_UNUSED(nc);
        // prefill: row chunks that stay in L2 while every column group passes over them
        static const int64_t chunk_bytes = (int64_t) env_int("GGML_KURN_CHUNK_KB", 1024) * 1024;
        const int64_t row_bytes = std::max<int64_t>(1, (int64_t) (stride / (size_t) nrows));
        int64_t chunk = std::max<int64_t>(64, chunk_bytes / row_bytes);
        chunk = (chunk + 63) / 64 * 64;
        for (int64_t a = r0; a < r1; a += chunk) {
            const int64_t b = std::min(r1, a + chunk);
            cols(pk, X, xrow, Y, K, N, M, a, b, xr);
        }
    }

    // bitmap + signs: kurn-tbm.cpp kernels on the raw Q8_K activations (no xprep); from GGML_KURN_TBM_AMX_MIN
    // (default 32) columns, multiples of 16 on AMX-INT8 (same per-column arithmetic, so batch invariant and
    // allowed in exact mode), the rest in groups of 8 columns over row chunks that stay in L2
    void run_tbm(int e, const char * X, size_t xrow, float * Y, int64_t K, int64_t N, int64_t M, int64_t r0, int64_t r1) const {
#if KURN_HAVE_AMX
        static const int64_t amx_cols = std::max(16, env_int("GGML_KURN_TBM_AMX_MIN", 32));
        if (amx_ok && amx_mm_enabled() && kurn_tbm_amx_available() && M >= amx_cols) {
            const int64_t full = M / 16 * 16;
            kurn_tbm_amx(data[e], X, xrow, Y, K, N, full, r0, r1);
            X += full * xrow;
            Y += full * N;
            M -= full;
        }
#endif
        if (M <= 0) {
            return;
        }
        static const int64_t chunk_bytes = (int64_t) env_int("GGML_KURN_CHUNK_KB", 1024) * 1024;
        const int64_t chunk = M <= KURN_VFY_MAX ? r1 - r0 : std::max<int64_t>(64, chunk_bytes / std::max<int64_t>(1, K / 4)) / 64 * 64;
        for (int64_t a = r0; a < r1; a += chunk) {
            const int64_t b = std::min(r1, a + chunk);
            for (int64_t c0 = 0; c0 < M; c0 += KURN_VFY_MAX) {
                kurn_tbm_cols(data[e], X + c0 * xrow, xrow, Y + c0 * N, K, N, std::min<int64_t>(KURN_VFY_MAX, M - c0), a, b);
            }
        }
    }

    size_t xprep_bytes(int64_t K, int64_t C) const {
        return k->xprep_bytes ? k->xprep_bytes(K, C) + 64 : 0;
    }

    size_t amx_bytes(int64_t rows, int64_t K) const {
        return amx ? (size_t) rows * (K / 32) * 8 + 128 : 0;
    }

    bool work_size(int n_threads, const ggml_tensor * op, size_t & size) override {
        const ggml_tensor * src0 = op->src[0];
        const ggml_tensor * src1 = op->src[1];
        const int64_t K = src0->ne[0];
        const size_t xrow = ggml_row_size(k ? (ggml_type) k->vec_dot_type : ggml_get_type_traits_cpu(ntype)->vec_dot_type, K);
        if (op->op == GGML_OP_MUL_MAT) {
            size = xrow * (size_t) ggml_nrows(src1) + 64;
            if (k) {
                size += amx_bytes(ggml_nrows(src1), K) + xprep_bytes(K, ggml_nrows(src1));
            }
            if (use_bf16(ggml_nrows(src1))) {
                size = std::max(size, (size_t) ((ggml_nrows(src1) + 31) / 32 * 32) * (size_t) K * 2 + 64);
            }
            size += 64 * ((size_t) n_threads + STEAL_RESERVE_SLOTS);  // steal_sched slots, at the end of the work buffer
            return true;
        }
        const ggml_tensor * ids = op->src[2];
        const size_t nsel = (size_t) ids->ne[0] * ids->ne[1];
        const size_t E = (size_t) src0->ne[2];
        size = xrow * (size_t) (src1->ne[1] * src1->ne[2]) + xrow * nsel + sizeof(float) * nsel * src0->ne[1] +
               sizeof(int64_t) * (4 * E + 2) + sizeof(int32_t) * nsel + amx_bytes((int64_t) nsel, K) +
               xprep_bytes(K, (int64_t) nsel) + 8 * 64;
        return true;
    }

    // Quantize every row of src1 into wq, split across threads by activation blocks (one decode
    // row is shared by all threads instead of quantized by thread 0 alone), and, when xw is
    // given, build the shared activation prep of the same blocks.
    void quantize_rows(const ggml_compute_params * p, const ggml_tensor * src1, ggml_from_float_t from_float, char * wq,
                       size_t xrow, void * xw) const {
        const int64_t ne11 = src1->ne[1], ne12 = src1->ne[2], K = src1->ne[0];
        const int64_t n = ggml_nrows(src1);
        const ggml_type vdt = (ggml_type) k->vec_dot_type;
        const int64_t bs = ggml_blck_size(vdt), ts = (int64_t) ggml_type_size(vdt), nb = K / bs;
        static const bool split = env_int("GGML_KURN_QSPLIT", 0) != 0;  // 1: split rows by blocks
        const int64_t total = n * nb;
        int64_t i0 = total * p->ith / p->nth, i1 = total * (p->ith + 1) / p->nth;
        if (!split) {
            const int64_t r0 = n * p->ith / p->nth, r1 = n * (p->ith + 1) / p->nth;
            i0 = r0 * nb;
            i1 = r1 * nb;
        }
        for (int64_t i = i0; i < i1;) {
            const int64_t c = i / nb, b0 = i % nb, b1 = std::min(nb, b0 + (i1 - i));
            const int64_t i11 = c % ne11, i12 = (c / ne11) % ne12, i13 = c / (ne11 * ne12);
            const float * src = (const float *) ((const char *) src1->data + i11 * src1->nb[1] + i12 * src1->nb[2] + i13 * src1->nb[3]);
            from_float(src + b0 * bs, wq + c * xrow + b0 * ts, (b1 - b0) * bs);
            if (xw) {
                k->xprep(wq, K, n, c, c + 1, b0 * (bs / 32), b1 * (bs / 32), xw);
            }
            i += b1 - b0;
        }
    }

    // AMX activation prep for rows [0, nrows) of X, split across threads (caller barriers)
    void prep(const ggml_compute_params * p, const char * X, size_t xrow, int64_t nrows, int64_t K, float * dx,
              int32_t * nc) const {
#if KURN_HAVE_AMX
        const int64_t a = nrows * p->ith / p->nth, b = nrows * (p->ith + 1) / p->nth;
        amx_prep_rows(X, xrow, K / 32, a, b, dx, nc);
#else
        GGML_UNUSED(p); GGML_UNUSED(X); GGML_UNUSED(xrow); GGML_UNUSED(nrows); GGML_UNUSED(K); GGML_UNUSED(dx); GGML_UNUSED(nc);
#endif
    }

    bool compute_forward(ggml_compute_params * params, ggml_tensor * op) override {
        ggml_kurn_numa_pin_thread(params->ith, params->nth);
        if (op->op == GGML_OP_MUL_MAT) {
            mul_mat(params, op);
            return true;
        }
        if (op->op == GGML_OP_MUL_MAT_ID) {
            mul_mat_id(params, op);
            return true;
        }
        return false;
    }

    static char * align64(void * p) {
        return (char *) (((uintptr_t) p + 63) & ~(uintptr_t) 63);
    }

#if KURN_HAVE_AMX
    // src1 rows -> bf16 rows (padded with zero rows to a multiple of 32), then the AMX-BF16 GEMM
    void mul_mat_bf16(ggml_compute_params * params, ggml_tensor * op) {
        const ggml_tensor * src0 = op->src[0];
        const ggml_tensor * src1 = op->src[1];
        const int64_t K = src0->ne[0], N = src0->ne[1], M = ggml_nrows(src1), Mp = (M + 31) / 32 * 32;
        const int64_t ne11 = src1->ne[1], ne12 = src1->ne[2];
        uint16_t * A = (uint16_t *) align64(params->wdata);
        for (int64_t c = Mp * params->ith / params->nth; c < Mp * (params->ith + 1) / params->nth; c++) {
            uint16_t * dst = A + c * K;
            if (c >= M) {
                memset(dst, 0, (size_t) K * 2);
                continue;
            }
            const int64_t i11 = c % ne11, i12 = (c / ne11) % ne12, i13 = c / (ne11 * ne12);
            const float * src = (const float *) ((const char *) src1->data + i11 * src1->nb[1] + i12 * src1->nb[2] + i13 * src1->nb[3]);
            int64_t k = 0;
            for (; k + 32 <= K; k += 32) {
                _mm512_storeu_si512((__m512i *) (dst + k),
                                    (__m512i) _mm512_cvtne2ps_pbh(_mm512_loadu_ps(src + k + 16), _mm512_loadu_ps(src + k)));
            }
            for (; k < K; k++) {
                dst[k] = (uint16_t) ((__m128i) _mm_cvtneps_pbh(_mm_set1_ps(src[k])))[0];
            }
        }
        rows_begin(params);
        ggml_barrier(params->threadpool);
        static const int64_t bpass = env_int("GGML_KURN_BF16_PASS", 64);
        for_rows(params, N, balign, std::max(balign, bpass / balign * balign), 1, [&](int64_t r0, int64_t r1, int64_t) {
            if (r0 < r1) {
                amx_bf16_gemm(data[0], bpack, balign, bkalign, A, M, (float *) op->data, K, N, r0, r1);
            }
        });
    }
#endif

    // native-layout weight, decode-sized M: src1 quantized to the type's vec_dot_type (Q8_K), then
    // kurn-native-vfy.cpp's kernels (one unpack per super-block for all columns; the same per-column
    // arithmetic at every M), or ggml's vec_dot per row and column with GGML_KURN_NATIVE_VFY=0
    void mul_mat_native(ggml_compute_params * params, ggml_tensor * op) {
        const ggml_tensor * src0 = op->src[0];
        const ggml_tensor * src1 = op->src[1];
        const int64_t K = src0->ne[0], N = src0->ne[1], M = ggml_nrows(src1);
        const ggml_type_traits_cpu * tt = ggml_get_type_traits_cpu(ntype);
        const ggml_type vdt = tt->vec_dot_type;
        const size_t xrow = ggml_row_size(vdt, K), wrow = ggml_row_size(ntype, K);
        char * wq = align64(params->wdata);
        const int64_t ne11 = src1->ne[1], ne12 = src1->ne[2];
        for (int64_t c = M * params->ith / params->nth; c < M * (params->ith + 1) / params->nth; c++) {
            const int64_t i11 = c % ne11, i12 = (c / ne11) % ne12, i13 = c / (ne11 * ne12);
            const float * src = (const float *) ((const char *) src1->data + i11 * src1->nb[1] + i12 * src1->nb[2] + i13 * src1->nb[3]);
            ggml_get_type_traits_cpu(vdt)->from_float(src, wq + c * xrow, K);
        }
        const bool steal = steal_enabled() && M <= KURN_VFY_MAX;
        steal_sched ss(params, rpass / ralign);
        rows_begin(params, steal ? &ss : nullptr, N, ralign);
        ggml_barrier(params->threadpool);
        float * Y = (float *) op->data;
        const char * W = (const char *) data[0];
        static_assert((int) KURN_NATIVE_Q6_K == (int) GGML_TYPE_Q6_K && (int) KURN_NATIVE_Q5_K == (int) GGML_TYPE_Q5_K, "kurn-native-vfy.h type ids");
        static const bool own = env_int("GGML_KURN_NATIVE_VFY", 1) != 0;  // 0: ggml's vec_dot per row and column
        for_rows(params, N, ralign, rpass, 1, [&](int64_t r0, int64_t r1, int64_t) {
            if (own && kurn_native_vfy(ntype, W, wrow, wq, xrow, Y, K, N, M, r0, r1)) {
                return;
            }
            for (int64_t n = r0; n < r1; n++) {
                for (int64_t m = 0; m < M; m++) {
                    tt->vec_dot((int) K, Y + m * N + n, 0, W + n * wrow, 0, wq + m * xrow, 0, 1);
                }
            }
        }, steal ? &ss : nullptr);
    }

    void mul_mat(ggml_compute_params * params, ggml_tensor * op) {
        const ggml_tensor * src0 = op->src[0];
        const ggml_tensor * src1 = op->src[1];
        const int64_t K = src0->ne[0], N = src0->ne[1], M = ggml_nrows(src1);
#if KURN_HAVE_AMX
        if (use_bf16(M)) {
            mul_mat_bf16(params, op);
            return;
        }
#endif
        if (!k) {
            mul_mat_native(params, op);
            return;
        }
        const ggml_type vdt = (ggml_type) k->vec_dot_type;
        const size_t xrow = ggml_row_size(vdt, K);
        char * wq = align64(params->wdata);
        float * dx = (float *) align64(wq + xrow * M);
        int32_t * nc = (int32_t *) (dx + M * (K / 32));
        static const bool use_xprep = env_int("GGML_KURN_XPREP", 1) != 0;
        void * xw = k->xprep && use_xprep && !tbm ? align64((char *) dx + amx_bytes(M, K)) : nullptr;
        const bool prof = g_prof.on && M <= KURN_VFY_MAX && params->ith < kurn_profile::MAXT;
        const uint64_t t0 = prof ? __rdtsc() : 0;
        quantize_rows(params, src1, ggml_get_type_traits_cpu(vdt)->from_float, wq, xrow, xw);
        const xprep_ref xr = { xw, M, 0 };
        const bool steal = steal_enabled() && M <= KURN_VFY_MAX;
        steal_sched ss(params, k->row_pass / k->row_align);
        rows_begin(params, steal ? &ss : nullptr, N, k->row_align);
        const uint64_t t1 = prof ? __rdtsc() : 0;
        ggml_barrier(params->threadpool);
        const uint64_t t2 = prof ? __rdtsc() : 0;
        if (use_amx(M) || use_amx_vfy(M)) {
            prep(params, wq, xrow, M, K, dx, nc);
            ggml_barrier(params->threadpool);
        }
        for_rows(params, N, k->row_align, k->row_pass, 1, [&](int64_t r0, int64_t r1, int64_t) {
            run(0, wq, xrow, (float *) op->data, K, N, M, r0, r1, xr, dx, nc);
        }, steal ? &ss : nullptr);
        if (prof) {
            const uint64_t t3 = __rdtsc();
            kurn_profile::slot & sl = g_prof.t[params->ith];
            sl.quant += t1 - t0; sl.wait += t2 - t1; sl.kern += t3 - t2; sl.ops++; sl.done = t3;
            ggml_barrier(params->threadpool);  // profiling only: read every thread's finish time
            if (params->ith == 0) {
                uint64_t lo = UINT64_MAX, hi = 0;
                for (int i = 0; i < params->nth && i < kurn_profile::MAXT; i++) {
                    lo = std::min(lo, g_prof.t[i].done);
                    hi = std::max(hi, g_prof.t[i].done);
                }
                g_prof.spread += hi - lo;
                g_prof.nops++;
            }
        }
    }

    void mul_mat_id(ggml_compute_params * params, ggml_tensor * op) {
        const ggml_tensor * src0 = op->src[0];
        const ggml_tensor * src1 = op->src[1];
        const ggml_tensor * ids = op->src[2];
        const int64_t K = src0->ne[0], N = src0->ne[1], E = src0->ne[2], nk = K / 32;
        const int64_t ne11 = src1->ne[1], ne12 = src1->ne[2];
        const int64_t n_ids = ids->ne[0], nsel = n_ids * ids->ne[1];
        const ggml_type vdt = (ggml_type) k->vec_dot_type;
        const size_t xrow = ggml_row_size(vdt, K);
        char * wq = align64(params->wdata);
        char * gx = align64(wq + xrow * ne11 * ne12);
        float * tmp = (float *) align64(gx + xrow * nsel);
        int64_t * cnt = (int64_t *) align64(tmp + nsel * N);
        int64_t * off = cnt + E;
        int32_t * sel = (int32_t *) align64(off + E + 1);  // sorted by expert: t * n_ids + i
        int64_t * act = (int64_t *) align64(sel + nsel);    // experts with tokens, act[E] = count
        int64_t * ord = act + E + 1;                        // streamed weights: compute order of act (hits first)
        float * dx = (float *) align64(ord + E);
        int32_t * nc = (int32_t *) (dx + nsel * nk);
        void * xw = k->xprep && !tbm ? align64((char *) dx + amx_bytes(nsel, K)) : nullptr;
        const int ith = params->ith, nth = params->nth;
#if KURN_XS
        kxs::tensor * const xst = xs;
#else
        void * const xst = nullptr;
#endif

        quantize_rows(params, src1, ggml_get_type_traits_cpu(vdt)->from_float, wq, xrow, nullptr);
        if (ith == 0) {
            std::fill(cnt, cnt + E, 0);
            for (int64_t t = 0; t < ids->ne[1]; t++) {
                for (int64_t i = 0; i < n_ids; i++) {
                    const int32_t e = *(const int32_t *) ((const char *) ids->data + i * ids->nb[0] + t * ids->nb[1]);
                    GGML_ASSERT(e >= 0 && e < E);
                    cnt[e]++;
                }
            }
            off[0] = 0;
            for (int64_t e = 0; e < E; e++) {
                off[e + 1] = off[e] + cnt[e];
            }
            act[E] = 0;
            for (int64_t e = 0; e < E; e++) {
                if (cnt[e] > 0) {
                    act[act[E]++] = e;
                }
            }
#if KURN_XS
            if (xst) {
                kxs::op_begin(xst, act, act[E], ids->ne[1], ord);
            }
#endif
            rows_begin(params);
            std::vector<int64_t> pos(off, off + E);
            for (int64_t t = 0; t < ids->ne[1]; t++) {
                for (int64_t i = 0; i < n_ids; i++) {
                    const int32_t e = *(const int32_t *) ((const char *) ids->data + i * ids->nb[0] + t * ids->nb[1]);
                    sel[pos[e]++] = (int32_t) (t * n_ids + i);
                }
            }
        }
        ggml_barrier(params->threadpool);
        for (int64_t j = ith; j < nsel; j += nth) {
            const int64_t t = sel[j] / n_ids, i = sel[j] % n_ids;
            memcpy(gx + j * xrow, wq + (t * ne11 + i % ne11) * xrow, xrow);
            if (xw) {
                k->xprep(gx, K, nsel, j, j + 1, 0, nk, xw);
            }
        }
        ggml_barrier(params->threadpool);
        const int64_t max_cnt = *std::max_element(cnt, cnt + E);
        if (use_amx(max_cnt) || (amx && max_cnt > KURN_VFY_MAX)) {
            prep(params, gx, xrow, nsel, K, dx, nc);
            ggml_barrier(params->threadpool);
        }
        auto item_rows = [&](int64_t r0, int64_t r1, int64_t item) {
            const int64_t e = xst ? ord[item] : act[item];
            float * Y = tmp + off[e] * N;
            if (cnt[e] == 1) {  // write straight into dst
                const int64_t t = sel[off[e]] / n_ids, i = sel[off[e]] % n_ids;
                Y = (float *) ((char *) op->data + i * op->nb[1] + t * op->nb[2]);
            }
            const xprep_ref xr = { xw, nsel, off[e] };
#if KURN_XS
            if (xst) {
                const void * pk;
                const uint8_t * w;
                kxs::wait(xst, item, &pk, &w);
                run_pk(pk, w, gx + off[e] * xrow, xrow, Y, K, N, cnt[e], r0, r1, xr, dx + off[e] * nk, nc + off[e] * nk);
                return;
            }
#endif
            run((int) e, gx + off[e] * xrow, xrow, Y, K, N, cnt[e], r0, r1, xr, dx + off[e] * nk, nc + off[e] * nk);
        };
#if KURN_XS
        if (xst) {
            // streamed experts: work items strictly in compute order on ggml's counter (a pending miss only gets a
            // slab when an earlier expert finishes; the NUMA split would interleave the items)
            const row_chunks rc(N, k->row_align, k->row_pass, nth);
            const int64_t nchunk = rc.n * act[E];
            for (int64_t c = ith; c < nchunk; c = nchunk > nth ? ggml_threadpool_chunk_add(params->threadpool, 1) : nchunk) {
                int64_t r0, r1;
                rc.range(N, c % rc.n, r0, r1);
                item_rows(r0, r1, c / rc.n);
                kxs::item_done(xst, c / rc.n, rc.n);
            }
        } else
#endif
        for_rows(params, N, k->row_align, k->row_pass, act[E], item_rows);
        ggml_barrier(params->threadpool);
#if KURN_XS
        if (xst && ith == 0) {
            kxs::op_end(xst);
        }
#endif
        for (int64_t j = ith; j < nsel; j += nth) {
            const int64_t t = sel[j] / n_ids, i = sel[j] % n_ids;
            const int64_t e = std::upper_bound(off, off + E + 1, j) - off - 1;
            if (cnt[e] > 1) {
                memcpy((char *) op->data + i * op->nb[1] + t * op->nb[2], tmp + j * N, sizeof(float) * N);
            }
        }
    }
};

// ------------------------------------------------------------------------------------------
// Fused MUL_MAT groups: G weights applied to the same activations (ffn gate + up, attention q + k + v) as one
// op. The activations are quantized / converted once, and one row schedule (row_chunks over the G weights'
// rows laid end to end) feeds every thread, so the group pays one tail and one set of barriers instead of G.
// With a SwiGLU node (gate, up -> silu(gate) * up), each chunk computes the same rows of gate and up into
// thread-local tiles and writes only the SwiGLU result, with ggml's own ggml_vec_swiglu_f32: gate and up never
// reach memory and the GLU op disappears. Every row is computed by the same kernel and arithmetic as the
// unfused op (chunks are whole records, SwiGLU spans multiples of 16), so outputs are bit-identical to the
// unfused graph, in exact mode too. GGML_KURN_FUSE=0 turns fusion off, GGML_KURN_FUSE_QKV=0 only the groups without
// a SwiGLU. llama.cpp's build_qkv emits q, k and v as consecutive nodes when wq is in the KURN buffer (apply.sh).
ggml_backend_buffer_type_t kurn_buft();

bool fuse_enabled() {
    static const bool v = env_int("GGML_KURN_FUSE", 1) != 0;
    return v;
}

// groups without a SwiGLU (attention q + k + v); GGML_KURN_FUSE_QKV=0 keeps only the gate + up + SwiGLU fusion
bool fuse_groups_enabled() {
    static const bool v = env_int("GGML_KURN_FUSE_QKV", 1) != 0;
    return v;
}

enum kurn_path { KP_K, KP_BF16, KP_NATIVE };

// traits of a MUL_MAT that the KURN buffer would compute (mirrors supports_op), else nullptr
tensor_traits * kurn_traits_of(const ggml_tensor * op) {
    if (!op || op->op != GGML_OP_MUL_MAT) {
        return nullptr;
    }
    const ggml_tensor * w = op->src[0];
    const ggml_tensor * x = op->src[1];
    if (!w->buffer || w->buffer->buft != kurn_buft() || !w->extra || !ggml_is_contiguous(w) || w->ne[2] != 1 ||
        x->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32 || !ggml_is_contiguous(op) || !ggml_is_contiguous(x)) {
        return nullptr;
    }
    if (x->buffer && !ggml_backend_buft_is_host(x->buffer->buft)) {
        return nullptr;
    }
    return (tensor_traits *) w->extra;
}

kurn_path path_of(const tensor_traits * t, int64_t M) {
    if (t->use_bf16(M)) {
        return KP_BF16;
    }
    return t->k ? KP_K : KP_NATIVE;
}

// same activation prep and the same row granularity for every member
bool group_compatible(const tensor_traits * a, const tensor_traits * b, int64_t M) {
    const kurn_path p = path_of(a, M);
    if (path_of(b, M) != p) {
        return false;
    }
    switch (p) {
        case KP_K: return a->k == b->k;
#if KURN_HAVE_AMX
        case KP_BF16: return a->bpack == b->bpack && a->balign == b->balign && a->bkalign == b->bkalign;
#else
        case KP_BF16: return false;
#endif
        case KP_NATIVE: return a->ntype == b->ntype;
    }
    return false;
}

bool fused_group(ggml_compute_params * params, ggml_tensor ** ops, int G, ggml_tensor * glu) {
    constexpr int GMAX = 4;
    if (!fuse_enabled() || G < 2 || G > GMAX || (!glu && !fuse_groups_enabled())) {
        return false;
    }
    // NUMA placement shards every member's rows per node; the fused schedule runs over the members' rows end to end
    // and would read other nodes' shards, so groups run unfused (node-local, through for_rows) on multi-node runs
    if (numa_nodes(params->nth) > 1) {
        return false;
    }
    tensor_traits * tt[GMAX];
    const ggml_tensor * src1 = ops[0]->src[1];
    const int64_t K = ops[0]->src[0]->ne[0], M = ggml_nrows(src1);
    for (int g = 0; g < G; g++) {
        tt[g] = kurn_traits_of(ops[g]);
        if (!tt[g] || ops[g]->src[1] != src1 || ops[g]->src[0]->ne[0] != K || !group_compatible(tt[0], tt[g], M)) {
            return false;
        }
    }
    if (glu) {
        if (G != 2 || ops[0]->src[0]->ne[1] != ops[1]->src[0]->ne[1] || !ggml_is_contiguous(glu) || glu->type != GGML_TYPE_F32 ||
            ggml_nrows(glu) != M || glu->ne[0] != ops[0]->src[0]->ne[1]) {
            return false;
        }
    }
    const tensor_traits & t0 = *tt[0];
    const kurn_path path = path_of(&t0, M);
    const int ith = params->ith, nth = params->nth;
    const int64_t ne11 = src1->ne[1], ne12 = src1->ne[2];

    // activation prep, as in mul_mat / mul_mat_bf16 / mul_mat_native
    char * wq = tensor_traits::align64(params->wdata);
    size_t xrow = 0;
    float * dx = nullptr;
    int32_t * nc = nullptr;
    void * xw = nullptr;
    const uint16_t * A = nullptr;
    int64_t align = t0.ralign, pass = t0.rpass;
    if (path == KP_BF16) {
#if KURN_HAVE_AMX
        const int64_t Mp = (M + 31) / 32 * 32;
        uint16_t * Aw = (uint16_t *) wq;
        for (int64_t c = Mp * ith / nth; c < Mp * (ith + 1) / nth; c++) {
            uint16_t * dst = Aw + c * K;
            if (c >= M) {
                memset(dst, 0, (size_t) K * 2);
                continue;
            }
            const int64_t i11 = c % ne11, i12 = (c / ne11) % ne12, i13 = c / (ne11 * ne12);
            const float * src = (const float *) ((const char *) src1->data + i11 * src1->nb[1] + i12 * src1->nb[2] + i13 * src1->nb[3]);
            int64_t k = 0;
            for (; k + 32 <= K; k += 32) {
                _mm512_storeu_si512((__m512i *) (dst + k),
                                    (__m512i) _mm512_cvtne2ps_pbh(_mm512_loadu_ps(src + k + 16), _mm512_loadu_ps(src + k)));
            }
            for (; k < K; k++) {
                dst[k] = (uint16_t) ((__m128i) _mm_cvtneps_pbh(_mm_set1_ps(src[k])))[0];
            }
        }
        A = Aw;
        static const int64_t bpass = env_int("GGML_KURN_BF16_PASS", 64);
        align = t0.balign;
        pass = std::max(t0.balign, bpass / t0.balign * t0.balign);
#endif
    } else if (path == KP_K) {
        const ggml_type vdt = (ggml_type) t0.k->vec_dot_type;
        xrow = ggml_row_size(vdt, K);
        dx = (float *) tensor_traits::align64(wq + xrow * M);
        nc = (int32_t *) (dx + M * (K / 32));
        static const bool use_xprep = env_int("GGML_KURN_XPREP", 1) != 0;
        xw = t0.k->xprep && use_xprep ? tensor_traits::align64((char *) dx + t0.amx_bytes(M, K)) : nullptr;
        t0.quantize_rows(params, src1, ggml_get_type_traits_cpu(vdt)->from_float, wq, xrow, xw);
        align = t0.k->row_align;
        pass = t0.k->row_pass;
    } else {
        const ggml_type vdt = ggml_get_type_traits_cpu(t0.ntype)->vec_dot_type;
        xrow = ggml_row_size(vdt, K);
        for (int64_t c = M * ith / nth; c < M * (ith + 1) / nth; c++) {
            const int64_t i11 = c % ne11, i12 = (c / ne11) % ne12, i13 = c / (ne11 * ne12);
            const float * src = (const float *) ((const char *) src1->data + i11 * src1->nb[1] + i12 * src1->nb[2] + i13 * src1->nb[3]);
            ggml_get_type_traits_cpu(vdt)->from_float(src, wq + c * xrow, K);
        }
    }
    int64_t base[GMAX + 1];
    base[0] = 0;
    for (int g = 0; g < G; g++) {
        base[g + 1] = base[g] + (glu && g > 0 ? 0 : ops[g]->src[0]->ne[1]);
    }
    const int64_t V = base[G];
    const bool steal = steal_enabled() && path != KP_BF16 && M <= KURN_VFY_MAX;
    steal_sched ss(params, pass / align);
    if (steal) {
        ss.init(ith, (V + align - 1) / align);
    }
    if (ith == 0) {
        ggml_threadpool_chunk_set(params->threadpool, nth);
    }
    ggml_barrier(params->threadpool);
    if (path == KP_K && (t0.use_amx(M) || t0.use_amx_vfy(M))) {
        t0.prep(params, wq, xrow, M, K, dx, nc);
        ggml_barrier(params->threadpool);
    }
    const tensor_traits::xprep_ref xr = { xw, M, 0 };

    // rows [r0, r1) of member g into Y (column stride ld; Y + m * ld + n for n in [r0, r1))
    auto rows = [&](int g, float * Y, int64_t ld, int64_t r0, int64_t r1) {
        const tensor_traits & t = *tt[g];
        switch (path) {
            case KP_BF16:
#if KURN_HAVE_AMX
                amx_bf16_gemm(t.data[0], t.bpack, t.balign, t.bkalign, A, M, Y, K, ld, r0, r1);
#endif
                break;
            case KP_K:
                t.run(0, wq, xrow, Y, K, ld, M, r0, r1, xr, dx, nc);
                break;
            case KP_NATIVE: {
                const size_t wrow = ggml_row_size(t.ntype, K);
                const char * W = (const char *) t.data[0];
                static const bool own = env_int("GGML_KURN_NATIVE_VFY", 1) != 0;
                if (own && kurn_native_vfy(t.ntype, W, wrow, wq, xrow, Y, K, ld, M, r0, r1)) {
                    break;
                }
                const ggml_type_traits_cpu * tc = ggml_get_type_traits_cpu(t.ntype);
                for (int64_t n = r0; n < r1; n++) {
                    for (int64_t m = 0; m < M; m++) {
                        tc->vec_dot((int) K, Y + m * ld + n, 0, W + n * wrow, 0, wq + m * xrow, 0, 1);
                    }
                }
                break;
            }
        }
    };

    const row_chunks rc(V, align, pass, nth);
    // SwiGLU: gate and up rows of a chunk go through thread-local tiles of at most ~256 KB each
    const int64_t sub = M <= KURN_VFY_MAX ? V : std::max(pass, (65536 / M) / pass * pass);
    static thread_local std::vector<float> tbuf;
    if (glu) {
        const size_t need = (size_t) 2 * M * (size_t) std::min(sub, V) + 32;
        if (tbuf.size() < need) {
            tbuf.resize(need);
        }
    }
    for (int64_t c = ith;;) {
        int64_t v0, v1;
        if (steal) {
            int64_t u0, u1;
            if (!ss.next(ith, u0, u1)) {
                break;
            }
            v0 = u0 * align;
            v1 = std::min(V, u1 * align);
        } else {
            if (c >= rc.n) {
                break;
            }
            rc.range(V, c, v0, v1);
            c = rc.n > nth ? ggml_threadpool_chunk_add(params->threadpool, 1) : rc.n;
        }
        if (glu) {
            float * Yo = (float *) glu->data;
            const int64_t N = glu->ne[0];
            for (int64_t s0 = v0; s0 < v1; s0 += sub) {
                const int64_t s1 = std::min(v1, s0 + sub), ld = s1 - s0;
                float * tg = tbuf.data();
                float * tu = tg + M * ld;
                rows(0, tg - s0, ld, s0, s1);
                rows(1, tu - s0, ld, s0, s1);
                for (int64_t m = 0; m < M; m++) {
                    ggml_vec_swiglu_f32((int) ld, Yo + m * N + s0, tg + m * ld, tu + m * ld);
                }
            }
            continue;
        }
        for (int g = 0; g < G; g++) {
            const int64_t a = std::max(v0, base[g]), b = std::min(v1, base[g + 1]);
            if (a < b) {
                rows(g, (float *) ops[g]->data, ops[g]->src[0]->ne[1], a - base[g], b - base[g]);
            }
        }
    }
    return true;
}

// ------------------------------------------------------------------------------------------
struct buffer_ctx {
    ggml_backend_buffer_t inner;  // CPU buffer over data
    void * data = nullptr;
    size_t size = 0;
    std::vector<std::unique_ptr<tensor_traits>> traits;
    std::mutex mu;
};

void buf_free(ggml_backend_buffer_t b) {
    auto * c = (buffer_ctx *) b->context;
    ggml_backend_buffer_free(c->inner);
    ggml_kurn_mem_free(c->data, c->size);
    delete c;
}

void * buf_base(ggml_backend_buffer_t b) {
    return ggml_backend_buffer_get_base(((buffer_ctx *) b->context)->inner);
}

enum ggml_status buf_init_tensor(ggml_backend_buffer_t b, ggml_tensor * t) {
    auto * c = (buffer_ctx *) b->context;
    t->extra = nullptr;
    if (t->view_src != nullptr) {
        return GGML_STATUS_SUCCESS;
    }
    const kurn_kernel * k = kernel_for(t);
    if (!k) {
#if KURN_HAVE_AMX
        if (native_for(t)) {
            auto tt = std::make_unique<tensor_traits>();
            tt->ntype = t->type;
            tt->stride = ggml_nbytes(t);
            tt->nrows = t->ne[1];
            tt->balign = 16;
            if (!exact_matmuls()) {
                tt->bpack = t->type == GGML_TYPE_Q6_K ? pack_bf16_native<deq_block_q6_K, sizeof(block_q6_K_n)>
                                                      : pack_bf16_native<deq_block_q5_K, sizeof(block_q5_K_n)>;
            }
            tt->data.push_back((uint8_t *) t->data);
            numa_place((uint8_t *) t->data, ggml_nbytes(t), t->ne[1], [&](int64_t r) { return (size_t) r * t->nb[1]; });
            if (verbose()) {
                GGML_LOG_INFO("kurn: %s [%lld x %lld] %s native layout (%s)\n", t->name, (long long) t->ne[0],
                              (long long) t->ne[1], ggml_type_name(t->type),
                              tt->bpack ? "AMX-BF16 prefill, native kernels below" : "native kernels at every width (exact mode)");
            }
            t->extra = tt.get();
            std::lock_guard<std::mutex> lock(c->mu);
            c->traits.push_back(std::move(tt));
        }
#endif
        return GGML_STATUS_SUCCESS;
    }
    auto tt = std::make_unique<tensor_traits>();
    tt->k = k;
    tt->stride = expert_bytes(k, t);
    tt->nrows = t->ne[1];
    tt->ralign = k->row_align;
    tt->rpass = k->row_pass;
#if KURN_HAVE_AMX
    tt->amx = amx_ok && k->type == GGML_TYPE_Q8_0 && strstr(k->config, "layout=i16") && strstr(k->config, "correction=act") &&
              amx_mm_enabled();
    if (amx_ok && amx_mm_enabled() && !exact_matmuls() && t->ne[2] == 1) {
        if (k->type == GGML_TYPE_Q4_K && strstr(k->config, "layout=i16") && k->row_align == 32) {
            tt->bpack = pack_bf16_q4_K;
            tt->balign = 32;
        }
        if (tt->amx && env_int("GGML_KURN_Q8_BF16", 1) != 0) {
            tt->bpack = pack_bf16_q8_0;
            tt->balign = 16;
            tt->bkalign = 32;
        }
    }
    if (verbose()) {
        GGML_LOG_INFO("kurn: %s %s prefill: %s\n", t->name, ggml_type_name(t->type),
                      tt->bpack ? "AMX-BF16" : tt->amx ? "AMX-INT8" : "verify kernels");
    }
#endif
#if KURN_XS
    if (kxs::tensor * xt = kxs::find(t->name)) {
        kxs::attach(xt, k->config, tt->stride, t->ne[0], t->ne[1], t->ne[2], k->view, k->pack);
        tt->xs = xt;
        t->extra = tt.get();
        std::lock_guard<std::mutex> lock(c->mu);
        c->traits.push_back(std::move(tt));
        return GGML_STATUS_SUCCESS;
    }
#endif
    for (int64_t e = 0; e < t->ne[2]; e++) {
        uint8_t * p = (uint8_t *) t->data + e * tt->stride;
        tt->data.push_back(p);
        tt->views.push_back(k->view(p, t->ne[0], t->ne[1]));
        // row r of the repacked layout starts at bytes(K, r) - bytes(K, 0) (row groups, then padding groups)
        const size_t pad = k->bytes(t->ne[0], 0);
        numa_place(p, tt->stride, t->ne[1], [&](int64_t r) { return k->bytes(t->ne[0], r) - pad; });
    }
    t->extra = tt.get();
    std::lock_guard<std::mutex> lock(c->mu);
    c->traits.push_back(std::move(tt));
    return GGML_STATUS_SUCCESS;
}

// TQ1_0 / TQ2_0: bitmap + signs when GGML_KURN_TBM says so (see tbm_source), else TQ2_0 records (TQ1_0
// converted to TQ2_0 blocks first). One layout per tensor (all experts), so MUL_MAT_ID runs one kernel family.
void set_ternary(tensor_traits * tt, ggml_tensor * t, const void * data) {
    const int64_t K = t->ne[0], N = t->ne[1], E = t->ne[2];
    const char * me = getenv("GGML_KURN_TBM");
    const int mode = me && *me ? atoi(me) : 2;
    const double zmin = 1e-3 * env_int("GGML_KURN_TBM_Z", 300);
    int64_t zeros = 0, sampled = 0;
    const int64_t step = std::max<int64_t>(1, N / 64);  // ~64 rows per expert decide auto mode
    if (mode == 2) {
        for (int64_t e = 0; e < E; e++) {
            zeros += kurn_tbm_zeros(t->type, (const char *) data + e * t->nb[2], K, N, step);
            sampled += (N + step - 1) / step * K;
        }
    }
    bool use = mode == 1 || (mode == 2 && zeros >= zmin * (double) sampled);
    size_t used = 0;
    int64_t dense = 0;
    if (use) {
        for (int64_t e = 0; e < E && use; e++) {
            int64_t de = 0;
            const size_t u = kurn_tbm_pack(t->type, (const char *) data + e * t->nb[2], K, N, tt->data[e], &de);
            use = u > 0;
            used += u;
            dense += de;
        }
    }
    tt->tbm = use;
    if (!use) {
        std::vector<uint8_t> tq2;
        for (int64_t e = 0; e < E; e++) {
            const char * src = (const char *) data + e * t->nb[2];
            if (t->type == GGML_TYPE_TQ1_0) {
                tq2.resize((size_t) (K / 256) * N * 66);
                kurn_tbm_tq1_to_tq2(src, tq2.data(), K / 256 * N);
                src = (const char *) tq2.data();
            }
            tt->k->pack(src, K, N, tt->data[e]);
        }
    }
    if (verbose()) {
        const double nat = (double) ggml_nbytes(t);
        if (use) {
            const double bpw = 8.0 * (double) used / ((double) K * N * E);
            GGML_LOG_INFO("kurn: repack %s [%lld x %lld x %lld] %s -> bitmap+signs, %.4f bpw (z ~ %.3f; %zu -> %zu bytes, %.1f%% of "
                          "native; %lld two-load records)\n", t->name, (long long) K, (long long) N, (long long) E,
                          ggml_type_name(t->type), bpw, 2.0625 - bpw, (size_t) nat, used, 100.0 * used / nat, (long long) dense);
        } else {
            GGML_LOG_INFO("kurn: repack %s [%lld x %lld x %lld] %s -> %s%s\n", t->name, (long long) K, (long long) N, (long long) E,
                          ggml_type_name(t->type), tt->k->config,
                          mode == 2 && sampled ? (" (bitmap+signs not used: sampled z below GGML_KURN_TBM_Z)") : "");
        }
    }
}

void buf_set_tensor(ggml_backend_buffer_t, ggml_tensor * t, const void * data, size_t offset, size_t size) {
    auto * tt = (tensor_traits *) t->extra;
    if (!tt || !tt->k) {
        memcpy((char *) t->data + offset, data, size);
        return;
    }
#if KURN_XS
    if (tt->xs) {
        return;  // streamed from the expert file
    }
#endif
    GGML_ASSERT(offset == 0 && size == ggml_nbytes(t));
    if (tbm_source(t)) {
        set_ternary(tt, t, data);
        return;
    }
    if (verbose()) {
        GGML_LOG_INFO("kurn: repack %s [%lld x %lld x %lld] %s -> %s (%zu -> %zu bytes)\n", t->name, (long long) t->ne[0],
                      (long long) t->ne[1], (long long) t->ne[2], ggml_type_name(t->type), tt->k->config, size,
                      tt->stride * (size_t) t->ne[2]);
    }
    for (int64_t e = 0; e < t->ne[2]; e++) {
        tt->k->pack((const char *) data + e * t->nb[2], t->ne[0], t->ne[1], tt->data[e]);
    }
}

void buf_memset_tensor(ggml_backend_buffer_t, ggml_tensor * t, uint8_t value, size_t offset, size_t size) {
    memset((char *) t->data + offset, value, size);
}

void buf_clear(ggml_backend_buffer_t b, uint8_t value) {
    ggml_backend_buffer_clear(((buffer_ctx *) b->context)->inner, value);
}

const ggml_backend_buffer_i buf_iface = {
    /* .free_buffer     = */ buf_free,
    /* .get_base        = */ buf_base,
    /* .init_tensor     = */ buf_init_tensor,
    /* .memset_tensor   = */ buf_memset_tensor,
    /* .set_tensor      = */ buf_set_tensor,
    /* .get_tensor      = */ nullptr,
    /* .set_tensor_2d   = */ nullptr,
    /* .get_tensor_2d   = */ nullptr,
    /* .cpy_tensor      = */ nullptr,
    /* .clear           = */ buf_clear,
    /* .reset           = */ nullptr,
};

const char * buft_name(ggml_backend_buffer_type_t) {
    return "KURN";
}

ggml_backend_buffer_t buft_alloc(ggml_backend_buffer_type_t buft, size_t size) {
    void * data = ggml_kurn_mem_alloc(size, GGML_KURN_MEM_WEIGHTS);
    if (!data) {
        return nullptr;
    }
    ggml_backend_buffer_t inner = ggml_backend_cpu_buffer_from_ptr(data, size);
    if (!inner) {
        ggml_kurn_mem_free(data, size);
        return nullptr;
    }
    auto * c = new buffer_ctx;
    c->inner = inner;
    c->data = data;
    c->size = size;
    return ggml_backend_buffer_init(buft, buf_iface, c, size);
}

size_t buft_alignment(ggml_backend_buffer_type_t) {
    return 64;
}

size_t buft_alloc_size(ggml_backend_buffer_type_t, const ggml_tensor * t) {
    const kurn_kernel * k = kernel_for(t);
#if KURN_XS
    if (t->view_src == nullptr && kxs::find(t->name)) {
        if (!k) {
            GGML_ABORT("kxs: %s is in the expert file but kurn has no kernel for it in this build", t->name);
        }
        return 64;  // placeholder: the experts live in kurn-xs.cpp's slabs
    }
#endif
    return k ? expert_bytes(k, t) * (size_t) t->ne[2] : ggml_nbytes(t);
}

ggml_backend_buffer_type_t kurn_buft();

class extra_buffer_type : ggml::cpu::extra_buffer_type {
    bool supports_op(ggml_backend_dev_t, const ggml_tensor * op) override {
        if (op->op != GGML_OP_MUL_MAT && op->op != GGML_OP_MUL_MAT_ID) {
            return false;
        }
        const ggml_tensor * src0 = op->src[0];
        const ggml_tensor * src1 = op->src[1];
        if (!src0->buffer || src0->buffer->buft != kurn_buft() || !ggml_is_contiguous(src0)) {
            return false;
        }
        if (!kernel_for(src0) && !(op->op == GGML_OP_MUL_MAT && native_for(src0))) {
            return false;
        }
        if (src1->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32 || !ggml_is_contiguous(op)) {
            return false;
        }
        if (src1->buffer && !ggml_backend_buft_is_host(src1->buffer->buft)) {
            return false;
        }
        if (op->op == GGML_OP_MUL_MAT) {
            return src0->ne[2] == 1;
        }
        return op->src[2]->type == GGML_TYPE_I32 && src1->ne[3] == 1 && op->nb[1] == (size_t) src0->ne[1] * sizeof(float);
    }

    ggml::cpu::tensor_traits * get_tensor_traits(const ggml_tensor * op) override {
        if ((op->op == GGML_OP_MUL_MAT || op->op == GGML_OP_MUL_MAT_ID) && op->src[0]->buffer &&
            op->src[0]->buffer->buft == kurn_buft()) {
            return (ggml::cpu::tensor_traits *) op->src[0]->extra;
        }
        return nullptr;
    }
};

ggml_backend_buffer_type_t kurn_buft() {
    static ggml_backend_buffer_type t = {
        /* .iface = */ {
            /* .get_name         = */ buft_name,
            /* .alloc_buffer     = */ buft_alloc,
            /* .alloc_buffer_n   = */ nullptr,
            /* .get_alignment    = */ buft_alignment,
            /* .get_max_size     = */ nullptr,
            /* .get_alloc_size   = */ buft_alloc_size,
            /* .get_alloc_size_n = */ nullptr,
            /* .is_host          = */ nullptr,
        },
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ new extra_buffer_type(),
    };
    return &t;
}

}  // namespace

bool ggml_kurn_fused_mul_mat(const struct ggml_compute_params * params, struct ggml_tensor ** ops, int n_ops,
                             struct ggml_tensor * glu) {
    if (!ggml_backend_cpu_kurn_buffer_type()) {
        return false;
    }
    return fused_group(const_cast<ggml_compute_params *>(params), ops, n_ops, glu);
}

bool ggml_kurn_ggml_amx_allowed(void) {
#if KURN_HAVE_AMX
    if (ggml_backend_cpu_kurn_buffer_type()) {
        return amx_ok;
    }
#endif
    return true;
}

int ggml_kurn_cap_threads(int n_threads) {
#if KURN_HAVE_AMX
    if (ggml_backend_cpu_kurn_buffer_type() && amx_lossy) {
        return kurn_amx_thread_cap(n_threads);
    }
#endif
    return n_threads;
}

bool ggml_kurn_xs_layout(int type, int64_t K, int64_t N, size_t * bytes, const char ** config) {
    ggml_tensor w = {};
    w.type = (ggml_type) type;
    w.ne[0] = K;
    w.ne[1] = N;
    w.ne[2] = 2;  // expert-stacked: the per-tensor fallback to ggml never applies
    w.ne[3] = 1;
    const kurn_kernel * k = kernel_for(&w);
    if (!k) {
        return false;
    }
    *bytes = expert_bytes(k, &w);
    *config = k->config;
    return true;
}

bool ggml_kurn_xs_pack(int type, int64_t K, int64_t N, const void * src, void * dst) {
    size_t bytes;
    const char * config;
    if (!ggml_kurn_xs_layout(type, K, N, &bytes, &config)) {
        return false;
    }
    find_kernel((ggml_type) type)->pack(src, K, N, dst);
    return true;
}

ggml_backend_buffer_type_t ggml_backend_cpu_kurn_buffer_type(void) {
    static const bool enabled = [] {
        if (env_int("GGML_KURN", 1) == 0) {
            return false;
        }
#if KURN_HAVE_AMX
        // ARCH_REQ_XCOMP_PERM for XTILEDATA (same request as ggml's AMX buffer), then the AMX self-test
        const kurn_amx_state st = kurn_amx_check();
        amx_ok = st != KURN_AMX_UNAVAILABLE && env_int("GGML_KURN_AMX", 1) != 0;
        amx_lossy = amx_ok && st == KURN_AMX_GUARDED;
        kurn_tbm_amx_set_lossy(amx_lossy);
#endif
        return true;
    }();
    return enabled ? kurn_buft() : nullptr;
}

#else

ggml_backend_buffer_type_t ggml_backend_cpu_kurn_buffer_type(void) {
    return nullptr;
}

int ggml_kurn_cap_threads(int n_threads) {
    return n_threads;
}

bool ggml_kurn_ggml_amx_allowed(void) {
    return true;
}

bool ggml_kurn_fused_mul_mat(const struct ggml_compute_params *, struct ggml_tensor **, int, struct ggml_tensor *) {
    return false;
}

bool ggml_kurn_xs_layout(int, int64_t, int64_t, size_t *, const char **) {
    return false;
}

bool ggml_kurn_xs_pack(int, int64_t, int64_t, const void *, void *) {
    return false;
}

#endif
