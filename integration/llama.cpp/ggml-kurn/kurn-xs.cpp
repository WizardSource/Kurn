// kurn expert streaming (KXS): run MoE models whose routed experts do not fit in RAM.
//
// Layout. llama-kurn-xs-convert writes every 3D expert weight (`*_exps.weight`) of a GGUF in
// kurn's repacked layout, one record per expert, 4 KiB aligned, so a read lands ready to use: no
// per-load repack. kurn's records end in all-zero padding row groups; the file stores only the
// bytes up to the last nonzero 4 KiB block (for Q4_0 that is exactly the GGUF's own bytes) and
// the slab keeps the zero tail from allocation. Everything else (attention, router, shared/dense
// FFN, embeddings, head) loads as usual and stays resident.
//
// Cache. Each streamed tensor owns a pool of slabs (all tensors of a layer get the same slot
// count; the RAM target is split evenly over the MoE layers). Replacement is LFU with a recency
// tie-break (GGML_KXS_POLICY=lru for plain LRU). Slabs in use by the running op are pinned.
//
// Reads. A small pool of I/O threads does O_DIRECT pread of up to GGML_KXS_IO_CHUNK_KB per
// request (bounded parallelism: GGML_KXS_IO_THREADS). Demand reads always go first; speculative
// reads are dispatched only when no demand read is queued and at most GGML_KXS_SPEC_INFLIGHT of
// them are in flight, and queued ones are dropped as soon as their layer is planned.
//
// Overlap. A plan node (ggml_kurn_xs_plan_op) sits right after each MoE layer's top-k and is
// placed ahead of the layer's shared expert in the graph. It pins cached experts and issues the
// misses' reads for every streamed tensor of the layer, then returns, so the reads proceed while
// the shared expert and the expert matmuls of cached experts run. MUL_MAT_ID computes cached
// experts first and waits per expert for the rest; each expert's rows are computed by the same
// kernel calls as in RAM, so the output is bit-identical to the all-in-RAM run whatever the order.
// When an op needs more experts than its pool holds (prefill, tiny budgets) slots are recycled
// as experts finish (a read/compute pipeline as deep as the pool).
//
// Prediction. With GGML_KXS_PREFETCH=1 the plan node of layer L applies layer L+1's router to
// layer L's router input (Pre-gated MoE / ProMoE style), with L+1's per-channel router scale in
// place of L's (GGML_KXS_PRED_MODE=raw: as is), and queues speculative reads of the top
// GGML_KXS_PREFETCH_K experts. GGML_KXS_MEASURE=1 only records how well that (and two simple
// baselines) predicts the next layer's experts during decode.
//
// GGML_KXS_FROM_GGUF=1 (ablation): read each expert's GGUF bytes instead and repack it after the read,
// i.e. streaming without the pre-repacked file.
#include "kurn-xs.h"

#if KURN_XS

#include "ggml-impl.h"
#include "gguf.h"

#include <fcntl.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <immintrin.h>
#include <x86intrin.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace kxs {

// ---------------------------------------------------------------- file format (llama-kurn-xs-convert)
struct file_header {
    char     magic[8];      // "KURNXS01"
    uint32_t version;       // 1
    uint32_t n_tensors;
    uint64_t data_offset;   // first record, 4 KiB aligned
    uint64_t align;         // 4096
    uint64_t gguf_size;     // size of the source GGUF (sanity check)
    char     source[232];
};
struct file_tensor {
    char     name[128];
    char     config[384];   // kurn kernel config of the layout
    int32_t  type, layer;
    int64_t  K, N, E;
    uint64_t expert_bytes;  // kurn's repacked bytes per expert
    uint64_t data_bytes;    // bytes stored per expert (4 KiB multiple); the rest of the record is zero
    uint64_t offset;        // file offset of expert 0
    uint64_t stride;        // file bytes per expert
};
static_assert(sizeof(file_header) == 272, "file_header");
static_assert(sizeof(file_tensor) == 576, "file_tensor");

static int env_int(const char * name, int def) {
    const char * v = getenv(name);
    return v && *v ? atoi(v) : def;
}

static uint64_t now_ns() {
    return (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch()).count();
}

enum { S_EMPTY = 0, S_LOADING = 1, S_READY = 2 };

struct slot {
    uint8_t * data = nullptr;
    void *    view = nullptr;
    int32_t   expert = -1;
    uint32_t  freq = 0;
    uint64_t  last = 0;
    int32_t   pins = 0;
    int32_t   parts_left = 0, parts_queued = 0, parts_inflight = 0;
    bool      spec = false;         // read issued speculatively and not demanded since
    bool      spec_loaded = false;  // contents came from a speculative read ...
    bool      used = false;         // ... and have (not) been used since
    std::atomic<int32_t> state{S_EMPTY};
};

struct tensor {
    std::string name;
    int     layer = -1, type = -1;
    int64_t K = 0, N = 0, E = 0;
    size_t  expert_bytes = 0, data_bytes = 0, slab_stride = 0;
    uint64_t file_off = 0, file_stride = 0;
    std::string config;
    bool    attached = false;
    void (*pack)(const void *, int64_t, int64_t, void *) = nullptr;
    uint64_t raw_off = 0, raw_bytes = 0;  // GGML_KXS_FROM_GGUF: expert 0's bytes in the GGUF, bytes per expert

    std::unique_ptr<slot[]> slots;
    int32_t n_slots = 0;
    uint8_t * pool = nullptr;
    std::vector<int32_t> slot_of;  // expert -> slot (-1: not resident)

    // the running (or planned) op
    bool planned = false;
    std::vector<int32_t> plan_set;  // ascending expert ids
    std::vector<int64_t> order;
    std::unique_ptr<std::atomic<int32_t>[]> op_slot;
    std::unique_ptr<std::atomic<int64_t>[]> op_done;
    size_t pend_next = 0;
    bool counts = true;  // this op's uses count toward LFU frequency (decode-sized ops only)
};

struct request {
    tensor * t;
    int32_t  s;
    uint64_t off;
    uint8_t * dst;
    size_t   len;
    bool     spec;  // issued speculatively (stays set when promoted to a demand read)
};

struct layer_group {
    int layer = -1;
    std::vector<tensor *> ts;
    int next = -1;                  // index of the next MoE layer group
    // predictor: this layer's router (bf16, [E][n_embd]) and per-channel router scale
    std::vector<uint16_t> router;
    std::vector<float> scale;
    int64_t n_embd = 0, E = 0;
    // measurement
    std::vector<int32_t> pred_ratio, pred_raw;  // top-16 predicted for this layer by the previous layer
    std::vector<int32_t> prev_tok;              // this layer's experts at the previous decode token
    std::vector<int32_t> last_set;              // this layer's experts at the current token
    bool pred_valid = false;
};

struct counters {
    uint64_t uses = 0, hits = 0, hits_inflight = 0, misses = 0, evictions = 0;
    uint64_t spec_issued = 0, spec_useful = 0, spec_wasted = 0, spec_cancelled = 0;
    uint64_t bytes_demand = 0, bytes_spec = 0, reads = 0, io_ns = 0;  // completed reads, by how they were issued
    uint64_t plan_ops = 0, ops = 0;
    // decode-only prediction recall (sum of |P ∩ X| and of |X|)
    uint64_t pred_n = 0, pred_x = 0, pr_ratio8 = 0, pr_ratio12 = 0, pr_ratio16 = 0, pr_raw8 = 0, pr_raw16 = 0, pr_prevtok = 0,
             pr_prevlayer = 0, prev_n = 0, prev_x = 0;
};

struct state {
    bool open = false;
    int fd = -1, gfd = -1;
    bool direct = true, from_gguf = false;
    size_t raw_max = 0;
    std::vector<std::unique_ptr<tensor>> tensors;
    std::unordered_map<std::string, tensor *> by_name;
    std::vector<layer_group> groups;
    std::map<int, int> group_of_layer;
    size_t cache_bytes = 0, pred_bytes = 0;

    // knobs
    int    io_threads = 4;
    size_t io_chunk = 1 << 20;
    bool   overlap = true, prefetch = false, measure = false, lru = false, pred_raw = false;
    int    prefetch_k = 4, spec_cap = 4;
    int64_t freq_max_tokens = 16;  // ops with more tokens (prefill) do not count toward LFU frequency
    uint32_t spec_max_freq = UINT32_MAX;

    std::mutex mu;
    std::condition_variable cv_io;
    std::deque<request> qd, qs;
    int spec_inflight = 0;
    bool stop = false;
    std::vector<std::thread> workers;
    uint64_t tick = 0;

    counters c;
    std::atomic<uint64_t> stall_cycles{0};
    double tsc_ghz = 0;

    ~state() {
        {
            std::lock_guard<std::mutex> lk(mu);
            stop = true;
        }
        cv_io.notify_all();
        for (auto & w : workers) {
            w.join();
        }
        if (fd >= 0) {
            close(fd);
        }
        if (gfd >= 0) {
            close(gfd);
        }
    }
};

static state & S() {
    static state s;
    return s;
}

// ---------------------------------------------------------------- I/O
static void io_worker() {
    state & st = S();
    uint8_t * stage = nullptr;
    if (st.from_gguf && posix_memalign((void **) &stage, 4096, st.raw_max) != 0) {
        GGML_ABORT("kxs: out of memory");
    }
    std::unique_ptr<uint8_t, decltype(&free)> stage_guard(stage, free);
    std::unique_lock<std::mutex> lk(st.mu);
    for (;;) {
        st.cv_io.wait(lk, [&] {
            return st.stop || !st.qd.empty() || (!st.qs.empty() && st.spec_inflight < st.spec_cap);
        });
        if (st.stop) {
            return;
        }
        request r;
        bool spec = false;
        if (!st.qd.empty()) {
            r = st.qd.front();
            st.qd.pop_front();
        } else {
            r = st.qs.front();
            st.qs.pop_front();
            spec = true;
            st.spec_inflight++;
        }
        slot & sl = r.t->slots[r.s];
        sl.parts_queued--;
        sl.parts_inflight++;
        lk.unlock();

        const uint64_t t0 = now_ns();
        int fd = st.fd;
        uint8_t * dst = r.dst;
        uint64_t off = r.off;
        size_t len = r.len;
        if (st.from_gguf) {
            // the expert's GGUF bytes, widened to 4 KiB for O_DIRECT, then repacked into the slab
            fd = st.gfd;
            off = r.off & ~(uint64_t) 4095;
            len = (r.off + r.len - off + 4095) & ~(uint64_t) 4095;
            dst = stage;
        }
        size_t done = 0;
        while (done < len) {
            const ssize_t n = pread(fd, dst + done, len - done, (off_t) (off + done));
            if (n <= 0) {
                if (n < 0 && errno == EINTR) {
                    continue;
                }
                if (n == 0 && st.from_gguf && done >= r.off + r.len - off) {
                    break;  // end of the GGUF inside the widened read
                }
                GGML_ABORT("kxs: read of %s failed at offset %llu: %s", r.t->name.c_str(),
                           (unsigned long long) (off + done), n < 0 ? strerror(errno) : "short file");
            }
            done += (size_t) n;
        }
        if (st.from_gguf) {
            r.t->pack(stage + (r.off - off), r.t->K, r.t->N, r.dst);
        }
        const uint64_t dt = now_ns() - t0;

        lk.lock();
        sl.parts_inflight--;
        if (spec) {
            st.spec_inflight--;
        }
        st.c.reads++;
        st.c.io_ns += dt;
        (r.spec ? st.c.bytes_spec : st.c.bytes_demand) += r.len;
        if (--sl.parts_left == 0) {
            sl.state.store(S_READY, std::memory_order_release);
        }
        if (spec || !st.qs.empty()) {
            st.cv_io.notify_one();
        }
    }
}

// queue the reads of slot s (expert already assigned)
static void enqueue(tensor * t, int32_t s, bool spec) {
    state & st = S();
    slot & sl = t->slots[s];
    if (st.from_gguf) {
        (spec ? st.qs : st.qd).push_back({ t, s, t->raw_off + (uint64_t) sl.expert * t->raw_bytes, sl.data, t->raw_bytes, spec });
        sl.parts_left = sl.parts_queued = 1;
        sl.state.store(S_LOADING, std::memory_order_relaxed);
        return;
    }
    const uint64_t base = t->file_off + (uint64_t) sl.expert * t->file_stride;
    int parts = 0;
    for (size_t o = 0; o < t->data_bytes; o += st.io_chunk) {
        const size_t len = std::min(st.io_chunk, t->data_bytes - o);
        (spec ? st.qs : st.qd).push_back({ t, s, base + o, sl.data + o, len, spec });
        parts++;
    }
    sl.parts_left = parts;
    sl.parts_queued = parts;
    sl.state.store(S_LOADING, std::memory_order_relaxed);
}

// move a speculative slot's queued parts to the demand queue
static void promote(tensor * t, int32_t s) {
    state & st = S();
    slot & sl = t->slots[s];
    sl.spec = false;
    if (sl.parts_queued == 0) {
        return;
    }
    for (auto it = st.qs.begin(); it != st.qs.end();) {
        if (it->t == t && it->s == s) {
            st.qd.push_back(*it);
            it = st.qs.erase(it);
        } else {
            ++it;
        }
    }
}

// drop queued speculative reads whose slot has no part in flight yet
static void cancel_spec() {
    state & st = S();
    std::vector<std::pair<tensor *, int32_t>> drop;
    for (const request & r : st.qs) {
        slot & sl = r.t->slots[r.s];
        if (sl.spec && sl.parts_inflight == 0 && sl.parts_queued == sl.parts_left) {
            if (drop.empty() || drop.back() != std::make_pair(r.t, r.s)) {
                drop.emplace_back(r.t, r.s);
            }
        }
    }
    if (drop.empty()) {
        return;
    }
    for (auto it = st.qs.begin(); it != st.qs.end();) {
        slot & sl = it->t->slots[it->s];
        if (sl.spec && sl.parts_inflight == 0 && sl.parts_queued == sl.parts_left) {
            it = st.qs.erase(it);
        } else {
            ++it;
        }
    }
    for (auto & d : drop) {
        slot & sl = d.first->slots[d.second];
        if (sl.expert < 0 || sl.state.load(std::memory_order_relaxed) != S_LOADING) {
            continue;
        }
        d.first->slot_of[sl.expert] = -1;
        sl.expert = -1;
        sl.parts_left = sl.parts_queued = 0;
        sl.spec = sl.spec_loaded = false;
        sl.state.store(S_EMPTY, std::memory_order_relaxed);
        st.c.spec_cancelled++;
    }
}

// ---------------------------------------------------------------- cache
static int32_t victim(tensor * t, bool spec) {
    state & st = S();
    int32_t best = -1;
    for (int32_t s = 0; s < t->n_slots; s++) {
        slot & sl = t->slots[s];
        if (sl.pins > 0 || sl.state.load(std::memory_order_relaxed) == S_LOADING) {
            continue;
        }
        if (sl.expert < 0) {
            return s;
        }
        if (spec && sl.freq > st.spec_max_freq) {
            continue;
        }
        if (best < 0) {
            best = s;
            continue;
        }
        const slot & b = t->slots[best];
        const bool better = st.lru ? sl.last < b.last : (sl.freq < b.freq || (sl.freq == b.freq && sl.last < b.last));
        if (better) {
            best = s;
        }
    }
    return best;
}

static void assign(tensor * t, int32_t s, int32_t e, bool spec) {
    state & st = S();
    slot & sl = t->slots[s];
    if (sl.expert >= 0) {
        if (sl.spec_loaded && !sl.used) {
            st.c.spec_wasted++;
        }
        t->slot_of[sl.expert] = -1;
        st.c.evictions++;
    }
    sl.expert = e;
    t->slot_of[e] = s;
    sl.freq = spec || !t->counts ? 0 : 1;
    sl.last = ++st.tick;
    sl.spec = spec;
    sl.spec_loaded = spec;
    sl.used = !spec;
    if (spec) {
        st.c.spec_issued++;
    }
    enqueue(t, s, spec);
}

// pin the experts of `set` (ascending) for t's next op and issue reads for the misses
static void plan(tensor * t, const std::vector<int32_t> & set, int64_t n_tok) {
    state & st = S();
    const size_t n = set.size();
    // prefill touches most experts; counting it would freeze the decode cache on whatever prefill loaded last
    t->counts = n_tok <= st.freq_max_tokens;
    std::vector<int64_t> hits, infl, miss;
    for (int32_t e : set) {
        const int32_t s = t->slot_of[e];
        st.c.uses++;
        if (s < 0) {
            miss.push_back(e);
            st.c.misses++;
            continue;
        }
        slot & sl = t->slots[s];
        sl.pins++;
        sl.freq += t->counts;
        sl.last = ++st.tick;
        if (sl.spec_loaded && !sl.used) {
            st.c.spec_useful++;
        }
        sl.used = true;
        if (sl.state.load(std::memory_order_relaxed) == S_READY) {
            hits.push_back(e);
            st.c.hits++;
        } else {
            if (sl.spec) {
                promote(t, s);
            }
            infl.push_back(e);
            st.c.hits_inflight++;
        }
    }
    t->order.clear();
    t->order.insert(t->order.end(), hits.begin(), hits.end());
    t->order.insert(t->order.end(), infl.begin(), infl.end());
    t->order.insert(t->order.end(), miss.begin(), miss.end());
    for (size_t i = 0; i < n; i++) {
        const int32_t s = i < hits.size() + infl.size() ? t->slot_of[t->order[i]] : -1;
        t->op_slot[i].store(s, std::memory_order_relaxed);
        t->op_done[i].store(0, std::memory_order_relaxed);
    }
    t->pend_next = hits.size() + infl.size();
    while (t->pend_next < n) {
        const int32_t v = victim(t, false);
        if (v < 0) {
            break;
        }
        assign(t, v, (int32_t) t->order[t->pend_next], false);
        t->slots[v].pins++;
        t->op_slot[t->pend_next].store(v, std::memory_order_release);
        t->pend_next++;
    }
    t->plan_set = set;
    t->planned = true;
    std::atomic_thread_fence(std::memory_order_release);
}

static void unplan(tensor * t) {
    for (size_t i = 0; i < t->plan_set.size(); i++) {
        const int32_t s = t->op_slot[i].load(std::memory_order_relaxed);
        if (s >= 0) {
            t->slots[s].pins--;
        }
    }
    t->planned = false;
}

static void speculate(tensor * t, const int32_t * pred, int k) {
    for (int i = 0; i < k; i++) {
        const int32_t e = pred[i];
        if (t->slot_of[e] >= 0) {
            continue;
        }
        const int32_t v = victim(t, true);
        if (v < 0) {
            return;
        }
        assign(t, v, e, true);
    }
}

// ---------------------------------------------------------------- predictor
static inline float dot_bf16(const uint16_t * w, const float * x, int64_t n) {
    __m512 acc = _mm512_setzero_ps();
    int64_t i = 0;
    for (; i + 16 <= n; i += 16) {
        const __m512i wi = _mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i *) (w + i))), 16);
        acc = _mm512_fmadd_ps(_mm512_castsi512_ps(wi), _mm512_loadu_ps(x + i), acc);
    }
    float s = _mm512_reduce_add_ps(acc);
    for (; i < n; i++) {
        uint32_t b = (uint32_t) w[i] << 16;
        float f;
        memcpy(&f, &b, 4);
        s += f * x[i];
    }
    return s;
}

static void topk(const float * v, int64_t n, int k, std::vector<int32_t> & out) {
    std::vector<int32_t> idx(n);
    for (int64_t i = 0; i < n; i++) {
        idx[i] = (int32_t) i;
    }
    k = (int) std::min<int64_t>(k, n);
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), [&](int32_t a, int32_t b) {
        return v[a] > v[b] || (v[a] == v[b] && a < b);
    });
    out.assign(idx.begin(), idx.begin() + k);
}

// predicted experts of group g_next from the router input x of group g (one token)
static void predict(const layer_group & g, const layer_group & gn, const float * x, bool raw, std::vector<int32_t> & out, int k) {
    const int64_t n = gn.n_embd;
    std::vector<float> xin(n), logits(gn.E);
    for (int64_t i = 0; i < n; i++) {
        const float r = raw || g.scale.empty() || gn.scale.empty() || g.scale[i] == 0.0f ? 1.0f : gn.scale[i] / g.scale[i];
        xin[i] = x[i] * r;
    }
    for (int64_t e = 0; e < gn.E; e++) {
        logits[e] = dot_bf16(gn.router.data() + e * n, xin.data(), n);
    }
    topk(logits.data(), gn.E, k, out);
}

static uint64_t overlap_count(const std::vector<int32_t> & p, size_t k, const std::vector<int32_t> & x) {
    uint64_t c = 0;
    for (size_t i = 0; i < std::min(k, p.size()); i++) {
        c += std::find(x.begin(), x.end(), p[i]) != x.end();
    }
    return c;
}

static bool load_predictor(const char * gguf_path) {
    state & st = S();
    ggml_context * meta = nullptr;
    gguf_init_params ip = { true, &meta };
    gguf_context * g = gguf_init_from_file(gguf_path, ip);
    if (!g) {
        GGML_LOG_WARN("kxs: cannot read %s for the router predictor\n", gguf_path);
        return false;
    }
    const int fd = open(gguf_path, O_RDONLY);
    const size_t data_off = gguf_get_data_offset(g);
    auto read_f32 = [&](const char * name, std::vector<float> & out, int64_t & ne0, int64_t & ne1) -> bool {
        const int64_t id = gguf_find_tensor(g, name);
        ggml_tensor * t = meta ? ggml_get_tensor(meta, name) : nullptr;
        if (id < 0 || !t) {
            return false;
        }
        const size_t nb = ggml_nbytes(t);
        std::vector<uint8_t> raw(nb);
        if (pread(fd, raw.data(), nb, (off_t) (data_off + gguf_get_tensor_offset(g, id))) != (ssize_t) nb) {
            return false;
        }
        ne0 = t->ne[0];
        ne1 = t->ne[1];
        out.resize(ggml_nelements(t));
        if (t->type == GGML_TYPE_F32) {
            memcpy(out.data(), raw.data(), nb);
        } else {
            const ggml_type_traits * tt = ggml_get_type_traits(t->type);
            if (!tt->to_float) {
                return false;
            }
            tt->to_float(raw.data(), out.data(), ggml_nelements(t));
        }
        return true;
    };
    bool ok = fd >= 0;
    for (layer_group & lg : st.groups) {
        if (!ok) {
            break;
        }
        char name[128];
        std::vector<float> w;
        int64_t ne0 = 0, ne1 = 0;
        snprintf(name, sizeof(name), "blk.%d.ffn_gate_inp.weight", lg.layer);
        if (!read_f32(name, w, ne0, ne1)) {
            ok = false;
            break;
        }
        lg.n_embd = ne0;
        lg.E = ne1;
        lg.router.resize(w.size());
        for (size_t i = 0; i < w.size(); i++) {
            uint32_t b;
            memcpy(&b, &w[i], 4);
            lg.router[i] = (uint16_t) ((b + 0x7fff + ((b >> 16) & 1)) >> 16);
        }
        // per-channel router input scale: Gemma 4's ffn_gate_inp.scale, else the FFN norm (Qwen3-MoE style)
        snprintf(name, sizeof(name), "blk.%d.ffn_gate_inp.scale", lg.layer);
        if (!read_f32(name, lg.scale, ne0, ne1)) {
            snprintf(name, sizeof(name), "blk.%d.ffn_norm.weight", lg.layer);
            if (!read_f32(name, lg.scale, ne0, ne1)) {
                lg.scale.clear();
            }
        }
        st.pred_bytes += lg.router.size() * 2 + lg.scale.size() * 4;
    }
    if (fd >= 0) {
        close(fd);
    }
    gguf_free(g);
    if (meta) {
        ggml_free(meta);
    }
    if (!ok) {
        GGML_LOG_WARN("kxs: router weights not found; prediction disabled\n");
        for (layer_group & lg : st.groups) {
            lg.router.clear();
        }
        st.pred_bytes = 0;
    }
    return ok;
}

// ---------------------------------------------------------------- open
static bool open_file(const char * path, const char * gguf_path, size_t cache_bytes) {
    state & st = S();
    st.direct = env_int("GGML_KXS_DIRECT", 1) != 0;
    st.fd = open(path, O_RDONLY | (st.direct ? O_DIRECT : 0));
    if (st.fd < 0 && st.direct) {
        st.direct = false;
        st.fd = open(path, O_RDONLY);
    }
    if (st.fd < 0) {
        GGML_LOG_ERROR("kxs: cannot open %s: %s\n", path, strerror(errno));
        return false;
    }
    void * hbuf = nullptr;
    if (posix_memalign(&hbuf, 4096, 4096) != 0) {
        return false;
    }
    std::unique_ptr<void, decltype(&free)> hguard(hbuf, free);
    if (pread(st.fd, hbuf, 4096, 0) != 4096) {
        GGML_LOG_ERROR("kxs: %s: short header\n", path);
        return false;
    }
    file_header h;
    memcpy(&h, hbuf, sizeof(h));
    if (memcmp(h.magic, "KURNXS01", 8) != 0 || h.version != 1 || h.align != 4096) {
        GGML_LOG_ERROR("kxs: %s is not a kurn expert file (version 1)\n", path);
        return false;
    }
    struct stat sb;
    if (gguf_path && stat(gguf_path, &sb) == 0 && (uint64_t) sb.st_size != h.gguf_size) {
        GGML_LOG_ERROR("kxs: %s was converted from a different file (%llu bytes, %s is %llu)\n", path,
                       (unsigned long long) h.gguf_size, gguf_path, (unsigned long long) sb.st_size);
        return false;
    }
    const size_t tab = (sizeof(file_header) + (size_t) h.n_tensors * sizeof(file_tensor) + 4095) / 4096 * 4096;
    void * tbuf = nullptr;
    if (posix_memalign(&tbuf, 4096, tab) != 0) {
        return false;
    }
    std::unique_ptr<void, decltype(&free)> tguard(tbuf, free);
    if (pread(st.fd, tbuf, tab, 0) != (ssize_t) tab) {
        return false;
    }
    for (uint32_t i = 0; i < h.n_tensors; i++) {
        file_tensor ft;
        memcpy(&ft, (const char *) tbuf + sizeof(file_header) + i * sizeof(file_tensor), sizeof(ft));
        auto t = std::make_unique<tensor>();
        t->name = std::string(ft.name, strnlen(ft.name, sizeof(ft.name)));
        t->config = std::string(ft.config, strnlen(ft.config, sizeof(ft.config)));
        t->type = ft.type;
        t->layer = ft.layer;
        t->K = ft.K;
        t->N = ft.N;
        t->E = ft.E;
        t->expert_bytes = ft.expert_bytes;
        t->data_bytes = ft.data_bytes;
        t->slab_stride = (ft.expert_bytes + 4095) / 4096 * 4096;
        t->file_off = ft.offset;
        t->file_stride = ft.stride;
        if (t->data_bytes % 4096 || t->file_off % 4096 || t->file_stride % 4096 || t->data_bytes > t->slab_stride) {
            GGML_LOG_ERROR("kxs: %s: bad record geometry for %s\n", path, t->name.c_str());
            return false;
        }
        t->slot_of.assign((size_t) t->E, -1);
        t->op_slot.reset(new std::atomic<int32_t>[t->E]);
        t->op_done.reset(new std::atomic<int64_t>[t->E]);
        st.by_name[t->name] = t.get();
        if (!st.group_of_layer.count(t->layer)) {
            st.group_of_layer[t->layer] = 0;
        }
        st.tensors.push_back(std::move(t));
    }
    int gi = 0;
    for (auto & kv : st.group_of_layer) {
        kv.second = gi++;
    }
    st.groups.resize(st.group_of_layer.size());
    for (auto & kv : st.group_of_layer) {
        st.groups[kv.second].layer = kv.first;
        st.groups[kv.second].next = kv.second + 1 < (int) st.groups.size() ? kv.second + 1 : -1;
    }
    for (auto & t : st.tensors) {
        st.groups[st.group_of_layer[t->layer]].ts.push_back(t.get());
    }

    // slot counts: the cache budget split evenly over the MoE layers
    const int min_slots = std::max(1, env_int("GGML_KXS_MIN_SLOTS", 1));
    const size_t per_layer = cache_bytes / std::max<size_t>(1, st.groups.size());
    size_t total = 0;
    int smin = INT32_MAX, smax = 0;
    for (layer_group & lg : st.groups) {
        size_t bytes = 0;
        int64_t E = INT64_MAX;
        for (tensor * t : lg.ts) {
            bytes += t->slab_stride;
            E = std::min(E, t->E);
        }
        int n = (int) std::min<int64_t>(E, std::max<int64_t>(min_slots, (int64_t) (per_layer / std::max<size_t>(1, bytes))));
        smin = std::min(smin, n);
        smax = std::max(smax, n);
        for (tensor * t : lg.ts) {
            t->n_slots = n;
            const size_t sz = (size_t) n * t->slab_stride;
            void * p = mmap(nullptr, sz, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE, -1, 0);
            if (p == MAP_FAILED) {
                GGML_LOG_ERROR("kxs: cannot allocate %zu MiB of slabs\n", sz >> 20);
                return false;
            }
            t->pool = (uint8_t *) p;
            t->slots.reset(new slot[n]);
            for (int s = 0; s < n; s++) {
                t->slots[s].data = t->pool + (size_t) s * t->slab_stride;
            }
            total += sz;
        }
    }
    st.cache_bytes = total;

    st.io_threads = std::max(1, env_int("GGML_KXS_IO_THREADS", 4));
    st.io_chunk = (size_t) std::max(4, env_int("GGML_KXS_IO_CHUNK_KB", 1024)) / 4 * 4 * 1024;
    st.overlap = env_int("GGML_KXS_OVERLAP", 1) != 0;
    st.prefetch = env_int("GGML_KXS_PREFETCH", 0) != 0;
    st.measure = env_int("GGML_KXS_MEASURE", 0) != 0;
    st.prefetch_k = std::max(1, env_int("GGML_KXS_PREFETCH_K", 4));
    st.spec_cap = std::max(1, env_int("GGML_KXS_SPEC_INFLIGHT", 4));
    st.freq_max_tokens = env_int("GGML_KXS_FREQ_MAX_TOKENS", 16);
    st.spec_max_freq = (uint32_t) env_int("GGML_KXS_SPEC_MAX_FREQ", -1);
    const char * pol = getenv("GGML_KXS_POLICY");
    st.lru = pol && strcmp(pol, "lru") == 0;
    const char * pm = getenv("GGML_KXS_PRED_MODE");
    st.pred_raw = pm && strcmp(pm, "raw") == 0;
    if ((st.prefetch || st.measure) && gguf_path) {
        load_predictor(gguf_path);
    }
    st.from_gguf = env_int("GGML_KXS_FROM_GGUF", 0) != 0 && gguf_path;
    if (st.from_gguf) {
        gguf_init_params ip = { true, nullptr };
        gguf_context * g = gguf_init_from_file(gguf_path, ip);
        st.gfd = open(gguf_path, O_RDONLY | (st.direct ? O_DIRECT : 0));
        if (!g || st.gfd < 0) {
            GGML_LOG_ERROR("kxs: GGML_KXS_FROM_GGUF: cannot read %s\n", gguf_path);
            return false;
        }
        for (auto & t : st.tensors) {
            const int64_t id = gguf_find_tensor(g, t->name.c_str());
            if (id < 0) {
                GGML_LOG_ERROR("kxs: %s not in %s\n", t->name.c_str(), gguf_path);
                return false;
            }
            t->raw_off = gguf_get_data_offset(g) + gguf_get_tensor_offset(g, id);
            t->raw_bytes = ggml_row_size((ggml_type) t->type, t->K) * (size_t) t->N;
            st.raw_max = std::max<size_t>(st.raw_max, (t->raw_bytes + 2 * 4096 + 4095) & ~(size_t) 4095);
        }
        gguf_free(g);
    }
    const uint64_t c0 = __rdtsc(), n0 = now_ns();
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    st.tsc_ghz = (double) (__rdtsc() - c0) / (double) (now_ns() - n0);

    for (int i = 0; i < st.io_threads; i++) {
        st.workers.emplace_back(io_worker);
    }
    st.open = true;
    GGML_LOG_INFO("kxs: %s: %zu streamed tensors in %zu MoE layers, %d-%d slots per tensor, %.1f MiB of slabs, "
                  "%d I/O threads, %s%s, overlap %d, prefetch %d (k=%d), predictor %.1f MiB\n",
                  path, st.tensors.size(), st.groups.size(), smin, smax, total / 1048576.0, st.io_threads,
                  st.direct ? "O_DIRECT" : "buffered", st.from_gguf ? ", from the GGUF with repack" : "", st.overlap, st.prefetch,
                  st.prefetch_k, st.pred_bytes / 1048576.0);
    return true;
}

tensor * find(const char * name) {
    state & st = S();
    if (!st.open || !name) {
        return nullptr;
    }
    auto it = st.by_name.find(name);
    return it == st.by_name.end() ? nullptr : it->second;
}

void attach(tensor * t, const char * config, size_t expert_bytes, int64_t K, int64_t N, int64_t E,
            void * (*view)(void *, int64_t, int64_t), void (*pack)(const void *, int64_t, int64_t, void *)) {
    t->pack = pack;
    if (t->config != config || t->expert_bytes != expert_bytes || t->K != K || t->N != N || t->E != E) {
        GGML_ABORT("kxs: %s: the expert file's layout (%s, %zu bytes) does not match this build's kernel (%s, %zu bytes); "
                   "re-run llama-kurn-xs-convert", t->name.c_str(), t->config.c_str(), t->expert_bytes, config, expert_bytes);
    }
    if (!t->attached) {
        for (int s = 0; s < t->n_slots; s++) {
            t->slots[s].view = view(t->slots[s].data, K, N);
        }
        t->attached = true;
    }
}

// ---------------------------------------------------------------- op
void op_begin(tensor * t, const int64_t * act, int64_t n_act, int64_t n_tok, int64_t * order) {
    state & st = S();
    std::vector<int32_t> set(act, act + n_act);
    std::sort(set.begin(), set.end());
    {
        std::lock_guard<std::mutex> lk(st.mu);
        if (t->planned && t->plan_set != set) {
            unplan(t);
        }
        if (!t->planned) {
            plan(t, set, n_tok);
        }
        st.c.ops++;
        std::copy(t->order.begin(), t->order.end(), order);
    }
    st.cv_io.notify_all();
    if (!st.overlap) {
        // no overlap: every expert the pool can hold is read before any is computed
        const uint64_t c0 = __rdtsc();
        for (size_t i = 0; i < std::min(t->pend_next, set.size()); i++) {
            const int32_t s = t->op_slot[i].load(std::memory_order_acquire);
            while (t->slots[s].state.load(std::memory_order_acquire) != S_READY) {
                _mm_pause();
            }
        }
        st.stall_cycles.fetch_add(__rdtsc() - c0, std::memory_order_relaxed);
    }
}

// give free slots to pending misses (a slot can also free up when a speculative read finishes)
static bool assign_pending(tensor * t) {
    state & st = S();
    bool issued = false;
    {
        std::lock_guard<std::mutex> lk(st.mu);
        while (t->planned && t->pend_next < t->plan_set.size()) {
            const int32_t v = victim(t, false);
            if (v < 0) {
                break;
            }
            assign(t, v, (int32_t) t->order[t->pend_next], false);
            t->slots[v].pins++;
            t->op_slot[t->pend_next].store(v, std::memory_order_release);
            t->pend_next++;
            issued = true;
        }
    }
    if (issued) {
        st.cv_io.notify_all();
    }
    return issued;
}

void wait(tensor * t, int64_t i, const void ** view, const uint8_t ** data) {
    int32_t s = t->op_slot[i].load(std::memory_order_acquire);
    if (s >= 0 && t->slots[s].state.load(std::memory_order_acquire) == S_READY) {
        *view = t->slots[s].view;
        *data = t->slots[s].data;
        return;
    }
    const uint64_t c0 = __rdtsc();
    for (int it = 0;; it++) {
        s = t->op_slot[i].load(std::memory_order_acquire);
        if (s >= 0 && t->slots[s].state.load(std::memory_order_acquire) == S_READY) {
            break;
        }
        if (it < 4096) {
            _mm_pause();
        } else {
            if (s < 0 && it % 64 == 0) {
                assign_pending(t);
            }
            sched_yield();
        }
    }
    S().stall_cycles.fetch_add(__rdtsc() - c0, std::memory_order_relaxed);
    *view = t->slots[s].view;
    *data = t->slots[s].data;
}

void item_done(tensor * t, int64_t i, int64_t n_items) {
    if (t->op_done[i].fetch_add(1, std::memory_order_acq_rel) + 1 != n_items) {
        return;
    }
    state & st = S();
    bool issued = false;
    {
        std::lock_guard<std::mutex> lk(st.mu);
        const int32_t s = t->op_slot[i].load(std::memory_order_relaxed);
        t->slots[s].pins--;
        if (t->pend_next < t->plan_set.size()) {
            const int32_t v = victim(t, false);
            if (v >= 0) {
                assign(t, v, (int32_t) t->order[t->pend_next], false);
                t->slots[v].pins++;
                t->op_slot[t->pend_next].store(v, std::memory_order_release);
                t->pend_next++;
                issued = true;
            }
        }
    }
    if (issued) {
        st.cv_io.notify_all();
    }
}

void op_end(tensor * t) {
    state & st = S();
    std::lock_guard<std::mutex> lk(st.mu);
    GGML_ASSERT(t->pend_next == t->plan_set.size());
    t->planned = false;
}

}  // namespace kxs

using namespace kxs;

bool ggml_kurn_xs_open(const char * kxs_path, const char * gguf_path, size_t cache_bytes) {
    if (S().open) {
        GGML_LOG_ERROR("kxs: an expert file is already open\n");
        return false;
    }
    return open_file(kxs_path, gguf_path, cache_bytes);
}

bool ggml_kurn_xs_is_streamed(const char * name) {
    return find(name) != nullptr;
}

void ggml_kurn_xs_plan_op(struct ggml_tensor * dst, int ith, int nth, void * userdata) {
    GGML_UNUSED(nth);
    if (ith != 0) {
        return;
    }
    state & st = S();
    const ggml_tensor * w = (const ggml_tensor *) userdata;
    tensor * t0 = find(ggml_get_name(w));
    if (!t0) {
        return;
    }
    layer_group & g = st.groups[st.group_of_layer[t0->layer]];
    const ggml_tensor * ids = dst->src[0];
    const ggml_tensor * rin = dst->src[1];
    const int64_t n_used = ids->ne[0], n_tok = ids->ne[1];
    std::vector<uint8_t> mark((size_t) t0->E, 0);
    for (int64_t tk = 0; tk < n_tok; tk++) {
        for (int64_t i = 0; i < n_used; i++) {
            const int32_t e = *(const int32_t *) ((const char *) ids->data + i * ids->nb[0] + tk * ids->nb[1]);
            if (e >= 0 && e < t0->E) {
                mark[e] = 1;
            }
        }
    }
    std::vector<int32_t> set;
    for (int32_t e = 0; e < (int32_t) t0->E; e++) {
        if (mark[e]) {
            set.push_back(e);
        }
    }
    const bool decode = n_tok == 1;
    {
        std::lock_guard<std::mutex> lk(st.mu);
        st.c.plan_ops++;
        // speculative reads of experts this layer needs become demand reads; the rest are dropped
        for (tensor * t : g.ts) {
            for (int32_t e : set) {
                const int32_t s = t->slot_of[e];
                if (s >= 0 && t->slots[s].spec) {
                    promote(t, s);
                }
            }
        }
        cancel_spec();
        if (st.overlap) {
            for (tensor * t : g.ts) {
                if (t->planned) {
                    unplan(t);
                }
                plan(t, set, n_tok);
            }
        }
        if (st.measure && decode) {
            if (g.pred_valid) {
                st.c.pred_n++;
                st.c.pred_x += set.size();
                st.c.pr_ratio8 += overlap_count(g.pred_ratio, 8, set);
                st.c.pr_ratio12 += overlap_count(g.pred_ratio, 12, set);
                st.c.pr_ratio16 += overlap_count(g.pred_ratio, 16, set);
                st.c.pr_raw8 += overlap_count(g.pred_raw, 8, set);
                st.c.pr_raw16 += overlap_count(g.pred_raw, 16, set);
            }
            if (!g.prev_tok.empty()) {
                st.c.prev_n++;
                st.c.prev_x += set.size();
                st.c.pr_prevtok += overlap_count(g.prev_tok, g.prev_tok.size(), set);
                // the previous MoE layer's experts (same token) as the guess for this layer
                const int gi = st.group_of_layer[t0->layer];
                if (gi > 0 && !st.groups[gi - 1].last_set.empty()) {
                    st.c.pr_prevlayer += overlap_count(st.groups[gi - 1].last_set, st.groups[gi - 1].last_set.size(), set);
                }
            }
            g.prev_tok = set;
            g.last_set = set;
        }
        g.pred_valid = false;
    }
    st.cv_io.notify_all();

    if (!decode || g.next < 0 || !rin || rin->type != GGML_TYPE_F32 || g.router.empty()) {
        return;
    }
    layer_group & gn = st.groups[g.next];
    if (gn.router.empty() || rin->ne[0] != gn.n_embd) {
        return;
    }
    const float * x = (const float *) rin->data;
    if (st.measure) {
        predict(g, gn, x, false, gn.pred_ratio, 16);
        predict(g, gn, x, true, gn.pred_raw, 16);
        gn.pred_valid = true;
    }
    if (st.prefetch) {
        std::vector<int32_t> pred;
        predict(g, gn, x, st.pred_raw, pred, st.prefetch_k);
        {
            std::lock_guard<std::mutex> lk(st.mu);
            for (tensor * t : gn.ts) {
                speculate(t, pred.data(), (int) pred.size());
            }
        }
        st.cv_io.notify_all();
    }
}

size_t ggml_kurn_xs_stats(char * buf, size_t n) {
    state & st = S();
    std::lock_guard<std::mutex> lk(st.mu);
    const counters & c = st.c;
    auto ratio = [](uint64_t a, uint64_t b) { return b ? (double) a / (double) b : 0.0; };
    int smin = INT32_MAX, smax = 0;
    for (auto & t : st.tensors) {
        smin = std::min(smin, t->n_slots);
        smax = std::max(smax, t->n_slots);
    }
    const double stall_s = st.tsc_ghz > 0 ? (double) st.stall_cycles.load() / (st.tsc_ghz * 1e9) : 0.0;
    const int len = snprintf(buf, n,
        "{\"open\":%d,\"tensors\":%zu,\"layers\":%zu,\"slots_min\":%d,\"slots_max\":%d,\"cache_mib\":%.1f,\"pred_mib\":%.1f,"
        "\"io_threads\":%d,\"io_chunk_kib\":%zu,\"direct\":%d,\"overlap\":%d,\"prefetch\":%d,\"prefetch_k\":%d,\"spec_inflight\":%d,"
        "\"policy\":\"%s\",\"pred_mode\":\"%s\",\"from_gguf\":%d,"
        "\"uses\":%llu,\"hits\":%llu,\"hits_inflight\":%llu,\"misses\":%llu,\"hit_rate\":%.4f,\"evictions\":%llu,"
        "\"spec_issued\":%llu,\"spec_useful\":%llu,\"spec_wasted\":%llu,\"spec_cancelled\":%llu,"
        "\"bytes_demand\":%llu,\"bytes_spec\":%llu,\"reads\":%llu,\"io_busy_s\":%.3f,\"stall_thread_s\":%.3f,"
        "\"plan_ops\":%llu,\"ops\":%llu,"
        "\"pred_n\":%llu,\"recall_ratio8\":%.4f,\"recall_ratio12\":%.4f,\"recall_ratio16\":%.4f,\"recall_raw8\":%.4f,"
        "\"recall_raw16\":%.4f,\"recall_prevtok\":%.4f,\"recall_prevlayer\":%.4f}",
        (int) st.open, st.tensors.size(), st.groups.size(), st.tensors.empty() ? 0 : smin, smax, st.cache_bytes / 1048576.0,
        st.pred_bytes / 1048576.0, st.io_threads, st.io_chunk / 1024, (int) st.direct, (int) st.overlap, (int) st.prefetch,
        st.prefetch_k, st.spec_cap, st.lru ? "lru" : "lfu", st.pred_raw ? "raw" : "ratio", (int) st.from_gguf,
        (unsigned long long) c.uses, (unsigned long long) c.hits, (unsigned long long) c.hits_inflight,
        (unsigned long long) c.misses, ratio(c.hits + c.hits_inflight, c.uses), (unsigned long long) c.evictions,
        (unsigned long long) c.spec_issued, (unsigned long long) c.spec_useful, (unsigned long long) c.spec_wasted,
        (unsigned long long) c.spec_cancelled, (unsigned long long) c.bytes_demand, (unsigned long long) c.bytes_spec,
        (unsigned long long) c.reads, c.io_ns * 1e-9, stall_s, (unsigned long long) c.plan_ops, (unsigned long long) c.ops,
        (unsigned long long) c.pred_n, ratio(c.pr_ratio8, c.pred_x), ratio(c.pr_ratio12, c.pred_x), ratio(c.pr_ratio16, c.pred_x),
        ratio(c.pr_raw8, c.pred_x), ratio(c.pr_raw16, c.pred_x), ratio(c.pr_prevtok, c.prev_x), ratio(c.pr_prevlayer, c.prev_x));
    return len < 0 ? 0 : std::min((size_t) len, n ? n - 1 : 0);
}

void ggml_kurn_xs_stats_reset(void) {
    state & st = S();
    std::lock_guard<std::mutex> lk(st.mu);
    st.c = counters();
    st.stall_cycles = 0;
}

#else

bool ggml_kurn_xs_open(const char *, const char *, size_t) {
    return false;
}
bool ggml_kurn_xs_is_streamed(const char *) {
    return false;
}
void ggml_kurn_xs_plan_op(struct ggml_tensor *, int, int, void *) {}
size_t ggml_kurn_xs_stats(char * buf, size_t n) {
    return n ? (size_t) snprintf(buf, n, "{\"open\":0}") : 0;
}
void ggml_kurn_xs_stats_reset(void) {}

#endif
