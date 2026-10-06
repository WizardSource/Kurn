// kurn attention as ggml-cpu's FLASH_ATTN_EXT (see kurn/integration/llama.cpp/README.md).
//
// ggml_compute_forward_flash_attn_ext calls ggml_kurn_flash_attn_ext first; when it returns false
// ggml's own kernel runs. kurn takes a node when
//   - Q is F32, K and V have the same type F16, BF16 or Q8_0, with contiguous rows (V not transposed),
//   - the head dims are a generated configuration (kattn_dispatch.h: 64, 128, 256, 576/512),
//   - there is no ALiBi (max_bias), logit softcap or attention sinks,
//   - the mask (if any) is F16 and shared by all heads (mask->ne[2] == 1).
// GQA is n_head / n_head_kv; streams (ne[3]: llama.cpp's per-sequence KV streams) are run one after
// another, with ggml's dim-3 broadcast. The mask carries causality, so kurn runs with causal = 0 and
// skips KV tiles the mask hides for every row of a tile. Everything else falls back to ggml.
//
// Kernels (gen_ggml_attn.py) and modes:
//   fast (default)  engine bf16 (AVX512-BF16 tile engine), amx (AMX-BF16 tile engine) or f32, with
//                   kurn's f32 decode row engine for n_q * G <= 8 and auto KV splits (flash-decoding).
//   exact           f32 tile engine for every batch size and a single KV split: a token's output does
//                   not depend on how many tokens are computed with it, the KV length or the thread
//                   count, so speculative verification reproduces one-token decoding bit for bit (with
//                   the KURN buffer type's batch-invariant matmuls).
// Prefill-sized calls of the bf16 engines pack K/V once per call into kurn's AMX-ready layout
// (kattn_pack) and read every query tile from it (kattn_packed), instead of packing each KV tile
// again for every query tile.
//
// Environment: GGML_KURN_FA=0 (or GGML_KURN=0) turns it off; GGML_KURN_FA_MODE=fast|exact;
// GGML_KURN_FA_ENGINE=bf16|amx|f32 (default bf16, or amx with GGML_KURN_AMX=1); GGML_KURN_FA_PACK=0
// turns the per-call packing off; GGML_KURN_VERBOSE=1 logs the configuration, each new node shape
// with the kernel or the fallback reason, and call counts at exit.
#include "kurn-attn.h"

#include "ggml-cpu-impl.h"
#include "ggml-impl.h"
#include "kattn_dispatch.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>

#if defined(__linux__)
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace {

struct fa_config {
    bool on = false;
    bool exact = false;
    bool pack = true;
    bool verbose = false;
    int engine = 0;  // preferred tile engine, kurn_fa_kernel::engine
};

int env_int(const char * name, int def) {
    const char * v = getenv(name);
    return v && *v ? atoi(v) : def;
}

bool compiled(int engine) {
    for (const kurn_fa_kernel * k = kurn_fa_kernels; k->name; k++) {
        if (k->engine == engine && !k->exact) {
            return true;
        }
    }
    return false;
}

bool amx_permitted() {
#if defined(__linux__)
    // ARCH_REQ_XCOMP_PERM for XTILEDATA; fails on kernels or VMs without AMX state support
    return syscall(SYS_arch_prctl, 0x1023, 18) == 0;
#else
    return false;
#endif
}

const char * engine_name(int e) {
    return e == 2 ? "amx" : e == 1 ? "bf16" : "f32";
}

const fa_config & cfg() {
    static const fa_config c = [] {
        fa_config c;
        c.on = env_int("GGML_KURN", 1) != 0 && env_int("GGML_KURN_FA", 1) != 0;
        const char * mode = getenv("GGML_KURN_FA_MODE");
        c.exact = mode && strcmp(mode, "exact") == 0;
        c.pack = env_int("GGML_KURN_FA_PACK", 1) != 0;
        c.verbose = env_int("GGML_KURN_VERBOSE", 0) != 0;
        const char * eng = getenv("GGML_KURN_FA_ENGINE");
        // f32 by default: on Emerald Rapids its tile engine prefills faster than the AVX512-BF16 one
        // (pp512 at depth 4096: 1.81x vs 1.38x ggml's FA) and it is exact to f32 rounding
        int want = env_int("GGML_KURN_AMX", 0) ? 2 : 0;
        if (eng && *eng) {
            want = strcmp(eng, "amx") == 0 ? 2 : strcmp(eng, "bf16") == 0 ? 1 : 0;
        }
        if (want == 2 && !(compiled(2) && amx_permitted())) {
            want = 1;
        }
        if (want == 1 && !compiled(1)) {
            want = 0;
        }
        c.engine = want;
        if (c.on && !compiled(0) && !compiled(1)) {
            c.on = false;  // no kernels in this build (no AVX-512)
        }
        if (c.verbose) {
            GGML_LOG_INFO("kurn fa: %s, mode %s, engine %s, per-call packing %s\n", c.on ? "on" : "off",
                          c.exact ? "exact" : "fast", c.exact ? "f32" : engine_name(c.engine), c.pack ? "on" : "off");
        }
        return c;
    }();
    return c;
}

int kv_format(ggml_type t) {
    switch (t) {
        case GGML_TYPE_F16:  return KATTN_KV_F16;
        case GGML_TYPE_BF16: return KATTN_KV_BF16;
        case GGML_TYPE_Q8_0: return KATTN_KV_Q8_0;
        default:             return -1;
    }
}

const kurn_fa_kernel * find_kernel(int dk, int dv, int kv) {
    const fa_config & c = cfg();
    if (c.exact) {
        for (const kurn_fa_kernel * k = kurn_fa_kernels; k->name; k++) {
            if (k->exact && k->dk == dk && k->dv == dv && k->kv == kv) {
                return k;
            }
        }
        return nullptr;
    }
    for (int e = c.engine; e >= 0; e--) {
        for (const kurn_fa_kernel * k = kurn_fa_kernels; k->name; k++) {
            if (!k->exact && k->engine == e && k->dk == dk && k->dv == dv && k->kv == kv) {
                return k;
            }
        }
    }
    return nullptr;
}

// Why a node is not taken (nullptr: taken).
const char * unsupported(const ggml_tensor * dst) {
    const ggml_tensor * q = dst->src[0], * k = dst->src[1], * v = dst->src[2], * mask = dst->src[3];
    float max_bias, softcap;
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&softcap,  (const float *) dst->op_params + 2, sizeof(float));
    if (max_bias != 0.0f)      return "ALiBi";
    if (softcap != 0.0f)       return "logit softcap";
    if (dst->src[4])           return "sinks";
    if (q->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) return "Q/dst type";
    if (k->type != v->type)    return "K/V types differ";
    if (kv_format(k->type) < 0) return "KV type";
    if (q->nb[0] != sizeof(float) || q->nb[1] % sizeof(float) || q->nb[2] % sizeof(float) || q->nb[3] % sizeof(float) ||
        dst->nb[0] != sizeof(float) || dst->nb[1] % sizeof(float) || dst->nb[2] % sizeof(float)) return "Q/dst layout";
    if (k->nb[0] != ggml_type_size(k->type) || v->nb[0] != ggml_type_size(v->type)) return "K/V rows not contiguous";
    if (k->ne[2] == 0 || q->ne[2] % k->ne[2] || v->ne[2] != k->ne[2] || v->ne[1] != k->ne[1] ||
        k->ne[3] == 0 || q->ne[3] % k->ne[3] || v->ne[3] != k->ne[3]) return "head/stream shape";
    if (q->ne[1] < 1 || k->ne[1] < 1)                     return "empty";
    if (mask && (mask->type != GGML_TYPE_F16 || mask->nb[0] != 2 || mask->ne[2] != 1 || mask->ne[0] < k->ne[1] ||
                 mask->ne[1] < q->ne[1] || mask->nb[1] % 2)) return "mask layout";
    if (!find_kernel((int) k->ne[0], (int) v->ne[0], kv_format(k->type))) return "head dims";
    return nullptr;
}

struct fa_stats {
    std::atomic<long> calls{0}, fallbacks{0};
    std::mutex m;
    std::map<std::string, long> seen;
    ~fa_stats() {
        if (cfg().verbose && (calls || fallbacks)) {
            fprintf(stderr, "kurn fa: %ld FLASH_ATTN_EXT nodes on kurn, %ld on ggml\n", calls.load(), fallbacks.load());
        }
    }
};

fa_stats & stats() {
    static fa_stats s;
    return s;
}

void log_shape(const ggml_tensor * dst, const kurn_fa_kernel * kern, const char * why) {
    const ggml_tensor * q = dst->src[0], * k = dst->src[1], * v = dst->src[2];
    char key[256];
    snprintf(key, sizeof key, "%s dk %lld dv %lld heads %lld/%lld streams %lld %s n_q %s", kern ? kern->name : why,
             (long long) k->ne[0], (long long) v->ne[0], (long long) q->ne[2], (long long) k->ne[2], (long long) q->ne[3],
             ggml_type_name(k->type), q->ne[1] == 1 ? "1" : "many");
    fa_stats & s = stats();
    std::lock_guard<std::mutex> lock(s.m);
    if (s.seen[key]++ == 0) {
        GGML_LOG_INFO("kurn fa: %s: %s\n", kern ? "kurn" : "ggml", key);
    }
}

// Buffers owned by the thread that runs ith 0 of a graph (the caller of ggml_graph_compute), so
// graphs computed concurrently from different threads never share them.
struct fa_state {
    void * ws = nullptr;   // kattn workspace: zero-filled when allocated, kurn resets it after every call
    size_t ws_size = 0;
    void * kvp = nullptr;  // kattn_pack buffer
    size_t kvp_size = 0;
    ~fa_state() {
        free(ws);
        free(kvp);
    }
};

thread_local fa_state tls_state;

// Grow a zero-filled buffer to `need` bytes: every thread computes the same `need` from the same
// arguments and sees the same old size, so either all of them take the barriers or none does.
void grow(const ggml_compute_params * params, void ** buf, size_t * size, size_t need) {
    if (need <= *size) {
        return;
    }
    ggml_barrier(params->threadpool);
    if (params->ith == 0) {
        free(*buf);
        // 64-byte aligned: kattn_pack stores whole cache lines (the workspace would align itself)
        need = (need + 63) & ~(size_t) 63;
        *buf = aligned_alloc(64, need);
        GGML_ASSERT(*buf && "kurn fa: out of memory");
        memset(*buf, 0, need);
        *size = need;
    }
    ggml_barrier(params->threadpool);
}

}  // namespace

extern "C" bool ggml_kurn_flash_attn_ext(const ggml_compute_params * params, ggml_tensor * dst) {
    const fa_config & c = cfg();
    if (!c.on || params->use_ref) {
        return false;
    }
    const char * why = unsupported(dst);
    const ggml_tensor * q = dst->src[0], * k = dst->src[1], * v = dst->src[2], * mask = dst->src[3];
    const kurn_fa_kernel * kern = why ? nullptr : find_kernel((int) k->ne[0], (int) v->ne[0], kv_format(k->type));
    if (params->ith == 0) {
        (kern ? stats().calls : stats().fallbacks)++;
        if (c.verbose) {
            log_shape(dst, kern, why);
        }
    }
    if (!kern) {
        return false;
    }

    // publish ith 0's buffers to the other threads through the node's work buffer (unused by kurn)
    GGML_ASSERT(params->wsize >= sizeof(fa_state *));
    if (params->ith == 0) {
        *(fa_state **) params->wdata = &tls_state;
    }
    ggml_barrier(params->threadpool);
    fa_state * st = *(fa_state * const *) params->wdata;

    float scale;
    memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));
    const int ith = params->ith, nth = params->nth;
    const int64_t n_stream = q->ne[3], rk3 = q->ne[3] / k->ne[3];
    const int64_t G = q->ne[2] / k->ne[2];
    const bool pack = c.pack && !c.exact && kern->engine != 0 && q->ne[1] > kern->tile_q && q->ne[1] * G > kern->dec_rows;

    for (int64_t s = 0; s < n_stream; s++) {
        kattn_args a;
        memset(&a, 0, sizeof a);
        a.n_q = q->ne[1];
        a.n_kv = k->ne[1];
        a.q_pos0 = 0;
        a.n_head = (int32_t) q->ne[2];
        a.n_head_kv = (int32_t) k->ne[2];
        a.causal = 0;
        a.scale = scale;
        a.q = (const float *) ((const char *) q->data + s * q->nb[3]);
        a.q_s_tok = q->nb[1] / sizeof(float);
        a.q_s_head = q->nb[2] / sizeof(float);
        a.k = (const char *) k->data + (s / rk3) * k->nb[3];
        a.k_s_tok = k->nb[1];
        a.k_s_head = k->nb[2];
        a.v = (const char *) v->data + (s / rk3) * v->nb[3];
        a.v_s_tok = v->nb[1];
        a.v_s_head = v->nb[2];
        if (mask) {
            a.mask = (const uint16_t *) ((const char *) mask->data + (s % mask->ne[3]) * mask->nb[3]);
            a.mask_s_tok = mask->nb[1] / 2;
        }
        a.out = (float *) ((char *) dst->data + s * dst->nb[3]);
        a.o_s_head = dst->nb[1] / sizeof(float);
        a.o_s_tok = dst->nb[2] / sizeof(float);

        if (s > 0) {
            ggml_barrier(params->threadpool);  // the previous stream's kattn call must be done on every thread
        }
        grow(params, &st->ws, &st->ws_size, kern->workspace(&a, nth));
        const int64_t cap = (a.n_kv + 1023) / 1024 * 1024;
        const size_t kvp_need = pack ? kern->pack_bytes(&a, cap) : 0;
        if (kvp_need > 0) {
            grow(params, &st->kvp, &st->kvp_size, kvp_need);
            kern->pack(&a, st->kvp, cap, 0, a.n_kv, ith, nth);
            ggml_barrier(params->threadpool);
            kern->packed(&a, st->kvp, cap, st->ws, ith, nth);
        } else {
            kern->run(&a, st->ws, ith, nth);
        }
    }
    return true;
}
