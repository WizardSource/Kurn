// kurn attn measurement hook for ggml-cpu (included into ggml/src/ggml-cpu/ops.cpp).
// KURN_ATTN_LIB=a.so[:b.so...] lists kurn attn builds (`kurn attn build`); a FLASH_ATTN_EXT
// node runs on the first library whose (DK, DV, KV format) matches and falls back to ggml
// otherwise (sinks, ALiBi, softcap, mixed K/V types, multi-sequence batches, other head dims).
// KURN_ATTN_VERBOSE=1 prints the library table and one line per new node shape.
#include <dlfcn.h>
#include <mutex>
#include <string>
#include <vector>

namespace kurn_attn {

struct args {
    int64_t n_q, n_kv, q_pos0;
    int32_t n_head, n_head_kv, causal;
    float scale;
    const float * q; int64_t q_s_tok, q_s_head;
    const void * k; int64_t k_s_tok, k_s_head;
    const void * v; int64_t v_s_tok, v_s_head;
    const uint16_t * mask; int64_t mask_s_tok;
    float * out; int64_t o_s_tok, o_s_head;
};

struct lib {
    int dk, dv, kv;
    size_t (*workspace)(const args *, int);
    void (*run)(const args *, void *, int, int);
};

static std::vector<lib> libs;
static std::once_flag once;
static bool verbose;
static void * ws;
static size_t ws_size;

static void load() {
    const char * env = getenv("KURN_ATTN_LIB");
    verbose = getenv("KURN_ATTN_VERBOSE") != nullptr;
    if (!env) return;
    std::string s(env);
    size_t p = 0;
    while (p <= s.size()) {
        size_t e = s.find(':', p);
        if (e == std::string::npos) e = s.size();
        std::string path = s.substr(p, e - p);
        p = e + 1;
        if (path.empty()) continue;
        void * h = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (!h) { fprintf(stderr, "kurn attn: %s\n", dlerror()); continue; }
        auto cfg = (int64_t (*)(int *, int *, int *)) dlsym(h, "kattn_config");
        lib l;
        l.workspace = (size_t (*)(const args *, int)) dlsym(h, "kattn_workspace");
        l.run = (void (*)(const args *, void *, int, int)) dlsym(h, "kattn");
        if (!cfg || !l.workspace || !l.run) { fprintf(stderr, "kurn attn: %s lacks kattn symbols\n", path.c_str()); continue; }
        cfg(&l.dk, &l.dv, &l.kv);
        libs.push_back(l);
        if (verbose) fprintf(stderr, "kurn attn: %s dk=%d dv=%d kv=%d\n", path.c_str(), l.dk, l.dv, l.kv);
    }
}

static int kv_format(ggml_type t) {
    switch (t) {
        case GGML_TYPE_F16:  return 0;
        case GGML_TYPE_BF16: return 1;
        case GGML_TYPE_Q8_0: return 2;
        default:             return -1;
    }
}

static bool try_run(const ggml_compute_params * params, ggml_tensor * dst) {
    std::call_once(once, load);
    if (libs.empty() || params->use_ref) return false;
    const ggml_tensor * q = dst->src[0], * k = dst->src[1], * v = dst->src[2], * mask = dst->src[3];
    float scale, max_bias, softcap;
    memcpy(&scale,    (const float *) dst->op_params + 0, sizeof(float));
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&softcap,  (const float *) dst->op_params + 2, sizeof(float));
    const int fmt = kv_format(k->type);
    if (q->type != GGML_TYPE_F32 || k->type != v->type || fmt < 0 || max_bias != 0.0f || softcap != 0.0f ||
        dst->src[4] != nullptr || q->ne[3] != 1 || k->ne[3] != 1 || v->ne[3] != 1 || q->ne[2] % k->ne[2] != 0 ||
        v->ne[2] != k->ne[2] || v->ne[1] != k->ne[1] || dst->type != GGML_TYPE_F32 || dst->nb[0] != sizeof(float) ||
        q->nb[0] != sizeof(float) || q->nb[1] % sizeof(float) || q->nb[2] % sizeof(float)) {
        return false;
    }
    if (mask && (mask->type != GGML_TYPE_F16 || mask->ne[2] != 1 || mask->ne[3] != 1 || mask->ne[0] < k->ne[1] ||
                 mask->ne[1] < q->ne[1])) {
        return false;
    }
    const lib * L = nullptr;
    for (const lib & l : libs) {
        if (l.dk == k->ne[0] && l.dv == v->ne[0] && l.kv == fmt) { L = &l; break; }
    }
    if (!L) return false;

    args a;
    a.n_q = q->ne[1]; a.n_kv = k->ne[1]; a.q_pos0 = 0;
    a.n_head = (int32_t) q->ne[2]; a.n_head_kv = (int32_t) k->ne[2]; a.causal = 0;
    a.scale = scale;
    a.q = (const float *) q->data; a.q_s_tok = q->nb[1] / sizeof(float); a.q_s_head = q->nb[2] / sizeof(float);
    a.k = k->data; a.k_s_tok = k->nb[1]; a.k_s_head = k->nb[2];
    a.v = v->data; a.v_s_tok = v->nb[1]; a.v_s_head = v->nb[2];
    a.mask = mask ? (const uint16_t *) mask->data : nullptr; a.mask_s_tok = mask ? mask->nb[1] / 2 : 0;
    a.out = (float *) dst->data; a.o_s_head = dst->nb[1] / sizeof(float); a.o_s_tok = dst->nb[2] / sizeof(float);

    const int ith = params->ith, nth = params->nth;
    const size_t need = L->workspace(&a, nth);
    if (need > ws_size) {
        ggml_barrier(params->threadpool);
        if (ith == 0) {
            free(ws);
            ws = calloc(1, need + 64);
            ws_size = need;
            if (verbose) fprintf(stderr, "kurn attn: workspace %zu bytes (n_q %lld n_kv %lld heads %d/%d)\n", need,
                                 (long long) a.n_q, (long long) a.n_kv, a.n_head, a.n_head_kv);
        }
        ggml_barrier(params->threadpool);
    }
    L->run(&a, ws, ith, nth);
    return true;
}

}  // namespace kurn_attn
