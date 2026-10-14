// K-shift through llama.cpp (what llama-server's context shift and --cache-reuse do): prefill A + B + C, remove B,
// shift C back by |B| (llama_memory_seq_rm + llama_memory_seq_add, applied as a K-shift graph on the next decode), then
// score D teacher-forced; and a fresh context that prefills A + C and scores D. Even with F16 KV the two differ (C's K/V
// were computed with B in context), so each KV type is compared with F16 under the same operation: its KL to F16 after
// the shift should match its KL to F16 without one (full-vocabulary KL, top-1 agreement).
//
//   test_kshift MODEL TEXT [KV...]     KV = q8_0 | k4c_q4 | k4c_q8 (default: all); exit code = failures
//
// Build against a llama.cpp checkout with apply.sh and k4c/apply.sh applied:
//   g++ -O2 -std=c++17 -I$L/include -I$L/ggml/include test_kshift.cpp -L$L/build/bin -lllama -lggml -lggml-base \
//       -Wl,-rpath,$L/build/bin -o test_kshift
#include "llama.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static const int NA = 256, NB = 300, NC = 700, ND = 200;

static llama_context * make_ctx(llama_model * m, const std::string & kv) {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 2048; cp.n_batch = 512; cp.n_ubatch = 512; cp.n_threads = 8; cp.n_threads_batch = 8;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    if (kv == "q8_0") { cp.type_k = GGML_TYPE_Q8_0; cp.type_v = GGML_TYPE_Q8_0; }
    else if (kv == "k4c_q4") { cp.type_k = GGML_TYPE_K4C; cp.type_v = GGML_TYPE_Q4_0; }
    else if (kv == "k4c_q8") { cp.type_k = GGML_TYPE_K4C; cp.type_v = GGML_TYPE_Q8_0; }
    return llama_init_from_model(m, cp);
}

// decode tokens t at positions pos0.. in 512-token batches; log-softmax of every row into lp (if given)
static void decode(llama_context * ctx, const llama_token * t, int n, int pos0, int nv, std::vector<float> * lp) {
    for (int i0 = 0; i0 < n; i0 += 512) {
        const int m = std::min(512, n - i0);
        llama_batch b = llama_batch_init(m, 0, 1);
        for (int i = 0; i < m; i++) {
            b.token[i] = t[i0 + i]; b.pos[i] = pos0 + i0 + i; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = lp != nullptr;
        }
        b.n_tokens = m;
        if (llama_decode(ctx, b)) { fprintf(stderr, "decode failed\n"); exit(1); }
        for (int i = 0; lp && i < m; i++) {
            const float * l = llama_get_logits_ith(ctx, i);
            double mx = -1e30, s = 0;
            for (int v = 0; v < nv; v++) mx = std::max(mx, (double) l[v]);
            for (int v = 0; v < nv; v++) s += std::exp(l[v] - mx);
            const double lse = mx + std::log(s);
            for (int v = 0; v < nv; v++) lp->push_back((float) (l[v] - lse));
        }
        llama_batch_free(b);
    }
}

int main(int argc, char ** argv) {
    if (argc < 3) { fprintf(stderr, "usage: test_kshift MODEL TEXT [KV...]\n"); return 2; }
    llama_backend_init();
    llama_model * model = llama_model_load_from_file(argv[1], llama_model_default_params());
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int nv = llama_vocab_n_tokens(vocab);
    std::ifstream f(argv[2]); std::stringstream ss; ss << f.rdbuf(); const std::string text = ss.str().substr(0, 40000);
    std::vector<llama_token> t(text.size());
    t.resize(llama_tokenize(vocab, text.c_str(), (int) text.size(), t.data(), (int) t.size(), false, false));
    if ((int) t.size() < NA + NB + NC + ND) { fprintf(stderr, "text too short\n"); return 2; }
    std::vector<std::string> kvs;
    for (int i = 3; i < argc; i++) kvs.push_back(argv[i]);
    if (kvs.empty()) kvs = { "q8_0", "k4c_q4", "k4c_q8" };
    const llama_token * A = t.data(), * B = A + NA, * C = B + NB, * D = C + NC;
    // the two runs for one KV type: lp of D after a fresh A + C prefill, and after A + B + C with B removed and C shifted
    auto runs = [&](const std::string & kv, std::vector<float> & fresh, std::vector<float> & shifted) -> bool {
        llama_context * c0 = make_ctx(model, kv);
        std::vector<llama_token> ac(A, A + NA);
        ac.insert(ac.end(), C, C + NC);
        decode(c0, ac.data(), (int) ac.size(), 0, nv, nullptr);
        decode(c0, D, ND, NA + NC, nv, &fresh);
        llama_free(c0);
        llama_context * c1 = make_ctx(model, kv);
        std::vector<llama_token> abc(A, A + NA + NB + NC);
        decode(c1, abc.data(), (int) abc.size(), 0, nv, nullptr);
        llama_memory_t mem = llama_get_memory(c1);
        if (!llama_memory_can_shift(mem) || !llama_memory_seq_rm(mem, 0, NA, NA + NB)) {
            llama_free(c1);
            return false;
        }
        llama_memory_seq_add(mem, 0, NA + NB, -1, -NB);
        decode(c1, D, ND, NA + NC, nv, &shifted);
        llama_free(c1);
        return true;
    };
    auto kl = [&](const std::vector<float> & p_all, const std::vector<float> & q_all, double & agree) {
        double s = 0; int a = 0;
        for (int i = 0; i < ND; i++) {
            const float * p = p_all.data() + (size_t) i * nv, * q = q_all.data() + (size_t) i * nv;
            int ap = 0, aq = 0;
            for (int v = 0; v < nv; v++) {
                s += std::exp((double) p[v]) * (p[v] - q[v]);
                if (p[v] > p[ap]) ap = v;
                if (q[v] > q[aq]) aq = v;
            }
            a += ap == aq;
        }
        agree = (double) a / ND;
        return s / ND;
    };
    std::vector<float> f16_fresh, f16_shift;
    if (!runs("f16", f16_fresh, f16_shift)) { printf("f16: K-shift not supported\n"); return 1; }
    double ag;
    const double sem = kl(f16_fresh, f16_shift, ag);
    printf("f16     shifted vs fresh (context-shift effect, not an error): KL %.5f, top-1 agree %.3f\n", sem, ag);
    int fails = 0;
    for (const auto & kv : kvs) {
        std::vector<float> fr, sh;
        if (!runs(kv, fr, sh)) { printf("%-7s K-shift not supported  FAIL\n", kv.c_str()); fails++; continue; }
        double af, as;
        const double kf = kl(f16_fresh, fr, af), ks = kl(f16_shift, sh, as);
        const bool ok = std::isfinite(ks) && ks <= 1.5 * kf + 0.002;
        printf("%-7s KL vs f16: fresh %.5f (top-1 %.3f), after K-shift %.5f (top-1 %.3f)  %s\n", kv.c_str(), kf, af, ks, as, ok ? "ok" : "FAIL");
        fails += !ok;
    }
    llama_model_free(model);
    return fails;
}
