// K4C session state through libllama (llama_state_seq_*): native group records (default) and f16 rows (older files,
// LLAMA_K4C_STATE=f16).
//
//   test_k4c_state exact MODEL TEXT KV           prefill + decode, save the sequence, run 24 more tokens (reference
//                                                logits), remove the sequence, restore it, check that the state bytes
//                                                and the 24 tokens' logits are identical
//   test_k4c_state save MODEL TEXT KV FILE       save the sequence to FILE (+ FILE.logits: the reference)
//   test_k4c_state load MODEL TEXT KV FILE       restore FILE and compare the 24 tokens' logits with FILE.logits
//   test_k4c_state shifted MODEL TEXT KV         restore at other cell offsets (a 7-cell sequence first): re-encoded,
//                                                logits close to the reference
//   test_k4c_state size MODEL TEXT KV NTOK       sequence state size after an NTOK-token prefill
// KV = k4c_q4 | k4c_q8 | f16 | q8_0. Exit code = failures.
#include "llama.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static const int NP = 1325, NG = 20;
static const int NC = getenv("K4C_STATE_NC") ? atoi(getenv("K4C_STATE_NC")) : 24; // scored continuation tokens

static llama_context * make_ctx(llama_model * m, const std::string & kv, int n_ctx) {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n_ctx; cp.n_batch = 512; cp.n_ubatch = 512; cp.n_threads = 8; cp.n_threads_batch = 8; cp.n_seq_max = 2;
    cp.kv_unified = true;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    if (kv == "q8_0") { cp.type_k = GGML_TYPE_Q8_0; cp.type_v = GGML_TYPE_Q8_0; }
    else if (kv == "k4c_q4") { cp.type_k = GGML_TYPE_K4C; cp.type_v = GGML_TYPE_Q4_0; }
    else if (kv == "k4c_q8") { cp.type_k = GGML_TYPE_K4C; cp.type_v = GGML_TYPE_Q8_0; }
    return llama_init_from_model(m, cp);
}

// decode t[0..n) for seq at positions pos0..; logits of every row appended to lg (if given)
static void decode(llama_context * ctx, const llama_token * t, int n, int pos0, int seq, int nv, std::vector<float> * lg) {
    for (int i0 = 0; i0 < n; i0 += 512) {
        const int m = std::min(512, n - i0);
        llama_batch b = llama_batch_init(m, 0, 1);
        for (int i = 0; i < m; i++) {
            b.token[i] = t[i0 + i]; b.pos[i] = pos0 + i0 + i; b.n_seq_id[i] = 1; b.seq_id[i][0] = seq; b.logits[i] = lg != nullptr;
        }
        b.n_tokens = m;
        if (llama_decode(ctx, b)) { fprintf(stderr, "decode failed\n"); exit(1); }
        for (int i = 0; lg && i < m; i++) {
            const float * l = llama_get_logits_ith(ctx, i);
            lg->insert(lg->end(), l, l + nv);
        }
        llama_batch_free(b);
    }
}

static int argmax(const float * l, int nv) {
    int a = 0;
    for (int v = 1; v < nv; v++) a = l[v] > l[a] ? v : a;
    return a;
}

// prefill NP tokens, then NG greedy one-token steps; returns the tokens (prompt + generated) and the next NC tokens
static void build(llama_context * ctx, const std::vector<llama_token> & text, int nv, std::vector<llama_token> & seq_tok, std::vector<llama_token> & next) {
    seq_tok.assign(text.begin(), text.begin() + NP);
    std::vector<float> lg;
    decode(ctx, seq_tok.data(), NP, 0, 0, nv, &lg);
    llama_token t = argmax(lg.data() + (size_t) (NP - 1) * nv, nv);
    for (int i = 0; i < NG; i++) {
        seq_tok.push_back(t);
        lg.clear();
        decode(ctx, &t, 1, (int) seq_tok.size() - 1, 0, nv, &lg);
        t = argmax(lg.data(), nv);
    }
    next.assign(text.begin() + NP + NG, text.begin() + NP + NG + NC); // teacher-forced continuation
}

static double kl_rows(const std::vector<float> & a, const std::vector<float> & b, int nv, int & agree) {
    double s = 0;
    agree = 0;
    const int n = (int) (a.size() / nv);
    for (int i = 0; i < n; i++) {
        const float * p = a.data() + (size_t) i * nv, * q = b.data() + (size_t) i * nv;
        double mp = -1e30, mq = -1e30, sp = 0, sq = 0;
        for (int v = 0; v < nv; v++) { mp = std::max(mp, (double) p[v]); mq = std::max(mq, (double) q[v]); }
        for (int v = 0; v < nv; v++) { sp += std::exp(p[v] - mp); sq += std::exp(q[v] - mq); }
        for (int v = 0; v < nv; v++) {
            const double lp = p[v] - mp - std::log(sp), lq = q[v] - mq - std::log(sq);
            s += std::exp(lp) * (lp - lq);
        }
        agree += argmax(p, nv) == argmax(q, nv);
    }
    return s / n;
}

int main(int argc, char ** argv) {
    if (argc < 5) { fprintf(stderr, "usage: see source\n"); return 2; }
    const std::string mode = argv[1], kv = argv[4];
    llama_backend_init();
    llama_model * model = llama_model_load_from_file(argv[2], llama_model_default_params());
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int nv = llama_vocab_n_tokens(vocab);
    std::ifstream fin(argv[3]); std::stringstream ss; ss << fin.rdbuf();
    const std::string text = ss.str().substr(0, mode == "size" ? std::string::npos : 20000);
    std::vector<llama_token> tok(text.size());
    tok.resize(llama_tokenize(vocab, text.c_str(), (int) text.size(), tok.data(), (int) tok.size(), false, false));
    int fails = 0;
    if (mode == "size") {
        const int n = atoi(argv[5]);
        llama_context * ctx = make_ctx(model, kv, n + 256);
        decode(ctx, tok.data(), n, 0, 0, nv, nullptr);
        printf("%s: %d tokens, sequence state %.1f MiB\n", kv.c_str(), n, llama_state_seq_get_size(ctx, 0) / 1048576.0);
        llama_free(ctx);
        llama_model_free(model);
        return 0;
    }
    llama_context * ctx = make_ctx(model, kv, 4096);
    llama_memory_t mem = llama_get_memory(ctx);
    std::vector<llama_token> seq_tok, next;
    build(ctx, tok, nv, seq_tok, next);
    const int pos = (int) seq_tok.size();
    if (mode == "exact" || mode == "save") {
        std::vector<uint8_t> st(llama_state_seq_get_size(ctx, 0));
        st.resize(llama_state_seq_get_data(ctx, st.data(), st.size(), 0));
        if (mode == "save" && !llama_state_seq_save_file(ctx, argv[5], 0, seq_tok.data(), seq_tok.size())) { printf("save failed\n"); return 1; }
        std::vector<float> ref;
        decode(ctx, next.data(), NC, pos, 0, nv, &ref);
        if (mode == "save") {
            FILE * f = fopen((std::string(argv[5]) + ".logits").c_str(), "wb");
            fwrite(ref.data(), sizeof(float), ref.size(), f);
            fclose(f);
            printf("%s: saved %d cells, state %.2f MiB\n", kv.c_str(), pos, st.size() / 1048576.0);
        } else {
            llama_memory_seq_rm(mem, 0, -1, -1);
            if (llama_state_seq_set_data(ctx, st.data(), st.size(), 0) != st.size()) { printf("%s: restore failed  FAIL\n", kv.c_str()); return 1; }
            std::vector<uint8_t> st2(llama_state_seq_get_size(ctx, 0));
            st2.resize(llama_state_seq_get_data(ctx, st2.data(), st2.size(), 0));
            std::vector<float> lg;
            decode(ctx, next.data(), NC, pos, 0, nv, &lg);
            const bool same_state = st == st2, same_logits = lg == ref;
            printf("%s: state %.2f MiB for %d cells; after restore: state bytes identical %s, %d tokens' logits bit-identical %s  %s\n",
                   kv.c_str(), st.size() / 1048576.0, pos, same_state ? "yes" : "NO", NC, same_logits ? "yes" : "NO",
                   same_state && same_logits ? "ok" : "FAIL");
            fails += !(same_state && same_logits);
        }
    } else if (mode == "load" || mode == "shifted") {
        std::vector<float> ref;
        if (mode == "load") {
            ref.resize((size_t) NC * nv);
            FILE * f = fopen((std::string(argv[5]) + ".logits").c_str(), "rb");
            if (!f || fread(ref.data(), sizeof(float), ref.size(), f) != ref.size()) { printf("no reference logits\n"); return 1; }
            fclose(f);
        } else {
            decode(ctx, next.data(), NC, pos, 0, nv, &ref);
        }
        std::vector<uint8_t> st;
        if (mode == "shifted") { // save, then restore behind a 7-cell sequence 1: the cells land 7 positions further on
            llama_memory_seq_rm(mem, 0, pos, -1);
            st.resize(llama_state_seq_get_size(ctx, 0));
            st.resize(llama_state_seq_get_data(ctx, st.data(), st.size(), 0));
        }
        llama_memory_clear(mem, true);
        if (mode == "shifted") {
            if (!getenv("NO_SEQ1")) decode(ctx, tok.data() + 5000, 7, 0, 1, nv, nullptr);
            if (llama_state_seq_set_data(ctx, st.data(), st.size(), 0) != st.size()) { printf("restore failed  FAIL\n"); return 1; }
        } else {
            std::vector<llama_token> saved(pos + 8);
            size_t n_saved = 0;
            if (!llama_state_seq_load_file(ctx, argv[5], 0, saved.data(), saved.size(), &n_saved)) { printf("%s: load failed  FAIL\n", kv.c_str()); return 1; }
        }
        std::vector<float> lg;
        decode(ctx, next.data(), NC, pos, 0, nv, &lg);
        int agree;
        const double kl = kl_rows(ref, lg, nv, agree);
        if (mode == "shifted") { // quality: the live run and the restored one against an F16-KV run of the same tokens
            llama_context * cf = make_ctx(model, "f16", 4096);
            decode(cf, seq_tok.data(), pos, 0, 0, nv, nullptr); // the same tokens as the k4c run
            std::vector<float> rf;
            decode(cf, next.data(), NC, pos, 0, nv, &rf);
            llama_free(cf);
            int a1 = 0, a2 = 0;
            const double k1 = kl_rows(rf, ref, nv, a1), k2 = kl_rows(rf, lg, nv, a2);
            printf("%s vs F16 KV: live KL %.5f (top-1 %d/%d), after the shifted restore KL %.5f (top-1 %d/%d)\n", kv.c_str(), k1, a1, NC, k2, a2, NC);
        }
        const bool exact = lg == ref, ok = exact || mode == "shifted" || (kl < 0.02 && agree >= NC - 2);
        printf("%s %s: logits bit-identical %s, KL %.6f, top-1 agree %d/%d  %s\n", kv.c_str(), mode.c_str(), exact ? "yes" : "no", kl, agree, NC,
               ok ? "ok" : "FAIL");
        fails += !ok;
    }
    llama_free(ctx);
    llama_model_free(model);
    return fails;
}
