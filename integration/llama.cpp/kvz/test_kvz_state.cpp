// kvz session state on a real model (llama.cpp with kvz/apply.sh): for each KV cache type, a sequence's state written
// with LLAMA_STATE_SEQ_FLAGS_KVZ must
//   - be smaller than the plain state, within llama_state_seq_get_size_ext's bound;
//   - restore to the exact plain state bytes, and give bit-identical logits for the next tokens;
//   - restore into another sequence at other (32-unaligned) cells, to the same state;
//   - round-trip through llama_state_seq_save_file_ext / llama_state_seq_load_file;
//   - be rejected when truncated (restore returns 0, the sequence is left empty), after which a good restore works.
// usage: test_kvz_state MODEL.gguf [N_PROMPT]    exit code = failures
#include "llama.h"
#include "ggml.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <unistd.h>

static int n_pass = 0, n_fail = 0;
static std::string cur_cfg;

static void check(bool ok, const char * what) {
    if (ok) {
        n_pass++;
    } else {
        n_fail++;
        printf("FAIL [%s] %s\n", cur_cfg.c_str(), what);
    }
}

static std::vector<uint8_t> get_state(llama_context * ctx, llama_seq_id s, llama_state_seq_flags flags, size_t * bound = nullptr) {
    const size_t n = llama_state_seq_get_size_ext(ctx, s, flags);
    if (bound) *bound = n;
    std::vector<uint8_t> buf(n);
    const size_t w = llama_state_seq_get_data_ext(ctx, buf.data(), n, s, flags);
    buf.resize(w);
    return buf;
}

// the state with its sequence ids zeroed (header and per-cell lists), for comparing states of different sequences
static std::vector<uint8_t> normalized(std::vector<uint8_t> st) {
    size_t off = 4;
    auto u32 = [&](size_t o) { uint32_t v; memcpy(&v, &st[o], 4); return v; };
    memset(&st[off], 0, 4);
    off += 4;
    const uint32_t n_stream = u32(off);
    off += 4;
    for (uint32_t s = 0; s < n_stream && off < st.size(); ++s) {
        const uint32_t n = u32(off);
        off += 4;
        if (!n) continue;
        for (uint32_t i = 0; i < n; ++i) {
            const uint32_t nsid = u32(off + 4);
            memset(&st[off + 8], 0, 4 * nsid);
            off += 8 + 4 * nsid;
        }
        break; // data follows; one stream with cells is enough here
    }
    return st;
}

static std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & text, int n) {
    std::vector<llama_token> t(text.size() + 16);
    const int nt = llama_tokenize(vocab, text.data(), (int) text.size(), t.data(), (int) t.size(), true, false);
    t.resize(std::max(0, nt));
    while ((int) t.size() < n) {
        t.insert(t.end(), t.begin() + 1, t.end());
    }
    t.resize(n);
    return t;
}

static bool decode(llama_context * ctx, const llama_token * t, int n, llama_pos p0, llama_seq_id s, bool logits_last) {
    llama_batch b = llama_batch_init(n, 0, 1);
    for (int i = 0; i < n; ++i) {
        b.token[i] = t[i];
        b.pos[i] = p0 + i;
        b.n_seq_id[i] = 1;
        b.seq_id[i][0] = s;
        b.logits[i] = logits_last && i == n - 1;
    }
    b.n_tokens = n;
    const bool ok = llama_decode(ctx, b) == 0;
    llama_batch_free(b);
    return ok;
}

// greedy continuation of sequence s from position p0: tokens and every step's logits
static void continue_greedy(llama_context * ctx, const llama_vocab * vocab, llama_token first, llama_pos p0, llama_seq_id s, int n,
                            std::vector<llama_token> & toks, std::vector<float> & logits) {
    const int nv = llama_vocab_n_tokens(vocab);
    llama_token t = first;
    for (int i = 0; i < n; ++i) {
        if (!decode(ctx, &t, 1, p0 + i, s, true)) {
            return;
        }
        const float * l = llama_get_logits_ith(ctx, -1);
        logits.insert(logits.end(), l, l + nv);
        int best = 0;
        for (int k = 1; k < nv; ++k) {
            if (l[k] > l[best]) best = k;
        }
        t = best;
        toks.push_back(t);
    }
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: test_kvz_state MODEL.gguf [N_PROMPT]\n");
        return 2;
    }
    const int n_prompt = argc > 2 ? atoi(argv[2]) : 700;
    const int n_gen = 8;
    llama_backend_init();
    llama_log_set([](ggml_log_level lvl, const char * msg, void *) { if (lvl >= GGML_LOG_LEVEL_ERROR) fputs(msg, stderr); }, nullptr);
    auto mp = llama_model_default_params();
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) {
        return 2;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::string text;
    for (int i = 0; i < 40; ++i) {
        text += "The entropy coder stores each channel of the key and value cache under its own table; restore must give back "
                "every byte. Sample " + std::to_string(i * 7919 % 1000) + " of the session, with numbers 3.14159 and 2.71828. ";
    }
    const auto prompt = tokenize(vocab, text, n_prompt);
    const auto other = tokenize(vocab, "An unrelated prefix that occupies the first cells of the cache. " + text.substr(0, 400), 100);

    struct cfg { const char * name; ggml_type k, v; bool fa; };
    const cfg cfgs[] = {
        {"f16", GGML_TYPE_F16, GGML_TYPE_F16, true},
        {"q8_0", GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, true},
        {"k4c_q4", GGML_TYPE_K4C, GGML_TYPE_Q4_0, true},
        {"k4c_q8", GGML_TYPE_K4C, GGML_TYPE_Q8_0, true},
        {"f16 no-FA (V transposed)", GGML_TYPE_F16, GGML_TYPE_F16, false},
    };
    for (const auto & c : cfgs) {
        cur_cfg = c.name;
        auto cp = llama_context_default_params();
        cp.n_ctx = 2048;
        cp.n_batch = 512;
        cp.n_ubatch = 512;
        cp.n_seq_max = 2;
        cp.kv_unified = true;
        cp.n_threads = cp.n_threads_batch = 8;
        cp.flash_attn_type = c.fa ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
        cp.type_k = c.k;
        cp.type_v = c.v;
        llama_context * ctx = llama_init_from_model(model, cp);
        if (!ctx) {
            check(false, "context");
            continue;
        }
        llama_memory_t mem = llama_get_memory(ctx);

        for (int i = 0; i < n_prompt; i += 256) {
            decode(ctx, prompt.data() + i, std::min(256, n_prompt - i), i, 0, false);
        }
        const llama_token next = prompt[n_prompt - 1] == 13 ? 11 : 13;

        size_t bound = 0;
        const auto raw0 = get_state(ctx, 0, 0);
        const auto z0 = get_state(ctx, 0, LLAMA_STATE_SEQ_FLAGS_KVZ, &bound);
        printf("[%s] state %zu bytes, kvz %zu bytes (%.3fx), bound %zu\n", c.name, raw0.size(), z0.size(),
               (double) raw0.size() / z0.size(), bound);
        check(!z0.empty() && z0.size() <= bound, "kvz state within the size bound");
        check(z0.size() < raw0.size(), "kvz state smaller than the plain state");

        std::vector<llama_token> t_ref, t_res;
        std::vector<float> l_ref, l_res;
        continue_greedy(ctx, vocab, next, n_prompt, 0, n_gen, t_ref, l_ref);
        check((int) t_ref.size() == n_gen, "reference continuation");

        // restore into the same sequence
        llama_memory_seq_rm(mem, 0, -1, -1);
        check(llama_state_seq_set_data_ext(ctx, z0.data(), z0.size(), 0, 0) == z0.size(), "restore kvz state");
        check(get_state(ctx, 0, 0) == raw0, "restored state equals the original, byte for byte");
        continue_greedy(ctx, vocab, next, n_prompt, 0, n_gen, t_res, l_res);
        check(t_res == t_ref, "same continuation tokens after restore");
        check(l_res.size() == l_ref.size() && memcmp(l_res.data(), l_ref.data(), l_ref.size() * sizeof(float)) == 0,
              "bit-identical logits after restore");

        // into another sequence, behind 100 cells of another one (k4c: other 32-cell groups)
        llama_memory_clear(mem, true);
        decode(ctx, other.data(), (int) other.size(), 0, 0, false);
        check(llama_state_seq_set_data_ext(ctx, z0.data(), z0.size(), 1, 0) == z0.size(), "restore kvz state into seq 1");
        // k4c session state as native group records (the default; LLAMA_K4C_STATE=f16 writes f16 rows): restored at
        // another 32-cell group alignment, the records are re-encoded into the destination groups, so the re-saved
        // state is not byte-identical. The f16-row form is alignment-independent and is checked strictly.
        const char * k4c_state = getenv("LLAMA_K4C_STATE");
        if (c.k == GGML_TYPE_K4C && !(k4c_state && !strcmp(k4c_state, "f16"))) {
            check(!get_state(ctx, 1, 0).empty(), "seq 1 state re-saved (native k4c records: re-encoded at another group alignment)");
        } else {
            check(normalized(get_state(ctx, 1, 0)) == normalized(raw0), "seq 1 state equals the original");
        }

        // slot files
        const std::string fz = "/tmp/test_kvz_state_" + std::to_string(getpid()) + ".kvz", fr = fz + ".raw";
        llama_memory_clear(mem, true);
        llama_state_seq_set_data_ext(ctx, raw0.data(), raw0.size(), 0, 0);
        const size_t nz = llama_state_seq_save_file_ext(ctx, fz.c_str(), 0, prompt.data(), prompt.size(), LLAMA_STATE_SEQ_FLAGS_KVZ);
        const size_t nr = llama_state_seq_save_file(ctx, fr.c_str(), 0, prompt.data(), prompt.size());
        check(nz > 0 && nz < nr, "kvz slot file smaller than the plain one");
        llama_memory_seq_rm(mem, 0, -1, -1);
        std::vector<llama_token> toks(prompt.size() + 8);
        size_t n_tok = 0;
        check(llama_state_seq_load_file(ctx, fz.c_str(), 0, toks.data(), toks.size(), &n_tok) == nz, "load kvz slot file");
        check(n_tok == prompt.size() && std::equal(prompt.begin(), prompt.end(), toks.begin()), "slot file tokens");
        check(get_state(ctx, 0, 0) == raw0, "state from the kvz slot file equals the original");
        remove(fz.c_str());
        remove(fr.c_str());

        // damaged
        llama_memory_seq_rm(mem, 0, -1, -1);
        std::vector<uint8_t> cut(z0.begin(), z0.begin() + z0.size() * 2 / 3);
        check(llama_state_seq_set_data_ext(ctx, cut.data(), cut.size(), 0, 0) == 0, "truncated kvz state rejected");
        check(llama_memory_seq_pos_max(mem, 0) == -1, "sequence empty after a rejected restore");
        check(llama_state_seq_set_data_ext(ctx, z0.data(), z0.size(), 0, 0) == z0.size() && get_state(ctx, 0, 0) == raw0,
              "good restore after a rejected one");

        llama_free(ctx);
    }
    llama_model_free(model);
    printf("kvz state: %d passed, %d failed\n", n_pass, n_fail);
    return n_fail;
}
