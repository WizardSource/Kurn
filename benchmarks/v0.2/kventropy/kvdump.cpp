// kvdump MODEL TEXT N_TOK CTK CTV OUT.seq [THREADS]
// prefill the first N_TOK tokens of TEXT and save sequence 0's state (llama_state_seq_save_file format)
#include "llama.h"
#include "ggml.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static ggml_type parse_type(const char * s) {
    if (!strcmp(s, "f16")) return GGML_TYPE_F16;
    if (!strcmp(s, "q8_0")) return GGML_TYPE_Q8_0;
    if (!strcmp(s, "q4_0")) return GGML_TYPE_Q4_0;
    if (!strcmp(s, "k4c")) return GGML_TYPE_K4C;
    fprintf(stderr, "bad type %s\n", s);
    exit(1);
}

int main(int argc, char ** argv) {
    if (argc < 7) {
        fprintf(stderr, "usage: kvdump MODEL TEXT N_TOK CTK CTV OUT [THREADS]\n");
        return 1;
    }
    const int n_tok = atoi(argv[3]);
    const int n_thr = argc > 7 ? atoi(argv[7]) : 8;
    llama_backend_init();
    auto mp = llama_model_default_params();
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) return 1;
    const llama_vocab * vocab = llama_model_get_vocab(model);

    std::ifstream f(argv[2]);
    std::stringstream ss;
    ss << f.rdbuf();
    std::string text = ss.str();
    if (text.size() > (size_t) n_tok * 12) text.resize((size_t) n_tok * 12);
    std::vector<llama_token> toks(text.size() + 16);
    int nt = llama_tokenize(vocab, text.data(), (int) text.size(), toks.data(), (int) toks.size(), true, false);
    if (nt < n_tok) {
        fprintf(stderr, "text too short: %d tokens\n", nt);
        return 1;
    }
    toks.resize(n_tok);

    auto cp = llama_context_default_params();
    cp.n_ctx = ((n_tok + 255) / 256) * 256;
    cp.n_batch = 2048;
    cp.n_ubatch = 512;
    cp.n_seq_max = 1;
    cp.n_threads = cp.n_threads_batch = n_thr;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.type_k = parse_type(argv[4]);
    cp.type_v = parse_type(argv[5]);
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) return 1;

    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < n_tok; i += cp.n_batch) {
        int n = std::min<int>(cp.n_batch, n_tok - i);
        llama_batch b = llama_batch_get_one(toks.data() + i, n);
        if (llama_decode(ctx, b)) {
            fprintf(stderr, "decode failed at %d\n", i);
            return 1;
        }
    }
    double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    fprintf(stderr, "prefill %d tokens: %.1f s (%.1f tok/s)\n", n_tok, dt, n_tok / dt);
    t0 = std::chrono::steady_clock::now();
    size_t sz = llama_state_seq_save_file(ctx, argv[6], 0, toks.data(), toks.size());
    dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    fprintf(stderr, "saved %zu bytes in %.2f s\n", sz, dt);
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
