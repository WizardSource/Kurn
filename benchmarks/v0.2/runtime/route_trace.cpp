// Dump MoE routing from llama.cpp for expert-prefetch analysis (route_predict.py):
// per layer, the selected experts (ffn_moe_topk), the router input (ffn_norm) and the
// residual stream after the layer (l_out), for every token of one prompt batch.
//   route_trace MODEL.gguf TEXT.txt OUTDIR [n_tokens=512]
// Build: g++ -O2 -std=c++17 route_trace.cpp -I$LLAMA/include -I$LLAMA/ggml/include \
//          -L$LLAMA/build/bin -lllama -lggml -lggml-base -Wl,-rpath,$LLAMA/build/bin
#include "ggml-backend.h"
#include "llama.h"
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static std::string g_out;

static bool cb(struct ggml_tensor *t, bool ask, void *) {
    const char *n = t->name;
    const bool want = !strncmp(n, "ffn_moe_topk-", 13) || !strncmp(n, "ffn_norm-", 9) || !strncmp(n, "l_out-", 6);
    if (ask) return want;
    if (!want) return true;
    std::string file = g_out + "/" + n + ".bin";
    FILE *f = fopen(file.c_str(), "wb");
    const size_t es = ggml_type_size(t->type);
    std::vector<char> row(t->ne[0] * es);
    for (int64_t i = 0; i < t->ne[1]; i++) { // 2-D, rows may be strided (top-k is a view)
        ggml_backend_tensor_get(t, row.data(), i * t->nb[1], row.size());
        fwrite(row.data(), 1, row.size(), f);
    }
    fclose(f);
    return true;
}

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s model.gguf text.txt outdir [n_tokens]\n", argv[0]); return 2; }
    g_out = argv[3];
    const int ntok = argc > 4 ? atoi(argv[4]) : 512;
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    llama_model *model = llama_model_load_from_file(argv[1], mp);
    if (!model) return 1;
    std::ifstream in(argv[2]);
    std::stringstream ss;
    ss << in.rdbuf();
    std::string text = ss.str();
    const llama_vocab *vocab = llama_model_get_vocab(model);
    std::vector<llama_token> toks(text.size() + 8);
    int n = llama_tokenize(vocab, text.c_str(), (int)text.size(), toks.data(), (int)toks.size(), true, false);
    if (n > ntok) n = ntok;
    toks.resize(n);
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n + 16; cp.n_batch = n; cp.n_ubatch = n; cp.n_threads = 8; cp.n_threads_batch = 8;
    cp.cb_eval = cb; cp.cb_eval_user_data = nullptr;
    llama_context *ctx = llama_init_from_model(model, cp);
    llama_batch b = llama_batch_init(n, 0, 1);
    for (int i = 0; i < n; i++) {
        b.token[i] = toks[i]; b.pos[i] = i; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = 1;
    }
    b.n_tokens = n;
    if (llama_decode(ctx, b)) { fprintf(stderr, "decode failed\n"); return 1; }
    printf("traced %d tokens into %s\n", n, argv[3]);
    llama_batch_free(b);
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
