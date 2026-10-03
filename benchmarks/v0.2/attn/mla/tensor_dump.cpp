// Dump named graph tensors of one layer while llama.cpp evaluates the first --ctx tokens of TEXT.
//   tensor_dump MODEL TEXT --layer L --names q_states,kv_compressed,k_pe,kqv_out --out DIR [--ctx 512]
// DIR/<name>.bin = int64 ne[4] then float32 data (contiguous, ne0 fastest); a name emitted more
// than once (e.g. a view and its normalised copy) keeps the last one. Repacking is off (no AMX).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "ggml.h"
#include "llama.h"

struct state {
    std::vector<std::string> names;
    std::string out;
};

static bool cb(struct ggml_tensor * t, bool ask, void * ud) {
    state & S = *(state *) ud;
    for (const std::string & n : S.names) {
        if (n != t->name) continue;
        if (ask) return true;
        if (t->type != GGML_TYPE_F32) { fprintf(stderr, "%s: type %d skipped\n", t->name, t->type); return true; }
        FILE * f = fopen((S.out + "/" + n.substr(0, n.rfind('-')) + ".bin").c_str(), "wb");
        int64_t ne[4] = {t->ne[0], t->ne[1], t->ne[2], t->ne[3]};
        fwrite(ne, sizeof ne, 1, f);
        for (int64_t i3 = 0; i3 < ne[3]; i3++)
            for (int64_t i2 = 0; i2 < ne[2]; i2++)
                for (int64_t i1 = 0; i1 < ne[1]; i1++)
                    for (int64_t i0 = 0; i0 < ne[0]; i0++) {
                        const float v = *(const float *) ((const char *) t->data + i0 * t->nb[0] + i1 * t->nb[1] + i2 * t->nb[2] + i3 * t->nb[3]);
                        fwrite(&v, 4, 1, f);
                    }
        fclose(f);
        return true;
    }
    return ask ? false : true;
}

static const char * arg(int argc, char ** argv, const char * k, const char * def) {
    for (int i = 1; i < argc - 1; i++)
        if (!strcmp(argv[i], k)) return argv[i + 1];
    return def;
}

int main(int argc, char ** argv) {
    if (argc < 3) { fprintf(stderr, "usage: tensor_dump MODEL TEXT --layer L --names a,b --out DIR\n"); return 1; }
    state S;
    S.out = arg(argc, argv, "--out", ".");
    const int layer = atoi(arg(argc, argv, "--layer", "0")), n_ctx = atoi(arg(argc, argv, "--ctx", "512"));
    std::string names = arg(argc, argv, "--names", "");
    for (size_t p = 0; p <= names.size();) {
        size_t e = names.find(',', p);
        if (e == std::string::npos) e = names.size();
        if (e > p) S.names.push_back(names.substr(p, e - p) + "-" + std::to_string(layer));
        p = e + 1;
    }
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.use_extra_bufts = false;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) return 1;
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = cp.n_batch = cp.n_ubatch = n_ctx;
    cp.n_threads = cp.n_threads_batch = atoi(arg(argc, argv, "--threads", "4"));
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.cb_eval = cb;
    cp.cb_eval_user_data = &S;
    llama_context * ctx = llama_init_from_model(model, cp);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    FILE * tf = fopen(argv[2], "rb");
    std::string text;
    char buf[65536];
    size_t got;
    while ((got = fread(buf, 1, sizeof buf, tf)) > 0) text.append(buf, got);
    fclose(tf);
    std::vector<llama_token> toks(text.size() + 16);
    const int nt = llama_tokenize(vocab, text.c_str(), (int) text.size(), toks.data(), (int) toks.size(), true, false);
    if (nt < n_ctx) { fprintf(stderr, "text too short\n"); return 1; }
    llama_batch batch = llama_batch_get_one(toks.data(), n_ctx);
    if (llama_decode(ctx, batch)) { fprintf(stderr, "decode failed\n"); return 1; }
    printf("dumped layer %d of %d tokens to %s\n", layer, n_ctx, S.out.c_str());
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
