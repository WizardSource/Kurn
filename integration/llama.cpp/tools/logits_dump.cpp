// logits_dump MODEL OUT [n_prompt=300] [n_decode=8] [n_verify=5] [n_ubatch=512]
// prefill n_prompt tokens in one batch, n_decode single-token steps, then one n_verify-token batch;
// writes every logit row (prefill: all rows) as raw f32 to OUT. Deterministic token ids.
#include "llama.h"
#include <cstdio>
#include <cstdlib>
#include <vector>
int main(int argc, char ** argv) {
    if (argc < 3) { fprintf(stderr, "usage\n"); return 1; }
    const int np = argc > 3 ? atoi(argv[3]) : 300, nd = argc > 4 ? atoi(argv[4]) : 8, nv = argc > 5 ? atoi(argv[5]) : 5;
    const int nub = argc > 6 ? atoi(argv[6]) : 512;
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 2048; cp.n_batch = 2048; cp.n_ubatch = nub; cp.n_threads = 8; cp.n_threads_batch = 8;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    llama_context * ctx = llama_init_from_model(model, cp);
    const int nvocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    FILE * f = fopen(argv[2], "wb");
    uint64_t r = 12345; auto tok = [&]() { r = r * 6364136223846793005ull + 1442695040888963407ull; return (llama_token) ((r >> 33) % 30000 + 100); };
    int pos = 0;
    auto run = [&](int n) {
        llama_batch b = llama_batch_init(n, 0, 1);
        for (int i = 0; i < n; i++) { b.token[i] = tok(); b.pos[i] = pos + i; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = 1; }
        b.n_tokens = n;
        if (llama_decode(ctx, b)) { fprintf(stderr, "decode failed\n"); exit(1); }
        for (int i = 0; i < n; i++) fwrite(llama_get_logits_ith(ctx, i), sizeof(float), nvocab, f);
        pos += n; llama_batch_free(b);
    };
    run(np);
    for (int i = 0; i < nd; i++) run(1);
    run(nv);
    fclose(f);
    llama_free(ctx); llama_model_free(model);
    return 0;
}
