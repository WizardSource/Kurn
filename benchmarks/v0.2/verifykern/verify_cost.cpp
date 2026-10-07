// verify_cost MODEL THREADS "M list" [REPS] [KV] : whole-forward time of an M-token batch with logits for
// all M tokens (a speculative verify step) at a KV depth of KV tokens. Reps interleave the widths and each
// batch is removed from the KV cache afterwards. Builds against llama.cpp (4ebdf2c) and ik_llama.cpp (-DIK).
// Prints "M ms_median ms_min" per width.
#include "llama.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>

static double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static void rm_from(llama_context * ctx, int p0) {
#ifdef IK
    llama_kv_cache_seq_rm(ctx, 0, p0, -1);
#else
    llama_memory_seq_rm(llama_get_memory(ctx), 0, p0, -1);
#endif
}

static int decode(llama_context * ctx, int p0, int n, bool all_logits) {
    llama_batch b = llama_batch_init(n, 0, 1);
    for (int i = 0; i < n; i++) {
        b.token[i] = 1000 + (p0 + i) * 7 % 20000;
        b.pos[i] = p0 + i;
        b.n_seq_id[i] = 1;
        b.seq_id[i][0] = 0;
        b.logits[i] = all_logits || i == n - 1;
    }
    b.n_tokens = n;
    const int r = llama_decode(ctx, b);
    llama_batch_free(b);
    return r;
}

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s MODEL THREADS \"M list\" [REPS] [KV]\n", argv[0]);
        return 2;
    }
    const int nt = atoi(argv[2]), reps = argc > 4 ? atoi(argv[4]) : 7, kv = argc > 5 ? atoi(argv[5]) : 292;
    std::vector<int> Ms;
    std::istringstream ss(argv[3]);
    for (int m; ss >> m;) Ms.push_back(m);
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
#ifdef IK
    mp.use_mmap = false;
    mp.repack_tensors = getenv("IK_RTR") ? atoi(getenv("IK_RTR")) != 0 : true;
#else
    mp.load_mode = LLAMA_LOAD_MODE_NONE;
#endif
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) return 1;
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 2048;
    cp.n_batch = 512;
    cp.n_ubatch = 512;
    cp.n_threads = nt;
    cp.n_threads_batch = nt;
#ifdef IK
    cp.flash_attn = true;
#else
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
#endif
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) return 1;
    if (decode(ctx, 0, kv, false)) return 1;
    for (int w = 0; w < 3; w++)
        for (int m : Ms) { decode(ctx, kv, m, true); rm_from(ctx, kv); }
    std::vector<std::vector<double>> t(Ms.size());
    for (int r = 0; r < reps; r++) {
        for (size_t i = 0; i < Ms.size(); i++) {
            const size_t j = (i + r) % Ms.size();
            const double t0 = now_ms();
            if (decode(ctx, kv, Ms[j], true)) return 1;
            t[j].push_back(now_ms() - t0);
            rm_from(ctx, kv);
        }
    }
    for (size_t i = 0; i < Ms.size(); i++) {
        std::sort(t[i].begin(), t[i].end());
        printf("%d %.2f %.2f\n", Ms[i], t[i][t[i].size() / 2], t[i][0]);
    }
    llama_free(ctx);
#ifdef IK
    llama_free_model(model);
#else
    llama_model_free(model);
#endif
    llama_backend_free();
    return 0;
}
