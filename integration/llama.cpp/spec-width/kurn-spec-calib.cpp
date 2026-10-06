// kurn-spec-calib: inputs of the cost-aware verify width policy (kurn-spec-width.h, kurn.specwidth).
//
//   kurn-spec-calib -m TARGET.gguf -md DRAFT.gguf -p PROMPT -n N [threads / cpu-mask flags]
//
// Same command line and thread pools as llama-speculative-simple. Writes, for KURN_CALIB_OUT=prefix:
//   prefix.cost    whole-forward cost table: `verify M ms` (target, M tokens, logits for all M, as a speculative
//                  verify batch) and `draft M ms` (draft, M tokens, last logit only, as the drafter's re-decode
//                  of a verified batch), M = 1..KURN_CALIB_MMAX (default 24), median of KURN_CALIB_REPS (5)
//                  repetitions interleaved over M, measured after the generation (KV holds prompt + N tokens)
//   prefix.trace   one line per generated token c: `match p`, match = the draft's top-1 given the true prefix
//                  is the target's greedy token, p = its probability under the drafter's top-10 sampler. For
//                  greedy chain drafting this decides every step of any width policy (kurn specwidth simulate).
// KURN_CALIB_TABLE=0 skips the cost table.
#include "arg.h"
#include "common.h"
#include "speculative.h"
#include "log.h"
#include "llama.h"

#include <algorithm>
#include <chrono>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static int env_int(const char * name, int def) {
    const char * v = getenv(name);
    return v && *v ? atoi(v) : def;
}

static double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// greedy token and its probability under a top-10 softmax (the draft-simple sampler's candidates)
static llama_token top1(const float * logits, int n_vocab, float * p) {
    std::vector<std::pair<float, llama_token>> top;
    top.reserve(11);
    for (llama_token t = 0; t < n_vocab; t++) {
        if (top.size() < 10 || logits[t] > top.back().first) {
            top.emplace_back(logits[t], t);
            std::sort(top.begin(), top.end(), [](auto & a, auto & b) { return a.first > b.first || (a.first == b.first && a.second < b.second); });
            if (top.size() > 10) top.pop_back();
        }
    }
    double s = 0.0;
    for (auto & x : top) s += std::exp((double) x.first - top[0].first);
    *p = (float) (1.0 / s);
    return top[0].second;
}

static bool process(llama_context * ctx, common_batch & b) {
    if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, b.get()) != 0) {
        LOG_ERR("llama_process failed\n");
        return false;
    }
    llama_synchronize(ctx);
    return true;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    common_params params;
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SPECULATIVE)) {
        return 1;
    }
    const int m_max = std::max(2, env_int("KURN_CALIB_MMAX", 24)), reps = std::max(1, env_int("KURN_CALIB_REPS", 5));
    const bool do_table = env_int("KURN_CALIB_TABLE", 1) != 0;
    const char * out = getenv("KURN_CALIB_OUT");
    if (!out || !*out) {
        LOG_ERR("set KURN_CALIB_OUT=prefix\n");
        return 1;
    }
    const auto lim = common_speculative_get_output_limits(params.n_batch, params.n_parallel, std::max(m_max, common_speculative_n_max(&params.speculative)));
    params.n_outputs_max = lim.total;
    params.n_outputs_max_per_seq = lim.per_seq;

    llama_backend_init();
    llama_numa_init(params.numa);
    auto init_tgt = common_init_from_params(params);
    llama_model * model_tgt = init_tgt->model();
    llama_context * ctx_tgt = init_tgt->context();
    common_params params_dft = common_base_params_to_speculative(params);
    auto spec_init = common_speculative_init_from_params(params_dft, model_tgt, ctx_tgt);
    llama_context * ctx_dft = spec_init->context();
    if (!ctx_tgt || !ctx_dft) {
        LOG_ERR("failed to load the target or the draft model\n");
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model_tgt);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    std::vector<llama_token> inp = common_tokenize(ctx_tgt, params.prompt, true, true);
    const int n_gen = params.n_predict > 0 ? params.n_predict : 128;
    if ((int) inp.size() + n_gen + m_max + 2 > (int) llama_n_ctx(ctx_tgt)) {
        LOG_ERR("prompt + n + MMAX exceeds the context\n");
        return 1;
    }
    const int n_inp = (int) inp.size();

    // target: prompt, then greedy generation one token at a time
    common_batch bt(ctx_tgt), bd(ctx_dft);
    for (int i = 0; i < n_inp; i++) bt.add(inp[i], i, 0, i == n_inp - 1);
    if (!process(ctx_tgt, bt)) return 1;
    std::vector<llama_token> gen;
    float pt;
    llama_token tok = top1(llama_get_logits_ith(ctx_tgt, -1), n_vocab, &pt);
    for (int c = 0; c < n_gen; c++) {
        gen.push_back(tok);
        if (llama_vocab_is_eog(vocab, tok) || c + 1 == n_gen) break;
        bt.clear();
        bt.add(tok, n_inp + c, 0, true);
        if (!process(ctx_tgt, bt)) return 1;
        tok = top1(llama_get_logits_ith(ctx_tgt, -1), n_vocab, &pt);
    }

    // draft: the prompt, then the true tokens one at a time; prediction for generated position c
    std::string trace_path = std::string(out) + ".trace";
    FILE * ft = fopen(trace_path.c_str(), "w");
    fprintf(ft, "# match p   (draft top-1 vs target greedy token, given the true prefix); prompt %d tokens\n", n_inp);
    for (int i = 0; i < n_inp; i++) bd.add(inp[i], i, 0, i == n_inp - 1);
    if (!process(ctx_dft, bd)) return 1;
    int n_match = 0;
    for (size_t c = 0; c < gen.size(); c++) {
        float p;
        const llama_token d = top1(llama_get_logits_ith(ctx_dft, -1), n_vocab, &p);
        n_match += d == gen[c];
        fprintf(ft, "%d %.6f\n", d == gen[c] ? 1 : 0, p);
        if (c + 1 == gen.size()) break;
        bd.clear();
        bd.add(gen[c], n_inp + (int) c, 0, true);
        if (!process(ctx_dft, bd)) return 1;
    }
    fclose(ft);
    LOG_INF("trace: %zu tokens, draft top-1 matches %.1f%% -> %s\n", gen.size(), 100.0 * n_match / gen.size(), trace_path.c_str());

    if (do_table) {
        const int kv_tgt = (int) llama_memory_seq_pos_max(llama_get_memory(ctx_tgt), 0) + 1;
        LOG_INF("timing at kv: target %d, draft %d tokens\n", kv_tgt, (int) llama_memory_seq_pos_max(llama_get_memory(ctx_dft), 0) + 1);
        std::vector<std::vector<double>> tv(m_max + 1), td(m_max + 1);
        auto time_forward = [&](llama_context * ctx, common_batch & b, int M, bool all_logits) {
            const int p0 = (int) llama_memory_seq_pos_max(llama_get_memory(ctx), 0) + 1;
            b.clear();
            for (int i = 0; i < M; i++) b.add(gen[i % gen.size()], p0 + i, 0, all_logits || i == M - 1);
            const double t0 = now_ms();
            if (!process(ctx, b)) {
                exit(1);
            }
            const double t = now_ms() - t0;
            llama_memory_seq_rm(llama_get_memory(ctx), 0, p0, -1);
            return t;
        };
        for (int M = 1; M <= m_max; M++) {  // warm-up
            time_forward(ctx_tgt, bt, M, true);
            time_forward(ctx_dft, bd, M, false);
        }
        for (int r = 0; r < reps; r++) {
            for (int M = 1; M <= m_max; M++) {
                tv[M].push_back(time_forward(ctx_tgt, bt, M, true));
                td[M].push_back(time_forward(ctx_dft, bd, M, false));
            }
        }
        auto median = [](std::vector<double> v) {
            std::sort(v.begin(), v.end());
            return v[v.size() / 2];
        };
        std::string cost_path = std::string(out) + ".cost";
        FILE * fc = fopen(cost_path.c_str(), "w");
        fprintf(fc, "# kurn verify cost table: target %s, draft %s\n", params.model.path.c_str(), params.speculative.draft.mparams.path.c_str());
        fprintf(fc, "# threads %d (draft %d), kv %d tokens, median of %d reps; GGML_KURN=%s GGML_KURN_AMX=%s\n", params.cpuparams.n_threads,
                params_dft.cpuparams.n_threads, kv_tgt, reps, getenv("GGML_KURN") ? getenv("GGML_KURN") : "1",
                getenv("GGML_KURN_AMX") ? getenv("GGML_KURN_AMX") : "0");
        for (int M = 1; M <= m_max; M++) fprintf(fc, "verify %d %.4f\n", M, median(tv[M]));
        for (int M = 1; M <= m_max; M++) fprintf(fc, "draft %d %.4f\n", M, median(td[M]));
        fclose(fc);
        LOG_INF("cost table: M = 1..%d -> %s\n", m_max, cost_path.c_str());
        for (int M = 1; M <= m_max; M++) {
            LOG_INF("  M %2d  verify %8.2f ms (%.2fx)  draft %6.2f ms\n", M, median(tv[M]), median(tv[M]) / median(tv[1]), median(td[M]));
        }
    }
    llama_backend_free();
    return 0;
}
