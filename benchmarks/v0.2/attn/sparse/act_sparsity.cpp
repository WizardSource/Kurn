// SwiGLU activation sparsity in a llama.cpp model, measured and applied through the eval callback.
//
//   act_sparsity MODEL TEXT [--ctx 512] [--chunks N] [--threads 8] [--skip-chunks K] MODE...
//
// The text is cut into chunks of --ctx tokens; perplexity is scored on the second half of each
// chunk (as llama-perplexity does). Modes:
//   --stats OUT.csv          per-layer histograms of |h| (h = silu(gate)*up) and |silu(gate)|
//                            (log10 bins of 0.01), and the per-token fraction of h that can be
//                            dropped keeping ||h_dropped|| <= eps*||h|| (eps = .05 .1 .2 .3)
//   --dump X.bin --dump-tokens N   also write the FFN inputs (ffn_norm) of the first N tokens
//   --thr T.txt --kind h|gate|pred [--pred P.bin]
//                            zero h where |h| < t_l (h: oracle), |silu(gate)| < t_l (gate:
//                            needs the full gate product), or |silu(B_l A_l x)| < t_l (pred:
//                            low-rank predictor from P.bin); T.txt holds one threshold per layer
//   --moe-tau TAU [--moe-tensor ffn_moe_weights]   MoE: drop routed experts whose routing weight
//                            (the named tensor, e.g. ffn_moe_weights_norm when renormalised) is < TAU
//   --act NAME               activation tensor to observe (default ffn_swiglu; PLM: "ffn_sqr(relu)")
// Weight repacking is off (use_extra_bufts = false), so no AMX code runs.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "ggml-cpu.h"
#include "ggml.h"
#include "llama.h"

static const int NB = 1000;  // log10 bins over [1e-7, 1e3)
static const double EPS[4] = {0.05, 0.1, 0.2, 0.3};
static const int BATCHES[4] = {1, 4, 8, 16};

static int bin_of(float v) {
    v = fabsf(v);
    if (!(v > 1e-7f)) return 0;
    int b = (int) ((log10f(v) + 7.0f) * 100.0f);
    return b < 0 ? 0 : b >= NB ? NB - 1 : b;
}

struct layer_stats {
    std::vector<double> hist_h = std::vector<double>(NB), hist_g = std::vector<double>(NB);
    double drop_eps[4] = {0, 0, 0, 0}, rows = 0;
    double zeroed = 0, total = 0, err2 = 0, norm2 = 0;
    double union_active[4] = {0, 0, 0, 0}, union_groups[4] = {0, 0, 0, 0};
    double moe_dropped = 0, moe_total = 0;
};

struct state {
    int n_layer = 0;
    std::vector<layer_stats> L;
    bool stats = false;
    std::string kind;  // "", h, gate, pred
    std::vector<float> thr;
    // predictor: per layer A [r][d], B [dff][r]
    int pr = 0, pd = 0, pff = 0;
    std::vector<std::vector<float>> PA, PB;
    std::vector<std::vector<uint8_t>> mask;  // per layer, [n_tok][dff] 1 = drop
    float moe_tau = -1;
    std::string moe_name = "ffn_moe_weights";
    std::string act = "ffn_swiglu";
    FILE * dump = nullptr;
    int dump_tokens = 0;
    std::vector<int> dumped;
    bool scoring = true;
};

static int layer_of(const char * name, const char * prefix) {
    size_t n = strlen(prefix);
    if (strncmp(name, prefix, n) != 0 || name[n] != '-') return -1;
    return atoi(name + n + 1);
}

static void union_density(state & S, int il, const uint8_t * drop, int64_t dff, int64_t ntok) {
    layer_stats & ls = S.L[il];
    for (int bi = 0; bi < 4; bi++) {
        const int b = BATCHES[bi];
        for (int64_t t0 = 0; t0 + b <= ntok; t0 += b) {
            int64_t act = 0;
            for (int64_t i = 0; i < dff; i++) {
                bool any = false;
                for (int t = 0; t < b && !any; t++) any = !drop[(t0 + t) * dff + i];
                act += any;
            }
            ls.union_active[bi] += (double) act / dff;
            ls.union_groups[bi] += 1;
        }
    }
}

static bool cb(struct ggml_tensor * t, bool ask, void * ud) {
    state & S = *(state *) ud;
    const char * nm = t->name;
    int il;
    if (ask) {
        if (layer_of(nm, S.act.c_str()) >= 0 || layer_of(nm, "ffn_moe_swiglu") >= 0) return S.stats || !S.kind.empty();
        if (layer_of(nm, "ffn_gate") >= 0) return S.stats || S.kind == "gate";
        if (layer_of(nm, "ffn_norm") >= 0) return S.kind == "pred" || S.dump;
        if (layer_of(nm, S.moe_name.c_str()) >= 0) return S.moe_tau >= 0;
        return false;
    }
    if (t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t)) return true;
    float * x = (float *) t->data;
    const int64_t ne0 = t->ne[0], rows = ggml_nrows(t);

    if ((il = layer_of(nm, "ffn_norm")) >= 0) {
        if (S.dump && S.dumped[il] < S.dump_tokens) {
            int64_t n = std::min<int64_t>(rows, S.dump_tokens - S.dumped[il]);
            int32_t hdr[3] = {il, (int32_t) n, (int32_t) ne0};
            fwrite(hdr, sizeof hdr, 1, S.dump);
            fwrite(x, sizeof(float), n * ne0, S.dump);
            S.dumped[il] += (int) n;
        }
        if (S.kind == "pred") {
            const int r = S.pr, d = S.pd, dff = S.pff;
            if (ne0 != d) return true;
            S.mask[il].assign(rows * dff, 0);
            const float th = S.thr[il];
            const float * A = S.PA[il].data(), * B = S.PB[il].data();
#pragma omp parallel for
            for (int64_t tk = 0; tk < rows; tk++) {
                std::vector<float> z(r);
                const float * xt = x + tk * d;
                for (int j = 0; j < r; j++) {
                    float s = 0;
                    for (int k = 0; k < d; k++) s += A[(size_t) j * d + k] * xt[k];
                    z[j] = s;
                }
                for (int i = 0; i < dff; i++) {
                    float g = 0;
                    for (int j = 0; j < r; j++) g += B[(size_t) i * r + j] * z[j];
                    const float a = g / (1.0f + expf(-g));
                    S.mask[il][tk * dff + i] = fabsf(a) < th;
                }
            }
        }
        return true;
    }
    if ((il = layer_of(nm, "ffn_gate")) >= 0) {
        if (S.stats) {
            layer_stats & ls = S.L[il];
            for (int64_t i = 0; i < rows * ne0; i++) ls.hist_g[bin_of(x[i] / (1.0f + expf(-x[i])))] += 1;
        }
        if (S.kind == "gate") {
            S.mask[il].assign(rows * ne0, 0);
            const float th = S.thr[il];
            for (int64_t i = 0; i < rows * ne0; i++) S.mask[il][i] = fabsf(x[i] / (1.0f + expf(-x[i]))) < th;
        }
        return true;
    }
    if (S.moe_tau >= 0 && (il = layer_of(nm, S.moe_name.c_str())) >= 0) {
        // [1, n_used, n_tok]: drop small routing weights (the expert's output is then zero)
        layer_stats & ls = S.L[il];
        for (int64_t i = 0; i < ggml_nelements(t); i++) {
            ls.moe_total += 1;
            if (x[i] < S.moe_tau) { x[i] = 0; ls.moe_dropped += 1; }
        }
        return true;
    }
    il = layer_of(nm, S.act.c_str());
    if (il < 0) il = layer_of(nm, "ffn_moe_swiglu");
    if (il < 0 || il >= S.n_layer) return true;
    layer_stats & ls = S.L[il];
    if (S.stats) {
        std::vector<double> hb(NB);
        for (int64_t r = 0; r < rows; r++) {
            const float * h = x + r * ne0;
            std::fill(hb.begin(), hb.end(), 0.0);
            double tot = 0;
            for (int64_t i = 0; i < ne0; i++) {
                const int b = bin_of(h[i]);
                ls.hist_h[b] += 1;
                hb[b] += (double) h[i] * h[i];
                tot += (double) h[i] * h[i];
            }
            // count of the smallest entries whose squared sum stays under eps^2 * total
            std::vector<double> cnt(NB, 0);
            for (int64_t i = 0; i < ne0; i++) cnt[bin_of(h[i])] += 1;
            for (int e = 0; e < 4; e++) {
                double acc = 0, n = 0;
                for (int b = 0; b < NB; b++) {
                    if (acc + hb[b] > EPS[e] * EPS[e] * tot) break;
                    acc += hb[b];
                    n += cnt[b];
                }
                ls.drop_eps[e] += n / ne0;
            }
            ls.rows += 1;
        }
    }
    if (!S.kind.empty()) {
        const int64_t n = rows * ne0;
        std::vector<uint8_t> drop(n);
        if (S.kind == "h") {
            for (int64_t i = 0; i < n; i++) drop[i] = fabsf(x[i]) < S.thr[il];
        } else {
            if ((int64_t) S.mask[il].size() != n) return true;  // MoE / shape mismatch: leave untouched
            drop = S.mask[il];
        }
        for (int64_t i = 0; i < n; i++) {
            ls.norm2 += (double) x[i] * x[i];
            if (drop[i]) {
                ls.err2 += (double) x[i] * x[i];
                ls.zeroed += 1;
                x[i] = 0;
            }
        }
        ls.total += n;
        if (t->ne[2] <= 1) union_density(S, il, drop.data(), ne0, rows);
    }
    return true;
}

static const char * arg(int argc, char ** argv, const char * k, const char * def) {
    for (int i = 1; i < argc - 1; i++)
        if (!strcmp(argv[i], k)) return argv[i + 1];
    return def;
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: act_sparsity MODEL TEXT [options]\n");
        return 1;
    }
    const int n_ctx = atoi(arg(argc, argv, "--ctx", "512"));
    const int max_chunks = atoi(arg(argc, argv, "--chunks", "1000"));
    const int skip_chunks = atoi(arg(argc, argv, "--skip-chunks", "0"));
    const int threads = atoi(arg(argc, argv, "--threads", "8"));
    const char * stats_out = arg(argc, argv, "--stats", nullptr);
    state S;
    S.stats = stats_out != nullptr;
    S.kind = arg(argc, argv, "--kind", "");
    S.moe_tau = (float) atof(arg(argc, argv, "--moe-tau", "-1"));
    S.moe_name = arg(argc, argv, "--moe-tensor", "ffn_moe_weights");
    S.act = arg(argc, argv, "--act", "ffn_swiglu");
    if (const char * d = arg(argc, argv, "--dump", nullptr)) {
        S.dump = fopen(d, "wb");
        S.dump_tokens = atoi(arg(argc, argv, "--dump-tokens", "1024"));
    }

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.use_extra_bufts = false;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) return 1;
    S.n_layer = llama_model_n_layer(model);
    S.L.resize(S.n_layer);
    S.mask.resize(S.n_layer);
    S.dumped.assign(S.n_layer, 0);
    if (!S.kind.empty() && S.kind != "h" && S.kind != "gate" && S.kind != "pred") {
        fprintf(stderr, "unknown --kind %s\n", S.kind.c_str());
        return 1;
    }
    if (!S.kind.empty()) {
        FILE * f = fopen(arg(argc, argv, "--thr", ""), "r");
        if (!f) { fprintf(stderr, "--kind needs --thr\n"); return 1; }
        float v;
        while (fscanf(f, "%f", &v) == 1) S.thr.push_back(v);
        fclose(f);
        if ((int) S.thr.size() < S.n_layer) { fprintf(stderr, "need %d thresholds\n", S.n_layer); return 1; }
    }
    if (S.kind == "pred") {
        FILE * f = fopen(arg(argc, argv, "--pred", ""), "rb");
        if (!f) { fprintf(stderr, "--kind pred needs --pred\n"); return 1; }
        int32_t hdr[4];
        if (fread(hdr, sizeof hdr, 1, f) != 1) return 1;
        S.pr = hdr[1]; S.pd = hdr[2]; S.pff = hdr[3];
        S.PA.resize(S.n_layer); S.PB.resize(S.n_layer);
        for (int l = 0; l < hdr[0] && l < S.n_layer; l++) {
            S.PA[l].resize((size_t) S.pr * S.pd);
            S.PB[l].resize((size_t) S.pff * S.pr);
            if (fread(S.PA[l].data(), 4, S.PA[l].size(), f) != S.PA[l].size()) return 1;
            if (fread(S.PB[l].data(), 4, S.PB[l].size(), f) != S.PB[l].size()) return 1;
        }
        fclose(f);
    }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n_ctx;
    cp.n_batch = n_ctx;
    cp.n_ubatch = n_ctx;
    cp.n_threads = cp.n_threads_batch = threads;
    cp.cb_eval = cb;
    cp.cb_eval_user_data = &S;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) return 1;
    // threads pinned 1:1 to vCPUs: ggml's AMX path loses tile state on migration on this VM
    ggml_threadpool_params tpp = ggml_threadpool_params_default(threads);
    for (int i = 0; i < threads; i++) tpp.cpumask[i] = true;
    tpp.strict_cpu = true;
    ggml_threadpool * tp = ggml_threadpool_new(&tpp);
    llama_attach_threadpool(ctx, tp, tp);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    FILE * tf = fopen(argv[2], "rb");
    std::string text;
    char buf[65536];
    size_t got;
    while ((got = fread(buf, 1, sizeof buf, tf)) > 0) text.append(buf, got);
    fclose(tf);
    std::vector<llama_token> toks(text.size() + 16);
    const int nt = llama_tokenize(vocab, text.c_str(), (int) text.size(), toks.data(), (int) toks.size(), true, false);
    toks.resize(nt);
    const int n_chunks = std::min(max_chunks, nt / n_ctx - skip_chunks);
    fprintf(stderr, "%d tokens, %d chunks of %d\n", nt, n_chunks, n_ctx);

    double nll = 0;
    int64_t count = 0;
    llama_batch batch = llama_batch_init(n_ctx, 0, 1);
    for (int c = 0; c < n_chunks; c++) {
        const int start = (c + skip_chunks) * n_ctx;
        llama_memory_clear(llama_get_memory(ctx), true);
        batch.n_tokens = n_ctx;
        for (int i = 0; i < n_ctx; i++) {
            batch.token[i] = toks[start + i];
            if (i == 0 && llama_vocab_get_add_bos(vocab)) batch.token[i] = llama_vocab_bos(vocab);
            batch.pos[i] = i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = i >= n_ctx / 2 - 1;
        }
        if (llama_decode(ctx, batch)) { fprintf(stderr, "decode failed\n"); return 1; }
        for (int i = n_ctx / 2 - 1; i < n_ctx - 1; i++) {
            const float * lg = llama_get_logits_ith(ctx, i);
            double mx = -1e30, se = 0;
            for (int v = 0; v < n_vocab; v++) mx = std::max(mx, (double) lg[v]);
            for (int v = 0; v < n_vocab; v++) se += exp((double) lg[v] - mx);
            nll += -((double) lg[toks[start + i + 1]] - mx - log(se));
            count++;
        }
        fprintf(stderr, "chunk %d ppl %.4f\n", c, exp(nll / count));
    }
    printf("PPL %.4f tokens %lld chunks %d\n", exp(nll / count), (long long) count, n_chunks);

    if (!S.kind.empty()) {
        double z = 0, tot = 0, e2 = 0, n2 = 0, ua[4] = {0}, ug[4] = {0};
        for (int l = 0; l < S.n_layer; l++) {
            const layer_stats & ls = S.L[l];
            z += ls.zeroed; tot += ls.total; e2 += ls.err2; n2 += ls.norm2;
            for (int b = 0; b < 4; b++) { ua[b] += ls.union_active[b]; ug[b] += ls.union_groups[b]; }
            printf("layer %d sparsity %.4f relerr %.4f union_density b1 %.4f b4 %.4f b8 %.4f b16 %.4f\n", l,
                   ls.total ? ls.zeroed / ls.total : 0, ls.norm2 ? sqrt(ls.err2 / ls.norm2) : 0,
                   ls.union_groups[0] ? ls.union_active[0] / ls.union_groups[0] : 0,
                   ls.union_groups[1] ? ls.union_active[1] / ls.union_groups[1] : 0,
                   ls.union_groups[2] ? ls.union_active[2] / ls.union_groups[2] : 0,
                   ls.union_groups[3] ? ls.union_active[3] / ls.union_groups[3] : 0);
        }
        printf("ALL sparsity %.4f relerr %.4f union_density b1 %.4f b4 %.4f b8 %.4f b16 %.4f\n", tot ? z / tot : 0,
               n2 ? sqrt(e2 / n2) : 0, ug[0] ? ua[0] / ug[0] : 0, ug[1] ? ua[1] / ug[1] : 0, ug[2] ? ua[2] / ug[2] : 0,
               ug[3] ? ua[3] / ug[3] : 0);
    }
    if (S.moe_tau >= 0) {
        double d = 0, t = 0;
        for (auto & ls : S.L) { d += ls.moe_dropped; t += ls.moe_total; }
        printf("MOE tau %.4f dropped %.4f of routed expert evaluations\n", S.moe_tau, t ? d / t : 0);
    }
    if (stats_out) {
        FILE * f = fopen(stats_out, "w");
        fprintf(f, "layer,kind,bin_lo,count\n");
        for (int l = 0; l < S.n_layer; l++) {
            for (int b = 0; b < NB; b++) {
                if (S.L[l].hist_h[b]) fprintf(f, "%d,h,%.2f,%.0f\n", l, -7 + b / 100.0, S.L[l].hist_h[b]);
                if (S.L[l].hist_g[b]) fprintf(f, "%d,silu_gate,%.2f,%.0f\n", l, -7 + b / 100.0, S.L[l].hist_g[b]);
            }
            for (int e = 0; e < 4; e++)
                fprintf(f, "%d,drop_eps,%.2f,%.6f\n", l, EPS[e], S.L[l].rows ? S.L[l].drop_eps[e] / S.L[l].rows : 0);
        }
        fclose(f);
    }
    if (S.dump) fclose(S.dump);
    llama_batch_free(batch);
    llama_free(ctx);
    ggml_threadpool_free(tp);
    llama_model_free(model);
    return 0;
}
