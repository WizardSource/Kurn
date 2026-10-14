// Cost-aware speculative verify width for llama.cpp's speculative decoding on the KURN buffer type.
// Header-only, no ggml / llama dependency. Line-by-line twin of kurn.specwidth.WidthPolicy (Python),
// which documents the method; tests/test_specwidth.py checks that the two agree.
//
// Each step: n_cap() bounds the draft length from the running acceptance rate; keep_drafting(probs) is
// asked after every drafted token (draft confidence = top-1 probability of the draft sampler); truncate()
// cuts the draft to the prefix whose verify width maximises expected accepted tokens - lambda * time;
// observe() feeds the verification result back. lambda is the running throughput (tokens / ms).
//
// The cost table (`kurn-spec-calib` writes it; `kurn specwidth show` prints it) has lines
//   verify M MS    target forward with M tokens (logits for all M)
//   draft M MS     draft forward with M tokens
// and `#` comments.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace kurn {

struct verify_cost_table {
    std::vector<double> verify_ms;  // [M - 1]
    std::vector<double> draft_ms;   // [M - 1], may be empty

    int m_max() const { return (int) verify_ms.size(); }
    double verify(int M) const { return verify_ms[M - 1]; }

    double draft_step() const {
        if (draft_ms.empty()) {
            return 0.0;
        }
        const size_t n = draft_ms.size();
        const double slope = n > 1 ? (draft_ms[n - 1] - draft_ms[0]) / (double) (n - 1) : 0.0;
        return draft_ms[0] + std::max(0.0, slope);
    }

    // false (and err set) when the file is missing or malformed
    bool load(const char * path, std::string * err = nullptr) {
        FILE * f = fopen(path, "r");
        if (!f) {
            if (err) *err = std::string("cannot open ") + path;
            return false;
        }
        std::vector<std::pair<int, double>> v, d;
        char line[256];
        while (fgets(line, sizeof line, f)) {
            char kind[16];
            int m;
            double ms;
            if (line[0] == '#' || sscanf(line, "%15s %d %lf", kind, &m, &ms) != 3) {
                continue;
            }
            if (!strcmp(kind, "verify")) v.emplace_back(m, ms);
            else if (!strcmp(kind, "draft")) d.emplace_back(m, ms);
        }
        fclose(f);
        auto fill = [](std::vector<std::pair<int, double>> & src, std::vector<double> & dst) {
            std::sort(src.begin(), src.end());
            dst.clear();
            for (size_t i = 0; i < src.size(); i++) {
                if (src[i].first != (int) i + 1 || !(src[i].second > 0) || !std::isfinite(src[i].second)) {
                    return false;
                }
                dst.push_back(src[i].second);
            }
            return true;
        };
        if (!fill(v, verify_ms) || verify_ms.empty() || !fill(d, draft_ms)) {
            if (err) *err = std::string(path) + ": verify/draft rows must cover M = 1..n with positive times";
            return false;
        }
        return true;
    }
};

class width_policy {
public:
    static constexpr int NB = 10;
    static constexpr double EDGES[NB + 1] = {0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.999, 1.0};

    double alpha0 = 0.6, prior = 4.0, decay = 0.98, lam_decay = 0.9;
    int probe_every = 16;
    bool use_conf = true;

    width_policy() = default;
    width_policy(const verify_cost_table & t, int k_max = -1) { init(t, k_max); }

    void init(const verify_cost_table & t, int k_max = -1) {
        t_ = t;
        k_max_ = std::min(k_max >= 0 ? k_max : t.m_max() - 1, t.m_max() - 1);
        draft_ms_ = t.draft_step();
        std::fill(bs_, bs_ + NB, 0.0);
        std::fill(bn_, bn_ + NB, 0.0);
        gs_ = gn_ = lam_num_ = lam_den_ = 0.0;
        idle_ = 0;
    }

    int k_max() const { return k_max_; }
    const verify_cost_table & table() const { return t_; }

    double alpha() const { return (gs_ + prior * alpha0) / (gn_ + prior); }

    // p < 0: no draft confidence (e.g. n-gram drafters)
    double acc(double p) const {
        if (p < 0 || !use_conf) {
            return alpha();
        }
        const int b = bin(p);
        const double mid = 0.5 * (EDGES[b] + EDGES[b + 1]);
        return (bs_[b] + prior * mid) / (bn_[b] + prior);
    }

    double lam() const { return lam_den_ > 0 ? lam_num_ / lam_den_ : 1.0 / t_.verify(1); }

    bool keep_drafting(const std::vector<float> & probs) const {
        const int j = (int) probs.size();
        if (j >= k_max_) {
            return false;
        }
        double q, g;
        const std::vector<double> u = values(probs, q, g);
        return best_future(j, q, g) > *std::max_element(u.begin(), u.end()) + 1e-9;
    }

    int truncate(const std::vector<float> & probs) const {
        double q, g;
        const std::vector<double> u = values(probs, q, g);
        int best = 0;
        for (int k = 1; k < (int) u.size(); k++) {
            if (u[k] > u[best] + 1e-9) {
                best = k;
            }
        }
        return best;
    }

    int n_cap() const {
        const double l = lam(), a = alpha();
        int best_k = 0;
        double best = -l * t_.verify(1), s = 0.0, r = 1.0;
        for (int k = 1; k <= k_max_; k++) {
            r *= a;
            s += r;
            const double v = s - l * (t_.verify(k + 1) + k * draft_ms_);
            if (v > best + 1e-9) {
                best_k = k;
                best = v;
            }
        }
        if (best_k == 0 && idle_ + 1 >= probe_every) {
            return 1;
        }
        return best_k;
    }

    void observe(const std::vector<float> & probs, int n_accepted, double step_ms, int n_tokens) {
        observe_acc(probs, n_accepted);
        observe_time(step_ms, n_tokens);
    }

    // acceptance feedback only (k = 0: a step this arm did not draft)
    void observe_acc(const std::vector<float> & probs, int n_accepted) {
        const int k = (int) probs.size();
        idle_ = k ? 0 : idle_ + 1;
        if (k) {
            for (int b = 0; b < NB; b++) {
                bs_[b] *= decay;
                bn_[b] *= decay;
            }
            gs_ *= decay;
            gn_ *= decay;
        }
        for (int i = 0; i < std::min(k, n_accepted + 1); i++) {
            const double ok = i < n_accepted ? 1.0 : 0.0;
            gs_ += ok;
            gn_ += 1.0;
            if (probs[i] >= 0) {
                const int b = bin(probs[i]);
                bs_[b] += ok;
                bn_[b] += 1.0;
            }
        }
    }

    // throughput feedback only: the step took step_ms and committed n_tokens
    void observe_time(double step_ms, int n_tokens) {
        lam_num_ = lam_decay * lam_num_ + n_tokens;
        lam_den_ = lam_decay * lam_den_ + step_ms;
    }

    // value of drafting k tokens at the global acceptance rate, k = 0: no draft (-lambda * verify(1))
    double value_at(int k, double draft_ms) const {
        const double l = lam(), a = alpha();
        double s = 0.0, r = 1.0;
        for (int i = 0; i < k; i++) {
            r *= a;
            s += r;
        }
        return s - l * (t_.verify(k + 1) + k * draft_ms);
    }

    // best value over k = 0..k_max at the global acceptance rate, with this arm's draft time
    double best_value() const {
        double best = value_at(0, draft_ms_);
        for (int k = 1; k <= k_max_; k++) {
            best = std::max(best, value_at(k, draft_ms_));
        }
        return best;
    }

    int idle() const { return idle_; }

private:
    verify_cost_table t_;
    int k_max_ = 0, idle_ = 0;
    double draft_ms_ = 0.0;
    double bs_[NB] = {}, bn_[NB] = {};
    double gs_ = 0.0, gn_ = 0.0, lam_num_ = 0.0, lam_den_ = 0.0;

    static int bin(double p) {
        for (int b = 0; b < NB - 1; b++) {
            if (p < EDGES[b + 1]) {
                return b;
            }
        }
        return NB - 1;
    }

    std::vector<double> values(const std::vector<float> & probs, double & q, double & g) const {
        const double l = lam();
        q = 1.0;
        g = 0.0;
        std::vector<double> u;
        u.reserve(probs.size() + 1);
        u.push_back(-l * t_.verify(1));
        for (float p : probs) {
            q *= acc(p);
            g += q;
            u.push_back(g - l * t_.verify((int) u.size() + 1));
        }
        return u;
    }

    double best_future(int j, double q, double g) const {
        const double l = lam(), a = alpha();
        double best = -std::numeric_limits<double>::infinity(), s = 0.0, r = 1.0;
        for (int k = j + 1; k <= k_max_; k++) {
            r *= a;
            s += r;
            best = std::max(best, g + q * s - l * (t_.verify(k + 1) + (k - j) * draft_ms_));
        }
        return best;
    }
};

// Two drafting arms chosen per step by the same objective (expected accepted tokens - lambda * time):
//   retrieval: n-gram / trajectory lookup in the prompt and the generated history. Costs ~no draft time
//              and has no confidences, so its acceptance is one online rate (decayed like width_policy's);
//   model:     a draft model, the width_policy above (confidence bins, keep_drafting / truncate).
// Both arms see every step's time, so they share one throughput estimate lambda. Each step:
//   take_retrieval(L) when a lookup proposes L tokens: the prefix length worth verifying, 0 = reject it
//                     (then the model arm may draft); a candidate is taken when its best value is at
//                     least the model arm's best value, which includes not drafting at all;
//   allow_model()     before the draft model runs: worth drafting now, or a probe is due;
//   observe(arm, ...) after verification.
class arm_selector {
public:
    enum arm { NONE = 0, RETRIEVAL = 1, MODEL = 2 };

    width_policy model, retr;
    bool has_model = false, has_retr = false;
    double retr_ms = 0.0;  // lookup time per drafted token (negligible; measured lookups are microseconds)

    void init(const verify_cost_table & t, int k_max_model, int k_max_retr, bool with_model, bool with_retr, double retr_alpha0 = 0.5) {
        has_model = with_model;
        has_retr = with_retr;
        model.init(t, k_max_model);
        verify_cost_table tr = t;
        tr.draft_ms.clear();
        retr.alpha0 = retr_alpha0;
        retr.use_conf = false;
        retr.init(tr, k_max_retr);
    }

    double lam() const { return model.lam(); }

    double model_value() const { return has_model ? model.best_value() : model.value_at(0, 0.0); }

    int take_retrieval(int L) const {
        if (!has_retr || L <= 0) {
            return 0;
        }
        const int k_lim = std::min(L, retr.k_max());
        int best_k = 0;
        double best = retr.value_at(0, retr_ms);
        for (int k = 1; k <= k_lim; k++) {
            const double v = retr.value_at(k, retr_ms);
            if (v > best + 1e-9) {
                best_k = k;
                best = v;
            }
        }
        if (best_k == 0) {
            return retr.idle() + 1 >= retr.probe_every ? 1 : 0;
        }
        return best + 1e-9 >= model_value() ? best_k : 0;
    }

    // catchup_ms: decode time the draft model needs before it can draft (lazy sync: tokens committed since it last
    // ran). It is paid once, so it is weighed against the model arm's gain over the request's remaining
    // horizon_tokens (<= 0: unknown, the catch-up is ignored). Probes run only when the catch-up costs less
    // than one token of throughput.
    bool allow_model(double catchup_ms = 0.0, int horizon_tokens = -1) const {
        if (!has_model) {
            return false;
        }
        const double none = model.value_at(0, 0.0);
        int kb = 0;
        double vb = none;
        for (int k = 1; k <= model.k_max(); k++) {
            const double v = model.value_at(k, model.table().draft_step());
            if (v > vb + 1e-9) {
                kb = k;
                vb = v;
            }
        }
        const double cost = horizon_tokens > 0 ? lam() * catchup_ms : 0.0;
        double steps = 1e9;
        if (horizon_tokens > 0) {
            double e = 1.0, r = 1.0;
            for (int i = 0; i < kb; i++) {
                r *= model.alpha();
                e += r;
            }
            steps = horizon_tokens / e;
        }
        if ((vb - none) * steps > cost + 1e-9) {
            return true;
        }
        return model.idle() + 1 >= model.probe_every && cost <= 1.0;
    }

    void observe(arm used, const std::vector<float> & probs, int n_accepted, double step_ms, int n_tokens) {
        static const std::vector<float> none;
        model.observe_acc(used == MODEL ? probs : none, used == MODEL ? n_accepted : 0);
        retr.observe_acc(used == RETRIEVAL ? probs : none, used == RETRIEVAL ? n_accepted : 0);
        model.observe_time(step_ms, n_tokens);
        retr.observe_time(step_ms, n_tokens);
    }
};

}  // namespace kurn
