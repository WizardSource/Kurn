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
        lam_num_ = lam_decay * lam_num_ + n_tokens;
        lam_den_ = lam_decay * lam_den_ + step_ms;
    }

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

}  // namespace kurn
