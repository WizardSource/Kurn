// Replay primitives for llama-server (kurn integration, see rsi/apply.sh):
//   - sha256 of files (model weights, cost tables) and strings, for the run manifest;
//   - a per-token hash chain over the committed tokens and their probabilities:
//       h_0 = sha256("kurn-replay-v1" || prompt token ids)
//       h_t = sha256(h_{t-1} || token id || prob bits of the sampled token || (id, prob bits) of the top-k)
//     Two runs with equal chains produced the same tokens with bit-identical probabilities; the first
//     differing per-token hash locates the first divergent position.
//   - a CPU fingerprint (model name and a hash of the feature flags) from /proc/cpuinfo.
// Integers and floats are hashed little-endian.
#pragma once

#include "hash/hash.h"

// written by rsi/apply.sh: KURN_INTEGRATION_SHA256 (sha256 of the kurn integration sources) and KURN_VERSION
#if defined(__has_include)
#if __has_include("kurn-build-info.h")
#include "kurn-build-info.h"
#endif
#endif

extern "C" {
#include "hash/sha256/sha256.h"
}

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace kurn {

inline std::string to_hex(const unsigned char * d, size_t n) {
    static const char hex[] = "0123456789abcdef";
    std::string out;
    out.reserve(2 * n);
    for (size_t i = 0; i < n; i++) {
        out += hex[d[i] >> 4];
        out += hex[d[i] & 0xf];
    }
    return out;
}

// empty string when the file cannot be read
inline std::string file_sha256(const std::string & path) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) {
        return "";
    }
    sha256_t st;
    sha256_init(&st);
    std::vector<unsigned char> buf(8u << 20);
    size_t n;
    while ((n = fread(buf.data(), 1, buf.size(), f)) > 0) {
        sha256_update(&st, buf.data(), n);
    }
    const bool ok = !ferror(f);
    fclose(f);
    if (!ok) {
        return "";
    }
    unsigned char d[SHA256_DIGEST_SIZE];
    sha256_final(&st, d);
    return to_hex(d, sizeof d);
}

inline std::string string_sha256(const std::string & s) {
    return hash_sha256_hex(s.data(), s.size());
}

class hash_chain {
public:
    void init(const std::vector<int32_t> & prompt, bool per_token) {
        sha256_t st;
        sha256_init(&st);
        static const char tag[] = "kurn-replay-v1";
        sha256_update(&st, (const unsigned char *) tag, sizeof tag - 1);
        for (int32_t id : prompt) {
            put_i32(&st, id);
        }
        sha256_final(&st, h_);
        n_ = 0;
        keep_ = per_token;
        per_token_.clear();
        active_ = true;
    }

    void reset() {
        active_ = false;
        n_ = 0;
        per_token_.clear();
    }

    bool active() const { return active_; }

    void add(int32_t token, float prob, const std::vector<std::pair<int32_t, float>> & top) {
        sha256_t st;
        sha256_init(&st);
        sha256_update(&st, h_, sizeof h_);
        put_i32(&st, token);
        put_f32(&st, prob);
        for (const auto & [id, p] : top) {
            put_i32(&st, id);
            put_f32(&st, p);
        }
        sha256_final(&st, h_);
        n_++;
        if (keep_) {
            per_token_.push_back(hex().substr(0, 16));
        }
    }

    std::string hex() const { return to_hex(h_, sizeof h_); }
    uint64_t n_tokens() const { return n_; }
    const std::vector<std::string> & per_token() const { return per_token_; }

private:
    unsigned char h_[SHA256_DIGEST_SIZE] = {};
    uint64_t n_ = 0;
    bool keep_ = false, active_ = false;
    std::vector<std::string> per_token_;

    static void put_i32(sha256_t * st, int32_t v) {
        unsigned char b[4];
        const uint32_t u = (uint32_t) v;
        for (int i = 0; i < 4; i++) b[i] = (unsigned char) (u >> (8 * i));
        sha256_update(st, b, 4);
    }
    static void put_f32(sha256_t * st, float v) {
        uint32_t u;
        memcpy(&u, &v, 4);
        put_i32(st, (int32_t) u);
    }
};

// {model name, sha256 of the flags line} of the first CPU in /proc/cpuinfo (empty strings elsewhere)
inline std::pair<std::string, std::string> cpu_fingerprint() {
    std::string name, flags;
    if (FILE * f = fopen("/proc/cpuinfo", "r")) {
        char line[8192];
        while (fgets(line, sizeof line, f) && (name.empty() || flags.empty())) {
            const char * colon = strchr(line, ':');
            if (!colon) continue;
            std::string val = colon + 1;
            while (!val.empty() && (val.back() == '\n' || val.back() == ' ')) val.pop_back();
            if (!val.empty() && val[0] == ' ') val.erase(0, 1);
            if (name.empty() && !strncmp(line, "model name", 10)) name = val;
            if (flags.empty() && !strncmp(line, "flags", 5)) flags = val;
        }
        fclose(f);
    }
    return { name, flags.empty() ? "" : string_sha256(flags) };
}

}  // namespace kurn
