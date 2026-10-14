// kvz codec tests: exact round trips for every layout over row counts that hit every tail case, KV-like and adversarial
// data (constant, uniform random, a single rare value, outlier channels), thread counts, and decoding of damaged blocks
// (truncated, bytes changed) without reading or writing out of bounds. Exit code = failures.
//   g++ -O1 -g -fsanitize=address,undefined -std=c++17 test_kvz.cpp llama-kvz.cpp -lpthread -o test_kvz && ./test_kvz
#include "llama-kvz.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

static int n_pass = 0, n_fail = 0;

static void check(bool ok, const char * what, int layout, size_t rows, size_t rb, int kind) {
    if (ok) {
        n_pass++;
    } else {
        n_fail++;
        printf("FAIL %s: layout %d rows %zu row_bytes %zu data %d\n", what, layout, rows, rb, kind);
    }
}

static uint16_t to_f16(float f) {
    // round-to-nearest f32 -> f16 for normal range values (enough for test data)
    uint32_t x;
    memcpy(&x, &f, 4);
    const uint32_t s = (x >> 16) & 0x8000;
    int e = (int) ((x >> 23) & 0xFF) - 127 + 15;
    uint32_t m = x & 0x7FFFFF;
    if (e <= 0) return (uint16_t) s;
    if (e >= 31) return (uint16_t) (s | 0x7BFF);
    uint32_t h = s | (e << 10) | (m >> 13);
    if ((m >> 12) & 1) h++;
    return (uint16_t) h;
}

// kind: 0 KV-like (per-channel scales, outlier channels), 1 uniform random bytes, 2 constant, 3 one rare value
static std::vector<uint8_t> make(kvz_layout l, size_t rows, size_t rb, int kind, std::mt19937 & g) {
    std::vector<uint8_t> d(rows * rb);
    std::uniform_int_distribution<int> byte(0, 255);
    std::normal_distribution<float> nd(0, 1);
    if (kind == 1) {
        for (auto & b : d) b = (uint8_t) byte(g);
        return d;
    }
    if (kind == 2) {
        std::fill(d.begin(), d.end(), 0x5A);
        return d;
    }
    if (kind == 3) {
        std::fill(d.begin(), d.end(), 0);
        if (!d.empty()) d[d.size() / 2] = 0xFF;
        return d;
    }
    for (size_t r = 0; r < rows; ++r) {
        uint8_t * row = &d[r * rb];
        if (l == KVZ_PLANES2 || l == KVZ_PLANES4 || l == KVZ_RAW) {
            const size_t w = l == KVZ_PLANES4 ? 4 : 2, n = rb / w;
            for (size_t c = 0; c < n; ++c) {
                const float scale = (c % 37 == 0) ? 40.0f : 0.5f + (c % 7);
                const float v = nd(g) * scale + (c % 11 == 0 ? 20.0f : 0.0f);
                if (w == 2) {
                    const uint16_t h = to_f16(v);
                    memcpy(row + 2 * c, &h, 2);
                } else {
                    memcpy(row + 4 * c, &v, 4);
                }
            }
            for (size_t k = n * w; k < rb; ++k) row[k] = (uint8_t) byte(g);
        } else if (l == KVZ_Q8_0) {
            for (size_t b = 0; b < rb / 34; ++b) {
                const uint16_t h = to_f16(0.01f + 0.001f * (b % 5) + 0.0001f * nd(g));
                memcpy(row + 34 * b, &h, 2);
                for (int j = 0; j < 32; ++j) {
                    const float v = nd(g) * (j % 9 == 0 ? 3.0f : 1.0f) * 30.0f;
                    row[34 * b + 2 + j] = (uint8_t) (int8_t) std::max(-127.0f, std::min(127.0f, std::round(v)));
                }
            }
        } else {
            for (size_t b = 0; b < rb / 18; ++b) {
                const uint16_t h = to_f16(0.02f + 0.0001f * nd(g));
                memcpy(row + 18 * b, &h, 2);
                for (int j = 0; j < 16; ++j) {
                    auto q = [&]() { return (uint8_t) std::max(0.0f, std::min(15.0f, std::round(8 + nd(g) * 2.5f))); };
                    row[18 * b + 2 + j] = (uint8_t) (q() | (q() << 4));
                }
            }
        }
    }
    return d;
}

static void roundtrip(kvz_layout l, size_t rows, size_t rb, int kind, int nt, std::mt19937 & g, bool fuzz) {
    const auto src = make(l, rows, rb, kind, g);
    std::vector<uint8_t> enc(kvz_bound(src.size()));
    const size_t n = kvz_encode(l, src.data(), rows, rb, enc.data(), nt);
    check(n <= kvz_bound(src.size()), "bound", l, rows, rb, kind);
    std::vector<uint8_t> dec(src.size() + 1, 0xCC);
    const bool ok = kvz_decode(enc.data(), n, dec.data(), rows, rb, nt);
    check(ok && (src.empty() || memcmp(dec.data(), src.data(), src.size()) == 0), "roundtrip", l, rows, rb, kind);
    check(dec[src.size()] == 0xCC, "no write past dst", l, rows, rb, kind);
    const bool fits = l == KVZ_PLANES2 ? rb % 2 == 0 : l == KVZ_PLANES4 ? rb % 4 == 0 : l == KVZ_Q8_0 ? rb % 34 == 0 : l == KVZ_Q4_0 ? rb % 18 == 0 : false;
    if (fits && src.size() >= 4096) { // smaller blocks are dominated by the header
        if (kind == 0) {
            check(n < src.size(), "KV-like data compresses", l, rows, rb, kind);
        }
        if (kind == 2) {
            check(n < src.size() / 8, "constant data compresses", l, rows, rb, kind);
        }
    }
    // other shapes must be rejected
    check(!kvz_decode(enc.data(), n, dec.data(), rows + 1, rb, nt), "rejects wrong rows", l, rows, rb, kind);
    if (!fuzz) {
        return;
    }
    // damaged blocks: any result is fine as long as nothing outside the buffers is touched (ASan / UBSan)
    std::vector<uint8_t> bad;
    for (size_t cut : {(size_t) 0, (size_t) 3, n / 2, n - 1}) {
        if (cut >= n) continue;
        bad.assign(enc.begin(), enc.begin() + cut);
        check(!kvz_decode(bad.data(), bad.size(), dec.data(), rows, rb, nt), "rejects truncated", l, rows, rb, kind);
    }
    std::uniform_int_distribution<size_t> pos(0, n - 1);
    std::uniform_int_distribution<int> byte(1, 255);
    for (int t = 0; t < 40; ++t) {
        bad = enc;
        const int k = 1 + t % 3;
        for (int j = 0; j < k; ++j) bad[pos(g)] ^= (uint8_t) byte(g);
        (void) kvz_decode(bad.data(), bad.size(), dec.data(), rows, rb, nt);
    }
    n_pass++;
}

int main(int argc, char ** argv) {
    const bool quick = argc > 1 && !strcmp(argv[1], "quick");
    std::mt19937 g(1234);
    const size_t rows_list[] = {0, 1, 2, 3, 4, 5, 7, 31, 32, 33, 63, 64, 65, 100, 257, 1000, 4099};
    struct lay { kvz_layout l; std::vector<size_t> rb; };
    const lay lays[] = {
        {KVZ_PLANES2, {2, 34, 256, 2048}},
        {KVZ_PLANES4, {4, 260}},
        {KVZ_Q8_0, {34, 34 * 4, 34 * 32}},
        {KVZ_Q4_0, {18, 18 * 4, 18 * 32}},
        {KVZ_RAW, {1, 77}},
        {KVZ_PLANES2, {3}}, // does not fit: stored raw
    };
    for (const auto & L : lays) {
        for (size_t rb : L.rb) {
            for (size_t rows : rows_list) {
                if (quick && rows > 257) continue;
                if (rows * rb > (size_t) 8 << 20) continue;
                for (int kind = 0; kind < 4; ++kind) {
                    for (int nt : {1, 3, 8}) {
                        roundtrip(L.l, rows, rb, kind, nt, g, nt == 3 && rows > 0 && rows * rb < (1 << 18));
                    }
                }
            }
        }
    }
    // a larger KV-like block with every chunking path in use
    roundtrip(KVZ_PLANES2, 8192, 2048, 0, 8, g, false);
    roundtrip(KVZ_Q8_0, 8192, 34 * 32, 0, 8, g, false);
    roundtrip(KVZ_Q4_0, 8192, 18 * 32, 0, 8, g, false);
    printf("kvz: %d passed, %d failed\n", n_pass, n_fail);
    return n_fail;
}
