#include "llama-kvz.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <thread>
#include <vector>

#if defined(__SSE2__)
#include <emmintrin.h>
#endif

namespace {

constexpr int      PROB_BITS = 12;
constexpr uint32_t PROB_M    = 1u << PROB_BITS;
constexpr uint32_t RANS_L    = 1u << 23;
constexpr uint8_t  VERSION   = 1;

template <typename F>
void parallel_for(size_t n, int n_threads, F && fn) {
    const int nt = (int) std::min<size_t>(n, (size_t) std::max(1, n_threads));
    if (nt <= 1) {
        for (size_t i = 0; i < n; ++i) {
            fn(i);
        }
        return;
    }
    std::atomic<size_t> next{0};
    std::vector<std::thread> th;
    th.reserve(nt - 1);
    auto work = [&]() {
        for (size_t i; (i = next.fetch_add(1)) < n;) {
            fn(i);
        }
    };
    for (int t = 1; t < nt; ++t) {
        th.emplace_back(work);
    }
    work();
    for (auto & t : th) {
        t.join();
    }
}

// byte buffer without value-initialization (kvz buffers are written before they are read)
struct bytes_t {
    std::unique_ptr<uint8_t[]> p;
    size_t n = 0;
    void alloc(size_t k) { p.reset(new uint8_t[k ? k : 1]); n = k; }
    void reserve(size_t k) { if (k > n || !p) alloc(k); } // grow-only: reused buffers stay faulted in
    uint8_t * data() { return p.get(); }
    const uint8_t * data() const { return p.get(); }
    size_t size() const { return n; }
};

// ---- byte-matrix transpose: in[R][C] (stride is) -> out[C][R] (stride os), rows [r0, r1) x cols [c0, c1) -----------

#if defined(__SSE2__)
inline void tr16(const uint8_t * in, size_t is, uint8_t * out, size_t os) {
    __m128i a[16], t[16], u[16], v[16];
    for (int i = 0; i < 16; ++i) {
        a[i] = _mm_loadu_si128((const __m128i *) (in + i * is));
    }
    for (int i = 0; i < 8; ++i) {
        t[i]     = _mm_unpacklo_epi8(a[2 * i], a[2 * i + 1]);
        t[8 + i] = _mm_unpackhi_epi8(a[2 * i], a[2 * i + 1]);
    }
    for (int h = 0; h < 2; ++h) {
        for (int i = 0; i < 4; ++i) {
            u[h * 8 + i]     = _mm_unpacklo_epi16(t[h * 8 + 2 * i], t[h * 8 + 2 * i + 1]);
            u[h * 8 + 4 + i] = _mm_unpackhi_epi16(t[h * 8 + 2 * i], t[h * 8 + 2 * i + 1]);
        }
    }
    for (int q = 0; q < 4; ++q) {
        for (int i = 0; i < 2; ++i) {
            v[q * 4 + i]     = _mm_unpacklo_epi32(u[q * 4 + 2 * i], u[q * 4 + 2 * i + 1]);
            v[q * 4 + 2 + i] = _mm_unpackhi_epi32(u[q * 4 + 2 * i], u[q * 4 + 2 * i + 1]);
        }
    }
    for (int q = 0; q < 4; ++q) {
        for (int p = 0; p < 2; ++p) {
            const int c = q * 4 + p * 2;
            _mm_storeu_si128((__m128i *) (out + c * os), _mm_unpacklo_epi64(v[c], v[c + 1]));
            _mm_storeu_si128((__m128i *) (out + (c + 1) * os), _mm_unpackhi_epi64(v[c], v[c + 1]));
        }
    }
}
#endif

void transpose(const uint8_t * in, size_t is, uint8_t * out, size_t os, size_t r0, size_t r1, size_t c0, size_t c1) {
    constexpr size_t T = 64; // 64 x 64 tiles keep both sides in L1
    for (size_t rt = r0; rt < r1; rt += T) {
        const size_t re = std::min(r1, rt + T);
        for (size_t ct = c0; ct < c1; ct += T) {
            const size_t ce = std::min(c1, ct + T);
            size_t r = rt;
#if defined(__SSE2__)
            for (; r + 16 <= re; r += 16) {
                size_t c = ct;
                for (; c + 16 <= ce; c += 16) {
                    tr16(in + r * is + c, is, out + c * os + r, os);
                }
                for (; c < ce; ++c) {
                    for (size_t k = r; k < r + 16; ++k) {
                        out[c * os + k] = in[k * is + c];
                    }
                }
            }
#endif
            for (; r < re; ++r) {
                for (size_t c = ct; c < ce; ++c) {
                    out[c * os + r] = in[r * is + c];
                }
            }
        }
    }
}

// ---- layouts: the block transposed to T[row_bytes][rows]; stream columns are rows of T ----------------------------
//
// PLANES2 / PLANES4: stream s, column c = byte s of element c. Q8_0 / Q4_0: streams 0 / 1 = the scale's low / high byte
// of block u, stream 2 = the codes of channel c (Q4_0: one nibble per symbol, split into a separate buffer).

struct stream_shape {
    size_t cols;
    int    alpha; // 256 or 16
};

int n_streams(kvz_layout l) {
    switch (l) {
        case KVZ_PLANES2: return 2;
        case KVZ_PLANES4: return 4;
        case KVZ_Q8_0:
        case KVZ_Q4_0:    return 3;
        default:          return 0;
    }
}

bool layout_fits(kvz_layout l, size_t rb) {
    switch (l) {
        case KVZ_PLANES2: return rb % 2 == 0;
        case KVZ_PLANES4: return rb % 4 == 0;
        case KVZ_Q8_0:    return rb % 34 == 0;
        case KVZ_Q4_0:    return rb % 18 == 0;
        default:          return true;
    }
}

stream_shape shape_of(kvz_layout l, int s, size_t rb) {
    switch (l) {
        case KVZ_PLANES2: return {rb / 2, 256};
        case KVZ_PLANES4: return {rb / 4, 256};
        case KVZ_Q8_0:    return s < 2 ? stream_shape{rb / 34, 256} : stream_shape{rb / 34 * 32, 256};
        case KVZ_Q4_0:    return s < 2 ? stream_shape{rb / 18, 256} : stream_shape{rb / 18 * 32, 16};
        default:          return {0, 256};
    }
}

// row stride of T: a power-of-two stride would put every row of a transpose tile in the same cache set
size_t t_stride(size_t rows) { return (rows + 63) / 64 * 64 + 64; }

// column pointers of every stream into T (and N, the Q4_0 nibble buffer [rb / 18 * 32][rows])
void column_ptrs(kvz_layout l, size_t rb, size_t rows, uint8_t * T, uint8_t * N, std::vector<std::vector<uint8_t *>> & cp) {
    const size_t ts = t_stride(rows);
    const int ns = n_streams(l);
    cp.assign(ns, {});
    for (int s = 0; s < ns; ++s) {
        const size_t cols = shape_of(l, s, rb).cols;
        cp[s].resize(cols);
        for (size_t c = 0; c < cols; ++c) {
            size_t k;
            switch (l) {
                case KVZ_PLANES2: k = 2 * c + s; break;
                case KVZ_PLANES4: k = 4 * c + s; break;
                case KVZ_Q8_0:    k = s < 2 ? 34 * c + s : 34 * (c / 32) + 2 + c % 32; break;
                default:          k = s < 2 ? 18 * c + s : SIZE_MAX; break;
            }
            cp[s][c] = k == SIZE_MAX ? N + c * rows : T + k * ts;
        }
    }
}

// Q4_0 codes: T rows (bytes of 16 channel pairs) <-> N rows (one nibble per symbol), for blocks [u0, u1)
void q4_split(uint8_t * T, uint8_t * N, size_t rows, size_t u0, size_t u1, bool join) {
    for (size_t u = u0; u < u1; ++u) {
        for (int j = 0; j < 16; ++j) {
            uint8_t * b = T + (18 * u + 2 + j) * t_stride(rows);
            uint8_t * lo = N + (32 * u + j) * rows, * hi = N + (32 * u + 16 + j) * rows;
            if (join) {
                for (size_t r = 0; r < rows; ++r) {
                    b[r] = (uint8_t) (lo[r] | (hi[r] << 4));
                }
            } else {
                for (size_t r = 0; r < rows; ++r) {
                    lo[r] = b[r] & 15;
                    hi[r] = b[r] >> 4;
                }
            }
        }
    }
}

// ---- frequency tables ---------------------------------------------------------------------------------------------

void normalize(const uint32_t * cnt, int A, uint16_t * f) {
    uint64_t N = 0;
    for (int s = 0; s < A; ++s) {
        N += cnt[s];
    }
    std::memset(f, 0, sizeof(uint16_t) * A);
    if (N == 0) {
        return;
    }
    int sum = 0, big = -1;
    for (int s = 0; s < A; ++s) {
        if (cnt[s]) {
            f[s] = (uint16_t) std::max<uint64_t>(1, (uint64_t) cnt[s] * PROB_M / N);
            sum += f[s];
            if (big < 0 || f[s] > f[big]) {
                big = s;
            }
        }
    }
    int diff = (int) PROB_M - sum;
    if (diff >= 0 || f[big] + diff >= 1) {
        f[big] = (uint16_t) (f[big] + diff);
        return;
    }
    // many rare symbols: take from the largest frequencies until the sum fits
    while (diff < 0) {
        int b = -1;
        for (int s = 0; s < A; ++s) {
            if (f[s] > 1 && (b < 0 || f[s] > f[b])) {
                b = s;
            }
        }
        const int take = std::min(-diff, f[b] - 1);
        f[b] = (uint16_t) (f[b] - take);
        diff += take;
    }
}

// n * log2(n) for small n from a table
struct nlogn_t {
    float v[4096];
    nlogn_t() {
        v[0] = 0;
        for (int i = 1; i < 4096; ++i) {
            v[i] = (float) (i * std::log2((double) i));
        }
    }
    double operator()(uint32_t n) const { return n < 4096 ? v[n] : n * std::log2((double) n); }
};
const nlogn_t nlogn;

// estimated coded bits of one context: entropy plus its table
double est_bits(const uint32_t * cnt, int A) {
    uint64_t N = 0;
    int nz = 0;
    double h = 0;
    for (int s = 0; s < A; ++s) {
        if (cnt[s]) {
            N += cnt[s];
            nz++;
            h -= nlogn(cnt[s]);
        }
    }
    if (!N) {
        return 8;
    }
    return h + N * std::log2((double) N) + 20.0 * nz + 16;
}

struct enc_sym {
    uint32_t x_max, rcp_freq, bias;
    uint16_t cmpl_freq, rcp_shift;
};

void build_enc(const uint16_t * f, int A, enc_sym * e) {
    uint32_t cum = 0;
    for (int s = 0; s < A; ++s) {
        const uint32_t fr = f[s];
        enc_sym & x = e[s];
        if (!fr) {
            x = {0, 0, 0, 0, 0};
            continue;
        }
        x.x_max = ((RANS_L >> PROB_BITS) << 8) * fr;
        x.cmpl_freq = (uint16_t) (PROB_M - fr);
        if (fr < 2) {
            x.rcp_freq = ~0u;
            x.rcp_shift = 0;
            x.bias = cum + PROB_M - 1;
        } else {
            uint32_t shift = 0;
            while (fr > (1u << shift)) {
                shift++;
            }
            x.rcp_freq = (uint32_t) (((1ull << (shift + 31)) + fr - 1) / fr);
            x.rcp_shift = (uint16_t) (shift - 1);
            x.bias = cum;
        }
        cum += fr;
    }
}

// slot -> symbol | (freq - 1) << 8 | (slot - cum) << 20
void build_dec(const uint16_t * f, int A, uint32_t * tab) {
    uint32_t cum = 0;
    for (int s = 0; s < A; ++s) {
        for (uint32_t k = 0; k < f[s]; ++k) {
            tab[cum + k] = (uint32_t) s | ((uint32_t) (f[s] - 1) << 8) | (k << 20);
        }
        cum += f[s];
    }
}

// ---- byte io ------------------------------------------------------------------------------------------------------

// writes into d up to cap bytes; past that only counts (the block is then stored raw)
struct writer {
    uint8_t * d;
    size_t cap, n = 0;
    void put(const void * src, size_t k) {
        if (k && n + k <= cap) {
            std::memcpy(d + n, src, k);
        }
        n += k;
    }
    void u8(uint8_t v) { put(&v, 1); }
    void u64(uint64_t v) {
        uint8_t b[8];
        for (int i = 0; i < 8; ++i) b[i] = (uint8_t) (v >> (8 * i));
        put(b, 8);
    }
    void var(uint64_t v) {
        uint8_t b[10];
        int k = 0;
        while (v >= 0x80) {
            b[k++] = (uint8_t) (v | 0x80);
            v >>= 7;
        }
        b[k++] = (uint8_t) v;
        put(b, k);
    }
    bool over() const { return n > cap; }
};

struct reader {
    const uint8_t * p, * end;
    bool ok = true;
    uint8_t u8() {
        if (p >= end) { ok = false; return 0; }
        return *p++;
    }
    uint64_t u64() {
        uint64_t v = 0;
        for (int i = 0; i < 8; ++i) v |= (uint64_t) u8() << (8 * i);
        return v;
    }
    uint64_t var() {
        uint64_t v = 0;
        for (int sh = 0; sh < 64; sh += 7) {
            const uint8_t c = u8();
            v |= (uint64_t) (c & 0x7F) << sh;
            if (!(c & 0x80)) return v;
        }
        ok = false;
        return 0;
    }
};

// ---- rANS over the columns of one chunk ---------------------------------------------------------------------------
//
// Symbol r of a column uses state r & 3; the encoder runs backwards over the chunk, the decoder forwards. Each column
// is coded under its context's table (context = column >> gran); delta columns code (x[r] - x[r-1]) mod alpha.

struct chunk_out;

struct stream_plan {
    stream_shape sh;
    int    gran = 0;
    bool   delta = false;
    bool   raw = false;
    size_t n_ctx = 0;
    std::vector<uint16_t> freq;      // [n_ctx][alpha]
    std::vector<size_t>   chunk_col; // chunk column starts, plus cols
    std::vector<struct chunk_out> chunk;
};

constexpr int NS = 4;

struct chunk_out {
    uint32_t        x[NS];
    const uint8_t * beg[NS];
    uint32_t        sz[NS];
    bool            overflow = false;
    size_t bytes() const {
        size_t b = 8 * NS;
        for (int j = 0; j < NS; ++j) b += sz[j];
        return b;
    }
};

inline void enc_put(uint32_t & x, uint8_t *& ptr, const enc_sym & e) {
    uint32_t v = x;
    while (v >= e.x_max) {
        *--ptr = (uint8_t) (v & 0xFF);
        v >>= 8;
    }
    const uint32_t q = (uint32_t) (((uint64_t) v * e.rcp_freq) >> 32) >> e.rcp_shift;
    x = v + e.bias + q * e.cmpl_freq;
}

// NS interleaved rANS states, each with its own byte stream (so the states' renormalizations do not serialize);
// symbol r of a column goes to state r % NS. A chunk is: NS final states (u32), NS stream sizes (u32), the streams.
// Each state writes backwards into its own `cap` bytes of region; a chunk that would not fit is flagged.
size_t state_cap(size_t rows, size_t n_cols) { return n_cols * (rows / NS + 1) + 2 * (rows / NS + 1) + 16; }

void encode_chunk(const stream_plan & p, uint8_t * const * cp, size_t rows, size_t c0, size_t c1, uint8_t * tmp, uint8_t * region,
                  size_t cap, chunk_out & out) {
    const int A = p.sh.alpha;
    const uint8_t mask = (uint8_t) (A - 1);
    uint8_t * ptr[NS], * lim[NS];
    uint32_t x[NS];
    for (int j = 0; j < NS; ++j) {
        lim[j] = region + cap * j;
        ptr[j] = lim[j] + cap;
        x[j] = RANS_L;
    }
    const size_t col_max = 2 * (rows / NS + 1); // bytes one column can write into one state's stream
    std::vector<enc_sym> tabv(A);
    const enc_sym * tab = tabv.data();
    size_t ctx_cur = SIZE_MAX;
    const size_t rN = rows / NS * NS;
    for (size_t c = c1; c-- > c0;) {
        for (int j = 0; j < NS; ++j) {
            if ((size_t) (ptr[j] - lim[j]) < col_max) {
                out.overflow = true;
                return;
            }
        }
        const size_t ctx = c >> p.gran;
        if (ctx != ctx_cur) {
            build_enc(&p.freq[ctx * A], A, tabv.data());
            ctx_cur = ctx;
        }
        const uint8_t * col = cp[c];
        if (p.delta) {
            tmp[0] = col[0];
            for (size_t r = 1; r < rows; ++r) {
                tmp[r] = (uint8_t) ((col[r] - col[r - 1]) & mask);
            }
            col = tmp;
        }
        for (size_t r = rows; r-- > rN;) {
            enc_put(x[r % NS], ptr[r % NS], tab[col[r]]);
        }
        for (size_t r = rN; r > 0; r -= NS) {
#pragma GCC unroll 4
            for (int j = NS - 1; j >= 0; --j) {
                enc_put(x[j], ptr[j], tab[col[r - NS + j]]);
            }
        }
    }
    for (int j = 0; j < NS; ++j) {
        out.x[j] = x[j];
        out.beg[j] = ptr[j];
        out.sz[j] = (uint32_t) (lim[j] + cap - ptr[j]);
    }
}

// the unchecked form needs 2 readable bytes at p
inline uint8_t dec_get_fast(uint32_t & x, const uint32_t * tab, const uint8_t *& p) {
    const uint32_t e = tab[x & (PROB_M - 1)];
    uint32_t v = ((e >> 8 & 0xFFF) + 1) * (x >> PROB_BITS) + (e >> 20);
    if (v < RANS_L) {
        v = (v << 8) | *p++;
        if (v < RANS_L) {
            v = (v << 8) | *p++;
        }
    }
    x = v;
    return (uint8_t) e;
}

inline uint8_t dec_get(uint32_t & x, const uint32_t * tab, const uint8_t *& p, const uint8_t * end, bool & bad) {
    const uint32_t e = tab[x & (PROB_M - 1)];
    uint32_t v = ((e >> 8 & 0xFFF) + 1) * (x >> PROB_BITS) + (e >> 20);
    while (v < RANS_L) {
        if (p >= end) {
            bad = true;
            break;
        }
        v = (v << 8) | *p++;
    }
    x = v;
    return (uint8_t) e;
}

bool decode_chunk(const uint16_t * freq, int A, int gran, bool delta, const uint8_t * src, size_t size, uint8_t * const * cp,
                  size_t rows, size_t c0, size_t c1) {
    if (size < 8 * NS) {
        return false;
    }
    uint32_t x[NS], sz[NS];
    std::memcpy(x, src, 4 * NS);
    std::memcpy(sz, src + 4 * NS, 4 * NS);
    const uint8_t * p[NS], * end[NS];
    size_t off = 8 * NS;
    for (int j = 0; j < NS; ++j) {
        if (sz[j] > size - off) {
            return false;
        }
        p[j] = src + off;
        off += sz[j];
        end[j] = src + off;
    }
    if (off != size) {
        return false;
    }
    std::vector<uint32_t> tabv(PROB_M);
    const uint32_t * tab = tabv.data();
    size_t ctx_cur = SIZE_MAX;
    const uint8_t mask = (uint8_t) (A - 1);
    const size_t rN = rows / NS * NS;
    const size_t need = 2 * (rows / NS + 1);
    bool bad = false;
    for (size_t c = c0; c < c1 && !bad; ++c) {
        const size_t ctx = c >> gran;
        if (ctx != ctx_cur) {
            build_dec(&freq[ctx * A], A, tabv.data());
            ctx_cur = ctx;
        }
        uint8_t * col = cp[c];
        bool fast = true;
        for (int j = 0; j < NS; ++j) {
            fast &= (size_t) (end[j] - p[j]) >= need;
        }
        if (fast) {
            for (size_t r = 0; r < rN; r += NS) {
#pragma GCC unroll 4
                for (int j = 0; j < NS; ++j) {
                    col[r + j] = dec_get_fast(x[j], tab, p[j]);
                }
            }
            for (size_t r = rN; r < rows; ++r) {
                col[r] = dec_get_fast(x[r % NS], tab, p[r % NS]);
            }
        } else {
            for (size_t r = 0; r < rows; ++r) {
                col[r] = dec_get(x[r % NS], tab, p[r % NS], end[r % NS], bad);
            }
        }
        if (delta) {
            for (size_t r = 1; r < rows; ++r) {
                col[r] = (uint8_t) ((col[r] + col[r - 1]) & mask);
            }
        }
    }
    for (int j = 0; j < NS; ++j) {
        bad |= x[j] != RANS_L || p[j] != end[j];
    }
    return !bad;
}

// choose the context width and delta mode from per-column histograms (cols x alpha each)
void plan_stream(stream_plan & p, const std::vector<uint32_t> & hp, const std::vector<uint32_t> & hd, size_t rows) {
    const int A = p.sh.alpha;
    const size_t cols = p.sh.cols;
    double best = INFINITY;
    for (int d = 0; d < 2; ++d) {
        std::vector<uint32_t> cur(d ? hd : hp);
        size_t ncur = cols;
        for (int g = 0;; ++g) {
            double bits = 0;
            for (size_t i = 0; i < ncur; ++i) {
                bits += est_bits(&cur[i * A], A);
            }
            if (bits < best) {
                best = bits;
                p.gran = g;
                p.delta = d;
            }
            if (ncur == 1) {
                break;
            }
            const size_t nn = (ncur + 1) / 2;
            for (size_t i = 0; i < nn; ++i) {
                for (int s = 0; s < A; ++s) {
                    cur[i * A + s] = cur[2 * i * A + s] + (2 * i + 1 < ncur ? cur[(2 * i + 1) * A + s] : 0);
                }
            }
            ncur = nn;
        }
    }
    // a 256-symbol stream that codes to more than 98% of its size is stored
    p.raw = A == 256 && best >= 0.98 * 8.0 * (double) (cols * rows);
}

// parallel transposes in tasks of whole 64-row / 64-column tiles
void transpose_par(const uint8_t * in, size_t is, uint8_t * out, size_t os, size_t R, size_t C, int n_threads, bool by_rows) {
    const size_t n = by_rows ? R : C;
    const size_t nt = std::max<size_t>(1, std::min<size_t>((size_t) 4 * n_threads, (n + 63) / 64));
    parallel_for(nt, n_threads, [&](size_t i) {
        const size_t a = i * n / nt / 64 * 64, b = i + 1 == nt ? n : (i + 1) * n / nt / 64 * 64;
        if (by_rows) {
            transpose(in, is, out, os, a, b, 0, C);
        } else {
            transpose(in, is, out, os, 0, R, a, b);
        }
    });
}

} // namespace

size_t kvz_bound(size_t raw_bytes) {
    return raw_bytes + 64;
}

int kvz_default_threads() {
    if (const char * e = std::getenv("LLAMA_KVZ_THREADS")) {
        return std::max(1, std::atoi(e));
    }
    return std::max(1, (int) std::thread::hardware_concurrency());
}

size_t kvz_encode(kvz_layout layout, const uint8_t * src, size_t rows, size_t rb, uint8_t * dst, int n_threads) {
    const size_t raw_bytes = rows * rb;
    writer w{dst, raw_bytes + 32};
    w.u8('K');
    w.u8('Z');
    w.u8(VERSION);
    if (!layout_fits(layout, rb) || rows == 0) {
        layout = KVZ_RAW;
    }
    const int ns = n_streams(layout);
    std::vector<stream_plan> plans(ns);
    std::vector<std::vector<uint8_t *>> cp;
    thread_local bytes_t T, N, arena; // reused across calls: fresh pages would cost a fault each, serialized by the kernel
    bool overflow = false;
    if (ns > 0) {
        T.reserve(t_stride(rows) * rb);
        transpose_par(src, rb, T.data(), t_stride(rows), rows, rb, n_threads, false);
        if (layout == KVZ_Q4_0) {
            N.reserve(rb / 18 * 32 * rows);
            const size_t nb = rb / 18;
            uint8_t * Tp = T.data(), * Np = N.data(); // thread_local: workers must not name them
            parallel_for(nb, n_threads, [&](size_t u) { q4_split(Tp, Np, rows, u, u + 1, false); });
        }
        column_ptrs(layout, rb, rows, T.data(), N.data(), cp);

        // per-column histograms of plain and delta symbols
        std::vector<std::vector<uint32_t>> hp(ns), hd(ns);
        struct htask { int s; size_t c0, c1; };
        std::vector<htask> htasks;
        for (int s = 0; s < ns; ++s) {
            plans[s].sh = shape_of(layout, s, rb);
            const size_t cols = plans[s].sh.cols, A = plans[s].sh.alpha;
            hp[s].assign(cols * A, 0);
            hd[s].assign(cols * A, 0);
            const size_t step = std::max<size_t>(1, cols / (4 * n_threads));
            for (size_t c = 0; c < cols; c += step) {
                htasks.push_back({s, c, std::min(cols, c + step)});
            }
        }
        parallel_for(htasks.size(), n_threads, [&](size_t i) {
            const htask & t = htasks[i];
            const int A = plans[t.s].sh.alpha;
            const uint8_t mask = (uint8_t) (A - 1);
            for (size_t c = t.c0; c < t.c1; ++c) {
                const uint8_t * col = cp[t.s][c];
                uint32_t * a = &hp[t.s][c * A], * b = &hd[t.s][c * A];
                uint8_t prev = 0;
                for (size_t r = 0; r < rows; ++r) {
                    a[col[r]]++;
                    b[(uint8_t) (col[r] - prev) & mask]++;
                    prev = col[r];
                }
            }
        });
        parallel_for(ns, n_threads, [&](size_t s) { plan_stream(plans[s], hp[s], hd[s], rows); });

        // tables and chunks
        struct ctask { int s; size_t k; size_t off, cap; };
        std::vector<ctask> ctasks;
        size_t arena_size = 0;
        for (int s = 0; s < ns; ++s) {
            stream_plan & p = plans[s];
            if (p.raw) {
                continue;
            }
            const int A = p.sh.alpha;
            const size_t cols = p.sh.cols, cw = (size_t) 1 << p.gran;
            p.n_ctx = (cols + cw - 1) / cw;
            p.freq.assign(p.n_ctx * A, 0);
            const auto & h = p.delta ? hd[s] : hp[s];
            std::vector<uint32_t> acc(A);
            for (size_t k = 0; k < p.n_ctx; ++k) {
                std::fill(acc.begin(), acc.end(), 0);
                for (size_t c = k * cw; c < std::min(cols, (k + 1) * cw); ++c) {
                    for (int a = 0; a < A; ++a) {
                        acc[a] += h[c * A + a];
                    }
                }
                normalize(acc.data(), A, &p.freq[k * A]);
            }
            const size_t n_sym = cols * rows;
            const size_t n_chunk = std::max<size_t>(1, std::min<size_t>({p.n_ctx, (size_t) 4 * n_threads, n_sym / 65536 + 1}));
            for (size_t k = 0; k <= n_chunk; ++k) {
                p.chunk_col.push_back(std::min(cols, (k * p.n_ctx / n_chunk) * cw));
            }
            p.chunk.resize(n_chunk);
            for (size_t k = 0; k < n_chunk; ++k) {
                const size_t cap = state_cap(rows, p.chunk_col[k + 1] - p.chunk_col[k]);
                ctasks.push_back({s, k, arena_size, cap});
                arena_size += cap * NS;
            }
        }
        arena.reserve(arena_size);
        uint8_t * Ap = arena.data();
        parallel_for(ctasks.size(), n_threads, [&](size_t i) {
            const ctask & t = ctasks[i];
            stream_plan & p = plans[t.s];
            std::vector<uint8_t> tmp(p.delta ? rows : 0);
            encode_chunk(p, cp[t.s].data(), rows, p.chunk_col[t.k], p.chunk_col[t.k + 1], tmp.data(), Ap + t.off, t.cap, p.chunk[t.k]);
        });
        for (const auto & t : ctasks) {
            overflow |= plans[t.s].chunk[t.k].overflow;
        }
    }

    // serialize
    w.u8((uint8_t) layout);
    w.u64(rows);
    w.u64(rb);
    for (int s = 0; s < ns && !w.over() && !overflow; ++s) {
        const stream_plan & p = plans[s];
        w.u8(p.raw ? 1 : 0);
        if (p.raw) {
            for (size_t c = 0; c < p.sh.cols; ++c) {
                w.put(cp[s][c], rows);
            }
            continue;
        }
        const int A = p.sh.alpha;
        w.u8(p.delta ? 1 : 0);
        w.u8((uint8_t) p.gran);
        for (size_t k = 0; k < p.n_ctx; ++k) {
            const uint16_t * f = &p.freq[k * A];
            int nz = 0;
            for (int a = 0; a < A; ++a) {
                nz += f[a] != 0;
            }
            w.var(nz);
            int last = -1;
            for (int a = 0; a < A; ++a) {
                if (f[a]) {
                    w.var(a - last - 1);
                    w.var(f[a] - 1);
                    last = a;
                }
            }
        }
        w.var(p.chunk.size());
        for (size_t k = 0; k < p.chunk.size(); ++k) {
            w.var(p.chunk_col[k + 1]);
            w.var(p.chunk[k].bytes());
        }
        for (const auto & c : p.chunk) {
            w.put(c.x, 4 * NS);
            w.put(c.sz, 4 * NS);
            for (int j = 0; j < NS; ++j) {
                w.put(c.beg[j], c.sz[j]);
            }
        }
    }
    if (std::getenv("KVZ_DEBUG")) {
        for (int s = 0; s < ns; ++s) {
            size_t z = 0;
            for (const auto & c : plans[s].chunk) {
                z += c.bytes();
            }
            fprintf(stderr, "kvz: layout %d stream %d: %s delta %d gran %d, %.3f bits/sym\n", layout, s, plans[s].raw ? "raw" : "coded",
                    plans[s].delta, plans[s].gran, plans[s].raw ? 8.0 : 8.0 * z / (plans[s].sh.cols * rows));
        }
    }
    if (layout == KVZ_RAW || overflow || w.over() || w.n >= raw_bytes + 20) {
        w.n = 3;
        w.u8((uint8_t) KVZ_RAW);
        w.u64(rows);
        w.u64(rb);
        w.put(src, raw_bytes);
    }
    return w.n;
}

bool kvz_decode(const uint8_t * src, size_t src_size, uint8_t * dst, size_t rows, size_t rb, int n_threads) {
    reader rd{src, src + src_size};
    if (rd.u8() != 'K' || rd.u8() != 'Z' || rd.u8() != VERSION) {
        return false;
    }
    const kvz_layout layout = (kvz_layout) rd.u8();
    if (rd.u64() != rows || rd.u64() != rb || !rd.ok || layout > KVZ_PLANES4 || !layout_fits(layout, rb)) {
        return false;
    }
    if (layout == KVZ_RAW) {
        if ((size_t) (rd.end - rd.p) != rows * rb) {
            return false;
        }
        if (rows * rb != 0) {
            std::memcpy(dst, rd.p, rows * rb);
        }
        return true;
    }
    const int ns = n_streams(layout);
    thread_local bytes_t T, N;
    T.reserve(t_stride(rows) * rb);
    if (layout == KVZ_Q4_0) {
        N.reserve(rb / 18 * 32 * rows);
    }
    std::vector<std::vector<uint8_t *>> cp;
    column_ptrs(layout, rb, rows, T.data(), N.data(), cp);
    struct sdec {
        stream_shape sh;
        bool delta;
        int gran;
        std::vector<uint16_t> freq;
        std::vector<size_t> col, off, size;
    };
    std::vector<sdec> st(ns);
    struct dtask { int s; size_t k; };
    std::vector<dtask> tasks;
    for (int s = 0; s < ns; ++s) {
        sdec & d = st[s];
        d.sh = shape_of(layout, s, rb);
        const int A = d.sh.alpha;
        const size_t cols = d.sh.cols;
        const uint8_t mode = rd.u8();
        if (mode == 1) {
            if (A != 256 || (size_t) (rd.end - rd.p) < cols * rows) {
                return false;
            }
            for (size_t c = 0; c < cols; ++c) {
                std::memcpy(cp[s][c], rd.p, rows);
                rd.p += rows;
            }
            continue;
        }
        if (mode != 0) {
            return false;
        }
        d.delta = rd.u8() == 1;
        d.gran = rd.u8();
        if (d.gran > 40) {
            return false;
        }
        const size_t cw = (size_t) 1 << d.gran, n_ctx = (cols + cw - 1) / cw;
        d.freq.assign(n_ctx * A, 0);
        for (size_t k = 0; k < n_ctx && rd.ok; ++k) {
            const uint64_t nz = rd.var();
            if (nz == 0 || nz > (uint64_t) A) {
                return false;
            }
            int a = -1;
            uint32_t sum = 0;
            for (uint64_t j = 0; j < nz && rd.ok; ++j) {
                a += (int) rd.var() + 1;
                const uint64_t f = rd.var() + 1;
                if (a >= A || f > PROB_M) {
                    return false;
                }
                d.freq[k * A + a] = (uint16_t) f;
                sum += (uint32_t) f;
            }
            if (sum != PROB_M) {
                return false;
            }
        }
        const uint64_t n_chunk = rd.var();
        if (!rd.ok || n_chunk == 0 || n_chunk > cols) {
            return false;
        }
        d.col.push_back(0);
        for (uint64_t k = 0; k < n_chunk; ++k) {
            const uint64_t c = rd.var(), sz = rd.var();
            if (c <= d.col.back() || c > cols) {
                return false;
            }
            d.col.push_back(c);
            d.size.push_back(sz);
        }
        if (d.col.back() != cols || !rd.ok) {
            return false;
        }
        for (uint64_t k = 0; k < n_chunk; ++k) {
            if ((size_t) (rd.end - rd.p) < d.size[k]) {
                return false;
            }
            d.off.push_back(rd.p - src);
            rd.p += d.size[k];
            tasks.push_back({s, (size_t) k});
        }
    }
    if (rd.p != rd.end) {
        return false;
    }
    std::atomic<bool> ok{true};
    parallel_for(tasks.size(), n_threads, [&](size_t i) {
        sdec & d = st[tasks[i].s];
        const size_t k = tasks[i].k;
        if (!decode_chunk(d.freq.data(), d.sh.alpha, d.gran, d.delta, src + d.off[k], d.size[k], cp[tasks[i].s].data(), rows,
                          d.col[k], d.col[k + 1])) {
            ok = false;
        }
    });
    if (!ok) {
        return false;
    }
    if (layout == KVZ_Q4_0) {
        const size_t nb = rb / 18;
        uint8_t * Tp = T.data(), * Np = N.data();
        parallel_for(nb, n_threads, [&](size_t u) { q4_split(Tp, Np, rows, u, u + 1, true); });
    }
    transpose_par(T.data(), t_stride(rows), dst, rb, rb, rows, n_threads, false);
    return true;
}
