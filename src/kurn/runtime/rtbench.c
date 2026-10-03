// rtbench: model-shaped decode microbenchmark for the kurn runtime (kurn_rt.h).
//
// A "token" is one layer step: a sequence of ops; an op is a set of independent
// GEMVs (e.g. the 8 selected experts' gate/up), followed by a barrier. Weights are
// random Q8_0, packed by the kernel's own *_prepare; experts are re-drawn every
// token and layer copies rotate so the weights touched per token stream from
// DRAM (--regime cold, footprint >= 1.2 GB by default) or stay cache-resident
// (--regime hot: only the picked experts exist). Every partitioner is checked
// against a double-precision reference before timing.
//
//   rtbench --impl kernel.so|read --workload W [--part static|balanced|streamk|steal|ggml]
//           [--threads T] [--regime hot|cold] [--footprint MB] [--secs S] [--ksplit S]
//           [--tile R] [--align A] [--wait spin|futex|hybrid:N|umwait] [--op-barriers B]
//           [--sync-each] [--gaps U0,U1,..] [--serial-us U] [--prefetch none|wait]
//           [--pf-kb KB] [--pf-hint 0..3] [--pf-predict P] [--static-w W] [--seed S]
//           [--csv out.csv] [--label L] [--dump F] [--rotate] [--producers P] [--list]
//
// --producers P: the last P threads only stream the next op's weights (one op ahead).
// --rotate: thread t runs thread (t + token) % T's share (no static ownership across tokens).
//
// --impl read: a bandwidth-only "kernel" that streams the native Q8_0 rows (no math;
// output not checked), for cache-cliff sweeps.
#define _GNU_SOURCE
#include "kurn_rt.h"
#include <dlfcn.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <time.h>
#include <unistd.h>
#if defined(__AVX512F__)
#include <immintrin.h>
#endif

#define PROXY_W_PER_CORE 5.47
#define MAX_KINDS 8
#define MAX_TASKS 64
#define MAX_OPS 8
#define RING 4096

typedef struct { uint16_t d; int8_t qs[32]; } q8blk;
typedef void *(*prep_fn)(const void *, int64_t, int64_t);
typedef void (*gemv_fn)(const void *, const void *, float *, int64_t, int64_t, int64_t);

typedef struct { const char *name; int64_t N, K; int pool, pick, op; } kind_t;
typedef struct { const char *name, *desc; int model_layers; int nk; kind_t k[MAX_KINDS]; } workload_t;

static workload_t WORKLOADS[] = {
    {"olmoe-moe", "OLMoE-1B-7B FFN, kurn engine graph: 8 of 64 experts, fused gate/up (2048x2048) then down (2048x1024)", 16, 2,
     {{"gu", 2048, 2048, 64, 8, 0}, {"down", 2048, 1024, 64, 8, 1}}},
    {"olmoe-moe-sep", "OLMoE-1B-7B FFN, ggml graph: gate, up (1024x2048) and down (2048x1024) as separate ops", 16, 3,
     {{"gate", 1024, 2048, 64, 8, 0}, {"up", 1024, 2048, 64, 8, 1}, {"down", 2048, 1024, 64, 8, 2}}},
    {"qwen3moe", "Qwen3-30B-A3B FFN, kurn graph: 8 of 128 experts, fused gate/up (1536x2048) then down (2048x768)", 48, 2,
     {{"gu", 1536, 2048, 128, 8, 0}, {"down", 2048, 768, 128, 8, 1}}},
    {"qwen3moe-sep", "Qwen3-30B-A3B FFN, ggml graph: gate, up (768x2048), down (2048x768) as separate ops", 48, 3,
     {{"gate", 768, 2048, 128, 8, 0}, {"up", 768, 2048, 128, 8, 1}, {"down", 2048, 768, 128, 8, 2}}},
    {"olmoe-layer", "OLMoE-1B-7B layer: QKV (6144x2048), O (2048x2048), MoE gate/up, down", 16, 4,
     {{"qkv", 6144, 2048, 1, 1, 0}, {"o", 2048, 2048, 1, 1, 1}, {"gu", 2048, 2048, 64, 8, 2}, {"down", 2048, 1024, 64, 8, 3}}},
    {"dense17", "Qwen3-1.7B layer, kurn graph: QKV (4096x2048), O (2048x2048), gate/up (12288x2048), down (2048x6144)", 28, 4,
     {{"qkv", 4096, 2048, 1, 1, 0}, {"o", 2048, 2048, 1, 1, 1}, {"gu", 12288, 2048, 1, 1, 2}, {"down", 2048, 6144, 1, 1, 3}}},
    {"dense17-sep", "Qwen3-1.7B layer, ggml graph: q, k, v, o, gate, up, down as separate ops", 28, 7,
     {{"q", 2048, 2048, 1, 1, 0}, {"k", 1024, 2048, 1, 1, 1}, {"v", 1024, 2048, 1, 1, 2}, {"o", 2048, 2048, 1, 1, 3},
      {"gate", 6144, 2048, 1, 1, 4}, {"up", 6144, 2048, 1, 1, 5}, {"down", 2048, 6144, 1, 1, 6}}},
    {"skinny", "small-N long-K GEMVs (N < threads x 128): 4 x (256 x 8192) in one op", 1, 1, {{"m", 256, 8192, 4, 4, 0}}},
    {"skinny1", "one small-N long-K GEMV: 384 x 16384", 1, 1, {{"m", 384, 16384, 1, 1, 0}}},
    {"tiny-moe", "test shapes: 8 of 16 experts, gate/up 272x512, down 200x256 (row tails, uneven slices)", 1, 2,
     {{"gu", 272, 512, 16, 8, 0}, {"down", 200, 256, 16, 8, 1}}},
    {"tiny-mixed", "test shapes: three unequal matrices in one op, then one", 1, 2,
     {{"a", 528, 512, 1, 1, 0}, {"b", 96, 1024, 1, 1, 0}, {"c", 1040, 256, 1, 1, 0}, {"d", 64, 2048, 1, 1, 1}}},
    {"sweep", "cache-cliff sweep: --footprint MB of 1024x2048 matrices, all in one op (static per-core ownership)", 1, 1,
     {{"w", 1024, 2048, 1, 1, 0}}},
};
#define NWL ((int)(sizeof WORKLOADS / sizeof WORKLOADS[0]))

static struct {
    workload_t W;
    int producers, rotate, threads, part, ksplit, op_barriers, sync_each, hot, read_impl, layout_vnni16, prefetch, pf_hint, nops;
    int64_t tile, align;
    double secs, run_secs, footprint_mb, static_w, pf_predict, gaps[MAX_OPS];
    long max_tokens;
    size_t pf_bytes;
    uint64_t seed;
    const char *impl, *csv, *label, *dump;
    krt_wait wait;
    prep_fn prep;
    gemv_fn gemv;
    int L;                       // layer copies
    void ***pk;                  // pk[l][kind] -> pool * ksplit packed handles
    uint8_t **X;                 // activation per kind (K/32 blocks)
    int ntask[MAX_OPS], tkind[MAX_OPS][MAX_TASKS], tj[MAX_OPS][MAX_TASKS];
    float *Y[MAX_OPS][MAX_TASKS], *Yref[MAX_OPS][MAX_TASKS];
    int sel[RING][MAX_KINDS][MAX_TASKS];
    krt_plan *plan[MAX_OPS][2];
    double bytes_per_tok;
    long stop_tok; // last token to run; a token id, not a flag, so a lagging thread can't stop a token early
    long tokens;
    double *tok_us;
    long tok_cap;
    double t_start;
} G;

typedef struct { long tok; int op; char pad[48]; } tctx_t;
static tctx_t TC[KRT_MAX_THREADS];
static krt_prefetch_queue PQ[KRT_MAX_THREADS];

static uint64_t rng;
static inline uint64_t next(void) {
    rng ^= rng >> 12; rng ^= rng << 25; rng ^= rng >> 27;
    return rng * 0x2545F4914F6CDD1Dull;
}
static float h2f(uint16_t h) {
    uint32_t s = (uint32_t)(h & 0x8000) << 16, e = (h >> 10) & 0x1f, m = h & 0x3ff, b;
    if (e == 0) {
        if (!m) b = s;
        else { e = 113; while (!(m & 0x400)) { m <<= 1; e--; } b = s | (e << 23) | ((m & 0x3ff) << 13); }
    } else if (e == 31) b = s | 0x7f800000 | (m << 13);
    else b = s | ((e + 112) << 23) | (m << 13);
    float f;
    memcpy(&f, &b, 4);
    return f;
}
static void gen_q8(q8blk *b, int64_t n) {
    for (int64_t i = 0; i < n; i++) {
        uint64_t r = next();
        b[i].d = (uint16_t)(((9 + r % 5) << 10) | ((r >> 8) & 0x3ff));
        for (int l = 0; l < 32; l += 8) {
            uint64_t v = next();
            for (int j = 0; j < 8; j++, v >>= 8) b[i].qs[l + j] = (int8_t)(v & 0xff) == -128 ? -127 : (int8_t)(v & 0xff);
        }
    }
}
static double ref_row(const q8blk *w, const q8blk *x, int64_t nb) {
    double acc = 0;
    for (int64_t b = 0; b < nb; b++) {
        int32_t s = 0;
        for (int l = 0; l < 32; l++) s += (int32_t)w[b].qs[l] * x[b].qs[l];
        acc += (double)h2f(w[b].d) * (double)h2f(x[b].d) * s;
    }
    return acc;
}
static void *xalloc(size_t n) {
    void *p = NULL;
    if (posix_memalign(&p, 2 << 20, (n + 4095) & ~(size_t)4095)) { perror("alloc"); exit(1); }
    return p;
}
static double now_s(clockid_t c) {
    struct timespec ts;
    clock_gettime(c, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}
static double cpu_s(void) {
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    return ru.ru_utime.tv_sec + ru.ru_utime.tv_usec * 1e-6 + ru.ru_stime.tv_sec + ru.ru_stime.tv_usec * 1e-6;
}
static size_t row_bytes(int64_t K) { return (size_t)(K / 32) * sizeof(q8blk); }

// ---------------------------------------------------------------- "read" kernel (bandwidth only)
typedef struct { int64_t rb; uint8_t *base; } native_t;
static void *read_prepare(const void *W, int64_t K, int64_t N) {
    native_t *n = malloc(sizeof *n);
    n->rb = (int64_t)row_bytes(K);
    n->base = xalloc((size_t)n->rb * N);
    memcpy(n->base, W, (size_t)n->rb * N);
    return n;
}
static void read_gemv(const void *pv, const void *x, float *y, int64_t K, int64_t r0, int64_t r1) {
    (void)x; (void)K;
    const native_t *n = pv;
    const uint8_t *p = n->base + r0 * n->rb, *e = n->base + r1 * n->rb;
#if defined(__AVX512F__)
    __m512i a0 = _mm512_setzero_si512(), a1 = a0, a2 = a0, a3 = a0;
    for (; p + 256 <= e; p += 256) {
        a0 = _mm512_xor_si512(a0, _mm512_loadu_si512(p));
        a1 = _mm512_xor_si512(a1, _mm512_loadu_si512(p + 64));
        a2 = _mm512_xor_si512(a2, _mm512_loadu_si512(p + 128));
        a3 = _mm512_xor_si512(a3, _mm512_loadu_si512(p + 192));
    }
    uint64_t s = (uint64_t)_mm512_reduce_add_epi64(_mm512_xor_si512(_mm512_xor_si512(a0, a1), _mm512_xor_si512(a2, a3)));
#else
    uint64_t s = 0;
    for (; p + 8 <= e; p += 8) { uint64_t v; memcpy(&v, p, 8); s ^= v; }
#endif
    for (int64_t r = r0; r < r1; r++) y[r] = (float)(s & 1);
}

static uint64_t touch(const uint8_t *p, size_t len) { // stream [p, p + len) through this core's caches
    const uint8_t *e = p + len;
#if defined(__AVX512F__)
    __m512i a0 = _mm512_setzero_si512(), a1 = a0;
    for (; p + 128 <= e; p += 128) {
        a0 = _mm512_xor_si512(a0, _mm512_loadu_si512(p));
        a1 = _mm512_xor_si512(a1, _mm512_loadu_si512(p + 64));
    }
    return (uint64_t)_mm512_reduce_add_epi64(_mm512_xor_si512(a0, a1));
#else
    uint64_t s = 0;
    for (; p + 8 <= e; p += 8) { uint64_t v; memcpy(&v, p, 8); s ^= v; }
    return s;
#endif
}

// native-layout kernels (no *_prepare): keep a private copy of the rows
static gemv_fn native_gemv;
static void *native_prepare(const void *W, int64_t K, int64_t N) {
    void *p = xalloc(row_bytes(K) * (size_t)N);
    memcpy(p, W, row_bytes(K) * (size_t)N);
    return p;
}

// ---------------------------------------------------------------- weights
static int kind_pool(int k) { return G.hot ? G.W.k[k].pick : G.W.k[k].pool; }
static size_t kind_bytes(int k) { return row_bytes(G.W.k[k].K) * (size_t)G.W.k[k].N; }

static void **pk_of(int l, int k, int e) { return (void **)G.pk[l][k] + (size_t)e * G.ksplit; }

// address range of rows [r0, r1) of one packed K tile (for prefetching)
static void rows_range(const void *pk, int64_t Kt, int64_t r0, int64_t r1, const uint8_t **p, size_t *len) {
    if (G.read_impl) {
        const native_t *n = pk;
        *p = n->base + r0 * n->rb;
        *len = (size_t)((r1 - r0) * n->rb);
    } else if (G.layout_vnni16) { // kurn vnni16 packed_t: { int64 nb, ngroups; 16-row groups of nb * 544 B }
        const int64_t *h = pk;
        const uint8_t *blk = *(uint8_t *const *)(h + 2);
        const size_t gb = (size_t)h[0] * 544;
        *p = blk + (size_t)(r0 / 16) * gb;
        *len = (size_t)((r1 + 15) / 16 - r0 / 16) * gb;
    } else {
        *p = pk;
        *len = (size_t)(r1 - r0) * row_bytes(Kt);
        *p += (size_t)r0 * row_bytes(Kt);
    }
}

static void make_data(void) {
    rng = G.seed ? G.seed : 0x9E3779B97F4A7C15ull;
    double layer_bytes = 0;
    for (int k = 0; k < G.W.nk; k++) layer_bytes += (double)kind_bytes(k) * kind_pool(k);
    G.L = 1;
    if (G.footprint_mb > 0) G.L = (int)ceil(G.footprint_mb * 1e6 / layer_bytes);
    if (G.L < 1) G.L = 1;
    // ops and tasks
    G.nops = 0;
    for (int k = 0; k < G.W.nk; k++) if (G.W.k[k].op + 1 > G.nops) G.nops = G.W.k[k].op + 1;
    G.bytes_per_tok = 0;
    for (int k = 0; k < G.W.nk; k++) {
        const int op = G.W.k[k].op;
        for (int j = 0; j < G.W.k[k].pick; j++) {
            const int t = G.ntask[op]++;
            if (t >= MAX_TASKS) { fprintf(stderr, "too many tasks\n"); exit(2); }
            G.tkind[op][t] = k; G.tj[op][t] = j;
            G.Y[op][t] = xalloc(sizeof(float) * (G.W.k[k].N + 16));
            G.Yref[op][t] = xalloc(sizeof(float) * (G.W.k[k].N + 16));
        }
        G.bytes_per_tok += (double)kind_bytes(k) * G.W.k[k].pick;
    }
    // expert selections: per token, `pick` distinct of `pool` (same sequence for kinds with equal pool/pick)
    for (int r = 0; r < RING; r++)
        for (int k = 0; k < G.W.nk; k++) {
            uint64_t s = (G.seed + 1) * 0x9E3779B97F4A7C15ull ^ (uint64_t)(r + 1) * 0xD1B54A32D192ED03ull;
            const int pool = kind_pool(k), pick = G.W.k[k].pick;
            int used[1024] = {0};
            for (int j = 0; j < pick; j++) {
                int e;
                do { s ^= s >> 12; s ^= s << 25; s ^= s >> 27; e = pool == pick ? j : (int)((s * 0x2545F4914F6CDD1Dull >> 33) % pool); } while (used[e]);
                used[e] = 1;
                G.sel[r][k][j] = e;
            }
        }
    // activations
    G.X = calloc((size_t)(G.W.nk > 0 ? G.W.nk : 1), sizeof *G.X);
    for (int k = 0; k < G.W.nk; k++) {
        G.X[k] = xalloc(row_bytes(G.W.k[k].K));
        gen_q8((q8blk *)G.X[k], G.W.k[k].K / 32);
    }
    // weights: generate native, reference (layer 0, token 0's picks), pack per K tile
    size_t maxb = 0;
    for (int k = 0; k < G.W.nk; k++) if (kind_bytes(k) > maxb) maxb = kind_bytes(k);
    q8blk *nat = xalloc(maxb), *sub = xalloc(maxb);
    G.pk = calloc(G.L, sizeof *G.pk);
    for (int l = 0; l < G.L; l++) {
        G.pk[l] = calloc(G.W.nk, sizeof(void *));
        for (int k = 0; k < G.W.nk; k++) {
            const kind_t *kd = &G.W.k[k];
            const int pool = kind_pool(k);
            G.pk[l][k] = calloc((size_t)pool * G.ksplit, sizeof(void *));
            const int64_t nb = kd->K / 32, nbt = nb / G.ksplit;
            for (int e = 0; e < pool; e++) {
                gen_q8(nat, nb * kd->N);
                if (l == 0)
                    for (int t = 0; t < G.ntask[kd->op]; t++)
                        if (G.tkind[kd->op][t] == k && G.sel[0][k][G.tj[kd->op][t]] == e)
                            for (int64_t r = 0; r < kd->N; r++)
                                G.Yref[kd->op][t][r] = (float)ref_row(nat + r * nb, (const q8blk *)G.X[k], nb);
                for (int s = 0; s < G.ksplit; s++) {
                    const q8blk *src = nat;
                    if (G.ksplit > 1) {
                        for (int64_t r = 0; r < kd->N; r++) memcpy(sub + r * nbt, nat + r * nb + s * nbt, sizeof(q8blk) * nbt);
                        src = sub;
                    }
                    pk_of(l, k, e)[s] = G.prep(src, kd->K / G.ksplit, kd->N);
                }
            }
        }
    }
    free(nat);
    free(sub);
}

// ---------------------------------------------------------------- token loop
static int layer_of(long tok) { return (int)(tok % G.L); }

static void kfn(void *user, int task, int kt, int64_t r0, int64_t r1, float *out) {
    const tctx_t *c = user;
    const int k = G.tkind[c->op][task];
    const int64_t Kt = G.W.k[k].K / G.ksplit;
    const int e = G.sel[c->tok % RING][k][G.tj[c->op][task]];
    const void *pk = pk_of(layer_of(c->tok), k, e)[kt];
    const uint8_t *x = G.X[k] + (size_t)kt * row_bytes(Kt);
    G.gemv(pk, x, out - r0, Kt, r0, r1);
}

// queue thread ith's first bytes of op `op` of token `tok` (its initial items)
static void queue_prefetch(int ith, long tok, int op) {
    krt_prefetch_queue *q = &PQ[ith];
    krt_prefetch_queue_init(q, G.pf_bytes, G.pf_hint);
    int n;
    const krt_item *it = krt_plan_items(G.plan[op][tok & 1], ith, &n);
    long ptok = tok;
    if (G.pf_predict < 1.0) { // MoE expert prediction with hit rate P: a miss prefetches another token's experts
        uint64_t h = (uint64_t)(tok * 2654435761u + op) * 0x9E3779B97F4A7C15ull;
        if ((double)(h >> 11) / 9007199254740992.0 >= G.pf_predict) ptok = tok + RING / 2;
    }
    for (int i = 0; i < n && q->n < 16; i++) {
        const int k = G.tkind[op][it[i].task];
        const int64_t Kt = G.W.k[k].K / G.ksplit;
        const long etok = G.W.k[k].pool > 1 ? ptok : tok;
        const int e = G.sel[etok % RING][k][G.tj[op][it[i].task]];
        for (int kt = it[i].kt0; kt < it[i].kt1 && q->n < 16; kt++) {
            const uint8_t *p;
            size_t len;
            if (it[i].r0 >= it[i].r1) continue;
            rows_range(pk_of(layer_of(tok), k, e)[kt], Kt, it[i].r0, it[i].r1, &p, &len);
            krt_prefetch_queue_add(q, p, len);
        }
    }
}

// --producers P: thread `pidx` of P streams its 1/P share of the next op's weights (every
// consumer's items) while the consumers compute the current op
static volatile uint64_t produce_sink;
static void produce(int pidx, long tok, int op) {
    const int last = op == G.nops - 1, nop = last ? 0 : op + 1;
    const long ntok = last ? tok + 1 : tok;
    const krt_plan *pl = G.plan[nop][ntok & 1];
    const int C = G.threads - G.producers;
    size_t tot = 0;
    for (int pass = 0; pass < 2; pass++) {
        size_t off = 0;
        const size_t lo = tot / G.producers * pidx, hi = pidx == G.producers - 1 ? tot : tot / G.producers * (pidx + 1);
        for (int c = 0; c < C; c++) {
            int n;
            const krt_item *it = krt_plan_items(pl, c, &n);
            for (int i = 0; i < n; i++) {
                if (it[i].r0 >= it[i].r1) continue;
                const int k = G.tkind[nop][it[i].task];
                const int64_t Kt = G.W.k[k].K / G.ksplit;
                const int e = G.sel[ntok % RING][k][G.tj[nop][it[i].task]];
                for (int kt = it[i].kt0; kt < it[i].kt1; kt++) {
                    const uint8_t *p;
                    size_t len;
                    rows_range(pk_of(layer_of(ntok), k, e)[kt], Kt, it[i].r0, it[i].r1, &p, &len);
                    if (pass == 1 && off + len > lo && off < hi) {
                        const size_t a = off < lo ? lo - off : 0, b = off + len > hi ? hi - off : len;
                        produce_sink += touch(p + a, b - a);
                    }
                    off += len;
                }
            }
        }
        tot = off;
    }
}

static void busy_wait_us(double us) {
    const double end = now_s(CLOCK_MONOTONIC) + us * 1e-6;
    while (now_s(CLOCK_MONOTONIC) < end) krt_cpu_relax();
}

static void op_sync(krt_pool *pool, int ith, long tok, int op) {
    const int last = op == G.nops - 1;
    const long ntok = last ? tok + 1 : tok;
    const int nop = last ? 0 : op + 1;
    const int pf = G.prefetch && G.threads > 1;
    if (pf) queue_prefetch(ith, ntok, nop);
    if (!(G.part == KRT_PART_STATIC && G.sync_each))
        for (int b = 0; b < G.op_barriers; b++) krt_barrier_idle(pool, ith, pf ? krt_prefetch_idle : NULL, &PQ[ith]);
    if (G.gaps[op] > 0) {
        if (ith == 0) busy_wait_us(G.gaps[op]);
        krt_barrier_idle(pool, ith, pf ? krt_prefetch_idle : NULL, &PQ[ith]);
    }
}

static void token_loop(krt_pool *pool, int ith, int nth, void *arg) {
    (void)nth;
    const long t_first = *(const long *)arg;
    for (long tok = t_first;; tok++) {
        TC[ith].tok = tok;
        double t0 = 0;
        if (ith == 0) {
            t0 = now_s(CLOCK_MONOTONIC);
            for (int op = 0; op < G.nops; op++) krt_plan_reset(G.plan[op][(tok + 1) & 1]);
        }
        for (int op = 0; op < G.nops; op++) {
            TC[ith].op = op;
            // decided before the last op's work, published by its barrier(s)
            if (ith == 0 && op == G.nops - 1) {
                const long done = tok - t_first + 1;
                if (G.run_secs <= 0 || now_s(CLOCK_MONOTONIC) - G.t_start >= G.run_secs || (G.max_tokens && done >= G.max_tokens))
                    __atomic_store_n(&G.stop_tok, tok, __ATOMIC_RELAXED);
            }
            if (ith >= G.threads - G.producers) produce(ith - (G.threads - G.producers), tok, op);
            else krt_plan_exec(pool, G.plan[op][tok & 1], G.rotate ? (int)((ith + tok) % G.threads) : ith, kfn, &TC[ith]);
            op_sync(pool, ith, tok, op);
        }
        if (ith == 0) {
            if (G.tokens < G.tok_cap) G.tok_us[G.tokens] = (now_s(CLOCK_MONOTONIC) - t0) * 1e6;
            G.tokens++;
        }
        if (__atomic_load_n(&G.stop_tok, __ATOMIC_RELAXED) == tok) break;
    }
}

static int cmp_d(const void *a, const void *b) {
    const double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : x > y;
}

static void run(krt_pool *pool, long first, double secs, long max_tokens) {
    G.stop_tok = LONG_MAX; G.tokens = 0; G.run_secs = secs; G.max_tokens = max_tokens;
    for (int op = 0; op < G.nops; op++) { krt_plan_reset(G.plan[op][0]); krt_plan_reset(G.plan[op][1]); }
    G.t_start = now_s(CLOCK_MONOTONIC);
    krt_pool_run(pool, token_loop, &first);
}

static double check(void) {
    double worst = 0;
    for (int op = 0; op < G.nops; op++)
        for (int t = 0; t < G.ntask[op]; t++) {
            const int64_t N = G.W.k[G.tkind[op][t]].N;
            double err = 0, mag = 0;
            for (int64_t r = 0; r < N; r++) {
                if (!isfinite(G.Y[op][t][r])) return INFINITY;
                const double d = fabs((double)G.Y[op][t][r] - G.Yref[op][t][r]);
                if (d > err) err = d;
                if (fabs(G.Yref[op][t][r]) > mag) mag = fabs(G.Yref[op][t][r]);
            }
            if (err / (mag > 0 ? mag : 1) > worst) worst = err / (mag > 0 ? mag : 1);
        }
    return worst;
}

static void build_plans(void) {
    for (int op = 0; op < G.nops; op++) {
        krt_task tasks[MAX_TASKS];
        for (int t = 0; t < G.ntask[op]; t++) {
            const kind_t *kd = &G.W.k[G.tkind[op][t]];
            tasks[t] = (krt_task){kd->N, G.ksplit, (double)row_bytes(kd->K), G.Y[op][t]};
        }
        for (int c = 0; c < 2; c++) {
            G.plan[op][c] = krt_plan_build(G.part, tasks, G.ntask[op], G.threads - G.producers, G.align, G.tile);
            if (!G.plan[op][c]) { fprintf(stderr, "plan build failed\n"); exit(2); }
            krt_plan_set_sync_each(G.plan[op][c], G.sync_each);
        }
    }
}

static void list_workloads(void) {
    for (int i = 0; i < NWL; i++) {
        printf("%-14s %s\n", WORKLOADS[i].name, WORKLOADS[i].desc);
    }
}

static int usage(const char *m) {
    fprintf(stderr, "rtbench: %s (see the header of rtbench.c; --list shows workloads)\n", m);
    return 2;
}

int main(int argc, char **argv) {
    G.threads = 1; G.part = KRT_PART_BALANCED; G.ksplit = 1; G.op_barriers = 1; G.tile = 128; G.align = 16;
    G.secs = 2; G.footprint_mb = -1; G.pf_bytes = 256 << 10; G.pf_hint = 0; G.pf_predict = 1.0;
    const char *wl = "olmoe-moe", *regime = "cold", *gaps = NULL, *waitname = "spin";
    double serial_us = 0;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i], *v = i + 1 < argc ? argv[i + 1] : "";
        if (!strcmp(a, "--list")) { list_workloads(); return 0; }
        else if (!strcmp(a, "--impl")) G.impl = v, i++;
        else if (!strcmp(a, "--workload")) wl = v, i++;
        else if (!strcmp(a, "--part")) { if ((G.part = krt_part_parse(v)) < 0) return usage("unknown --part"); i++; }
        else if (!strcmp(a, "--threads")) G.threads = atoi(v), i++;
        else if (!strcmp(a, "--regime")) regime = v, i++;
        else if (!strcmp(a, "--footprint")) G.footprint_mb = atof(v), i++;
        else if (!strcmp(a, "--secs")) G.secs = atof(v), i++;
        else if (!strcmp(a, "--ksplit")) G.ksplit = atoi(v), i++;
        else if (!strcmp(a, "--tile")) G.tile = atol(v), i++;
        else if (!strcmp(a, "--align")) G.align = atol(v), i++;
        else if (!strcmp(a, "--wait")) waitname = v, i++;
        else if (!strcmp(a, "--op-barriers")) G.op_barriers = atoi(v), i++;
        else if (!strcmp(a, "--sync-each")) G.sync_each = 1;
        else if (!strcmp(a, "--rotate")) G.rotate = 1;
        else if (!strcmp(a, "--producers")) G.producers = atoi(v), i++;
        else if (!strcmp(a, "--gaps")) gaps = v, i++;
        else if (!strcmp(a, "--serial-us")) serial_us = atof(v), i++;
        else if (!strcmp(a, "--prefetch")) { G.prefetch = !strcmp(v, "wait"); i++; }
        else if (!strcmp(a, "--pf-kb")) G.pf_bytes = (size_t)(atof(v) * 1024), i++;
        else if (!strcmp(a, "--pf-hint")) G.pf_hint = atoi(v), i++;
        else if (!strcmp(a, "--pf-predict")) G.pf_predict = atof(v), i++;
        else if (!strcmp(a, "--static-w")) G.static_w = atof(v), i++;
        else if (!strcmp(a, "--seed")) G.seed = strtoull(v, NULL, 0), i++;
        else if (!strcmp(a, "--csv")) G.csv = v, i++;
        else if (!strcmp(a, "--label")) G.label = v, i++;
        else if (!strcmp(a, "--dump")) G.dump = v, i++;
        else { fprintf(stderr, "unknown arg %s\n", a); return usage("bad arguments"); }
    }
    if (!G.impl) return usage("--impl is required");
    if (krt_wait_parse(waitname, &G.wait)) return usage("bad --wait");
    if (G.threads < 1 || G.threads > KRT_MAX_THREADS) return usage("bad --threads");
    int wi = -1;
    for (int i = 0; i < NWL; i++) if (!strcmp(wl, WORKLOADS[i].name)) wi = i;
    if (wi < 0) return usage("unknown --workload");
    G.W = WORKLOADS[wi];
    G.hot = !strcmp(regime, "hot");
    if (!strcmp(G.W.name, "sweep")) { // footprint = the matrices touched every token
        const double mb = G.footprint_mb > 0 ? G.footprint_mb : 64;
        int n = (int)ceil(mb * 1e6 / ((double)row_bytes(G.W.k[0].K) * G.W.k[0].N));
        if (n > MAX_TASKS) { G.W.k[0].N *= (n + MAX_TASKS - 1) / MAX_TASKS; n = (int)ceil(mb * 1e6 / ((double)row_bytes(G.W.k[0].K) * G.W.k[0].N)); }
        G.W.k[0].pool = G.W.k[0].pick = n < 1 ? 1 : n;
        G.footprint_mb = 0;
        G.hot = 1;
    } else if (G.footprint_mb < 0) G.footprint_mb = G.hot ? 0 : 1200;
    for (int op = 0; op < MAX_OPS; op++) G.gaps[op] = 0;
    if (gaps) {
        char *s = strdup(gaps), *p = strtok(s, ",");
        for (int op = 0; p && op < MAX_OPS; op++, p = strtok(NULL, ",")) G.gaps[op] = atof(p);
        free(s);
    } else if (serial_us > 0) {
        for (int op = 0; op < MAX_OPS; op++) G.gaps[op] = serial_us;
    }
    for (int k = 0; k < G.W.nk; k++)
        if (G.W.k[k].K % (64 * G.ksplit)) return usage("--ksplit must leave K tiles that are multiples of 64");
    if (G.tile % G.align) return usage("--tile must be a multiple of --align");
    if (G.op_barriers < 1 && !G.sync_each) return usage("--op-barriers must be >= 1");
    if (G.producers < 0 || G.producers >= G.threads || (G.producers && (G.rotate || G.prefetch || (G.part != KRT_PART_STATIC && G.part != KRT_PART_BALANCED))))
        return usage("--producers needs 0 <= P < threads, --part static|balanced, no --rotate / --prefetch");
    // kernel
    if (!strcmp(G.impl, "read")) {
        G.read_impl = 1; G.prep = read_prepare; G.gemv = read_gemv;
    } else {
        void *h = dlopen(G.impl, RTLD_NOW | RTLD_LOCAL);
        if (!h) { fprintf(stderr, "dlopen: %s\n", dlerror()); return 2; }
        G.prep = (prep_fn)dlsym(h, "kq8_gemv_prepare");
        G.gemv = (gemv_fn)dlsym(h, "kq8_gemv_packed");
        if (G.prep && G.gemv) {
            G.layout_vnni16 = 1; // kurn's packed Q8_0 GEMV layout (16-row groups)
        } else if ((native_gemv = (gemv_fn)dlsym(h, "kq8_gemv"))) {
            G.prep = native_prepare;
            G.gemv = native_gemv;
        } else {
            return usage("kernel must export kq8_gemv, or kq8_gemv_prepare / kq8_gemv_packed");
        }
    }
    char loadavg[64] = "?";
    FILE *la = fopen("/proc/loadavg", "r");
    if (la) { if (fscanf(la, "%63s", loadavg) != 1) strcpy(loadavg, "?"); fclose(la); }
    make_data();
    build_plans();
    G.tok_cap = 1 << 20;
    G.tok_us = calloc(G.tok_cap, sizeof(double));
    krt_pool *pool = krt_pool_create(G.threads, NULL, 1, G.wait);
    // correctness: token 0 (layer copy 0, token 0's experts) with every thread
    for (int op = 0; op < G.nops; op++)
        for (int t = 0; t < G.ntask[op]; t++) memset(G.Y[op][t], 0xff, sizeof(float) * G.W.k[G.tkind[op][t]].N);
    run(pool, 0, 0, 1);
    const double rel = G.read_impl ? 0 : check();
    if (G.dump) {
        FILE *f = fopen(G.dump, "wb");
        for (int op = 0; f && op < G.nops; op++)
            for (int t = 0; t < G.ntask[op]; t++) fwrite(G.Y[op][t], sizeof(float), G.W.k[G.tkind[op][t]].N, f);
        if (f) fclose(f);
    }
    // warm-up, then timing
    run(pool, 1, G.secs < 0.3 ? G.secs : 0.3, 0);
    krt_stats_reset(pool);
    krt_stats_enable(pool, 1);
    const double c0 = cpu_s(), w0 = now_s(CLOCK_MONOTONIC), r0 = now_s(CLOCK_REALTIME);
    run(pool, 1, G.secs, 0);
    const double wall = now_s(CLOCK_MONOTONIC) - w0, cpu = cpu_s() - c0, drift = (now_s(CLOCK_REALTIME) - r0) - wall;
    krt_stats_enable(pool, 0);
    double busy = 0, waitt = 0;
    uint64_t steals = 0;
    char spin_t[KRT_MAX_THREADS * 6 + 1] = "";
    for (int t = 0; t < G.threads; t++) {
        const krt_stats *s = krt_stats_of(pool, t);
        busy += s->busy_ns * 1e-9; waitt += s->wait_ns * 1e-9; steals += s->steals;
        snprintf(spin_t + strlen(spin_t), sizeof spin_t - strlen(spin_t), "%s%.0f", t ? ";" : "", 100 * s->wait_ns * 1e-9 / wall);
    }
    const long ntok = G.tokens;
    const long nmed = ntok < G.tok_cap ? ntok : G.tok_cap;
    qsort(G.tok_us, nmed, sizeof(double), cmp_d);
    const double us_mean = wall / ntok * 1e6, us_med = G.tok_us[nmed / 2];
    double imb = 0;
    for (int op = 0; op < G.nops; op++) imb = fmax(imb, krt_plan_imbalance(G.plan[op][0]));
    const double gbs = G.bytes_per_tok / (us_med * 1e-6) / 1e9;
    const double spin_share = waitt / (G.threads * wall);
    const double j_tok = cpu / ntok * PROXY_W_PER_CORE + G.static_w * wall / ntok;
    const double tok_s_model = 1e6 / (us_med * G.W.model_layers);
    const char *ok = rel < 1e-5 ? "ok" : rel < 1e-3 ? "approx" : "FAIL";
    char wbuf[32];
    krt_wait_name(&G.wait, wbuf, sizeof wbuf);
    char gbuf[96] = "";
    for (int op = 0; op < G.nops; op++) snprintf(gbuf + strlen(gbuf), sizeof gbuf - strlen(gbuf), "%s%g", op ? "/" : "", G.gaps[op]);
    const char *label = G.label ? G.label : krt_part_name(G.part);
    printf("spin %% per thread: %s\n", spin_t); // before the summary line, which stays last (`| tail -1`)
    printf("%-18s %-13s %-8s T=%d %s fp=%.0fMB L=%d ks=%d tile=%ld wait=%s gaps=%s pf=%s | med %8.1f us/tok mean %8.1f | %6.1f GB/s"
           " | cpu/wall %.2f spin %.1f%% imb %.3f steals %.1f/tok | %.3f mJ/tok | model %.1f tok/s | relerr %.1e %s%s load %s\n",
           label, G.W.name, krt_part_name(G.part), G.threads, G.hot ? "hot" : "cold", G.footprint_mb, G.L, G.ksplit, (long)G.tile, wbuf,
           gbuf, G.prefetch ? "wait" : "none", us_med, us_mean, gbs, cpu / wall, 100 * spin_share, imb, (double)steals / ntok, j_tok * 1e3,
           tok_s_model, rel, ok, fabs(drift) > 0.05 ? " DRIFT!" : "", loadavg);
    if (G.csv) {
        FILE *f = fopen(G.csv, "a");
        if (f) {
            fseek(f, 0, SEEK_END);
            if (ftell(f) == 0)
                fprintf(f, "label,workload,part,threads,regime,footprint_mb,layers,ksplit,tile,wait,gaps,prefetch,pf_kb,pf_predict,op_barriers,"
                           "tokens,wall_s,cpu_s,us_med,us_mean,bytes_per_tok,GBps,cpu_wall,busy_s,wait_s,spin_share,imbalance,steals_per_tok,"
                           "proxy_mJ_tok,static_w,model_tok_s,relerr,check,drift_s,load,spin_pct_per_thread\n");
            fprintf(f, "%s,%s,%s,%d,%s,%.0f,%d,%d,%ld,%s,%s,%s,%.0f,%.2f,%d,%ld,%.6f,%.6f,%.3f,%.3f,%.0f,%.3f,%.3f,%.6f,%.6f,%.5f,%.4f,%.3f,%.5f,%.1f,%.3f,%.3e,%s,%.4f,%s,%s\n",
                    label, G.W.name, krt_part_name(G.part), G.threads, G.hot ? "hot" : "cold", G.footprint_mb, G.L, G.ksplit, (long)G.tile, wbuf,
                    gbuf, G.prefetch ? "wait" : "none", G.pf_bytes / 1024.0, G.pf_predict, G.op_barriers, ntok, wall, cpu, us_med, us_mean,
                    G.bytes_per_tok, gbs, cpu / wall, busy, waitt, spin_share, imb, (double)steals / ntok, j_tok * 1e3, G.static_w, tok_s_model,
                    rel, ok, drift, loadavg, spin_t);
            fclose(f);
        }
    }
    krt_pool_destroy(pool);
    return rel < 1e-3 ? 0 : 1;
}
