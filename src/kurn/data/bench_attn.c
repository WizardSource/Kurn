/* kurn attention harness: times a generated kattn library and checks it against a
 * double-precision reference softmax(q kᵀ * scale + mask) v computed on exactly the
 * K/V values the kernel sees (dequantized from the stored format).
 *
 *   bench_attn --impl lib.so --nq 1 --nkv 4096 --heads 16 --kv-heads 8 --threads 8 [--regime cold]
 *   bench_attn --peak --threads 8        (AMX-BF16 / AVX-512 BF16 / FP32 FMA peak throughput)
 *
 * Inputs: k, v ~ N(0, 1), q ~ N(0, 3^2): logits have std ~3, so the softmax is peaked
 * (a near-uniform softmax would hide errors). The CSV row has the same leading
 * columns as data/bench.c.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <immintrin.h>
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#include "kurn_attn.h"

typedef size_t (*ws_fn)(const kattn_args *, int);
typedef void (*attn_fn)(const kattn_args *, void *, int, int);
typedef int64_t (*cfg_fn)(int *, int *, int *);
typedef size_t (*packb_fn)(const kattn_args *, int64_t);
typedef void (*pack_fn)(const kattn_args *, void *, int64_t, int64_t, int64_t, int, int);
typedef void (*packed_fn)(const kattn_args *, const void *, int64_t, void *, int, int);

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + 1e-9 * t.tv_nsec;
}
static double cpu_now(void) {
    struct timespec t;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t);
    return t.tv_sec + 1e-9 * t.tv_nsec;
}
static double drift(void) {
    struct timespec a, b;
    clock_gettime(CLOCK_REALTIME, &a);
    clock_gettime(CLOCK_MONOTONIC, &b);
    return (a.tv_sec - b.tv_sec) + 1e-9 * (a.tv_nsec - b.tv_nsec);
}

static uint64_t rng_s = 0x9E3779B97F4A7C15ull;
static double urand(void) {
    rng_s ^= rng_s << 13; rng_s ^= rng_s >> 7; rng_s ^= rng_s << 17;
    return ((rng_s >> 11) + 0.5) * (1.0 / 9007199254740992.0);
}
static float nrand(void) { return (float)(sqrt(-2.0 * log(urand())) * cos(6.283185307179586 * urand())); }

static uint16_t to_bf16(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    u += 0x7FFF + ((u >> 16) & 1);
    return (uint16_t)(u >> 16);
}
static float from_bf16(uint16_t h) {
    uint32_t u = (uint32_t)h << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}

static int KV, DK, DV;
#define ROW_Q4_0 100 /* V rows of KATTN_KV_K4C_Q4 */
static int is_k4c(void) { return KV == KATTN_KV_K4C_Q4 || KV == KATTN_KV_K4C_Q8; }
/* row format of V (and of K for the row formats) */
static int row_fmt(void) { return KV == KATTN_KV_K4C_Q4 ? ROW_Q4_0 : KV == KATTN_KV_K4C_Q8 ? KATTN_KV_Q8_0 : KV; }
static int64_t row_bytes(int d) {
    const int f = row_fmt();
    return f == KATTN_KV_Q8_0 ? d / 32 * 34 : f == ROW_Q4_0 ? d / 32 * 18 : 2 * (int64_t)d;
}

static void encode_row(const float *x, int d, uint8_t *dst) {
    const int f = row_fmt();
    if (f == KATTN_KV_F16) {
        for (int i = 0; i < d; i++) ((uint16_t *)dst)[i] = _cvtss_sh(x[i], 0);
    } else if (f == KATTN_KV_BF16) {
        for (int i = 0; i < d; i++) ((uint16_t *)dst)[i] = to_bf16(x[i]);
    } else if (f == ROW_Q4_0) { /* ggml quantize_row_q4_0_ref */
        for (int b = 0; b < d / 32; b++) {
            float amax = 0, mx = 0;
            for (int i = 0; i < 32; i++)
                if (amax < fabsf(x[32 * b + i])) { amax = fabsf(x[32 * b + i]); mx = x[32 * b + i]; }
            const float dd = mx / -8, id = dd ? 1.0f / dd : 0.0f;
            uint8_t *blk = dst + 18 * b;
            *(uint16_t *)blk = _cvtss_sh(dd, 0);
            for (int i = 0; i < 16; i++) {
                const int lo = (int8_t)(x[32 * b + i] * id + 8.5f), hi = (int8_t)(x[32 * b + 16 + i] * id + 8.5f);
                blk[2 + i] = (uint8_t)((lo < 15 ? lo : 15) | ((hi < 15 ? hi : 15) << 4));
            }
        }
    } else {
        for (int b = 0; b < d / 32; b++) {
            float amax = 0;
            for (int i = 0; i < 32; i++) amax = fmaxf(amax, fabsf(x[32 * b + i]));
            const float dd = amax / 127.0f, id = dd ? 1.0f / dd : 0.0f;
            uint8_t *blk = dst + 34 * b;
            *(uint16_t *)blk = _cvtss_sh(dd, 0);
            for (int i = 0; i < 32; i++) ((int8_t *)blk)[2 + i] = (int8_t)lrintf(x[32 * b + i] * id);
        }
    }
}

static void decode_row(const uint8_t *src, int d, double *out) {
    const int f = row_fmt();
    if (f == KATTN_KV_F16) {
        for (int i = 0; i < d; i++) out[i] = _cvtsh_ss(((const uint16_t *)src)[i]);
    } else if (f == KATTN_KV_BF16) {
        for (int i = 0; i < d; i++) out[i] = from_bf16(((const uint16_t *)src)[i]);
    } else if (f == ROW_Q4_0) {
        for (int b = 0; b < d / 32; b++) {
            const double dd = _cvtsh_ss(*(const uint16_t *)(src + 18 * b));
            for (int i = 0; i < 16; i++) {
                out[32 * b + i] = dd * ((src[18 * b + 2 + i] & 15) - 8);
                out[32 * b + 16 + i] = dd * ((src[18 * b + 2 + i] >> 4) - 8);
            }
        }
    } else {
        for (int b = 0; b < d / 32; b++) {
            const double dd = _cvtsh_ss(*(const uint16_t *)(src + 34 * b));
            for (int i = 0; i < 32; i++) out[32 * b + i] = dd * ((const int8_t *)(src + 34 * b + 2))[i];
        }
    }
}

/* ------------------------------------------------------------------ pre-RoPE 4-bit per-channel K */
static int64_t k4c_tok_bytes(void) { return KATTN_K4C_BLOCK_BYTES(DK) / KATTN_K4C_GROUP; }

/* K [nkv][nhkv][DK] pre-RoPE -> group blocks [nkv / 32][nhkv] + f16 tail rows [32][nhkv][DK] */
static void k4c_encode(const float *K, int64_t nkv, int nhkv, uint8_t *blocks, uint16_t *tail) {
    const int64_t nfull = nkv / KATTN_K4C_GROUP * KATTN_K4C_GROUP;
    for (int64_t b = 0; b < nfull / KATTN_K4C_GROUP; b++)
        for (int g = 0; g < nhkv; g++) {
            uint8_t *blk = blocks + (b * nhkv + g) * KATTN_K4C_BLOCK_BYTES(DK);
            uint16_t *sc = (uint16_t *)blk, *mn = sc + DK;
            uint8_t *qs = blk + 4 * DK;
            memset(qs, 0, 16 * (size_t)DK);
            for (int c = 0; c < DK; c++) {
                float lo = INFINITY, hi = -INFINITY;
                for (int t = 0; t < KATTN_K4C_GROUP; t++) {
                    const float x = K[((b * KATTN_K4C_GROUP + t) * nhkv + g) * DK + c];
                    lo = fminf(lo, x);
                    hi = fmaxf(hi, x);
                }
                sc[c] = _cvtss_sh((hi - lo) / 15.0f, 0);
                mn[c] = _cvtss_sh(lo, 0);
                const float s = _cvtsh_ss(sc[c]), m = _cvtsh_ss(mn[c]);
                for (int t = 0; t < KATTN_K4C_GROUP; t++) {
                    const float x = K[((b * KATTN_K4C_GROUP + t) * nhkv + g) * DK + c];
                    long q = s > 0 ? lrintf((x - m) / s) : 0;
                    q = q < 0 ? 0 : q > 15 ? 15 : q;
                    qs[t * (DK / 2) + (c / 32) * 16 + (c % 16)] |= (uint8_t)(q << ((c % 32) >= 16 ? 4 : 0));
                }
            }
        }
    for (int64_t j = nfull; j < nkv; j++)
        for (int g = 0; g < nhkv; g++)
            for (int c = 0; c < DK; c++) tail[((j % KATTN_K4C_GROUP) * nhkv + g) * DK + c] = _cvtss_sh(K[(j * nhkv + g) * DK + c], 0);
}

/* dequantized, rotated K row j of head g (the reference the kernel must reproduce) */
static void k4c_decode(const kattn_args *a, int64_t j, int g, double *out) {
    const int64_t nfull = a->n_kv / KATTN_K4C_GROUP * KATTN_K4C_GROUP;
    if (j < nfull) {
        const uint8_t *blk = (const uint8_t *)a->k + (j / KATTN_K4C_GROUP) * a->k_s_tok + g * a->k_s_head;
        const uint16_t *sc = (const uint16_t *)blk, *mn = sc + DK;
        const uint8_t *qs = blk + 4 * DK + (j % KATTN_K4C_GROUP) * (DK / 2);
        for (int c = 0; c < DK; c++) {
            const int q = (qs[(c / 32) * 16 + (c % 16)] >> ((c % 32) >= 16 ? 4 : 0)) & 15;
            out[c] = (double)q * _cvtsh_ss(sc[c]) + _cvtsh_ss(mn[c]);
        }
    } else {
        const uint16_t *tr = (const uint16_t *)((const uint8_t *)a->k_tail + (j % KATTN_K4C_GROUP) * a->kt_s_tok + g * a->kt_s_head);
        for (int c = 0; c < DK; c++) out[c] = _cvtsh_ss(tr[c]);
    }
    const int rd = a->rope_dim;
    for (int i = 0; i < rd / 2; i++) {
        const double th = (double)(a->k_pos0 + j) * a->rope_freq[i], c = cos(th), s = sin(th);
        const int i0 = a->rope_mode == KATTN_ROPE_NEOX ? i : 2 * i, i1 = a->rope_mode == KATTN_ROPE_NEOX ? i + rd / 2 : 2 * i + 1;
        const double x0 = out[i0], x1 = out[i1];
        out[i0] = x0 * c - x1 * s;
        out[i1] = x0 * s + x1 * c;
    }
}

/* ------------------------------------------------------------------ thread pool */
static int NTH;
static attn_fn ATTN;
static kattn_args *CUR;
static void *WS;
static atomic_int gen, arrived, quit;

/* one thread per vCPU: spinning waiters must never share a CPU with the thread they wait
 * for, and AMX tile state is not preserved across migrations on this class of VM */
static void pin(int t) {
    cpu_set_t s;
    CPU_ZERO(&s);
    CPU_SET(t % sysconf(_SC_NPROCESSORS_ONLN), &s);
    sched_setaffinity(0, sizeof(s), &s);
}

static void *worker(void *arg) {
    const int ith = (int)(intptr_t)arg;
    pin(ith);
    int seen = 0;
    for (;;) {
        int g;
        while ((g = atomic_load_explicit(&gen, memory_order_acquire)) == seen) _mm_pause();
        seen = g;
        if (atomic_load(&quit)) return NULL;
        ATTN(CUR, WS, ith, NTH);
        atomic_fetch_add_explicit(&arrived, 1, memory_order_acq_rel);
    }
}

/* --packed: kattn_pack / kattn_packed through the same pool (the layer's buffer is found by args index) */
static kattn_args *AA;
static void **KVP;
static int64_t CAP, PJ0, PJ1;
static pack_fn PACK;
static packed_fn PACKED;
static void pack_wrap(const kattn_args *a, void *ws, int ith, int nth) { (void)ws; PACK(a, KVP[a - AA], CAP, PJ0, PJ1, ith, nth); }
static void packed_wrap(const kattn_args *a, void *ws, int ith, int nth) { PACKED(a, KVP[a - AA], CAP, ws, ith, nth); }

static void run_call(kattn_args *a) {
    CUR = a;
    atomic_store(&arrived, 0);
    atomic_fetch_add_explicit(&gen, 1, memory_order_acq_rel);
    ATTN(a, WS, 0, NTH);
    while (atomic_load_explicit(&arrived, memory_order_acquire) < NTH - 1) _mm_pause();
}

/* ------------------------------------------------------------------ peak mode */
__attribute__((target("amx-tile,amx-bf16"))) static double peak_amx(double secs) {
    syscall(SYS_arch_prctl, 0x1023, 18);
    struct __attribute__((aligned(64))) { uint8_t p, s, r[14]; uint16_t c[16]; uint8_t rows[16]; } cfg;
    memset(&cfg, 0, sizeof cfg);
    cfg.p = 1;
    for (int i = 0; i < 8; i++) { cfg.rows[i] = 16; cfg.c[i] = 64; }
    _tile_loadconfig(&cfg);
    static __attribute__((aligned(64))) uint16_t buf[4][512];
    memset(buf, 0, sizeof buf);
    _tile_loadd(4, buf[0], 64); _tile_loadd(5, buf[1], 64); _tile_loadd(6, buf[2], 64); _tile_loadd(7, buf[3], 64);
    _tile_zero(0); _tile_zero(1); _tile_zero(2); _tile_zero(3);
    double t0 = now(), t;
    long n = 0;
    do {
        for (int i = 0; i < 1000; i++) {
            _tile_dpbf16ps(0, 4, 6); _tile_dpbf16ps(1, 4, 7); _tile_dpbf16ps(2, 5, 6); _tile_dpbf16ps(3, 5, 7);
        }
        n += 4000;
    } while ((t = now() - t0) < secs);
    _tile_release();
    return n * 16.0 * 16 * 32 * 2 / t; /* FLOP/s */
}

__attribute__((target("avx512bf16,avx512f"))) static double peak_dpbf16(double secs) {
    __m512 a[12];
    for (int i = 0; i < 12; i++) a[i] = _mm512_setzero_ps();
    const __m512bh x = (__m512bh)_mm512_set1_epi32(0x3f803f80), y = (__m512bh)_mm512_set1_epi32(0x3c003c00);
    double t0 = now(), t;
    long n = 0;
    do {
        for (int i = 0; i < 1000; i++)
            for (int k = 0; k < 12; k++) a[k] = _mm512_dpbf16_ps(a[k], x, y);
        n += 12000;
        __asm__ volatile("" : "+v"(a[0]), "+v"(a[1]), "+v"(a[2]), "+v"(a[3]), "+v"(a[4]), "+v"(a[5]));
    } while ((t = now() - t0) < secs);
    float s = 0;
    for (int k = 0; k < 12; k++) s += _mm512_reduce_add_ps(a[k]);
    if (s == 12345.f) puts("");
    return n * 64.0 / t;
}

__attribute__((target("avx512f"))) static double peak_fma(double secs) {
    __m512 a[12];
    for (int i = 0; i < 12; i++) a[i] = _mm512_setzero_ps();
    const __m512 x = _mm512_set1_ps(1e-7f), y = _mm512_set1_ps(1.0f);
    double t0 = now(), t;
    long n = 0;
    do {
        for (int i = 0; i < 1000; i++)
            for (int k = 0; k < 12; k++) a[k] = _mm512_fmadd_ps(a[k], y, x);
        n += 12000;
        __asm__ volatile("" : "+v"(a[0]), "+v"(a[1]), "+v"(a[2]), "+v"(a[3]), "+v"(a[4]), "+v"(a[5]));
    } while ((t = now() - t0) < secs);
    float s = 0;
    for (int k = 0; k < 12; k++) s += _mm512_reduce_add_ps(a[k]);
    if (s == 12345.f) puts("");
    return n * 32.0 / t;
}

static double (*PEAK_FN)(double);
static double peak_res[256];
static void *peak_worker(void *arg) {
    pin((int)(intptr_t)arg);
    peak_res[(intptr_t)arg] = PEAK_FN(1.0);
    return NULL;
}
static void peak(int nth) {
    const char *names[] = {"amx_bf16", "avx512_bf16", "fp32_fma"};
    double (*fns[])(double) = {peak_amx, peak_dpbf16, peak_fma};
    for (int f = 0; f < 3; f++) {
        PEAK_FN = fns[f];
        pthread_t th[256];
        for (int i = 0; i < nth; i++) pthread_create(&th[i], NULL, peak_worker, (void *)(intptr_t)i);
        double tot = 0;
        for (int i = 0; i < nth; i++) { pthread_join(th[i], NULL); tot += peak_res[i]; }
        printf("peak %s: %.1f GFLOP/s (threads=%d)\n", names[f], tot * 1e-9, nth);
    }
}

/* ------------------------------------------------------------------ main */
static const char *arg_s(int argc, char **argv, const char *k, const char *def) {
    for (int i = 1; i < argc - 1; i++)
        if (!strcmp(argv[i], k)) return argv[i + 1];
    return def;
}
static long arg_i(int argc, char **argv, const char *k, long def) {
    const char *s = arg_s(argc, argv, k, NULL);
    return s ? atol(s) : def;
}
static int has(int argc, char **argv, const char *k) {
    for (int i = 1; i < argc; i++)
        if (!strcmp(argv[i], k)) return 1;
    return 0;
}

int main(int argc, char **argv) {
    NTH = (int)arg_i(argc, argv, "--threads", 1);
    if (has(argc, argv, "--peak")) {
        peak(NTH);
        return 0;
    }
    const char *impl = arg_s(argc, argv, "--impl", NULL);
    if (!impl) {
        fprintf(stderr, "usage: bench_attn --impl lib.so [--nq N --nkv N --heads H --kv-heads H --threads T --secs S --regime hot|cold]\n");
        return 2;
    }
    void *lib = dlopen(impl, RTLD_NOW | RTLD_LOCAL);
    if (!lib) { fprintf(stderr, "dlopen: %s\n", dlerror()); return 2; }
    ws_fn WSF = (ws_fn)dlsym(lib, "kattn_workspace");
    ATTN = (attn_fn)dlsym(lib, "kattn");
    cfg_fn CFG = (cfg_fn)dlsym(lib, "kattn_config");
    if (!WSF || !ATTN || !CFG) { fprintf(stderr, "%s: missing kattn symbols\n", impl); return 2; }
    CFG(&DK, &DV, &KV);

    const int64_t nq = arg_i(argc, argv, "--nq", 1), nkv = arg_i(argc, argv, "--nkv", 4096);
    const int64_t pos0 = arg_i(argc, argv, "--pos0", nkv - nq);
    const int nh = (int)arg_i(argc, argv, "--heads", 16), nhkv = (int)arg_i(argc, argv, "--kv-heads", 8);
    const int causal = (int)arg_i(argc, argv, "--causal", 1), use_mask = has(argc, argv, "--mask");
    const int mla = has(argc, argv, "--mla");
    /* --k-bias B: add +-B to the last 3 channels of each half of every K row (per kv head, constant over
     * tokens), like the post-RoPE k_proj bias of Qwen2/2.5 (up to ~300-430 in layer 0) */
    const float kbias = (float)atof(arg_s(argc, argv, "--k-bias", "0"));
    const double secs = atof(arg_s(argc, argv, "--secs", "1"));
    const double tol = atof(arg_s(argc, argv, "--tol", "1e-2"));
    const char *regime = arg_s(argc, argv, "--regime", "hot");
    const char *csv = arg_s(argc, argv, "--csv", NULL);
    const int64_t check_toks = arg_i(argc, argv, "--check-toks", 24) > 64 ? 64 : arg_i(argc, argv, "--check-toks", 24);
    rng_s ^= (uint64_t)arg_i(argc, argv, "--seed", 1) * 0x2545F4914F6CDD1Dull;
    if (nh % nhkv || (mla && DV > DK)) { fprintf(stderr, "bad head config\n"); return 2; }

    const int k4c = is_k4c();
    if (k4c && mla) { fprintf(stderr, "pre-RoPE K formats do not support --mla\n"); return 2; }
    const int rope_dim = (int)arg_i(argc, argv, "--rope-dim", -1) < 0 ? DK : (int)arg_i(argc, argv, "--rope-dim", -1);
    const int rope_mode = (int)arg_i(argc, argv, "--rope-mode", KATTN_ROPE_NEOX);
    const double rope_base = atof(arg_s(argc, argv, "--rope-base", "1000000"));
    const int64_t kpos0 = arg_i(argc, argv, "--kpos0", 0);
    float *rope_freq = malloc(sizeof(float) * (DK / 2 + 16));
    for (int i = 0; i < DK / 2 + 16; i++) rope_freq[i] = i < rope_dim / 2 ? (float)pow(rope_base, -2.0 * i / rope_dim) : 0.0f;
    const int64_t rk = k4c ? k4c_tok_bytes() : row_bytes(DK), rv = mla ? 0 : row_bytes(DV);
    const size_t kv_layer = (size_t)nkv * nhkv * (rk + rv);
    const size_t kblk_bytes = (size_t)(nkv / KATTN_K4C_GROUP) * nhkv * KATTN_K4C_BLOCK_BYTES(DK) + 64;
    const size_t ktail_bytes = (size_t)KATTN_K4C_GROUP * nhkv * DK * 2;
    uint16_t **kt = calloc(64, sizeof *kt);
    int nl = 1;
    if (!strcmp(regime, "cold")) {
        const double target = atof(arg_s(argc, argv, "--cold-bytes", "7e8"));
        nl = (int)ceil(target / (double)kv_layer);
        if (nl < 2) nl = 2;
        if (nl > 64) nl = 64;
    }
    const size_t qn = (size_t)nq * nh * DK, on = (size_t)nq * nh * DV;
    float *q = aligned_alloc(64, ((qn * 4 + 63) / 64) * 64);
    float *out = aligned_alloc(64, ((on * 4 + 63) / 64) * 64);
    uint8_t **kb = calloc(nl, sizeof *kb), **vb = calloc(nl, sizeof *vb);
    float *tmp = malloc(sizeof(float) * (DK > DV ? DK : DV));
    for (size_t i = 0; i < qn; i++) q[i] = 3.0f * nrand();
    for (int L = 0; L < nl && k4c; L++) {
        kb[L] = aligned_alloc(64, (kblk_bytes + 63) / 64 * 64);
        kt[L] = aligned_alloc(64, (ktail_bytes + 63) / 64 * 64);
        vb[L] = aligned_alloc(64, ((size_t)nkv * nhkv * rv + 63) / 64 * 64);
        if (L == 0) {
            float *Kf = malloc(sizeof(float) * nkv * nhkv * DK);
            for (int64_t i = 0; i < nkv * nhkv * DK; i++) Kf[i] = nrand();
            k4c_encode(Kf, nkv, nhkv, kb[0], kt[0]);
            free(Kf);
            for (int64_t j = 0; j < nkv * nhkv; j++) {
                for (int i = 0; i < DV; i++) tmp[i] = nrand();
                encode_row(tmp, DV, vb[0] + j * rv);
            }
        } else {
            memcpy(kb[L], kb[0], kblk_bytes);
            memcpy(kt[L], kt[0], ktail_bytes);
            memcpy(vb[L], vb[0], (size_t)nkv * nhkv * rv);
        }
    }
    for (int L = 0; L < nl && !k4c; L++) {
        kb[L] = aligned_alloc(64, ((size_t)nkv * nhkv * rk + 63) / 64 * 64);
        vb[L] = mla ? kb[L] : aligned_alloc(64, ((size_t)nkv * nhkv * rv + 63) / 64 * 64);
        if (L == 0) {
            for (int64_t j = 0; j < nkv * nhkv; j++) {
                for (int i = 0; i < DK; i++) tmp[i] = nrand();
                if (kbias != 0.0f)
                    for (int c = 0; c < 6; c++) {
                        const int i = mla ? DK - 1 - c : (c < 3 ? DK / 2 : DK) - 1 - c % 3; /* MLA: rope dims only, not v */
                        tmp[i] += ((((j % nhkv) * 7 + c * 3) >> 1) & 1 ? -kbias : kbias);
                    }
                encode_row(tmp, DK, kb[0] + j * rk);
                if (!mla) {
                    for (int i = 0; i < DV; i++) tmp[i] = nrand();
                    encode_row(tmp, DV, vb[0] + j * rv);
                }
            }
        } else {
            memcpy(kb[L], kb[0], (size_t)nkv * nhkv * rk);
            if (!mla) memcpy(vb[L], vb[0], (size_t)nkv * nhkv * rv);
        }
    }
    uint16_t *mask = NULL;
    const int64_t mask_s = (nkv + 63) / 64 * 64;
    if (use_mask) { /* ggml-style causal mask, padded rows */
        mask = aligned_alloc(64, (size_t)nq * mask_s * 2);
        for (int64_t t = 0; t < nq; t++)
            for (int64_t j = 0; j < mask_s; j++) mask[t * mask_s + j] = (j <= pos0 + t && j < nkv) ? 0 : 0xFC00;
    }
    kattn_args *A = calloc(nl, sizeof *A);
    for (int L = 0; L < nl; L++) {
        kattn_args a = {nq, nkv, pos0, nh, nhkv, use_mask ? 0 : causal, (float)(1.0 / sqrt((double)DK)),
                        q, (int64_t)nh * DK, DK,
                        kb[L], nhkv * rk, rk,
                        vb[L], mla ? nhkv * rk : nhkv * rv, mla ? rk : rv,
                        mask, mask_s,
                        out, (int64_t)nh * DV, DV};
        if (k4c) {
            a.k_s_tok = (int64_t)nhkv * KATTN_K4C_BLOCK_BYTES(DK);
            a.k_s_head = KATTN_K4C_BLOCK_BYTES(DK);
            a.k_tail = kt[L];
            a.kt_s_tok = (int64_t)nhkv * DK * 2;
            a.kt_s_head = (int64_t)DK * 2;
            a.rope_freq = rope_freq;
            a.k_pos0 = kpos0;
            a.rope_dim = rope_dim;
            a.rope_mode = rope_mode;
        }
        A[L] = a;
    }
    const size_t wsz = WSF(&A[0], NTH);
    WS = aligned_alloc(64, (wsz + 63) / 64 * 64);
    memset(WS, 0, (wsz + 63) / 64 * 64);
    pthread_t th[256];
    pin(0);
    for (int i = 1; i < NTH; i++) pthread_create(&th[i], NULL, worker, (void *)(intptr_t)i);

    if (has(argc, argv, "--packed")) {
        packb_fn PB = (packb_fn)dlsym(lib, "kattn_pack_bytes");
        PACK = (pack_fn)dlsym(lib, "kattn_pack");
        PACKED = (packed_fn)dlsym(lib, "kattn_packed");
        const size_t pb = PB ? PB(&A[0], nkv) : 0;
        if (!PACK || !PACKED || !pb) { fprintf(stderr, "%s: no packed KV support\n", impl); return 2; }
        const int64_t split = arg_i(argc, argv, "--packed-split", 0);
        AA = A;
        CAP = nkv;
        KVP = calloc(nl, sizeof *KVP);
        ATTN = pack_wrap;
        double tpack = 0;
        for (int L = 0; L < nl; L++) {
            KVP[L] = aligned_alloc(64, (pb + 63) / 64 * 64);
            memset(KVP[L], 0, (pb + 63) / 64 * 64);
            const double t0 = now();
            if (split > 0 && split < nkv) {
                PJ0 = 0; PJ1 = split; run_call(&A[L]);
                PJ0 = split; PJ1 = nkv; run_call(&A[L]);
            } else {
                PJ0 = 0; PJ1 = nkv; run_call(&A[L]);
            }
            if (L == 0) tpack = now() - t0;
        }
        /* cost of one decode-step append (repacks the token's 16-token group) */
        int reps = 0;
        const double t1 = now();
        do { PJ0 = nkv - 1; PJ1 = nkv; run_call(&A[0]); reps++; } while (now() - t1 < 0.05);
        fprintf(stderr, "KATTN_PACK bytes %zu full_pack_us %.1f per_token_us %.4f append1_us %.2f\n", pb, tpack * 1e6,
                tpack * 1e6 / nkv, (now() - t1) * 1e6 / reps);
        ATTN = packed_wrap;
    }

    /* correctness on layer 0 */
    for (size_t i = 0; i < on; i++) out[i] = NAN;
    run_call(&A[0]);
    double maxerr = 0, maxref = 0;
    int nan_seen = 0;
    {
        double *kd = malloc(sizeof(double) * nkv * DK), *vd = malloc(sizeof(double) * nkv * DV), *s = malloc(sizeof(double) * nkv);
        int64_t toks[64], nt = 0;
        if (nq <= check_toks) {
            for (int64_t t = 0; t < nq; t++) toks[nt++] = t;
        } else {
            toks[nt++] = 0; toks[nt++] = nq - 1;
            while (nt < check_toks && nt < 64) toks[nt++] = (int64_t)(urand() * nq);
        }
        for (int g = 0; g < nhkv; g++) {
            for (int64_t j = 0; j < nkv; j++) {
                if (k4c) k4c_decode(&A[0], j, g, kd + j * DK);
                else decode_row(kb[0] + (j * nhkv + g) * rk, DK, kd + j * DK);
                if (mla) for (int i = 0; i < DV; i++) vd[j * DV + i] = kd[j * DK + i];
                else decode_row(vb[0] + (j * nhkv + g) * rv, DV, vd + j * DV);
            }
            for (int hh = 0; hh < nh / nhkv; hh++) {
                const int h = g * (nh / nhkv) + hh;
                for (int64_t ti = 0; ti < nt; ti++) {
                    const int64_t t = toks[ti];
                    const float *qr = q + (t * nh + h) * DK;
                    double mx = -INFINITY;
                    const int64_t lim = (causal || use_mask) ? (pos0 + t < nkv - 1 ? pos0 + t : nkv - 1) : nkv - 1;
                    for (int64_t j = 0; j <= lim; j++) {
                        double acc = 0;
                        for (int i = 0; i < DK; i++) acc += (double)qr[i] * kd[j * DK + i];
                        s[j] = acc / sqrt((double)DK);
                        if (s[j] > mx) mx = s[j];
                    }
                    double sum = 0;
                    for (int64_t j = 0; j <= lim; j++) { s[j] = exp(s[j] - mx); sum += s[j]; }
                    const float *o = out + (t * nh + h) * DV;
                    for (int i = 0; i < DV; i++) {
                        double r = 0;
                        for (int64_t j = 0; j <= lim; j++) r += s[j] * vd[j * DV + i];
                        r /= sum;
                        if (isnan(o[i])) nan_seen = 1;
                        maxerr = fmax(maxerr, fabs(o[i] - r));
                        maxref = fmax(maxref, fabs(r));
                    }
                }
            }
        }
        free(kd); free(vd); free(s);
    }
    const double relerr = nan_seen ? INFINITY : maxerr / (maxref > 0 ? maxref : 1);
    const int ok = relerr <= tol;

    /* timing */
    const double d0 = drift();
    int64_t calls = 0;
    for (int L = 0; L < nl; L++) run_call(&A[L]); /* warm-up */
    const double t0 = now(), c0 = cpu_now();
    double t;
    do {
        run_call(&A[calls % nl]);
        calls++;
    } while ((t = now() - t0) < secs || calls < 2);
    const double cpu = cpu_now() - c0;
    const double d1 = drift();
    atomic_store(&quit, 1);
    atomic_fetch_add(&gen, 1);
    for (int i = 1; i < NTH; i++) pthread_join(th[i], NULL);

    double pairs = 0; /* (query, key) pairs per head */
    for (int64_t tt = 0; tt < nq; tt++) {
        int64_t n = nkv;
        if (causal || use_mask) n = pos0 + tt + 1 < nkv ? pos0 + tt + 1 : nkv;
        pairs += n;
    }
    const double kv_seen = (causal || use_mask) ? (pos0 + nq < nkv ? pos0 + nq : nkv) : nkv;
    const double flop = 2.0 * pairs * (DK + DV) * nh;
    const double bytes = kv_seen * nhkv * (rk + rv) + 4.0 * (qn + on);
    const double us = t / calls * 1e6;
    const char *kvn[] = {"f16", "bf16", "q8_0", "k4c_q4", "k4c_q8"};
    char row[1024];
    snprintf(row, sizeof row, "kurn,attn_%s_d%d,%s,%d,%lld,%lld,%d,%d,%lld,%.6f,%.6f,%.3f,%.2f,%.2f,%.3f,%.3e,%s,%.6f,%d,%d",
             kvn[KV], DK, regime, NTH, (long long)nq, (long long)nkv, nh, nhkv, (long long)calls, t, cpu, us,
             flop / (us * 1e3), bytes / (us * 1e3), cpu / calls * 1e6 * 5.47, relerr, ok ? "ok" : "FAIL", d1 - d0, DV, mla);
    if (csv) {
        FILE *f = fopen(csv, "w");
        fprintf(f, "%s\n", row);
        fclose(f);
    }
    printf("impl,kernel,regime,threads,n_q,n_kv,heads,kv_heads,calls,wall_s,cpu_s,us_per_call,GFLOPs,GBps,proxy_uJ_per_call,relerr,check,drift_s,dv,mla\n%s\n", row);
    return ok ? 0 : 1;
}
