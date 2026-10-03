/* ggml attention baselines on the same problem as kurn's bench_attn.c, laid out the way
 * llama.cpp's KV cache views are (q [DK, n_head, n_q] and k/v [D, n_head_kv, n_kv],
 * permuted), with a causal fp16 mask:
 *
 *   --path fa     ggml_flash_attn_ext (prec F32): tiled FP32 path when n_q >= 64 and K/V are
 *                 F16, split-KV path for single-query decode with F16 K/V and n_kv >= 512,
 *                 one-row path otherwise (e.g. Q8_0 K/V)
 *   --path nofa   mul_mat(k, q) -> soft_max_ext(mask, scale) -> mul_mat(v_trans, kq) (F16 only)
 *
 * One graph holds L independent attention ops (cold: L layers of KV > --cold-bytes), so
 * the time per op includes ggml's per-op thread synchronisation like a real model.
 * Checked against a float64 reference on the dequantized K/V (ggml's own to_float).
 *
 * L=~/src/llama.cpp; gcc -O3 -march=native -I $L/ggml/include bench_ggml_attn.c -o bench_ggml_attn \
 *     -L$L/build/bin -lggml -lggml-base -lggml-cpu -Wl,-rpath,$L/build/bin -lm
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

static double now(clockid_t c) {
    struct timespec t;
    clock_gettime(c, &t);
    return t.tv_sec + 1e-9 * t.tv_nsec;
}
static uint64_t rng_s = 0x9E3779B97F4A7C15ull;
static double urand(void) {
    rng_s ^= rng_s << 13; rng_s ^= rng_s >> 7; rng_s ^= rng_s << 17;
    return ((rng_s >> 11) + 0.5) * (1.0 / 9007199254740992.0);
}
static float nrand(void) { return (float)(sqrt(-2.0 * log(urand())) * cos(6.283185307179586 * urand())); }

static const char *arg_s(int argc, char **argv, const char *k, const char *def) {
    for (int i = 1; i < argc - 1; i++)
        if (!strcmp(argv[i], k)) return argv[i + 1];
    return def;
}
static long arg_i(int argc, char **argv, const char *k, long def) {
    const char *s = arg_s(argc, argv, k, NULL);
    return s ? atol(s) : def;
}

int main(int argc, char **argv) {
    const char *path = arg_s(argc, argv, "--path", "fa"), *kvs = arg_s(argc, argv, "--kv", "f16");
    const char *regime = arg_s(argc, argv, "--regime", "hot"), *csv = arg_s(argc, argv, "--csv", NULL);
    const int D = (int)arg_i(argc, argv, "--dk", 128), nth = (int)arg_i(argc, argv, "--threads", 8);
    const int64_t nq = arg_i(argc, argv, "--nq", 1), nkv = arg_i(argc, argv, "--nkv", 4096);
    const int64_t pos0 = arg_i(argc, argv, "--pos0", nkv - nq);
    const int nh = (int)arg_i(argc, argv, "--heads", 16), nhkv = (int)arg_i(argc, argv, "--kv-heads", 8);
    const double secs = atof(arg_s(argc, argv, "--secs", "1")), tol = atof(arg_s(argc, argv, "--tol", "1e-2"));
    const int64_t check_toks = arg_i(argc, argv, "--check-toks", 8) > 64 ? 64 : arg_i(argc, argv, "--check-toks", 8);
    const int fa = !strcmp(path, "fa");
    /* --mla: absorbed MLA, V is a view of the first DV values of each K row (DeepSeek in llama.cpp) */
    int mla = 0;
    for (int i = 1; i < argc; i++) mla |= !strcmp(argv[i], "--mla");
    const int DV = mla ? (int)arg_i(argc, argv, "--dv", 512) : D;
    if (mla && !fa) { fprintf(stderr, "--mla needs --path fa\n"); return 2; }
    const enum ggml_type kt = !strcmp(kvs, "q8_0") ? GGML_TYPE_Q8_0 : !strcmp(kvs, "bf16") ? GGML_TYPE_BF16 : GGML_TYPE_F16;
    if (!fa && kt != GGML_TYPE_F16) { fprintf(stderr, "nofa needs f16 (llama.cpp needs FA for a quantized V cache)\n"); return 2; }

    const size_t rowb = ggml_row_size(kt, D);
    const size_t kv_layer = (size_t)nkv * nhkv * rowb * (mla ? 1 : 2);
    int nl = 1;
    if (!strcmp(regime, "cold")) {
        nl = (int)ceil(atof(arg_s(argc, argv, "--cold-bytes", "7e8")) / (double)kv_layer);
        if (nl < 2) nl = 2;
        if (nl > 64) nl = 64;
    }
    const float scale = (float)(1.0 / sqrt((double)D));

    struct ggml_init_params ip = {(size_t)(16 * nl + 16) * ggml_tensor_overhead() + ggml_graph_overhead_custom(16 * nl + 16, false), NULL, true};
    struct ggml_context *ctx = ggml_init(ip);
    struct ggml_tensor *q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, nh, nq);
    struct ggml_tensor *mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, nkv, nq);
    struct ggml_tensor **K = calloc(nl, sizeof *K), **V = calloc(nl, sizeof *V), **O = calloc(nl, sizeof *O);
    struct ggml_cgraph *gf = ggml_new_graph_custom(ctx, 16 * nl + 16, false);
    struct ggml_tensor *qp = ggml_permute(ctx, q, 0, 2, 1, 3); /* [D, nq, nh] */
    for (int L = 0; L < nl; L++) {
        K[L] = ggml_new_tensor_3d(ctx, kt, D, nhkv, nkv);
        struct ggml_tensor *kp = ggml_permute(ctx, K[L], 0, 2, 1, 3); /* [D, nkv, nhkv] */
        struct ggml_tensor *cur;
        if (fa) {
            V[L] = mla ? ggml_view_3d(ctx, K[L], DV, nhkv, nkv, K[L]->nb[1], K[L]->nb[2], 0) : ggml_new_tensor_3d(ctx, kt, D, nhkv, nkv);
            cur = ggml_flash_attn_ext(ctx, qp, kp, ggml_permute(ctx, V[L], 0, 2, 1, 3), mask, scale, 0.0f, 0.0f);
            ggml_prec_set_acc(cur, GGML_PREC_F32); /* [D, nh, nq] */
        } else {
            V[L] = ggml_new_tensor_3d(ctx, kt, nkv, D, nhkv); /* llama.cpp's transposed V cache */
            struct ggml_tensor *kq = ggml_mul_mat(ctx, kp, qp); /* [nkv, nq, nh] */
            ggml_prec_set_acc(kq, GGML_PREC_F32);
            kq = ggml_soft_max_ext(ctx, kq, mask, scale, 0.0f);
            struct ggml_tensor *kqv = ggml_mul_mat(ctx, V[L], kq); /* [D, nq, nh] */
            cur = ggml_cont(ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3)); /* [D, nh, nq] */
        }
        O[L] = cur;
        ggml_build_forward_expand(gf, cur);
    }
    ggml_backend_t be = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(be, nth);
    /* pinned 1:1 like bench_attn.c (llama.cpp's --cpu-mask 0xff --cpu-strict 1) */
    struct ggml_threadpool_params tpp = ggml_threadpool_params_default(nth);
    for (int i = 0; i < nth; i++) tpp.cpumask[i] = true;
    tpp.strict_cpu = true;
    ggml_backend_cpu_set_threadpool(be, ggml_threadpool_new(&tpp));
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    if (!buf) { fprintf(stderr, "alloc failed\n"); return 2; }

    /* data: q ~ N(0, 9), k, v ~ N(0, 1), same statistics as bench_attn.c */
    float *qf = malloc(sizeof(float) * D * nh * nq);
    for (int64_t i = 0; i < (int64_t)D * nh * nq; i++) qf[i] = 3.0f * nrand();
    ggml_backend_tensor_set(q, qf, 0, sizeof(float) * D * nh * nq);
    uint16_t *mk = malloc(2 * nkv * nq);
    for (int64_t t = 0; t < nq; t++)
        for (int64_t j = 0; j < nkv; j++) mk[t * nkv + j] = ggml_fp32_to_fp16(j <= pos0 + t ? 0.0f : -INFINITY);
    ggml_backend_tensor_set(mask, mk, 0, 2 * nkv * nq);
    const int64_t nrows = nkv * nhkv;
    float *kf = malloc(sizeof(float) * D * nrows), *vf = malloc(sizeof(float) * D * nrows);
    const int vs = D; /* float stride of a V row in vf (MLA: vf = kf) */
    for (int64_t i = 0; i < D * nrows; i++) kf[i] = nrand();
    for (int64_t i = 0; i < D * nrows; i++) vf[i] = nrand();
    void *kq8 = malloc(rowb * nrows), *vq8 = malloc(rowb * nrows);
    ggml_quantize_chunk(kt, kf, kq8, 0, nrows, D, NULL);
    ggml_quantize_chunk(kt, vf, vq8, 0, nrows, D, NULL);
    /* exactly what the kernel sees, back in float */
    const struct ggml_type_traits *tr = ggml_get_type_traits(kt);
    for (int64_t r = 0; r < nrows; r++) {
        tr->to_float((const char *)kq8 + r * rowb, kf + r * D, D);
        tr->to_float((const char *)vq8 + r * rowb, vf + r * D, D);
    }
    if (mla) memcpy(vf, kf, sizeof(float) * D * nrows);
    void *vt = NULL;
    if (!fa) { /* V transposed: [nkv][D][nhkv] -> per head h: [D][nkv] */
        vt = malloc(2 * (size_t)nkv * D * nhkv);
        for (int h = 0; h < nhkv; h++)
            for (int d = 0; d < D; d++)
                for (int64_t j = 0; j < nkv; j++)
                    ((uint16_t *)vt)[((size_t)h * D + d) * nkv + j] = ((uint16_t *)vq8)[(j * nhkv + h) * D + d];
    }
    for (int L = 0; L < nl; L++) {
        ggml_backend_tensor_set(K[L], kq8, 0, rowb * nrows);
        if (!mla) ggml_backend_tensor_set(V[L], fa ? vq8 : vt, 0, rowb * nrows);
    }

    ggml_backend_graph_compute(be, gf);
    float *out = malloc(sizeof(float) * DV * nh * nq);
    ggml_backend_tensor_get(O[0], out, 0, sizeof(float) * DV * nh * nq);
    double maxerr = 0, maxref = 0;
    int nan_seen = 0;
    {
        double *s = malloc(sizeof(double) * nkv);
        int64_t toks[64], nt = 0;
        if (nq <= check_toks) for (int64_t t = 0; t < nq; t++) toks[nt++] = t;
        else { toks[nt++] = 0; toks[nt++] = nq - 1; while (nt < check_toks && nt < 64) toks[nt++] = (int64_t)(urand() * nq); }
        for (int h = 0; h < nh; h++) {
            const int g = h / (nh / nhkv);
            for (int64_t ti = 0; ti < nt; ti++) {
                const int64_t t = toks[ti], lim = pos0 + t < nkv - 1 ? pos0 + t : nkv - 1;
                const float *qr = qf + (t * nh + h) * D;
                double mx = -INFINITY, sum = 0;
                for (int64_t j = 0; j <= lim; j++) {
                    double a = 0;
                    for (int i = 0; i < D; i++) a += (double)qr[i] * kf[(j * nhkv + g) * D + i];
                    s[j] = a * scale;
                    if (s[j] > mx) mx = s[j];
                }
                for (int64_t j = 0; j <= lim; j++) { s[j] = exp(s[j] - mx); sum += s[j]; }
                for (int i = 0; i < DV; i++) {
                    double r = 0;
                    for (int64_t j = 0; j <= lim; j++) r += s[j] * vf[(j * nhkv + g) * vs + i];
                    r /= sum;
                    const float o = out[(t * nh + h) * DV + i];
                    if (isnan(o)) nan_seen = 1;
                    if (fabs(o - r) > maxerr) maxerr = fabs(o - r);
                    if (fabs(r) > maxref) maxref = fabs(r);
                }
            }
        }
        free(s);
    }
    const double relerr = nan_seen ? INFINITY : maxerr / (maxref > 0 ? maxref : 1);

    struct timespec dr;
    clock_gettime(CLOCK_REALTIME, &dr);
    const double d0 = dr.tv_sec + 1e-9 * dr.tv_nsec - now(CLOCK_MONOTONIC);
    const double w0 = now(CLOCK_MONOTONIC), c0 = now(CLOCK_PROCESS_CPUTIME_ID);
    int64_t graphs = 0;
    double w;
    do { ggml_backend_graph_compute(be, gf); graphs++; } while ((w = now(CLOCK_MONOTONIC) - w0) < secs || graphs < 2);
    const double cpu = now(CLOCK_PROCESS_CPUTIME_ID) - c0;
    clock_gettime(CLOCK_REALTIME, &dr);
    const double d1 = dr.tv_sec + 1e-9 * dr.tv_nsec - now(CLOCK_MONOTONIC);
    const int64_t calls = graphs * nl;
    double pairs = 0;
    for (int64_t t = 0; t < nq; t++) pairs += pos0 + t + 1 < nkv ? pos0 + t + 1 : nkv;
    const double flop = 2.0 * pairs * (D + DV) * nh, bytes = (double)kv_layer + 4.0 * (D + DV) * nh * nq;
    const double us = w / calls * 1e6;
    char row[1024];
    snprintf(row, sizeof row, "ggml-%s,attn_%s_d%d,%s,%d,%lld,%lld,%d,%d,%lld,%.6f,%.6f,%.3f,%.2f,%.2f,%.3f,%.3e,%s,%.6f,%d,%d",
             path, kvs, D, regime, nth, (long long)nq, (long long)nkv, nh, nhkv, (long long)calls, w, cpu, us,
             flop / (us * 1e3), bytes / (us * 1e3), cpu / calls * 1e6 * 5.47, relerr, relerr <= tol ? "ok" : "FAIL", d1 - d0, DV, mla);
    if (csv) { FILE *f = fopen(csv, "w"); fprintf(f, "%s\n", row); fclose(f); }
    printf("impl,kernel,regime,threads,n_q,n_kv,heads,kv_heads,calls,wall_s,cpu_s,us_per_call,GFLOPs,GBps,proxy_uJ_per_call,relerr,check,drift_s,dv,mla\n%s\n", row);
    ggml_backend_buffer_free(buf);
    ggml_backend_free(be);
    ggml_free(ctx);
    return relerr <= tol ? 0 : 1;
}
