// Checks of GGML_TYPE_K4C (k4c/) in a patched llama.cpp build:
//   - SET_ROWS into a K4C cache in llama.cpp's write patterns (a prefill that ends inside a group, one-row appends,
//     a rollback that rewrites the last rows, out-of-order rows) keeps every row within 4-bit per-channel error,
//     leaves unwritten rows empty, and clearing rows does not touch the others;
//   - FLASH_ATTN_EXT on the K4C cache (kurn's k4c kernels, V Q4_0 or Q8_0) matches ggml's FA on the same keys
//     dequantized to F16 (mixed K/V types, so ggml computes it);
//   - `invariance`: with GGML_KURN_FA_MODE=exact, every row of a batch equals the same row computed alone.
//
//   test_k4c [threads]          exit code = number of failing checks
//   test_k4c invariance         run with GGML_KURN_FA_MODE=exact
//
// Build against a llama.cpp checkout with apply.sh and k4c/apply.sh applied:
//   gcc -O2 -I$L/ggml/include test_k4c.c -L$L/build/bin -lggml -lggml-base -lggml-cpu -lm -Wl,-rpath,$L/build/bin
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define KV 256  // cache cells

static uint64_t rng = 88172645463325252ull;
static float frand(void) {
    rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
    return (float)((rng >> 40) * (1.0 / (1ull << 23))) - 1.0f;
}

static int nfail, npass, threads = 4;
static void check(int ok, const char *what, double v) {
    printf("%-64s %10.3g  %s\n", what, v, ok ? "ok" : "FAIL");
    ok ? npass++ : nfail++;
}

typedef struct {
    int dk, nhkv, nh;
    enum ggml_type vt;
    int64_t n_embd;
    ggml_backend_t be;
    struct ggml_context *ctx;
    ggml_backend_buffer_t buf;
    struct ggml_tensor *kc, *vc;  // caches [n_embd, KV]
    float *truth;                 // [KV][n_embd], what was written
    int written[KV];
} cache_t;

static void cache_init(cache_t *c, int dk, int nhkv, int G, enum ggml_type vt) {
    memset(c, 0, sizeof *c);
    c->dk = dk; c->nhkv = nhkv; c->nh = nhkv * G; c->vt = vt; c->n_embd = (int64_t)dk * nhkv;
    c->be = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(c->be, threads);
    struct ggml_init_params ip = { 4 * ggml_tensor_overhead(), NULL, true };
    c->ctx = ggml_init(ip);
    c->kc = ggml_new_tensor_2d(c->ctx, GGML_TYPE_K4C, c->n_embd, KV);
    c->vc = ggml_new_tensor_2d(c->ctx, vt, c->n_embd, KV);
    c->buf = ggml_backend_alloc_ctx_tensors(c->ctx, c->be);
    ggml_backend_buffer_clear(c->buf, 0);
    c->truth = calloc((size_t)KV * c->n_embd, sizeof(float));
}

static void cache_free(cache_t *c) {
    ggml_backend_buffer_free(c->buf);
    ggml_free(c->ctx);
    ggml_backend_free(c->be);
    free(c->truth);
}

// llama.cpp-like keys: per-channel offsets (a few large) plus noise
static float key_value(int64_t ch) {
    const float off = (ch % 37 == 3) ? 6.0f : (ch % 11 == 5) ? -2.0f : 0.3f * (float)((ch * 7) % 5);
    return off + (frand() + frand() + frand());
}

// one SET_ROWS graph writing rows `cells` (n of them) of both caches
static void write_rows(cache_t *c, const int *cells, int n) {
    struct ggml_init_params ip = { 16 * ggml_tensor_overhead() + ggml_graph_overhead(), NULL, true };
    struct ggml_context *g = ggml_init(ip);
    struct ggml_tensor *src = ggml_new_tensor_2d(g, GGML_TYPE_F32, c->n_embd, n);
    struct ggml_tensor *idx = ggml_new_tensor_1d(g, GGML_TYPE_I64, n);
    struct ggml_tensor *wk = ggml_set_rows_k4c(g, c->kc, src, idx, c->dk);
    struct ggml_tensor *wv = ggml_set_rows(g, c->vc, src, idx);
    struct ggml_cgraph *gf = ggml_new_graph(g);
    ggml_build_forward_expand(gf, wk);
    ggml_build_forward_expand(gf, wv);
    ggml_backend_buffer_t b = ggml_backend_alloc_ctx_tensors(g, c->be);
    float *f = malloc(sizeof(float) * c->n_embd * n);
    int64_t *ix = malloc(sizeof(int64_t) * n);
    for (int i = 0; i < n; i++) {
        ix[i] = cells[i];
        for (int64_t e = 0; e < c->n_embd; e++) f[i * c->n_embd + e] = key_value(e);
        memcpy(c->truth + (size_t)cells[i] * c->n_embd, f + i * c->n_embd, sizeof(float) * c->n_embd);
        c->written[cells[i]] = 1;
    }
    ggml_backend_tensor_set(src, f, 0, ggml_nbytes(src));
    ggml_backend_tensor_set(idx, ix, 0, ggml_nbytes(idx));
    ggml_backend_graph_compute(c->be, gf);
    free(f);
    free(ix);
    ggml_backend_buffer_free(b);
    ggml_free(g);
}

static const uint8_t *group(const cache_t *c, int g) {
    return (const uint8_t *)c->kc->data + (size_t)g * GGML_K4C_GROUP * c->kc->nb[1];
}

// worst per-row NMSE of the stored keys against what was written; unwritten rows must be empty
static double row_error(const cache_t *c, int *bad_empty) {
    float *r = malloc(sizeof(float) * c->n_embd);
    double worst = 0;
    *bad_empty = 0;
    for (int i = 0; i < KV; i++) {
        const int ok = ggml_k4c_get_row(group(c, i / GGML_K4C_GROUP), c->nhkv, c->dk, i % GGML_K4C_GROUP, r);
        if (!c->written[i]) {
            *bad_empty += ok;
            continue;
        }
        if (!ok) {
            (*bad_empty)++;
            continue;
        }
        double e = 0, s = 0;
        const float *t = c->truth + (size_t)i * c->n_embd;
        for (int64_t k = 0; k < c->n_embd; k++) {
            e += (r[k] - t[k]) * (r[k] - t[k]);
            s += t[k] * t[k];
        }
        worst = fmax(worst, e / s);
    }
    free(r);
    return worst;
}

// FLASH_ATTN_EXT over the whole cache (mask: unwritten cells hidden, plus a causal-like limit per query row).
// ref = 0: K4C keys (kurn); ref = 1: the same keys dequantized to F16 (ggml, mixed K/V types).
static void attention(cache_t *c, const float *q, int n_q, const int *lim, int ref, float *out) {
    struct ggml_init_params ip = { 16 * ggml_tensor_overhead() + ggml_graph_overhead(), NULL, true };
    struct ggml_context *g = ggml_init(ip);
    struct ggml_tensor *kf = NULL, *kt = c->kc;
    if (ref) {
        kf = ggml_new_tensor_2d(g, GGML_TYPE_F16, c->n_embd, KV);
        kt = kf;
    }
    struct ggml_tensor *qt = ggml_new_tensor_3d(g, GGML_TYPE_F32, c->dk, n_q, c->nh);
    struct ggml_tensor *m = ggml_new_tensor_2d(g, GGML_TYPE_F16, KV, n_q);
    struct ggml_tensor *k = ggml_permute(g, ggml_view_4d(g, kt, c->dk, c->nhkv, KV, 1, ggml_row_size(kt->type, c->dk),
                                                        ggml_row_size(kt->type, c->n_embd), ggml_row_size(kt->type, c->n_embd * KV), 0), 0, 2, 1, 3);
    struct ggml_tensor *v = ggml_permute(g, ggml_view_4d(g, c->vc, c->dk, c->nhkv, KV, 1, ggml_row_size(c->vt, c->dk),
                                                        ggml_row_size(c->vt, c->n_embd), ggml_row_size(c->vt, c->n_embd * KV), 0), 0, 2, 1, 3);
    struct ggml_tensor *o = ggml_flash_attn_ext(g, qt, k, v, m, 1.0f / sqrtf((float)c->dk), 0.0f, 0.0f);
    ggml_prec_set_acc(o, GGML_PREC_F32);
    struct ggml_cgraph *gf = ggml_new_graph(g);
    ggml_build_forward_expand(gf, o);
    ggml_backend_buffer_t b = ggml_backend_alloc_ctx_tensors(g, c->be);
    if (ref) {
        float *r = malloc(sizeof(float) * c->n_embd);
        ggml_fp16_t *h = malloc(sizeof(ggml_fp16_t) * c->n_embd * KV);
        for (int i = 0; i < KV; i++) {
            ggml_k4c_get_row(group(c, i / GGML_K4C_GROUP), c->nhkv, c->dk, i % GGML_K4C_GROUP, r);
            ggml_fp32_to_fp16_row(r, h + (size_t)i * c->n_embd, c->n_embd);
        }
        ggml_backend_tensor_set(kf, h, 0, ggml_nbytes(kf));
        free(r);
        free(h);
    }
    ggml_fp16_t *mh = malloc(sizeof(ggml_fp16_t) * KV * n_q);
    for (int t = 0; t < n_q; t++)
        for (int j = 0; j < KV; j++) mh[t * KV + j] = ggml_fp32_to_fp16(c->written[j] && j <= lim[t] ? 0.0f : -INFINITY);
    // q [dk][n_q][nh]
    ggml_backend_tensor_set(qt, q, 0, ggml_nbytes(qt));
    ggml_backend_tensor_set(m, mh, 0, ggml_nbytes(m));
    ggml_backend_graph_compute(c->be, gf);
    ggml_backend_tensor_get(o, out, 0, ggml_nbytes(o));
    free(mh);
    ggml_backend_buffer_free(b);
    ggml_free(g);
}

static void fill_q(float *q, int64_t n) {
    for (int64_t i = 0; i < n; i++) q[i] = 2.0f * (frand() + frand());
}

static void run_case(int dk, int nhkv, int G, enum ggml_type vt) {
    cache_t c;
    cache_init(&c, dk, nhkv, G, vt);
    char what[160];
    int cells[KV], n = 0;
    for (int i = 0; i < 100; i++) cells[n++] = i;  // prefill ending inside group 3
    write_rows(&c, cells, n);
    for (int i = 100; i < 141; i++) write_rows(&c, &i, 1);  // decode appends
    n = 0;
    for (int i = 130; i < 141; i++) cells[n++] = i;  // rollback: rewrite the last 11 rows
    write_rows(&c, cells, n);
    const int ooo[] = { 200, 170, 171, 250, 169 };  // out of order, several groups
    write_rows(&c, ooo, 5);
    int bad;
    const double e = row_error(&c, &bad);
    snprintf(what, sizeof what, "k4c dk %d heads %d v %s: stored-row NMSE (worst row)", dk, nhkv, ggml_type_name(vt));
    check(e < 0.02, what, e);
    snprintf(what, sizeof what, "k4c dk %d heads %d v %s: rows empty iff never written", dk, nhkv, ggml_type_name(vt));
    check(bad == 0, what, bad);

    // FA against ggml on the dequantized keys
    for (int n_q = 1; n_q <= 40; n_q += 19) {
        const int64_t nq_el = (int64_t)dk * n_q * c.nh, no = (int64_t)dk * c.nh * n_q;
        float *q = malloc(sizeof(float) * nq_el), *o1 = malloc(sizeof(float) * no), *o2 = malloc(sizeof(float) * no);
        int lim[64];
        for (int t = 0; t < n_q; t++) lim[t] = 120 + 4 * t;
        fill_q(q, nq_el);
        attention(&c, q, n_q, lim, 0, o1);
        attention(&c, q, n_q, lim, 1, o2);
        double es = 0, ss = 0;
        for (int64_t i = 0; i < no; i++) {
            es += (o1[i] - o2[i]) * (o1[i] - o2[i]);
            ss += o2[i] * o2[i];
        }
        snprintf(what, sizeof what, "FA k4c dk %d heads %d/%d v %s n_q %d: NMSE vs ggml (F16 keys)", dk, c.nh, nhkv, ggml_type_name(vt), n_q);
        check(es / ss < 1e-5 && isfinite(es), what, es / ss);
        free(q); free(o1); free(o2);
    }

    // clearing rows empties them; the group is re-encoded from the rows that stay (as if they had been written alone)
    ggml_k4c_update_group((uint8_t *)c.kc->data + 4 * GGML_K4C_GROUP * c.kc->nb[1], nhkv, dk, 0, NULL, (1u << 5) | (1u << 9));
    c.written[4 * GGML_K4C_GROUP + 5] = c.written[4 * GGML_K4C_GROUP + 9] = 0;
    const double e2 = row_error(&c, &bad);
    snprintf(what, sizeof what, "k4c dk %d: after clearing rows 5, 9 of group 4: worst NMSE (empty mismatches %d)", dk, bad);
    check(bad == 0 && e2 < 0.02, what, e2);
    cache_free(&c);
}

// the same rows written at once, or appended one by one with a rollback in between, give byte-identical groups
static void history(void) {
    cache_t a, b;
    cache_init(&a, 128, 4, 2, GGML_TYPE_Q4_0);
    cache_init(&b, 128, 4, 2, GGML_TYPE_Q4_0);
    const uint64_t seed = rng;
    int cells[KV];
    for (int i = 0; i < 77; i++) cells[i] = i;
    write_rows(&a, cells, 77);
    rng = seed;
    write_rows(&b, cells, 40);
    for (int i = 40; i < 77; i++) write_rows(&b, &i, 1);
    const uint64_t after = rng;
    for (int i = 0; i < 9; i++) cells[i] = 77 + i;  // drafted rows that get rejected
    write_rows(&b, cells, 9);
    for (int g = 2; g < 3; g++)
        ggml_k4c_update_group((uint8_t *)b.kc->data + g * GGML_K4C_GROUP * b.kc->nb[1], 4, 128, 0, NULL, 0x3FE000u);
    rng = after;
    check(memcmp(a.kc->data, b.kc->data, 3 * GGML_K4C_GROUP * a.kc->nb[1]) == 0,
          "k4c: prefill vs appends + rollback: groups byte-identical", 0);
    cache_free(&a);
    cache_free(&b);
}

static int invariance(void) {
    cache_t c;
    cache_init(&c, 128, 8, 2, GGML_TYPE_Q4_0);
    int cells[KV];
    for (int i = 0; i < 200; i++) cells[i] = i;
    write_rows(&c, cells, 200);
    const int n_q = 7;
    float *q = malloc(sizeof(float) * 128 * n_q * c.nh), *ob = malloc(sizeof(float) * 128 * c.nh * n_q);
    float *q1 = malloc(sizeof(float) * 128 * c.nh), *o1 = malloc(sizeof(float) * 128 * c.nh);
    int lim[8];
    for (int t = 0; t < n_q; t++) lim[t] = 190 + t;
    fill_q(q, 128 * n_q * c.nh);
    attention(&c, q, n_q, lim, 0, ob);
    int diff = 0;
    for (int t = 0; t < n_q; t++) {
        for (int h = 0; h < c.nh; h++) memcpy(q1 + h * 128, q + (h * n_q + t) * 128, sizeof(float) * 128);  // q [dk][n_q][nh]
        attention(&c, q1, 1, &lim[t], 0, o1);
        diff += memcmp(o1, ob + (size_t)t * c.nh * 128, sizeof(float) * c.nh * 128) != 0;  // out [dk][nh][n_q]
    }
    check(diff == 0, "k4c exact mode: batch rows equal single-row results (rows differing)", diff);
    cache_free(&c);
    return nfail;
}

int main(int argc, char **argv) {
    if (argc > 1 && !strcmp(argv[1], "invariance")) return invariance();
    if (argc > 1) threads = atoi(argv[1]);
    run_case(128, 8, 2, GGML_TYPE_Q4_0);
    run_case(128, 2, 4, GGML_TYPE_Q8_0);
    run_case(64, 4, 1, GGML_TYPE_Q4_0);
    run_case(256, 1, 8, GGML_TYPE_Q8_0);
    history();
    printf("k4c: %d passed, %d failed\n", npass, nfail);
    return nfail;
}
