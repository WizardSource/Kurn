// GGML_TYPE_K4C: kurn's k4c key format as a KV cache type (kurn/integration/llama.cpp/k4c).
//
// Keys are quantized per channel to 4 bits over groups of 32 consecutive cache rows (KIVI-style
// per-channel K). A K4C tensor [n_embd = n_head * dk, n_rows] (n_rows % 32 == 0) stores group g in the
// bytes of rows [32 g, 32 g + 32) (row size n_embd * 20 / 32 bytes = 5 bits per value), as one block of
// dk * 20 bytes per head:
//     f16 scale[dk], f16 min[dk], then 32 rows of dk / 2 bytes
// with value = q * scale[c] + min[c], q in 0..15; in each 16-byte run of a row the low nibbles are channels
// c0..c0+15 and the high nibbles c0+16..c0+31. This is kurn_attn.h's KATTN_KV_K4C_* block, read by kurn's
// attention kernels (rope_dim 0: llama.cpp caches keys after RoPE).
//
// Rows are written one group at a time (ggml_k4c_update_group). A row whose nibbles are zero in every head
// is empty (never written, or cleared); written rows never look like that, because channels whose range is
// zero store code 8. Every update re-encodes the group from the exact values of its valid rows (kept in a
// bounded table, below), so a group filled one row at a time (decode), rolled back or partly cleared holds the
// same codes as the same rows written at once (prefill).
#include "ggml.h"
#include "ggml-impl.h"
#include "ggml-threading.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define K4C_G GGML_K4C_GROUP

static inline const uint8_t * k4c_row(const uint8_t * blk, int64_t dk, int r) {
    return blk + 4 * dk + (int64_t) r * (dk / 2);
}

static inline uint8_t * k4c_row_w(uint8_t * blk, int64_t dk, int r) {
    return blk + 4 * dk + (int64_t) r * (dk / 2);
}

static inline int k4c_code(const uint8_t * row, int64_t c) {
    const uint8_t b = row[(c / 32) * 16 + (c % 16)];
    return (c % 32) < 16 ? (b & 15) : (b >> 4);
}

static inline void k4c_set_code(uint8_t * row, int64_t c, int q) {
    uint8_t * b = &row[(c / 32) * 16 + (c % 16)];
    *b = (c % 32) < 16 ? (uint8_t) ((*b & 0xF0) | q) : (uint8_t) ((*b & 0x0F) | (q << 4));
}

static bool k4c_row_empty(const uint8_t * grp, int64_t n_head, int64_t dk, int r) {
    for (int64_t h = 0; h < n_head; h++) {
        const uint8_t * row = k4c_row(grp + h * dk * 20, dk, r);
        for (int64_t i = 0; i < dk / 2; i++) {
            if (row[i]) {
                return false;
            }
        }
    }
    return true;
}

bool ggml_k4c_get_row(const void * group, int64_t n_head, int64_t dk, int r, float * dst) {
    const uint8_t * grp = (const uint8_t *) group;
    if (k4c_row_empty(grp, n_head, dk, r)) {
        memset(dst, 0, sizeof(float) * n_head * dk);
        return false;
    }
    for (int64_t h = 0; h < n_head; h++) {
        const uint8_t *    blk = grp + h * dk * 20;
        const ggml_fp16_t * sc = (const ggml_fp16_t *) blk, * mn = sc + dk;
        const uint8_t *    row = k4c_row(blk, dk, r);
        for (int64_t c = 0; c < dk; c++) {
            dst[h * dk + c] = k4c_code(row, c) * GGML_FP16_TO_FP32(sc[c]) + GGML_FP16_TO_FP32(mn[c]);
        }
    }
    return true;
}

// f16 neighbours of a finite value, toward -inf / +inf
static ggml_fp16_t k4c_f16_dn(ggml_fp16_t h) {
    return (h & 0x8000) ? (ggml_fp16_t) (h + 1) : (h == 0 ? (ggml_fp16_t) 0x8001 : (ggml_fp16_t) (h - 1));
}

static ggml_fp16_t k4c_f16_up(ggml_fp16_t h) {
    return (h & 0x8000) ? (h == 0x8000 ? (ggml_fp16_t) 1 : (ggml_fp16_t) (h - 1)) : (ggml_fp16_t) (h + 1);
}

static int k4c_quant(float x, float lo, float s) {
    if (s == 0.0f) {
        return 8;
    }
    const int q = (int) lrintf((x - lo) / s);
    return q < 0 ? 0 : q > 15 ? 15 : q;
}

// Exact values of recently written groups: the values last written to each valid row (as f16: keys are rounded to f16
// before quantization, so session files that store f16 rows restore the same codes), so that a group's codes are
// always the one-shot quantization of its current rows, whatever the order of writes, rollbacks and clears. Entries are
// keyed by the group's address and checked against a hash of the block, so a group changed or freed behind the codec's
// back (buffer cleared, reallocated) just loses its entry; the least recently written groups are dropped beyond
// GGML_K4C_EXACT_MB (default 256). Without an entry, stored rows are re-encoded from their dequantized values.
typedef struct {
    const void * key;
    uint64_t     hash, stamp;
    uint32_t     valid;
    int64_t      n;
    ggml_fp16_t * x;  // [GGML_K4C_GROUP][n]
} k4c_exact_t;

static k4c_exact_t * k4c_tab;
static size_t        k4c_ntab, k4c_captab, k4c_bytes;
static uint64_t      k4c_clock;

static uint64_t k4c_hash(const uint8_t * grp, size_t bytes) {
    uint64_t h = 1469598103934665603ull, w;
    for (size_t i = 0; i + 8 <= bytes; i += 8) {
        memcpy(&w, grp + i, 8);
        h = (h ^ w) * 1099511628211ull;
    }
    return h;
}

static size_t k4c_budget(void) {
    static size_t b = 0;
    if (!b) {
        const char * e = getenv("GGML_K4C_EXACT_MB");
        b = (size_t) (e && *e ? atof(e) : 256.0) * (1u << 20) + 1;
    }
    return b;
}

static void k4c_drop(size_t i) {
    k4c_bytes -= sizeof(ggml_fp16_t) * K4C_G * k4c_tab[i].n;
    free(k4c_tab[i].x);
    k4c_tab[i] = k4c_tab[--k4c_ntab];
}

static k4c_exact_t * k4c_find(const void * key) {
    for (size_t i = 0; i < k4c_ntab; i++) {
        if (k4c_tab[i].key == key) {
            return &k4c_tab[i];
        }
    }
    return NULL;
}

// one-shot quantization of rows `valid` of x into the group (other rows empty)
static void k4c_encode(uint8_t * grp, int64_t n_head, int64_t dk, uint32_t valid, const ggml_fp16_t * xh, int64_t n) {
    float * x = (float *) malloc(sizeof(float) * K4C_G * n);
    for (int r = 0; r < K4C_G; r++) {
        if (valid >> r & 1) {
            ggml_fp16_to_fp32_row(xh + r * n, x + r * n, n);
        }
    }
    for (int64_t h = 0; h < n_head; h++) {
        uint8_t *     blk = grp + h * dk * 20;
        ggml_fp16_t * sc = (ggml_fp16_t *) blk, * mn = sc + dk;
        for (int64_t c = 0; c < dk; c++) {
            float lo = INFINITY, hi = -INFINITY;
            for (int r = 0; r < K4C_G; r++) {
                if (valid >> r & 1) {
                    lo = fminf(lo, x[r * n + h * dk + c]);
                    hi = fmaxf(hi, x[r * n + h * dk + c]);
                }
            }
            if (!(lo <= hi)) {
                lo = hi = 0.0f;
            }
            // rounded outward (min down, scale up): the stored range covers every value
            ggml_fp16_t mh = GGML_FP32_TO_FP16(lo);
            while (GGML_FP16_TO_FP32(mh) > lo) {
                mh = k4c_f16_dn(mh);
            }
            const float m = GGML_FP16_TO_FP32(mh);
            ggml_fp16_t sh = GGML_FP32_TO_FP16((hi - m) / 15.0f);
            while (m + 15.0f * GGML_FP16_TO_FP32(sh) < hi) {
                sh = k4c_f16_up(sh);
            }
            sc[c] = sh;
            mn[c] = mh;
            const float s = GGML_FP16_TO_FP32(sh);
            for (int r = 0; r < K4C_G; r++) {
                k4c_set_code(k4c_row_w(blk, dk, r), c, (valid >> r & 1) ? k4c_quant(x[r * n + h * dk + c], m, s) : 0);
            }
        }
    }
    free(x);
}

void ggml_k4c_update_group(void * group, int64_t n_head, int64_t dk, uint32_t set_mask, const float * const * src,
                           uint32_t clear_mask) {
    uint8_t *     grp = (uint8_t *) group;
    const int64_t n = n_head * dk;
    const size_t  bytes = (size_t) n * 20;
    clear_mask &= ~set_mask;
    ggml_critical_section_start();
    k4c_exact_t * e = k4c_find(group);
    if (e && (e->n != n || e->hash != k4c_hash(grp, bytes))) {
        k4c_drop((size_t) (e - k4c_tab));
        e = NULL;
    }
    if (!set_mask && !e) {  // clearing rows of a group without exact values: the rest stays as it is
        for (int64_t h = 0; h < n_head; h++) {
            for (int r = 0; r < K4C_G; r++) {
                if (clear_mask >> r & 1) {
                    memset(k4c_row_w(grp + h * dk * 20, dk, r), 0, dk / 2);
                }
            }
        }
        ggml_critical_section_end();
        return;
    }
    if (!e) {
        if (k4c_ntab == k4c_captab) {
            k4c_captab = k4c_captab ? 2 * k4c_captab : 64;
            k4c_tab = (k4c_exact_t *) realloc(k4c_tab, sizeof(k4c_exact_t) * k4c_captab);
        }
        e = &k4c_tab[k4c_ntab++];
        e->key = group;
        e->valid = 0;
        e->n = n;
        e->x = (ggml_fp16_t *) malloc(sizeof(ggml_fp16_t) * K4C_G * n);
        k4c_bytes += sizeof(ggml_fp16_t) * K4C_G * n;
    }
    uint32_t valid = 0;
    for (int r = 0; r < K4C_G; r++) {
        if ((set_mask >> r & 1)) {
            ggml_fp32_to_fp16_row(src[r], e->x + r * n, n);
            valid |= 1u << r;
        } else if (!(clear_mask >> r & 1) && !k4c_row_empty(grp, n_head, dk, r)) {
            if (!(e->valid >> r & 1)) {  // exact value lost: the stored one
                float * f = (float *) malloc(sizeof(float) * n);
                ggml_k4c_get_row(grp, n_head, dk, r, f);
                ggml_fp32_to_fp16_row(f, e->x + r * n, n);
                free(f);
            }
            valid |= 1u << r;
        }
    }
    e->valid = valid;
    k4c_encode(grp, n_head, dk, valid, e->x, n);
    e->hash = k4c_hash(grp, bytes);
    e->stamp = ++k4c_clock;
    if (!valid) {
        k4c_drop((size_t) (e - k4c_tab));
    }
    while (k4c_bytes > k4c_budget() && k4c_ntab > 1) {  // least recently written first
        size_t old = 0;
        for (size_t i = 1; i < k4c_ntab; i++) {
            old = k4c_tab[i].stamp < k4c_tab[old].stamp ? i : old;
        }
        k4c_drop(old);
    }
    ggml_critical_section_end();
}

bool ggml_k4c_get_row_src(const void * group, int64_t n_head, int64_t dk, int r, float * dst) {
    const int64_t n = n_head * dk;
    ggml_critical_section_start();
    const k4c_exact_t * e = k4c_find(group);
    const bool exact = e && e->n == n && (e->valid >> r & 1) && e->hash == k4c_hash((const uint8_t *) group, (size_t) n * 20);
    if (exact) {
        ggml_fp16_to_fp32_row(e->x + r * n, dst, n);
    }
    ggml_critical_section_end();
    return exact || ggml_k4c_get_row(group, n_head, dk, r, dst);
}

struct ggml_tensor * ggml_set_rows_k4c(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b,
                                       struct ggml_tensor * c, int dk) {
    GGML_ASSERT(a->type == GGML_TYPE_K4C && dk > 0 && dk % 32 == 0 && a->ne[0] % dk == 0);
    struct ggml_tensor * r = ggml_set_rows(ctx, a, b, c);
    ggml_set_op_params_i32(r, 0, dk);  // the CPU writer needs the head dim to find each head's block
    return r;
}
