// kurn lowbit measurement hook for ggml-cpu's MUL_MAT (decode GEMV, ne11 == 1) on AVX-512 VNNI builds.
//   GGML_KURN_LOWBIT=1          enable (weights must stay in the plain CPU buffer: --repack 0 / --no-repack)
//   GGML_KURN_<FMT>=<variant>   pick a compiled variant (see kurn_lowbit_dispatch.h), e.g. GGML_KURN_TQ2_0=lut
//   GGML_KURN_LUT_SHARED=0      layout=lut: every thread builds all tables itself (default: the threads build
//                               disjoint table ranges into one buffer, then ggml_barrier, then their rows)
//   GGML_KURN_VERBOSE=1         log repacks and the chosen variants
// Packed weights are cached per (data pointer, shape, content signature), as in integration/llama.cpp/kurn_hook.h.
#pragma once
#include "kurn_lowbit_dispatch.h"
#include <pthread.h>
#include <ctype.h>

#define KL_SLOTS 8192
typedef struct { const void * key; int64_t ne0, ne1; uint64_t sig; void * packed; } kl_entry;
static kl_entry * kl_cache[KL_SLOTS];
static pthread_mutex_t kl_mu = PTHREAD_MUTEX_INITIALIZER;

static uint64_t kl_signature(const struct ggml_tensor * t) {
    const size_t n = ggml_nbytes(t) / 8;
    const uint64_t * p = (const uint64_t *) t->data;
    uint64_t h = 0x9E3779B97F4A7C15ull ^ (uint64_t) t->ne[0] * 31 ^ (uint64_t) t->ne[1] * 131 ^ (uint64_t) t->type;
    for (int i = 0; i < 32; i++) {
        uint64_t v;
        memcpy(&v, p + (n - 1) * (size_t) i / 31, 8);
        h = (h ^ v) * 0x100000001B3ull;
        h ^= h >> 29;
    }
    return h;
}

static void * kl_get_packed(const struct ggml_tensor * t, void * (*prep)(const void *, int64_t, int64_t)) {
    const size_t h = ((uintptr_t) t->data >> 6) % KL_SLOTS;
    const uint64_t sig = kl_signature(t);
    for (size_t i = 0; i < KL_SLOTS; i++) {
        const kl_entry * e = __atomic_load_n(&kl_cache[(h + i) % KL_SLOTS], __ATOMIC_ACQUIRE);
        if (e == NULL) break;
        if (e->key == t->data) {
            if (e->ne0 == t->ne[0] && e->ne1 == t->ne[1] && e->sig == sig) return e->packed;
            break;
        }
    }
    pthread_mutex_lock(&kl_mu);
    void * res = NULL;
    for (size_t i = 0; i < KL_SLOTS; i++) {
        kl_entry ** slot = &kl_cache[(h + i) % KL_SLOTS];
        kl_entry * e = *slot;
        if (e != NULL && e->key != t->data) continue;
        if (e != NULL && e->ne0 == t->ne[0] && e->ne1 == t->ne[1] && e->sig == sig) { res = e->packed; break; }
        if (getenv("GGML_KURN_VERBOSE")) fprintf(stderr, "kurn-lowbit: repack %s [%lld x %lld] %s\n", t->name,
                                                 (long long) t->ne[0], (long long) t->ne[1], ggml_type_name(t->type));
        kl_entry * n = malloc(sizeof *n);
        *n = (kl_entry) { t->data, t->ne[0], t->ne[1], sig, prep(t->data, t->ne[0], t->ne[1]) };
        __atomic_store_n(slot, n, __ATOMIC_RELEASE);
        free(e);
        res = n->packed;
        break;
    }
    pthread_mutex_unlock(&kl_mu);
    GGML_ASSERT(res != NULL && "kurn-lowbit repack cache full");
    return res;
}

static const kurn_lowbit_kernel * kl_choice[64];
static int kl_on, kl_shared;
static void * kl_tabs;  // shared lut tables (largest K = 32768 of any variant)

static void kl_init(void) {
    kl_on = getenv("GGML_KURN_LOWBIT") != NULL && atoi(getenv("GGML_KURN_LOWBIT")) != 0;
    kl_shared = getenv("GGML_KURN_LUT_SHARED") == NULL || atoi(getenv("GGML_KURN_LUT_SHARED")) != 0;
    int64_t cap = 0;
    for (size_t i = 0; i < sizeof kurn_lowbit_kernels / sizeof kurn_lowbit_kernels[0]; i++) {
        const kurn_lowbit_kernel * k = &kurn_lowbit_kernels[i];
        char env[64], fmt[16];
        size_t j = 0;
        for (; k->fmt[j] && j < sizeof fmt - 1; j++) fmt[j] = (char) toupper((unsigned char) k->fmt[j]);
        fmt[j] = 0;
        snprintf(env, sizeof env, "GGML_KURN_%s", fmt);
        const char * want = getenv(env);
        if (kl_choice[k->type] == NULL || (want && !strcmp(want, k->name))) kl_choice[k->type] = k;
        if (k->lut_info) {
            int64_t b, u;
            k->lut_info(32768, &b, &u);
            if (b > cap) cap = b;
        }
    }
    if (cap) kl_tabs = aligned_alloc(64, (size_t) (cap + 63) & ~(size_t) 63);
    if (kl_on && getenv("GGML_KURN_VERBOSE"))
        for (int t = 0; t < 64; t++)
            if (kl_choice[t]) fprintf(stderr, "kurn-lowbit: %s -> %s%s\n", kl_choice[t]->fmt, kl_choice[t]->name,
                                      kl_choice[t]->lut_info ? (kl_shared ? " (shared tables)" : " (per-thread tables)") : "");
}

// Returns true if kurn computed this MUL_MAT (every thread calls it and does its own rows).
static bool kurn_lowbit_mul_mat(const struct ggml_compute_params * params, struct ggml_tensor * dst, const void * wdata) {
    static pthread_once_t once = PTHREAD_ONCE_INIT;
    pthread_once(&once, kl_init);
    if (!kl_on) return false;
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];
    if ((int) src0->type < 0 || (int) src0->type >= 64) return false;
    const kurn_lowbit_kernel * k = kl_choice[src0->type];
    if (k == NULL) return false;
    const int64_t K = src0->ne[0], N = src0->ne[1];
    if (src1->ne[1] != 1 || src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1 ||
        !ggml_is_contiguous(src0) || !ggml_is_contiguous(dst) || dst->nb[0] != sizeof(float) || K > 32768 || K % k->period != 0) {
        return false;
    }
    const int ith = params->ith, nth = params->nth;
    const int64_t per = ((N + nth - 1) / nth + k->row_align - 1) / k->row_align * k->row_align;
    const int64_t r0 = MIN(ith * per, N), r1 = MIN(r0 + per, N);
    void * pk = kl_get_packed(src0, k->prep);
    if (k->lut_info && kl_shared && nth > 1) {
        int64_t bytes, units;
        k->lut_info(K, &bytes, &units);
        const int64_t up = (units + nth - 1) / nth, u0 = MIN(ith * up, units), u1 = MIN(u0 + up, units);
        if (u0 < u1) k->lut_build(wdata, K, kl_tabs, u0, u1);
        ggml_barrier(params->threadpool);
        if (r0 < r1) k->lut_rows(pk, kl_tabs, (float *) dst->data, K, r0, r1);
    } else if (r0 < r1) {
        k->packed(pk, wdata, (float *) dst->data, K, r0, r1);
    }
    return true;
}
