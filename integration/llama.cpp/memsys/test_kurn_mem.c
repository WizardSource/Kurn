// Tests of the kurn memory system (ggml-kurn-mem.h): allocation in every GGML_KURN_HUGEPAGES mode, the NUMA
// thread / row split, and memory-policy calls under a fake topology.
//
//   test_kurn_mem            exit code = number of failures
// Run it once per mode and topology, e.g. GGML_KURN_HUGEPAGES=2m (with a hugetlb pool: sudo sysctl vm.nr_hugepages=64)
// or GGML_KURN_NUMA_FAKE=3. The hugetlb checks only require hugetlb pages when the pool has free pages.
//
// Build against a llama.cpp checkout with the kurn memsys applied:
//   gcc -O2 -I$L/ggml/include test_kurn_mem.c -L$L/build/bin -lggml-base -Wl,-rpath,$L/build/bin -o test_kurn_mem
#include "ggml-kurn-mem.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int nfail, npass;

#define CHECK(cond, ...)                         \
    do {                                         \
        if (cond) {                              \
            npass++;                             \
        } else {                                 \
            nfail++;                             \
            printf("FAIL %s:%d: ", __FILE__, __LINE__); \
            printf(__VA_ARGS__);                 \
            printf("\n");                        \
        }                                        \
    } while (0)

static long pool_free(const char * size) {
    char p[128];
    long v = 0;
    snprintf(p, sizeof p, "/sys/kernel/mm/hugepages/hugepages-%s/free_hugepages", size);
    FILE * f = fopen(p, "r");
    if (f) {
        if (fscanf(f, "%ld", &v) != 1) v = 0;
        fclose(f);
    }
    return v;
}

static void test_alloc(void) {
    const size_t sizes[] = { 64, 4096, (1u << 20) + 3, (4u << 20) - 1, 4u << 20, (6u << 20) + 12345, 64u << 20, 300u << 20 };
    void * ps[8];
    for (int i = 0; i < 8; i++) {
        ps[i] = ggml_kurn_mem_alloc(sizes[i], (enum ggml_kurn_mem_tag) (i % GGML_KURN_MEM_N_TAGS));
        CHECK(ps[i] != NULL, "alloc %zu", sizes[i]);
        if (!ps[i]) continue;
        CHECK(((uintptr_t) ps[i] & 63) == 0, "alignment of %zu: %p", sizes[i], ps[i]);
        memset(ps[i], 0x5a, sizes[i]);
        const unsigned char * c = (const unsigned char *) ps[i];
        CHECK(c[0] == 0x5a && c[sizes[i] - 1] == 0x5a, "write/read %zu", sizes[i]);
    }
    const char * mode = ggml_kurn_mem_mode();
    const char * sum = ggml_kurn_mem_summary();
    printf("mode %s; %s\n", mode, sum);
    if (!strcmp(mode, "off")) {
        CHECK(strstr(sum, "malloc") != NULL, "off: big buffers from ggml's allocator: %s", sum);
    } else if (!strcmp(mode, "4k")) {
        CHECK(strstr(sum, "4K 100%") != NULL, "4k: %s", sum);
    } else if (!strcmp(mode, "thp")) {
        CHECK(strstr(sum, "thp") != NULL && strstr(sum, "2M") == NULL, "thp: %s", sum);
    } else if (pool_free("2048kB") >= 160) {
        CHECK(strstr(sum, "2M") != NULL, "%s with a free 2 MB pool: %s", mode, sum);
    }
    for (int i = 0; i < 8; i++) {
        ggml_kurn_mem_free(ps[i], sizes[i]);
    }
    // a 1 GB buffer takes 1 GB pages when the pool has them (single node; auto / 1g)
    if ((!strcmp(mode, "1g") || !strcmp(mode, "auto")) && pool_free("1048576kB") >= 1 && pool_free("2048kB") >= 8 &&
        ggml_kurn_numa_n_nodes() == 1) {
        const size_t sz = (1u << 30) + (5u << 20);
        unsigned char * p = (unsigned char *) ggml_kurn_mem_alloc(sz, GGML_KURN_MEM_WEIGHTS);
        CHECK(p && ((uintptr_t) p & ((1u << 30) - 1)) == 0, "1 GB buffer is 1 GB aligned: %p", (void *) p);
        if (p) {
            memset(p, 1, sz);
            const char * s2 = ggml_kurn_mem_summary();
            CHECK(strstr(s2, "1G") != NULL, "1G pages used: %s", s2);
            printf("1 GB buffer: %s\n", s2);
            ggml_kurn_mem_free(p, sz);
        }
    }
    // tags
    enum ggml_kurn_mem_tag old = ggml_kurn_mem_set_tag(GGML_KURN_MEM_KV);
    CHECK(ggml_kurn_mem_get_tag() == GGML_KURN_MEM_KV, "tag set");
    ggml_kurn_mem_set_tag(old);
    CHECK(ggml_kurn_mem_get_tag() == old, "tag restored");
}

static void test_numa_split(void) {
    const int nn = ggml_kurn_numa_n_nodes();
    printf("numa: %d node(s)%s\n", nn, ggml_kurn_numa_is_fake() ? " (fake)" : "");
    for (int nth = 1; nth <= 64; nth++) {
        int prev = 0, ok = 1;
        for (int n = 0; n <= nn; n++) {
            const int f = ggml_kurn_numa_first_thread(n, nth);
            ok &= f >= prev && f <= nth;
            prev = f;
        }
        ok &= ggml_kurn_numa_first_thread(0, nth) == 0 && ggml_kurn_numa_first_thread(nn, nth) == nth;
        for (int i = 0; i < nth; i++) {
            const int node = ggml_kurn_numa_node_of_thread(i, nth);
            ok &= node >= 0 && node < nn && i >= ggml_kurn_numa_first_thread(node, nth) &&
                  i < ggml_kurn_numa_first_thread(node + 1, nth);
        }
        CHECK(ok, "thread split nth=%d", nth);
    }
    const int64_t Ns[] = { 1, 100, 128, 1024, 1000, 4096, 12288, 151936 };
    for (int i = 0; i < 8; i++) {
        int64_t prev = 0;
        int ok = 1;
        for (int n = 0; n <= nn; n++) {
            const int64_t s = ggml_kurn_numa_split(Ns[i], n, 128);
            ok &= s >= prev && s <= Ns[i] && (s % 128 == 0 || s == Ns[i]);
            prev = s;
        }
        ok &= ggml_kurn_numa_split(Ns[i], 0, 128) == 0 && ggml_kurn_numa_split(Ns[i], nn, 128) == Ns[i];
        CHECK(ok, "row split N=%lld", (long long) Ns[i]);
    }
}

static void test_numa_policy(void) {
    const int nn = ggml_kurn_numa_n_nodes();
    const size_t sz = 64u << 20;
    unsigned char * p = (unsigned char *) ggml_kurn_mem_alloc(sz, GGML_KURN_MEM_WEIGHTS);
    CHECK(p != NULL, "alloc for policy test");
    if (!p) return;
    uint64_t n0, b0, f0, n1, b1, f1;
    ggml_kurn_numa_stats(&n0, &b0, &f0);
    for (int n = 0; n < nn; n++) {
        ggml_kurn_numa_bind(p + sz * n / nn, sz / nn, n);
    }
    ggml_kurn_numa_interleave(p, 4096);
    memset(p, 3, sz);
    ggml_kurn_numa_stats(&n1, &b1, &f1);
    if (nn > 1) {
        uint64_t nb[64] = { 0 }, il = 0;
        ggml_kurn_numa_placement(nb, &il);
        for (int n = 0; n < nn; n++) {
            CHECK(nb[n] >= sz / nn, "node %d placement %llu >= %zu", n, (unsigned long long) nb[n], sz / nn);
        }
        CHECK(il >= 4096, "interleaved bytes %llu", (unsigned long long) il);
        CHECK(n1 - n0 == (uint64_t) nn + 1, "policy calls: %llu", (unsigned long long) (n1 - n0));
        CHECK(f1 == f0, "mbind failures: %llu", (unsigned long long) (f1 - f0));
    } else {
        CHECK(n1 == n0, "single node: no policy calls (%llu)", (unsigned long long) (n1 - n0));
    }
    ggml_kurn_numa_pin_thread(0, 4);
    ggml_kurn_mem_free(p, sz);
}

int main(void) {
    test_alloc();
    test_numa_split();
    test_numa_policy();
    printf("kurn mem: %d passed, %d failed\n", npass, nfail);
    return nfail;
}
