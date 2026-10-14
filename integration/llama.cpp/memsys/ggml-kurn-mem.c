// kurn memory system: huge-page allocation and NUMA placement for ggml's CPU buffers (see ggml-kurn-mem.h).
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "ggml-kurn-mem.h"
#include "ggml-impl.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__linux__)
#include <stdatomic.h>
#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
#define KMEM_LINUX 1

#define KMEM_2M ((size_t) 2 << 20)
#define KMEM_1G ((size_t) 1 << 30)
#define KMEM_MAX_NODES 64

enum kmem_mode { KMEM_OFF, KMEM_4K, KMEM_THP, KMEM_2M_MODE, KMEM_1G_MODE, KMEM_AUTO };
enum kmem_kind { KIND_1G, KIND_2M, KIND_THP, KIND_4K, KIND_LEGACY, KIND_N };
static const char * const kind_names[KIND_N] = { "1G", "2M", "thp", "4K", "malloc" };
static const char * const tag_names[GGML_KURN_MEM_N_TAGS] = { "buffer", "weights", "kv", "compute", "work" };

static int env_int(const char * name, int def) {
    const char * v = getenv(name);
    return v && *v ? atoi(v) : def;
}

static int verbose(void) {
    static int v = -1;
    if (v < 0) {
        v = env_int("GGML_KURN_MEM_VERBOSE", 0);
    }
    return v;
}

// ------------------------------------------------------------------------------------------------ NUMA topology
typedef struct {
    int on, fake, pin;
    int n;                       // active nodes
    int os_node[KMEM_MAX_NODES]; // memory-policy target of each active node
    int ncpu[KMEM_MAX_NODES];
    int cum[KMEM_MAX_NODES + 1]; // prefix sums of ncpu
#if KMEM_LINUX
    cpu_set_t cpus[KMEM_MAX_NODES];
#endif
} numa_topo;

static numa_topo g_topo;
static atomic_uint_fast64_t g_bind_n, g_bind_bytes, g_bind_failed;
static atomic_uint_fast64_t g_node_bytes[KMEM_MAX_NODES], g_ilv_bytes;  // placement requested, per logical node

#if KMEM_LINUX
static int parse_cpulist(const char * s, cpu_set_t * set) {
    CPU_ZERO(set);
    int n = 0;
    while (*s && *s != '\n') {
        char * e;
        long a = strtol(s, &e, 10), b = a;
        if (e == s) {
            break;
        }
        if (*e == '-') {
            s = e + 1;
            b = strtol(s, &e, 10);
        }
        for (long c = a; c <= b && c < CPU_SETSIZE; c++) {
            CPU_SET(c, set);
            n++;
        }
        s = *e == ',' ? e + 1 : e;
    }
    return n;
}

static long sys_mbind(void * addr, unsigned long len, int mode, const unsigned long * mask, unsigned long maxnode, unsigned flags) {
    return syscall(SYS_mbind, addr, len, mode, mask, maxnode, flags);
}
#endif

static void topo_init_once(void) {
    numa_topo * t = &g_topo;
    t->n = 1;
    t->cum[0] = 0;
    t->cum[1] = 1;
    t->ncpu[0] = 1;
#if KMEM_LINUX
    const char * mode = getenv("GGML_KURN_NUMA");
    const int off = mode && (!strcmp(mode, "0") || !strcmp(mode, "off") || !strcmp(mode, "false"));
    t->pin = env_int("GGML_KURN_NUMA_PIN", 1) != 0;
    cpu_set_t allowed;
    if (sched_getaffinity(0, sizeof allowed, &allowed) != 0) {
        return;
    }
    const int fake = env_int("GGML_KURN_NUMA_FAKE", 0);
    int n = 0;
    if (fake > 1) {
        int cpus[CPU_SETSIZE], nc = 0;
        for (int c = 0; c < CPU_SETSIZE; c++) {
            if (CPU_ISSET(c, &allowed)) {
                cpus[nc++] = c;
            }
        }
        const int nf = fake < nc ? (fake < KMEM_MAX_NODES ? fake : KMEM_MAX_NODES) : nc;
        for (int f = 0; f < nf; f++) {
            CPU_ZERO(&t->cpus[f]);
            for (int i = nc * f / nf; i < nc * (f + 1) / nf; i++) {
                CPU_SET(cpus[i], &t->cpus[f]);
            }
            t->os_node[f] = 0;
            const int c0 = cpus[nc * f / nf];
            for (int nd = 0; nd < 1024; nd++) {  // the real node of the group's first CPU
                char p[96];
                snprintf(p, sizeof p, "/sys/devices/system/cpu/cpu%d/node%d", c0, nd);
                if (access(p, F_OK) == 0) {
                    t->os_node[f] = nd;
                    break;
                }
            }
        }
        n = nf;
        t->fake = 1;
    } else {
        for (int nd = 0; nd < 1024 && n < KMEM_MAX_NODES; nd++) {
            char p[96], buf[4096];
            snprintf(p, sizeof p, "/sys/devices/system/node/node%d/cpulist", nd);
            FILE * f = fopen(p, "r");
            if (!f) {
                continue;
            }
            const size_t r = fread(buf, 1, sizeof buf - 1, f);
            fclose(f);
            buf[r] = 0;
            cpu_set_t set;
            parse_cpulist(buf, &set);
            CPU_AND(&set, &set, &allowed);
            if (CPU_COUNT(&set) == 0) {  // memory-only node or outside our cpuset
                continue;
            }
            t->cpus[n] = set;
            t->os_node[n] = nd;
            n++;
        }
    }
    if (n <= 1 || off) {
        if (n > 1 && verbose()) {
            GGML_LOG_INFO("kurn numa: %d nodes, off (GGML_KURN_NUMA=%s)\n", n, mode);
        }
        return;
    }
    t->n = n;
    t->cum[0] = 0;
    for (int i = 0; i < n; i++) {
        t->ncpu[i] = CPU_COUNT(&t->cpus[i]);
        t->cum[i + 1] = t->cum[i] + t->ncpu[i];
    }
    t->on = 1;
    char desc[512];
    int o = 0;
    for (int i = 0; i < n && o < (int) sizeof desc - 32; i++) {
        o += snprintf(desc + o, sizeof desc - o, "%s%d cpus -> node %d", i ? ", " : "", t->ncpu[i], t->os_node[i]);
    }
    GGML_LOG_INFO("kurn numa: %d %snodes (%s); weights split by rows per node, threads pinned%s\n", n,
                  t->fake ? "fake " : "", desc, t->pin ? "" : " (GGML_KURN_NUMA_PIN=0: not by kurn)");
#endif
}

static pthread_once_t g_topo_once = PTHREAD_ONCE_INIT;

static const numa_topo * topo(void) {
    pthread_once(&g_topo_once, topo_init_once);
    return &g_topo;
}

int ggml_kurn_numa_n_nodes(void) {
    return topo()->n;
}

int ggml_kurn_numa_is_fake(void) {
    return topo()->fake;
}

int ggml_kurn_numa_first_thread(int node, int nth) {
    const numa_topo * t = topo();
    if (node <= 0) {
        return 0;
    }
    if (node >= t->n) {
        return nth;
    }
    return (int) ((int64_t) nth * t->cum[node] / t->cum[t->n]);
}

int ggml_kurn_numa_node_of_thread(int ith, int nth) {
    const numa_topo * t = topo();
    int n = 0;
    while (n + 1 < t->n && ith >= ggml_kurn_numa_first_thread(n + 1, nth)) {
        n++;
    }
    return n;
}

int64_t ggml_kurn_numa_split(int64_t n, int node, int64_t unit) {
    const numa_topo * t = topo();
    if (node <= 0) {
        return 0;
    }
    if (node >= t->n) {
        return n;
    }
    const int64_t units = (n + unit - 1) / unit;
    const int64_t u = units * t->cum[node] / t->cum[t->n];
    return u * unit < n ? u * unit : n;
}

void ggml_kurn_numa_pin_thread(int ith, int nth) {
#if KMEM_LINUX
    const numa_topo * t = topo();
    if (!t->on || !t->pin) {
        return;
    }
    static __thread int pinned_ith = -1, pinned_nth = -1;
    if (pinned_ith == ith && pinned_nth == nth) {
        return;
    }
    pinned_ith = ith;
    pinned_nth = nth;
    const int node = ggml_kurn_numa_node_of_thread(ith, nth);
    if (sched_setaffinity(0, sizeof(cpu_set_t), &t->cpus[node]) != 0 && verbose()) {
        GGML_LOG_WARN("kurn numa: pinning thread %d/%d to node %d failed: %s\n", ith, nth, node, strerror(errno));
    }
#else
    (void) ith, (void) nth;
#endif
}

// ------------------------------------------------------------------------------------------------ region registry
typedef struct {
    uintptr_t base;
    size_t len;       // mapped bytes (whole region)
    size_t size;      // requested bytes
    size_t align;     // policy granularity: the largest page size inside the region
    size_t bytes[KIND_N];
    int tag;
} kmem_region;

static pthread_mutex_t g_mu = PTHREAD_MUTEX_INITIALIZER;
static kmem_region * g_regions;
static int g_nregions, g_cap;
static size_t g_legacy[GGML_KURN_MEM_N_TAGS];  // live bytes from ggml's allocator, per tag (sizes >= min only)

static void region_add(const kmem_region * r) {
    pthread_mutex_lock(&g_mu);
    if (g_nregions == g_cap) {
        g_cap = g_cap ? 2 * g_cap : 64;
        g_regions = (kmem_region *) realloc(g_regions, sizeof(kmem_region) * g_cap);
        GGML_ASSERT(g_regions);
    }
    g_regions[g_nregions++] = *r;
    pthread_mutex_unlock(&g_mu);
}

static int region_take(uintptr_t base, kmem_region * out) {
    int found = 0;
    pthread_mutex_lock(&g_mu);
    for (int i = 0; i < g_nregions; i++) {
        if (g_regions[i].base == base) {
            *out = g_regions[i];
            g_regions[i] = g_regions[--g_nregions];
            found = 1;
            break;
        }
    }
    pthread_mutex_unlock(&g_mu);
    return found;
}

static size_t region_align_of(uintptr_t p) {
    size_t a = 4096;
    pthread_mutex_lock(&g_mu);
    for (int i = 0; i < g_nregions; i++) {
        if (p >= g_regions[i].base && p < g_regions[i].base + g_regions[i].len) {
            a = g_regions[i].align;
            break;
        }
    }
    pthread_mutex_unlock(&g_mu);
    return a;
}

// ------------------------------------------------------------------------------------------------ allocation
static enum kmem_mode g_mode = KMEM_AUTO;
static size_t g_min = 4u << 20;
static pthread_once_t g_mode_once = PTHREAD_ONCE_INIT;

static void mode_init_once(void) {
    const char * m = getenv("GGML_KURN_HUGEPAGES");
    if (m && *m) {
        if (!strcmp(m, "off") || !strcmp(m, "0") || !strcmp(m, "none")) g_mode = KMEM_OFF;
        else if (!strcmp(m, "4k") || !strcmp(m, "4K")) g_mode = KMEM_4K;
        else if (!strcmp(m, "thp") || !strcmp(m, "madvise")) g_mode = KMEM_THP;
        else if (!strcmp(m, "2m") || !strcmp(m, "2M")) g_mode = KMEM_2M_MODE;
        else if (!strcmp(m, "1g") || !strcmp(m, "1G")) g_mode = KMEM_1G_MODE;
        else if (!strcmp(m, "auto") || !strcmp(m, "1") || !strcmp(m, "on")) g_mode = KMEM_AUTO;
        else GGML_LOG_WARN("kurn mem: unknown GGML_KURN_HUGEPAGES=%s, using auto\n", m);
    }
    const char * mn = getenv("GGML_KURN_HUGEPAGES_MIN");
    if (mn && *mn) {
        g_min = (size_t) strtoull(mn, NULL, 10);
    }
#if !KMEM_LINUX
    g_mode = KMEM_OFF;
#endif
}

static enum kmem_mode mode(void) {
    pthread_once(&g_mode_once, mode_init_once);
    return g_mode;
}

const char * ggml_kurn_mem_mode(void) {
    static const char * const names[] = { "off", "4k", "thp", "2m", "1g", "auto" };
    return names[mode()];
}

static __thread int tls_tag = GGML_KURN_MEM_BUFFER;

enum ggml_kurn_mem_tag ggml_kurn_mem_set_tag(enum ggml_kurn_mem_tag tag) {
    const int old = tls_tag;
    tls_tag = tag;
    return (enum ggml_kurn_mem_tag) old;
}

enum ggml_kurn_mem_tag ggml_kurn_mem_get_tag(void) {
    return (enum ggml_kurn_mem_tag) tls_tag;
}

#if KMEM_LINUX
#ifndef MAP_HUGE_SHIFT
#define MAP_HUGE_SHIFT 26
#endif
#ifndef MAP_HUGE_2MB
#define MAP_HUGE_2MB (21 << MAP_HUGE_SHIFT)
#endif
#ifndef MAP_HUGE_1GB
#define MAP_HUGE_1GB (30 << MAP_HUGE_SHIFT)
#endif

static size_t hugetlb_avail(size_t page) {  // free pages not promised to other mappings
    char p[128];
    long v[2] = { 0, 0 };
    const char * f[2] = { "free_hugepages", "resv_hugepages" };
    for (int i = 0; i < 2; i++) {
        snprintf(p, sizeof p, "/sys/kernel/mm/hugepages/hugepages-%zukB/%s", page >> 10, f[i]);
        FILE * fp = fopen(p, "r");
        if (fp) {
            if (fscanf(fp, "%ld", &v[i]) != 1) {
                v[i] = 0;
            }
            fclose(fp);
        }
    }
    return v[0] > v[1] ? (size_t) (v[0] - v[1]) : 0;
}

static int map_fixed(uintptr_t at, size_t len, int extra) {
    void * p = mmap((void *) at, len, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED | extra, -1, 0);
    return p == (void *) at;
}

static void * alloc_mapped(size_t size, int tag, kmem_region * r) {
    const enum kmem_mode m = mode();
    const int numa = topo()->on;
    const size_t len = (size + KMEM_2M - 1) & ~(KMEM_2M - 1);
    // 1 GB pages only for whole GBs, and not under NUMA placement (it splits buffers at 2 MB granularity)
    const int try1g = (m == KMEM_AUTO || m == KMEM_1G_MODE) && len >= KMEM_1G && !numa;
    const int try2m = m == KMEM_AUTO || m == KMEM_1G_MODE || m == KMEM_2M_MODE;
    const size_t align = try1g ? KMEM_1G : KMEM_2M;
    void * res = mmap(NULL, len + align, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (res == MAP_FAILED) {
        return NULL;
    }
    const uintptr_t raw = (uintptr_t) res, base = (raw + align - 1) & ~(uintptr_t) (align - 1);
    if (base > raw) {
        munmap(res, base - raw);
    }
    if (raw + len + align > base + len) {
        munmap((void *) (base + len), raw + len + align - (base + len));
    }
    memset(r, 0, sizeof *r);
    r->base = base;
    r->len = len;
    r->size = size;
    r->tag = tag;
    r->align = 4096;
    size_t off = 0;
    if (try1g) {
        size_t n1 = len / KMEM_1G, av = hugetlb_avail(KMEM_1G);
        n1 = n1 < av ? n1 : av;
        if (n1 > 0 && map_fixed(base, n1 * KMEM_1G, MAP_HUGETLB | MAP_HUGE_1GB)) {
            off = n1 * KMEM_1G;
            r->bytes[KIND_1G] = off;
            r->align = KMEM_1G;
        }
    }
    if (try2m && off < len) {
        size_t n2 = (len - off) / KMEM_2M, av = hugetlb_avail(KMEM_2M);
        n2 = n2 < av ? n2 : av;
        if (n2 > 0 && map_fixed(base + off, n2 * KMEM_2M, MAP_HUGETLB | MAP_HUGE_2MB)) {
            r->bytes[KIND_2M] = n2 * KMEM_2M;
            off += n2 * KMEM_2M;
            r->align = r->align > KMEM_2M ? r->align : KMEM_2M;
        }
    }
    if (off < len) {
        if (!map_fixed(base + off, len - off, 0)) {
            munmap((void *) base, len);
            return NULL;
        }
        int kind = KIND_4K;
        if (m == KMEM_4K) {
            madvise((void *) (base + off), len - off, MADV_NOHUGEPAGE);
        } else if (madvise((void *) (base + off), len - off, MADV_HUGEPAGE) == 0) {
            kind = KIND_THP;
            r->align = r->align > KMEM_2M ? r->align : KMEM_2M;
        }
        r->bytes[kind] = len - off;
    }
    return (void *) base;
}

static void fmt_bytes(char * o, size_t n, size_t b) {
    if (b >= KMEM_1G) snprintf(o, n, "%.2f GiB", b / (double) KMEM_1G);
    else snprintf(o, n, "%.1f MiB", b / (double) (1 << 20));
}

static void log_region(const kmem_region * r) {
    char sz[32], d[256];
    int o = 0;
    fmt_bytes(sz, sizeof sz, r->size);
    for (int k = 0; k < KIND_N; k++) {
        if (!r->bytes[k]) continue;
        if (k == KIND_1G || k == KIND_2M) {
            o += snprintf(d + o, sizeof d - o, "%shugetlb %s x%zu", o ? " + " : "", kind_names[k],
                          r->bytes[k] / (k == KIND_1G ? KMEM_1G : KMEM_2M));
        } else {
            char b[32];
            fmt_bytes(b, sizeof b, r->bytes[k]);
            o += snprintf(d + o, sizeof d - o, "%s%s %s", o ? " + " : "", k == KIND_THP ? "THP (madvise)" : "4K pages", b);
        }
    }
    GGML_LOG_INFO("kurn mem: %s %s -> %s%s\n", tag_names[r->tag], sz, d,
                  r->bytes[KIND_4K] && mode() != KMEM_4K ? " (no hugetlb pages free, THP unavailable)" : "");
}
#endif

void * ggml_kurn_mem_alloc(size_t size, enum ggml_kurn_mem_tag tag) {
    const enum kmem_mode m = mode();
    if ((int) tag < 0 || tag >= GGML_KURN_MEM_N_TAGS) {
        tag = GGML_KURN_MEM_BUFFER;
    }
#if KMEM_LINUX
    if (m != KMEM_OFF && size >= g_min && size > 0) {
        kmem_region r;
        void * p = alloc_mapped(size, tag, &r);
        if (p) {
            region_add(&r);
            log_region(&r);
            if (topo()->on) {  // shared by every thread; KURN weight shards and KV streams then bind their own ranges
                ggml_kurn_numa_interleave(p, r.len);
            }
            return p;
        }
        GGML_LOG_WARN("kurn mem: mmap of %zu bytes failed (%s), using ggml's allocator\n", size, strerror(errno));
    }
#endif
    void * p = ggml_aligned_malloc(size);
    if (p && size >= g_min) {
        pthread_mutex_lock(&g_mu);
        g_legacy[tag] += size;
        pthread_mutex_unlock(&g_mu);
        if (m == KMEM_OFF && verbose()) {
            GGML_LOG_INFO("kurn mem: %s %zu bytes -> ggml allocator (GGML_KURN_HUGEPAGES=off)\n", tag_names[tag], size);
        }
    }
    return p;
}

void ggml_kurn_mem_free(void * ptr, size_t size) {
    if (!ptr) {
        return;
    }
#if KMEM_LINUX
    kmem_region r;
    if (region_take((uintptr_t) ptr, &r)) {
        munmap(ptr, r.len);
        return;
    }
#endif
    if (size >= g_min) {
        pthread_mutex_lock(&g_mu);
        for (int t = 0; t < GGML_KURN_MEM_N_TAGS; t++) {  // tag unknown here; any live tag with that many bytes
            if (g_legacy[t] >= size) {
                g_legacy[t] -= size;
                break;
            }
        }
        pthread_mutex_unlock(&g_mu);
    }
    ggml_aligned_free(ptr, size);
}

// ------------------------------------------------------------------------------------------------ NUMA policy
#define MPOL_PREFERRED_ 1
#define MPOL_INTERLEAVE_ 3
#define MPOL_MF_MOVE_ (1 << 1)

static void policy(void * ptr, size_t len, int mpol, const unsigned long * mask) {
#if KMEM_LINUX
    const size_t a = region_align_of((uintptr_t) ptr);
    const uintptr_t s = (uintptr_t) ptr & ~(uintptr_t) (a - 1);
    const uintptr_t e = ((uintptr_t) ptr + len + a - 1) & ~(uintptr_t) (a - 1);
    if (e <= s) {
        return;
    }
    atomic_fetch_add(&g_bind_n, 1);
    atomic_fetch_add(&g_bind_bytes, e - s);
    if (sys_mbind((void *) s, e - s, mpol, mask, KMEM_MAX_NODES + 1, MPOL_MF_MOVE_) != 0) {
        if (atomic_fetch_add(&g_bind_failed, 1) == 0) {
            GGML_LOG_WARN("kurn numa: mbind(%p, %zu) failed: %s (further failures counted silently)\n", (void *) s,
                          (size_t) (e - s), strerror(errno));
        }
    }
#else
    (void) ptr, (void) len, (void) mpol, (void) mask;
#endif
}

void ggml_kurn_numa_bind(void * ptr, size_t len, int node) {
    const numa_topo * t = topo();
    if (!t->on || len == 0 || node < 0 || node >= t->n) {
        return;
    }
    unsigned long mask[(KMEM_MAX_NODES + 63) / 64 + 16] = { 0 };
    const int os = t->os_node[node];
    mask[os / 64] |= 1ul << (os % 64);
    atomic_fetch_add(&g_node_bytes[node], len);
    policy(ptr, len, MPOL_PREFERRED_, mask);
}

void ggml_kurn_numa_interleave(void * ptr, size_t len) {
    const numa_topo * t = topo();
    if (!t->on || len == 0) {
        return;
    }
    unsigned long mask[(KMEM_MAX_NODES + 63) / 64 + 16] = { 0 };
    for (int i = 0; i < t->n; i++) {
        mask[t->os_node[i] / 64] |= 1ul << (t->os_node[i] % 64);
    }
    atomic_fetch_add(&g_ilv_bytes, len);
    policy(ptr, len, MPOL_INTERLEAVE_, mask);
}

void ggml_kurn_numa_placement(uint64_t * node_bytes, uint64_t * interleaved_bytes) {
    for (int i = 0; node_bytes && i < topo()->n; i++) node_bytes[i] = atomic_load(&g_node_bytes[i]);
    if (interleaved_bytes) *interleaved_bytes = atomic_load(&g_ilv_bytes);
}

void ggml_kurn_numa_stats(uint64_t * n_bind, uint64_t * bytes_bind, uint64_t * n_failed) {
    if (n_bind) *n_bind = atomic_load(&g_bind_n);
    if (bytes_bind) *bytes_bind = atomic_load(&g_bind_bytes);
    if (n_failed) *n_failed = atomic_load(&g_bind_failed);
}

// ------------------------------------------------------------------------------------------------ summary
const char * ggml_kurn_mem_summary(void) {
    static char out[1024];
    size_t bytes[GGML_KURN_MEM_N_TAGS][KIND_N];
    size_t thp_huge[GGML_KURN_MEM_N_TAGS];
    memset(bytes, 0, sizeof bytes);
    memset(thp_huge, 0, sizeof thp_huge);
    pthread_mutex_lock(&g_mu);
    const int n = g_nregions;
    kmem_region * rs = (kmem_region *) malloc(sizeof(kmem_region) * (n ? n : 1));
    if (rs && n) memcpy(rs, g_regions, sizeof(kmem_region) * n);
    for (int t = 0; t < GGML_KURN_MEM_N_TAGS; t++) {
        bytes[t][KIND_LEGACY] = g_legacy[t];
    }
    pthread_mutex_unlock(&g_mu);
    if (!rs) {
        return "kurn mem: out of memory";
    }
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < KIND_N; k++) {
            bytes[rs[i].tag][k] += rs[i].bytes[k];
        }
    }
#if KMEM_LINUX
    FILE * f = fopen("/proc/self/smaps", "r");  // AnonHugePages of each VMA inside a THP region
    if (f) {
        char line[512];
        int cur = -1;
        while (fgets(line, sizeof line, f)) {
            unsigned long a, b;
            if (sscanf(line, "%lx-%lx ", &a, &b) == 2 && strchr(line, '-') < strchr(line, ' ')) {
                cur = -1;
                for (int i = 0; i < n; i++) {
                    if (rs[i].bytes[KIND_THP] && a >= rs[i].base && b <= rs[i].base + rs[i].len) {
                        cur = rs[i].tag;
                        break;
                    }
                }
            } else if (cur >= 0 && !strncmp(line, "AnonHugePages:", 14)) {
                thp_huge[cur] += (size_t) strtoull(line + 14, NULL, 10) << 10;
            }
        }
        fclose(f);
    }
#endif
    free(rs);
    int o = snprintf(out, sizeof out, "mode %s", ggml_kurn_mem_mode());
    if (topo()->on) {  // placement requested since start (cumulative; bound ranges override interleaved ones)
        o += snprintf(out + o, sizeof out - o, " | numa%s bound:", topo()->fake ? " (fake)" : "");
        for (int i = 0; i < topo()->n && o < (int) sizeof out - 96; i++) {
            char b[32];
            fmt_bytes(b, sizeof b, atomic_load(&g_node_bytes[i]));
            o += snprintf(out + o, sizeof out - o, " N%d %s", i, b);
        }
        char b[32];
        fmt_bytes(b, sizeof b, atomic_load(&g_ilv_bytes));
        o += snprintf(out + o, sizeof out - o, ", interleaved %s", b);
    }
    for (int t = 0; t < GGML_KURN_MEM_N_TAGS && o < (int) sizeof out - 64; t++) {
        size_t tot = 0;
        for (int k = 0; k < KIND_N; k++) tot += bytes[t][k];
        if (!tot) continue;
        char b[32];
        fmt_bytes(b, sizeof b, tot);
        o += snprintf(out + o, sizeof out - o, " | %s %s:", tag_names[t], b);
        for (int k = 0; k < KIND_N && o < (int) sizeof out - 48; k++) {
            if (!bytes[t][k]) continue;
            o += snprintf(out + o, sizeof out - o, " %s %.0f%%", kind_names[k], 100.0 * bytes[t][k] / tot);
            if (k == KIND_THP) {
                o += snprintf(out + o, sizeof out - o, " (%.0f%% huge)", 100.0 * thp_huge[t] / bytes[t][k]);
            }
        }
    }
    return out;
}

#else  // !__linux__: ggml's allocator, one node

void * ggml_kurn_mem_alloc(size_t size, enum ggml_kurn_mem_tag tag) { (void) tag; return ggml_aligned_malloc(size); }
void ggml_kurn_mem_free(void * ptr, size_t size) { ggml_aligned_free(ptr, size); }
enum ggml_kurn_mem_tag ggml_kurn_mem_set_tag(enum ggml_kurn_mem_tag tag) { (void) tag; return GGML_KURN_MEM_BUFFER; }
enum ggml_kurn_mem_tag ggml_kurn_mem_get_tag(void) { return GGML_KURN_MEM_BUFFER; }
const char * ggml_kurn_mem_summary(void) { return "mode off (not Linux)"; }
const char * ggml_kurn_mem_mode(void) { return "off"; }
int ggml_kurn_numa_n_nodes(void) { return 1; }
int ggml_kurn_numa_is_fake(void) { return 0; }
int ggml_kurn_numa_first_thread(int node, int nth) { return node <= 0 ? 0 : nth; }
int ggml_kurn_numa_node_of_thread(int ith, int nth) { (void) ith, (void) nth; return 0; }
int64_t ggml_kurn_numa_split(int64_t n, int node, int64_t unit) { (void) unit; return node <= 0 ? 0 : n; }
void ggml_kurn_numa_pin_thread(int ith, int nth) { (void) ith, (void) nth; }
void ggml_kurn_numa_bind(void * ptr, size_t len, int node) { (void) ptr, (void) len, (void) node; }
void ggml_kurn_numa_interleave(void * ptr, size_t len) { (void) ptr, (void) len; }
void ggml_kurn_numa_stats(uint64_t * a, uint64_t * b, uint64_t * c) { if (a) *a = 0; if (b) *b = 0; if (c) *c = 0; }
void ggml_kurn_numa_placement(uint64_t * node_bytes, uint64_t * il) { if (node_bytes) node_bytes[0] = 0; if (il) *il = 0; }

#endif
