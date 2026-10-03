// kurn runtime implementation. See kurn_rt.h.
#define _GNU_SOURCE
#include "kurn_rt.h"
#include <limits.h>
#include <linux/futex.h>
#include <pthread.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>
#if defined(__x86_64__) || defined(__i386__)
#include <cpuid.h>
#include <immintrin.h>
#include <x86intrin.h>
#endif

#define LD(p) __atomic_load_n((p), __ATOMIC_ACQUIRE)
#define CL 64

static void *xcalloc(size_t n, size_t sz) {
    void *p = NULL;
    size_t bytes = (n * sz + CL - 1) / CL * CL;
    if (!bytes) bytes = CL;
    if (posix_memalign(&p, CL, bytes)) { perror("krt alloc"); abort(); }
    memset(p, 0, bytes);
    return p;
}

// ---------------------------------------------------------------- waits
void krt_cpu_relax(void) {
#if defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#elif defined(__aarch64__)
    __asm__ volatile("yield");
#endif
}

uint64_t krt_now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

int krt_has_waitpkg(void) {
#if defined(__x86_64__) || defined(__i386__)
    static int cached = -1;
    int v = __atomic_load_n(&cached, __ATOMIC_RELAXED);
    if (v < 0) {
        unsigned a, b, c, d;
        v = __get_cpuid_count(7, 0, &a, &b, &c, &d) && (c & (1u << 5)) ? 1 : 0;
        if (getenv("KURN_RT_NO_WAITPKG")) v = 0;
        __atomic_store_n(&cached, v, __ATOMIC_RELAXED);
    }
    return v;
#else
    return 0;
#endif
}

#if defined(__x86_64__)
__attribute__((target("waitpkg"))) static void umwait_ne(const uint32_t *word, uint32_t old, const krt_wait *w) {
    const uint64_t dt = w->umwait_tsc ? w->umwait_tsc : 100000;
    while (LD(word) == old) {
        _umonitor((void *)word);
        if (LD(word) != old) break;
        _umwait(w->umwait_c01 ? 1 : 0, __rdtsc() + dt);
    }
}
#endif

static long futex(uint32_t *addr, int op, uint32_t val) { return syscall(SYS_futex, addr, op, val, NULL, NULL, 0); }

void krt_wait_ne(const uint32_t *word, uint32_t old, const krt_wait *w, uint32_t *sleepers) {
    uint32_t spins = w->kind == KRT_WAIT_HYBRID ? w->spins : 0;
    switch (w->kind) {
    case KRT_WAIT_UMWAIT:
#if defined(__x86_64__)
        if (krt_has_waitpkg()) { umwait_ne(word, old, w); return; }
#endif
        /* fall through */
    case KRT_WAIT_SPIN:
        while (LD(word) == old) krt_cpu_relax();
        return;
    case KRT_WAIT_HYBRID:
    case KRT_WAIT_FUTEX:
    default:
        for (uint32_t i = 0; i < spins; i++) {
            if (LD(word) != old) return;
            krt_cpu_relax();
        }
        // Dekker with krt_wake: sleepers++ (seq_cst) before re-checking `word` in the kernel,
        // the waker bumps `word` (seq_cst) before reading `sleepers`.
        __atomic_fetch_add(sleepers, 1, __ATOMIC_SEQ_CST);
        while (__atomic_load_n(word, __ATOMIC_SEQ_CST) == old) futex((uint32_t *)word, FUTEX_WAIT_PRIVATE, old);
        __atomic_fetch_sub(sleepers, 1, __ATOMIC_SEQ_CST);
        return;
    }
}

void krt_wake(uint32_t *word, uint32_t *sleepers) {
    if (__atomic_load_n(sleepers, __ATOMIC_SEQ_CST)) futex(word, FUTEX_WAKE_PRIVATE, INT_MAX);
}

int krt_wait_parse(const char *s, krt_wait *w) {
    memset(w, 0, sizeof *w);
    if (!s || !strcmp(s, "spin")) w->kind = KRT_WAIT_SPIN;
    else if (!strcmp(s, "sleep")) { w->kind = KRT_WAIT_HYBRID; w->spins = 100; } // bench.c's historical meaning
    else if (!strcmp(s, "futex")) w->kind = KRT_WAIT_FUTEX;
    else if (!strcmp(s, "umwait")) w->kind = KRT_WAIT_UMWAIT;
    else if (!strcmp(s, "umwait01")) { w->kind = KRT_WAIT_UMWAIT; w->umwait_c01 = 1; }
    else if (!strncmp(s, "hybrid:", 7)) { w->kind = KRT_WAIT_HYBRID; w->spins = (uint32_t)strtoul(s + 7, NULL, 10); }
    else if (*s >= '0' && *s <= '9') { w->kind = KRT_WAIT_HYBRID; w->spins = (uint32_t)strtoul(s, NULL, 10); }
    else return -1;
    return 0;
}

const char *krt_wait_name(const krt_wait *w, char *buf, size_t n) {
    switch (w->kind) {
    case KRT_WAIT_SPIN: snprintf(buf, n, "spin"); break;
    case KRT_WAIT_HYBRID: snprintf(buf, n, "hybrid:%u", w->spins); break;
    case KRT_WAIT_FUTEX: snprintf(buf, n, "futex"); break;
    default: snprintf(buf, n, "%s%s", w->umwait_c01 ? "umwait01" : "umwait", krt_has_waitpkg() ? "" : "(spin)"); break;
    }
    return buf;
}

// ---------------------------------------------------------------- pool
struct krt_pool {
    _Alignas(CL) uint32_t count;
    _Alignas(CL) uint32_t gen;
    _Alignas(CL) uint32_t sleepers;
    _Alignas(CL) krt_wait wait;
    int nth, pin, stats_on, stop;
    krt_fn fn;
    void *arg;
    int cpus[KRT_MAX_THREADS];
    pthread_t th[KRT_MAX_THREADS];
    struct worker_arg { krt_pool *p; int ith; } wa[KRT_MAX_THREADS];
    _Alignas(CL) krt_stats st[KRT_MAX_THREADS];
};
typedef struct worker_arg worker_arg;

static void pin_to(int cpu) {
    cpu_set_t s;
    CPU_ZERO(&s);
    CPU_SET(cpu, &s);
    sched_setaffinity(0, sizeof s, &s);
}

void krt_barrier_idle(krt_pool *p, int ith, krt_idle_fn idle, void *idle_arg) {
    if (p->nth == 1) {
        if (idle) while (idle(idle_arg, ith)) {}
        return;
    }
    const uint32_t g = LD(&p->gen);
    if (__atomic_fetch_add(&p->count, 1, __ATOMIC_ACQ_REL) == (uint32_t)p->nth - 1) {
        __atomic_store_n(&p->count, 0, __ATOMIC_RELAXED);
        __atomic_fetch_add(&p->gen, 1, __ATOMIC_SEQ_CST);
        krt_wake(&p->gen, &p->sleepers);
        return;
    }
    const uint64_t t0 = p->stats_on ? krt_now_ns() : 0;
    if (idle)
        while (LD(&p->gen) == g && idle(idle_arg, ith)) {}
    krt_wait_ne(&p->gen, g, &p->wait, &p->sleepers);
    if (t0) { p->st[ith].wait_ns += krt_now_ns() - t0; p->st[ith].waits++; }
}

void krt_barrier(krt_pool *p, int ith) { krt_barrier_idle(p, ith, NULL, NULL); }

static void *worker(void *a) {
    worker_arg *wa = a;
    krt_pool *p = wa->p;
    const int ith = wa->ith;
    if (p->pin) pin_to(p->cpus[ith]);
    for (;;) {
        krt_barrier(p, ith);
        if (__atomic_load_n(&p->stop, __ATOMIC_ACQUIRE)) break;
        p->fn(p, ith, p->nth, p->arg);
        krt_barrier(p, ith);
    }
    return NULL;
}

krt_pool *krt_pool_create(int nth, const int *cpus, int pin, krt_wait wait) {
    if (nth < 1 || nth > KRT_MAX_THREADS) return NULL;
    krt_pool *p = xcalloc(1, sizeof *p);
    long ncpu = sysconf(_SC_NPROCESSORS_ONLN);
    if (ncpu < 1) ncpu = 1;
    p->nth = nth; p->pin = pin; p->wait = wait;
    for (int t = 0; t < nth; t++) p->cpus[t] = cpus ? cpus[t] : (int)(t % ncpu);
    if (pin) pin_to(p->cpus[0]);
    for (int t = 1; t < nth; t++) {
        p->wa[t] = (worker_arg){p, t};
        pthread_create(&p->th[t], NULL, worker, &p->wa[t]);
    }
    return p;
}

void krt_pool_destroy(krt_pool *p) {
    if (!p) return;
    __atomic_store_n(&p->stop, 1, __ATOMIC_RELEASE);
    krt_barrier(p, 0);
    for (int t = 1; t < p->nth; t++) pthread_join(p->th[t], NULL);
    free(p);
}

int krt_pool_threads(const krt_pool *p) { return p->nth; }
void krt_pool_set_wait(krt_pool *p, krt_wait wait) { p->wait = wait; }

void krt_pool_run(krt_pool *p, krt_fn fn, void *arg) {
    p->fn = fn; p->arg = arg;
    krt_barrier(p, 0);
    fn(p, 0, p->nth, arg);
    krt_barrier(p, 0);
}

void krt_stats_enable(krt_pool *p, int on) { p->stats_on = on; }
krt_stats *krt_stats_of(krt_pool *p, int ith) { return &p->st[ith]; }
void krt_stats_reset(krt_pool *p) { memset(p->st, 0, sizeof p->st); }

// ---------------------------------------------------------------- partitioners
static const char *PART_NAMES[] = {"static", "balanced", "streamk", "steal", "ggml"};
int krt_part_parse(const char *s) {
    for (int i = 0; i < 5; i++)
        if (!strcmp(s, PART_NAMES[i])) return i;
    return -1;
}
const char *krt_part_name(int kind) { return kind >= 0 && kind < 5 ? PART_NAMES[kind] : "?"; }

#define DQ_STRIDE 8   // uint64 per deque word (one cache line)
#define CTR_STRIDE 16 // uint32 per counter (one cache line)

struct krt_plan {
    int kind, nth, ntasks, sync_each, nitems, nsplit;
    int64_t align, tile, maxrows;
    krt_task *tasks;
    krt_item *items;
    int *first;          // nth + 1
    double *tcost;       // per-thread cost of the initial assignment
    // STREAMK split tiles
    uint32_t *arrive;    // nsplit * CTR_STRIDE
    float *partial;      // per split tile: ktiles * tile rows
    size_t *poff;
    float *scratch;      // nth * maxrows (full-K items of tasks with ktiles > 1)
    // STEAL
    uint64_t *dq;        // nth * DQ_STRIDE: head | tail << 32 (global item indices)
    // GGML
    uint32_t *ctr;       // ntasks * CTR_STRIDE
    int64_t *gn, *gdr;   // chunks and rows per chunk, per task
};

typedef struct { int task, kt0, kt1; int64_t r0, r1; double cost; } unit;

static int64_t round_up(int64_t x, int64_t a) { return (x + a - 1) / a * a; }

// Assign units (in order) to threads so each gets ~1/nth of the total cost: unit u goes to the
// thread containing the midpoint of its cost interval. Monotone, hence contiguous per thread.
static void assign(const unit *u, int n, int nth, int *owner) {
    double tot = 0, cum = 0;
    for (int i = 0; i < n; i++) tot += u[i].cost;
    for (int i = 0; i < n; i++) {
        int t = tot > 0 ? (int)((cum + u[i].cost / 2) / tot * nth) : 0;
        owner[i] = t < nth ? t : nth - 1;
        cum += u[i].cost;
    }
}

static void push_item(krt_plan *pl, int *cap, krt_item it) {
    if (pl->nitems == *cap) {
        *cap = *cap ? 2 * *cap : 64;
        krt_item *n = xcalloc(*cap, sizeof *n);
        if (pl->items) { memcpy(n, pl->items, sizeof *n * pl->nitems); free(pl->items); }
        pl->items = n;
    }
    pl->items[pl->nitems++] = it;
}

krt_plan *krt_plan_build(int kind, const krt_task *tasks, int ntasks, int nth, int64_t align, int64_t tile) {
    if (kind < 0 || kind > KRT_PART_GGML || ntasks < 1 || nth < 1 || nth > KRT_MAX_THREADS || align < 1) return NULL;
    if (tile < align) tile = align;
    tile = round_up(tile, align);
    krt_plan *pl = xcalloc(1, sizeof *pl);
    pl->kind = kind; pl->nth = nth; pl->ntasks = ntasks; pl->align = align; pl->tile = tile;
    pl->tasks = xcalloc(ntasks, sizeof *pl->tasks);
    memcpy(pl->tasks, tasks, sizeof *tasks * ntasks);
    pl->first = xcalloc(nth + 1, sizeof(int));
    pl->tcost = xcalloc(nth, sizeof(double));
    for (int k = 0; k < ntasks; k++) {
        if (pl->tasks[k].ktiles < 1) pl->tasks[k].ktiles = 1;
        if (pl->tasks[k].rows > pl->maxrows) pl->maxrows = pl->tasks[k].rows;
    }
    int cap = 0;
    if (kind == KRT_PART_STATIC || kind == KRT_PART_GGML) {
        // one item per (thread, task), possibly empty; GGML items are only the first chunk
        for (int t = 0; t < nth; t++) {
            pl->first[t] = pl->nitems;
            for (int k = 0; k < ntasks; k++) {
                const krt_task *T = &pl->tasks[k];
                int64_t per = round_up((T->rows + nth - 1) / nth, align), a = t * per, b = (t + 1) * per;
                if (a > T->rows) a = T->rows;
                if (b > T->rows) b = T->rows;
                push_item(pl, &cap, (krt_item){k, 0, T->ktiles, -1, a, b});
                pl->tcost[t] += (double)(b - a) * T->cost;
            }
        }
        pl->first[nth] = pl->nitems;
        if (kind == KRT_PART_GGML) {
            pl->ctr = xcalloc((size_t)ntasks * CTR_STRIDE, sizeof(uint32_t));
            pl->gn = xcalloc(ntasks, sizeof(int64_t));
            pl->gdr = xcalloc(ntasks, sizeof(int64_t));
            for (int k = 0; k < ntasks; k++) {
                const int64_t nr0 = pl->tasks[k].rows;
                int64_t nchunk = (nr0 + tile - 1) / tile;
                if (nchunk < 4 * nth) nchunk = nth;
                int64_t dr = round_up((nr0 + nchunk - 1) / nchunk, align); // ggml: no alignment
                pl->gn[k] = (nr0 + dr - 1) / dr;
                pl->gdr[k] = dr;
            }
        }
    } else {
        // flattened units
        int nu = 0, ucap = 0;
        unit *u = NULL;
        for (int k = 0; k < ntasks; k++) {
            const krt_task *T = &pl->tasks[k];
            const int64_t step = kind == KRT_PART_BALANCED ? align : tile;
            for (int64_t r = 0; r < T->rows; r += step) {
                const int64_t r1 = r + step < T->rows ? r + step : T->rows;
                const int nk = kind == KRT_PART_STREAMK ? T->ktiles : 1;
                for (int kt = 0; kt < nk; kt++) {
                    if (nu == ucap) {
                        ucap = ucap ? 2 * ucap : 256;
                        unit *n = xcalloc(ucap, sizeof *n);
                        if (u) { memcpy(n, u, sizeof *n * nu); free(u); }
                        u = n;
                    }
                    u[nu++] = (unit){k, nk == 1 ? 0 : kt, nk == 1 ? T->ktiles : kt + 1, r, r1, (double)(r1 - r) * T->cost / nk};
                }
            }
        }
        int *own = xcalloc(nu, sizeof(int));
        assign(u, nu, nth, own);
        // merge runs: same thread and task; rows contiguous for full-K units; K-contiguous within one tile otherwise
        int t_cur = 0;
        pl->first[0] = 0;
        for (int i = 0; i < nu; i++) {
            while (t_cur < own[i]) pl->first[++t_cur] = pl->nitems;
            pl->tcost[own[i]] += u[i].cost;
            const int full = u[i].kt0 == 0 && u[i].kt1 == pl->tasks[u[i].task].ktiles;
            krt_item *last = pl->nitems > pl->first[t_cur] ? &pl->items[pl->nitems - 1] : NULL;
            if (kind != KRT_PART_STEAL && last && last->task == u[i].task) {
                const int last_full = last->kt0 == 0 && last->kt1 == pl->tasks[last->task].ktiles;
                if (full && last_full && last->r1 == u[i].r0) { last->r1 = u[i].r1; continue; }
                if (!full && last->r0 == u[i].r0 && last->kt1 == u[i].kt0) { last->kt1 = u[i].kt1; continue; }
            }
            push_item(pl, &cap, (krt_item){u[i].task, u[i].kt0, u[i].kt1, -1, u[i].r0, u[i].r1});
        }
        while (t_cur < nth) pl->first[++t_cur] = pl->nitems;
        free(u);
        free(own);
        if (kind == KRT_PART_STREAMK) {
            // items covering part of a tile's K range: give the tile a partial buffer and an arrival counter
            size_t off = 0;
            int ns = 0;
            for (int i = 0; i < pl->nitems; i++) {
                krt_item *it = &pl->items[i];
                if (it->kt0 == 0 && it->kt1 == pl->tasks[it->task].ktiles) continue;
                int found = -1;
                for (int j = 0; j < i; j++) {
                    const krt_item *o = &pl->items[j];
                    if (o->tile >= 0 && o->task == it->task && o->r0 == it->r0) { found = o->tile; break; }
                }
                it->tile = found >= 0 ? found : ns++;
            }
            pl->nsplit = ns;
            pl->arrive = xcalloc((size_t)(ns ? ns : 1) * CTR_STRIDE, sizeof(uint32_t));
            pl->poff = xcalloc(ns ? ns : 1, sizeof(size_t));
            for (int s = 0; s < ns; s++)
                for (int i = 0; i < pl->nitems; i++)
                    if (pl->items[i].tile == s) {
                        pl->poff[s] = off;
                        off += (size_t)pl->tasks[pl->items[i].task].ktiles * (size_t)tile;
                        break;
                    }
            pl->partial = xcalloc(off ? off : 1, sizeof(float));
        }
        if (kind == KRT_PART_STEAL) pl->dq = xcalloc((size_t)nth * DQ_STRIDE, sizeof(uint64_t));
    }
    int need_scratch = 0;
    for (int k = 0; k < ntasks; k++) need_scratch |= pl->tasks[k].ktiles > 1;
    if (need_scratch) pl->scratch = xcalloc((size_t)nth * (size_t)round_up(pl->maxrows, 16), sizeof(float));
    krt_plan_reset(pl);
    return pl;
}

void krt_plan_free(krt_plan *pl) {
    if (!pl) return;
    free(pl->tasks); free(pl->items); free(pl->first); free(pl->tcost); free(pl->arrive); free(pl->partial);
    free(pl->poff); free(pl->scratch); free(pl->dq); free(pl->ctr); free(pl->gn); free(pl->gdr);
    free(pl);
}

void krt_plan_set_sync_each(krt_plan *pl, int on) { pl->sync_each = on; }

void krt_plan_reset(krt_plan *pl) {
    if (pl->dq)
        for (int t = 0; t < pl->nth; t++)
            __atomic_store_n(&pl->dq[t * DQ_STRIDE], (uint64_t)(uint32_t)pl->first[t] | ((uint64_t)(uint32_t)pl->first[t + 1] << 32),
                             __ATOMIC_RELAXED);
    if (pl->ctr)
        for (int k = 0; k < pl->ntasks; k++) __atomic_store_n(&pl->ctr[k * CTR_STRIDE], (uint32_t)pl->nth, __ATOMIC_RELAXED);
}

const krt_item *krt_plan_items(const krt_plan *pl, int ith, int *n) {
    *n = pl->first[ith + 1] - pl->first[ith];
    return pl->items + pl->first[ith];
}

double krt_plan_imbalance(const krt_plan *pl) {
    double mx = 0, sum = 0;
    for (int t = 0; t < pl->nth; t++) { sum += pl->tcost[t]; if (pl->tcost[t] > mx) mx = pl->tcost[t]; }
    return sum > 0 ? mx / (sum / pl->nth) : 1.0;
}

static void fold_add(float *y, const float *p, int64_t n) {
    for (int64_t i = 0; i < n; i++) y[i] += p[i];
}

static void run_item(krt_pool *pool, krt_plan *pl, int ith, const krt_item *it, krt_kernel_fn fn, void *user) {
    if (it->r0 >= it->r1) return;
    const krt_task *T = &pl->tasks[it->task];
    const uint64_t t0 = pool && pool->stats_on ? krt_now_ns() : 0;
    const int64_t n = it->r1 - it->r0;
    if (T->ktiles == 1) {
        fn(user, it->task, 0, it->r0, it->r1, T->y + it->r0);
    } else if (it->tile < 0) { // all K tiles here: same left-to-right fold as the split path
        float *tmp = pl->scratch + (size_t)ith * (size_t)round_up(pl->maxrows, 16);
        fn(user, it->task, 0, it->r0, it->r1, T->y + it->r0);
        for (int kt = 1; kt < T->ktiles; kt++) {
            fn(user, it->task, kt, it->r0, it->r1, tmp);
            fold_add(T->y + it->r0, tmp, n);
        }
    } else {
        float *buf = pl->partial + pl->poff[it->tile];
        for (int kt = it->kt0; kt < it->kt1; kt++) fn(user, it->task, kt, it->r0, it->r1, buf + (size_t)kt * pl->tile);
        uint32_t *ctr = &pl->arrive[it->tile * CTR_STRIDE];
        if (__atomic_add_fetch(ctr, (uint32_t)(it->kt1 - it->kt0), __ATOMIC_ACQ_REL) == (uint32_t)T->ktiles) {
            __atomic_store_n(ctr, 0, __ATOMIC_RELAXED); // next use is behind a barrier
            float *y = T->y + it->r0;
            memcpy(y, buf, sizeof(float) * n);
            for (int kt = 1; kt < T->ktiles; kt++) fold_add(y, buf + (size_t)kt * pl->tile, n);
        }
    }
    if (t0) { pool->st[ith].busy_ns += krt_now_ns() - t0; pool->st[ith].items++; }
}

static int dq_pop(uint64_t *w, int *idx) { // owner: take from the head
    uint64_t v = __atomic_load_n(w, __ATOMIC_ACQUIRE);
    for (;;) {
        uint32_t h = (uint32_t)v, t = (uint32_t)(v >> 32);
        if (h >= t) return 0;
        if (__atomic_compare_exchange_n(w, &v, (uint64_t)(h + 1) | ((uint64_t)t << 32), 0, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) {
            *idx = (int)h;
            return 1;
        }
    }
}

static int dq_steal(uint64_t *w, int *idx) { // thief: take from the tail
    uint64_t v = __atomic_load_n(w, __ATOMIC_ACQUIRE);
    for (;;) {
        uint32_t h = (uint32_t)v, t = (uint32_t)(v >> 32);
        if (h >= t) return 0;
        if (__atomic_compare_exchange_n(w, &v, (uint64_t)h | ((uint64_t)(t - 1) << 32), 0, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) {
            *idx = (int)(t - 1);
            return 1;
        }
    }
}

void krt_plan_exec(krt_pool *pool, krt_plan *pl, int ith, krt_kernel_fn fn, void *user) {
    const int nth = pl->nth;
    switch (pl->kind) {
    case KRT_PART_STATIC:
        for (int i = pl->first[ith]; i < pl->first[ith + 1]; i++) {
            run_item(pool, pl, ith, &pl->items[i], fn, user);
            if (pl->sync_each && pool) krt_barrier(pool, ith);
        }
        break;
    case KRT_PART_BALANCED:
    case KRT_PART_STREAMK:
        for (int i = pl->first[ith]; i < pl->first[ith + 1]; i++) run_item(pool, pl, ith, &pl->items[i], fn, user);
        break;
    case KRT_PART_STEAL: {
        int idx;
        while (dq_pop(&pl->dq[ith * DQ_STRIDE], &idx)) run_item(pool, pl, ith, &pl->items[idx], fn, user);
        for (;;) { // steal from the thread with the most work left
            int victim = -1;
            uint32_t most = 0;
            for (int d = 1; d < nth; d++) {
                const int v = (ith + d) % nth;
                const uint64_t w = __atomic_load_n(&pl->dq[v * DQ_STRIDE], __ATOMIC_RELAXED);
                const uint32_t left = (uint32_t)(w >> 32) > (uint32_t)w ? (uint32_t)(w >> 32) - (uint32_t)w : 0;
                if (left > most) { most = left; victim = v; }
            }
            if (victim < 0) break;
            if (dq_steal(&pl->dq[victim * DQ_STRIDE], &idx)) {
                if (pool && pool->stats_on) pool->st[ith].steals++;
                run_item(pool, pl, ith, &pl->items[idx], fn, user);
            }
        }
        break;
    }
    case KRT_PART_GGML:
        for (int k = 0; k < pl->ntasks; k++) {
            const krt_task *T = &pl->tasks[k];
            const int64_t nchunk = pl->gn[k], dr = pl->gdr[k];
            int64_t c = ith;
            while (c < nchunk) {
                const int64_t r1 = (c + 1) * dr < T->rows ? (c + 1) * dr : T->rows;
                const krt_item it = {k, 0, T->ktiles, -1, c * dr, r1};
                run_item(pool, pl, ith, &it, fn, user);
                if (nth >= nchunk) break;
                c = __atomic_fetch_add(&pl->ctr[k * CTR_STRIDE], 1, __ATOMIC_RELAXED);
            }
        }
        break;
    }
}

// ---------------------------------------------------------------- prefetch
size_t krt_prefetch(const void *p, size_t bytes, int hint) {
    const char *c = (const char *)((uintptr_t)p & ~(uintptr_t)(CL - 1)), *e = (const char *)p + bytes;
    size_t n = 0;
    for (; c < e; c += CL, n += CL) {
        switch (hint) {
        case 0: __builtin_prefetch(c, 0, 3); break;
        case 1: __builtin_prefetch(c, 0, 2); break;
        case 2: __builtin_prefetch(c, 0, 1); break;
        default: __builtin_prefetch(c, 0, 0); break;
        }
    }
    return n;
}

void krt_prefetch_queue_init(krt_prefetch_queue *q, size_t budget, int hint) {
    memset(q, 0, sizeof *q);
    q->budget = budget;
    q->hint = hint;
}

void krt_prefetch_queue_add(krt_prefetch_queue *q, const void *p, size_t bytes) {
    if (q->n < 16 && bytes) { q->ptr[q->n] = p; q->len[q->n] = bytes; q->n++; }
}

int krt_prefetch_idle(void *qv, int ith) {
    (void)ith;
    krt_prefetch_queue *q = qv;
    if (q->cur >= q->n || q->done >= q->budget) return 0;
    size_t chunk = 1024;
    if (q->off + chunk > q->len[q->cur]) chunk = q->len[q->cur] - q->off;
    q->done += krt_prefetch(q->ptr[q->cur] + q->off, chunk, q->hint);
    q->off += chunk;
    if (q->off >= q->len[q->cur]) { q->cur++; q->off = 0; }
    return q->cur < q->n && q->done < q->budget;
}
