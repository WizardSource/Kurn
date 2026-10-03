// Self-test of the kurn runtime: every partitioner covers each (task, row, K tile)
// exactly once (also under concurrent stealing / chunk counters), STREAMK's
// fix-up reduction is deterministic and independent of the thread count, and the
// barrier works with every wait policy. Prints "ok" and exits 0 on success.
//   rt_selftest [threads]
#define _GNU_SOURCE
#include "kurn_rt.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define FAIL(...) do { fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); exit(1); } while (0)
#define MAXR 4096
#define MAXK 8

typedef struct {
    int ntasks;
    krt_task t[16];
    uint32_t hits[16][MAXK][MAXR];
    float val[16][MAXK][MAXR]; // deterministic per-(task, kt, row) "partial dot products"
} work;

static void count_fn(void *u, int task, int kt, int64_t r0, int64_t r1, float *out) {
    work *w = u;
    for (int64_t r = r0; r < r1; r++) {
        __atomic_fetch_add(&w->hits[task][kt][r], 1, __ATOMIC_RELAXED);
        out[r - r0] = w->val[task][kt][r];
    }
}

typedef struct { krt_plan *pl; work *w; int reps; } job;
static void exec_job(krt_pool *p, int ith, int nth, void *a) {
    (void)nth;
    job *j = a;
    for (int r = 0; r < j->reps; r++) {
        if (ith == 0) krt_plan_reset(j->pl);
        krt_barrier(p, ith);
        krt_plan_exec(p, j->pl, ith, count_fn, j->w);
        krt_barrier(p, ith);
    }
}

static uint64_t rng = 12345;
static uint32_t rnd(uint32_t n) {
    rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
    return (uint32_t)(rng % n);
}

static void make_work(work *w, int ksplit) {
    memset(w, 0, sizeof *w);
    w->ntasks = 1 + (int)rnd(12);
    for (int k = 0; k < w->ntasks; k++) {
        const int64_t rows = 16 * (1 + (int64_t)rnd(MAXR / 16 - 1)) - (rnd(4) == 0 ? (int64_t)rnd(15) : 0);
        static float ys[16][MAXR];
        w->t[k] = (krt_task){rows, ksplit, 1.0 + rnd(4), ys[k]};
        for (int kt = 0; kt < ksplit; kt++)
            for (int64_t r = 0; r < rows; r++) w->val[k][kt][r] = (float)((int)rnd(2001) - 1000) * 1e-3f + 0.1f * kt;
    }
}

static void check_plan(krt_pool *pool, int kind, int nth, int ksplit, int64_t align, int64_t tile) {
    static work w;
    make_work(&w, ksplit);
    krt_plan *pl = krt_plan_build(kind, w.t, w.ntasks, nth, align, tile);
    if (!pl) FAIL("plan build failed");
    // static share: initial items cover every (task, row, kt) exactly once (STATIC/GGML list only first chunks)
    if (kind != KRT_PART_GGML) {
        static uint8_t cov[16][MAXK][MAXR];
        memset(cov, 0, sizeof cov);
        for (int t = 0; t < nth; t++) {
            int n;
            const krt_item *it = krt_plan_items(pl, t, &n);
            for (int i = 0; i < n; i++) {
                if (it[i].r0 % align) FAIL("%s: item r0 %ld not aligned", krt_part_name(kind), (long)it[i].r0);
                for (int kt = it[i].kt0; kt < it[i].kt1; kt++)
                    for (int64_t r = it[i].r0; r < it[i].r1; r++) cov[it[i].task][kt][r]++;
            }
        }
        for (int k = 0; k < w.ntasks; k++)
            for (int kt = 0; kt < ksplit; kt++)
                for (int64_t r = 0; r < w.t[k].rows; r++)
                    if (cov[k][kt][r] != 1) FAIL("%s nth=%d ks=%d: task %d kt %d row %ld covered %d times", krt_part_name(kind), nth, ksplit, k, kt, (long)r, cov[k][kt][r]);
    }
    // execution: every (task, kt, row) computed exactly once per run, outputs = left fold over kt
    const int reps = 5;
    job j = {pl, &w, reps};
    krt_pool_run(pool, exec_job, &j);
    for (int k = 0; k < w.ntasks; k++)
        for (int64_t r = 0; r < w.t[k].rows; r++) {
            float ref = w.val[k][0][r];
            for (int kt = 1; kt < ksplit; kt++) ref += w.val[k][kt][r];
            for (int kt = 0; kt < ksplit; kt++)
                if (w.hits[k][kt][r] != (uint32_t)reps)
                    FAIL("%s nth=%d ks=%d: task %d kt %d row %ld ran %u times (want %d)", krt_part_name(kind), nth, ksplit, k, kt, (long)r, w.hits[k][kt][r], reps);
            if (memcmp(&ref, &w.t[k].y[r], sizeof ref))
                FAIL("%s nth=%d ks=%d: task %d row %ld = %.9g, want %.9g (left fold)", krt_part_name(kind), nth, ksplit, k, (long)r, w.t[k].y[r], ref);
        }
    krt_plan_free(pl);
}

typedef struct { int iters; uint32_t phase[KRT_MAX_THREADS]; int bad; } bjob;
static void barrier_job(krt_pool *p, int ith, int nth, void *a) {
    bjob *b = a;
    for (int i = 0; i < b->iters; i++) {
        __atomic_store_n(&b->phase[ith], (uint32_t)i + 1, __ATOMIC_RELEASE);
        krt_barrier(p, ith);
        for (int t = 0; t < nth; t++)
            if (__atomic_load_n(&b->phase[t], __ATOMIC_ACQUIRE) < (uint32_t)i + 1) __atomic_store_n(&b->bad, 1, __ATOMIC_RELAXED);
        krt_barrier(p, ith);
    }
}

// --calibrate [T]: ns per pause, and barrier round trips (T threads, empty work) per wait policy
typedef struct { int iters; double gap_us; } cjob;
static void cal_job(krt_pool *p, int ith, int nth, void *a) {
    (void)nth;
    cjob *c = a;
    for (int i = 0; i < c->iters; i++) {
        if (ith == 0 && c->gap_us > 0) {
            const uint64_t end = krt_now_ns() + (uint64_t)(c->gap_us * 1e3);
            while (krt_now_ns() < end) krt_cpu_relax();
        }
        krt_barrier(p, ith);
    }
}
static int calibrate(int T) {
    const int n = 2000000;
    uint64_t t0 = krt_now_ns();
    for (int i = 0; i < n; i++) krt_cpu_relax();
    printf("pause: %.1f ns\n", (double)(krt_now_ns() - t0) / n);
    const char *waits[] = {"spin", "hybrid:500", "hybrid:5000", "futex", "umwait"};
    const double gaps[] = {0, 5, 20, 50, 200};
    for (int wi = 0; wi < 5; wi++) {
        krt_wait w;
        krt_wait_parse(waits[wi], &w);
        krt_pool *pool = krt_pool_create(T, NULL, 1, w);
        for (int gi = 0; gi < 5; gi++) {
            cjob c = {gaps[gi] > 0 ? (int)(200000 / (gaps[gi] + 2)) : 100000, gaps[gi]};
            if (c.iters < 2000) c.iters = 2000;
            krt_pool_run(pool, cal_job, &c); // warm-up
            struct timespec a, b;
            clock_gettime(CLOCK_MONOTONIC, &a);
            const double c0 = (double)clock() / CLOCKS_PER_SEC;
            krt_pool_run(pool, cal_job, &c);
            clock_gettime(CLOCK_MONOTONIC, &b);
            const double wall = (b.tv_sec - a.tv_sec) + (b.tv_nsec - a.tv_nsec) * 1e-9, cpu = (double)clock() / CLOCKS_PER_SEC - c0;
            printf("wait=%-12s T=%d gap=%5.0f us: %8.2f us per barrier cycle (overhead %6.2f us), cpu/wall %.2f\n", waits[wi], T, gaps[gi],
                   wall / c.iters * 1e6, wall / c.iters * 1e6 - gaps[gi], cpu / wall);
        }
        krt_pool_destroy(pool);
    }
    return 0;
}

int main(int argc, char **argv) {
    if (argc > 1 && !strcmp(argv[1], "--calibrate")) return calibrate(argc > 2 ? atoi(argv[2]) : 8);
    const int T = argc > 1 ? atoi(argv[1]) : 4;
    const char *waits[] = {"spin", "futex", "hybrid:200", "umwait"};
    for (int wi = 0; wi < 4; wi++) {
        krt_wait w;
        if (krt_wait_parse(waits[wi], &w)) FAIL("parse %s", waits[wi]);
        krt_pool *pool = krt_pool_create(T, NULL, 0, w);
        bjob b = {wi == 0 ? 20000 : 2000, {0}, 0};
        krt_pool_run(pool, barrier_job, &b);
        if (b.bad) FAIL("barrier broken with wait=%s", waits[wi]);
        krt_pool_destroy(pool);
    }
    krt_wait spin;
    krt_wait_parse("spin", &spin);
    for (int nth = 1; nth <= T; nth++) {
        krt_pool *pool = krt_pool_create(nth, NULL, 0, spin);
        for (int kind = 0; kind <= KRT_PART_GGML; kind++)
            for (int rep = 0; rep < 6; rep++) {
                const int ks = kind == KRT_PART_STREAMK ? 1 + (int)rnd(MAXK) : (rep % 3 == 2 ? 2 : 1);
                check_plan(pool, kind, nth, ks, 16, 16 * (1 + (int64_t)rnd(16)));
            }
        krt_pool_destroy(pool);
    }
    printf("ok waitpkg=%d threads=%d\n", krt_has_waitpkg(), T);
    return 0;
}
