/* Harness for the sparse SwiGLU FFN (sparse_ffn.c is compiled in):
 *
 *   bench_sparse --mode dense|gate|pred [--d 2048] [--dff 6144] [--rank 128] [--density 0.5]
 *                [--thr T] [--threads 8] [--secs 1] [--regime hot|cold] [--cold-bytes 7e8]
 *                [--seed 1] [--tol 1e-5] [--csv out.csv]
 *
 * Random Q8_0 weights (cold: enough layer copies to exceed --cold-bytes) and x ~ N(0,1).
 * With --density the active set is a random subset (Bernoulli) forced into the kernel; with
 * --thr the kernel selects rows itself. Checked against a float64 reference of the same
 * quantized math on the kernel's active set (active-set agreement is checked separately).
 * Bytes per call count what the mode must read: gate rows (or predictor) + active rows.
 */
#define _GNU_SOURCE
#include <pthread.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#include "sparse_ffn.c"

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
static double cpu_now(void) { struct timespec t; clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
static double drift(void) {
    struct timespec r, m;
    clock_gettime(CLOCK_REALTIME, &r);
    clock_gettime(CLOCK_MONOTONIC, &m);
    return (r.tv_sec - m.tv_sec) + 1e-9 * (r.tv_nsec - m.tv_nsec);
}
static uint64_t rs = 88172645463325252ull;
static double urand(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (rs >> 11) * (1.0 / 9007199254740992.0); }
static double nrand(void) { return sqrt(-2 * log(urand() + 1e-300)) * cos(6.283185307179586 * urand()); }

static const char *arg(int argc, char **argv, const char *k, const char *def) {
    for (int i = 1; i < argc - 1; i++)
        if (!strcmp(argv[i], k)) return argv[i + 1];
    return def;
}

static void rand_q8(ks_blk *w, size_t nblk, double scale) {
    for (size_t b = 0; b < nblk; b++) {
        w[b].d = _cvtss_sh((float)(scale * (0.5 + urand())), 0);
        for (int i = 0; i < 32; i++) w[b].q[i] = (int8_t)(lrint(urand() * 254.0) - 127);
    }
}

static double ref_dot(const ks_blk *w, const ks_xblk *x, int nb) {
    double s = 0;
    for (int b = 0; b < nb; b++) {
        long acc = 0;
        for (int i = 0; i < 32; i++) acc += (long)w[b].q[i] * x[b].q[i];
        s += (double)_cvtsh_ss(w[b].d) * x[b].d * acc;
    }
    return s;
}

static int NTH;
static ks_args *CUR;
static void *WS;
static atomic_int gen, arrived, quit;

/* one thread per vCPU: the barriers spin, so a waiter must never share a CPU with the
 * thread it waits for */
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
        ks_ffn(CUR, WS, ith, NTH);
        atomic_fetch_add_explicit(&arrived, 1, memory_order_acq_rel);
    }
}

static void run_call(ks_args *a) {
    CUR = a;
    atomic_store(&arrived, 0);
    atomic_fetch_add_explicit(&gen, 1, memory_order_release);
    ks_ffn(a, WS, 0, NTH);
    while (atomic_load_explicit(&arrived, memory_order_acquire) < NTH - 1) _mm_pause();
}

int main(int argc, char **argv) {
    const char *ms = arg(argc, argv, "--mode", "gate");
    const int mode = !strcmp(ms, "dense") ? KS_DENSE : !strcmp(ms, "pred") ? KS_PRED : KS_GATE;
    const int d = atoi(arg(argc, argv, "--d", "2048")), dff = atoi(arg(argc, argv, "--dff", "6144"));
    const int rank = mode == KS_PRED ? atoi(arg(argc, argv, "--rank", "128")) : 0;
    const double density = atof(arg(argc, argv, "--density", "-1"));
    const char *thr_s = arg(argc, argv, "--thr", NULL);
    NTH = atoi(arg(argc, argv, "--threads", "8"));
    const double secs = atof(arg(argc, argv, "--secs", "1"));
    const char *regime = arg(argc, argv, "--regime", "hot");
    const double tol = atof(arg(argc, argv, "--tol", "1e-5"));
    rs ^= (uint64_t)atoll(arg(argc, argv, "--seed", "1")) * 0x9E3779B97F4A7C15ull;
    const int nbd = d / 32;
    const size_t mat = (size_t)dff * nbd, pab = (size_t)rank * nbd, pbb = (size_t)dff * (rank / 32);
    const double layer_bytes = 34.0 * (3 * mat + pab + pbb);
    int nl = 1;
    if (!strcmp(regime, "cold")) {
        nl = (int)ceil(atof(arg(argc, argv, "--cold-bytes", "7e8")) / layer_bytes);
        if (nl < 2) nl = 2;
    }
    ks_blk **wg = malloc(sizeof(void *) * nl), **wu = malloc(sizeof(void *) * nl), **wd = malloc(sizeof(void *) * nl);
    ks_blk **pa = malloc(sizeof(void *) * nl), **pb = malloc(sizeof(void *) * nl);
    for (int l = 0; l < nl; l++) {
        wg[l] = aligned_alloc(64, (mat * 34 + 63) / 64 * 64);
        wu[l] = aligned_alloc(64, (mat * 34 + 63) / 64 * 64);
        wd[l] = aligned_alloc(64, (mat * 34 + 63) / 64 * 64);
        rand_q8(wg[l], mat, 0.02);
        rand_q8(wu[l], mat, 0.02);
        rand_q8(wd[l], mat, 0.02);
        pa[l] = pab ? aligned_alloc(64, (pab * 34 + 63) / 64 * 64) : NULL;
        pb[l] = pbb ? aligned_alloc(64, (pbb * 34 + 63) / 64 * 64) : NULL;
        if (pab) { rand_q8(pa[l], pab, 0.02); rand_q8(pb[l], pbb, 0.02); }
    }
    float *x = malloc(sizeof(float) * d), *y = malloc(sizeof(float) * d);
    for (int i = 0; i < d; i++) x[i] = (float)nrand();
    uint8_t *force = NULL;
    if (density >= 0) {
        force = malloc(dff);
        for (int i = 0; i < dff; i++) force[i] = urand() < density;
    }
    const float thr = thr_s ? (float)atof(thr_s) : 0.0f;
    int64_t n_active = 0;
    ks_args *A = malloc(sizeof(ks_args) * nl);
    for (int l = 0; l < nl; l++) {
        ks_args a = {d, dff, rank, mode, thr, wg[l], wu[l], wd[l], pa[l], pb[l], force, x, y, NULL};
        A[l] = a;
    }
    const size_t wsz = ks_workspace(&A[0], NTH);
    WS = aligned_alloc(64, (wsz + 63) / 64 * 64);
    memset(WS, 0, (wsz + 63) / 64 * 64);
    pthread_t th[256];
    pin(0);
    for (int i = 1; i < NTH; i++) pthread_create(&th[i], NULL, worker, (void *)(intptr_t)i);

    /* correctness on layer 0 */
    A[0].n_active = &n_active;
    for (int i = 0; i < d; i++) y[i] = NAN;
    run_call(&A[0]);
    A[0].n_active = NULL;
    ks_xblk *xq = malloc(sizeof(ks_xblk) * nbd);
    ks_quant(x, d, xq);
    double *yr = calloc(d, sizeof(double));
    float *zf = malloc(sizeof(float) * (rank + 1));
    ks_xblk *zq = malloc(sizeof(ks_xblk) * (rank / 32 + 1));
    if (mode == KS_PRED) {
        for (int j = 0; j < rank; j++) zf[j] = (float)ref_dot(pa[0] + (size_t)j * nbd, xq, nbd);
        ks_quant(zf, rank, zq);
    }
    int64_t ref_active = 0, sel_mismatch = 0;
    for (int i = 0; i < dff; i++) {
        const double g = ref_dot(wg[0] + (size_t)i * nbd, xq, nbd);
        int on;
        if (mode == KS_DENSE) on = 1;
        else if (force) on = force[i];
        else if (mode == KS_GATE) on = fabs(g / (1 + exp(-g))) >= thr;
        else {
            const double gp = ref_dot(pb[0] + (size_t)i * (rank / 32), zq, rank / 32);
            on = fabs(gp / (1 + exp(-gp))) >= thr;
        }
        if (!on) continue;
        ref_active++;
        const double h = g / (1 + exp(-g)) * ref_dot(wu[0] + (size_t)i * nbd, xq, nbd);
        const ks_blk *row = wd[0] + (size_t)i * nbd;
        for (int b = 0; b < nbd; b++)
            for (int k = 0; k < 32; k++) yr[32 * b + k] += h * _cvtsh_ss(row[b].d) * row[b].q[k];
    }
    sel_mismatch = llabs(ref_active - n_active);
    double maxerr = 0, maxref = 0;
    int nan_seen = 0;
    for (int i = 0; i < d; i++) {
        if (isnan(y[i])) nan_seen = 1;
        maxerr = fmax(maxerr, fabs(y[i] - yr[i]));
        maxref = fmax(maxref, fabs(yr[i]));
    }
    /* a threshold-selected row can flip on float rounding; tolerate a handful but report them */
    const double relerr = nan_seen ? INFINITY : maxerr / (maxref > 0 ? maxref : 1);
    const int ok = relerr <= tol || (sel_mismatch > 0 && sel_mismatch <= 2 && relerr <= 1e-3);

    const double d0 = drift();
    int64_t calls = 0;
    for (int l = 0; l < nl; l++) run_call(&A[l]);
    const double t0 = now(), c0 = cpu_now();
    double t;
    do {
        run_call(&A[calls % nl]);
        calls++;
    } while ((t = now() - t0) < secs || calls < 2);
    const double cpu = cpu_now() - c0, d1 = drift();
    atomic_store(&quit, 1);
    atomic_fetch_add(&gen, 1);
    for (int i = 1; i < NTH; i++) pthread_join(th[i], NULL);

    const double row = 34.0 * nbd, act = (double)n_active;
    double bytes = 0;
    if (mode == KS_DENSE) bytes = 3 * dff * row;
    else if (mode == KS_GATE) bytes = dff * row + 2 * act * row;
    else bytes = 34.0 * (pab + pbb) + 3 * act * row;
    const double us = t / calls * 1e6;
    printf("%-5s d %d dff %d rank %d active %lld/%d (%.3f) threads %d %s: %.1f us/call  %.1f GB/s  cpu/wall %.2f  relerr %.2e %s (sel diff %lld)\n",
           ms, d, dff, rank, (long long)n_active, dff, act / dff, NTH, regime, us, bytes / (us * 1e3), cpu / t, relerr,
           ok ? "ok" : "FAIL", (long long)sel_mismatch);
    const char *csv = arg(argc, argv, "--csv", NULL);
    if (csv) {
        FILE *f = fopen(csv, "w");
        fprintf(f, "sparse_ffn,%s,%s,%d,%d,%d,%d,%.4f,%lld,%.6f,%.6f,%.3f,%.3f,%.3f,%.3e,%s,%.3e\n", ms, regime, NTH, d, dff,
                rank, act / dff, (long long)calls, t, cpu, us, bytes / (us * 1e3), cpu / calls * 1e6 * 5.47, relerr,
                ok ? "ok" : "FAIL", d1 - d0);
        fclose(f);
    }
    return ok ? 0 : 2;
}
