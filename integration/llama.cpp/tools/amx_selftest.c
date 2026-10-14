// amx_selftest: does AMX return correct tile matmul results on this machine?
//
// Standard Linux AMX setup, then a loop of small int8 (TDPBSSD) and bf16 (TDPBF16PS) tile matmuls, each
// compared bit-for-bit against a scalar reference. Inputs are small integers, so the bf16 products and fp32
// sums are exact and any difference is a wrong result, not rounding.
//
//   1. arch_prctl(ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA) before any thread is created
//   2. each thread loads a tile configuration (palette 1) once
//   3. one computation: zero C, then K steps of {load A_k, load B_k, C += A_k B_k} with C kept in its tile,
//      store C (16 x 16) and compare. Four input sets rotate so a stale tile from an earlier computation
//      cannot look correct.
//
// Options:
//   -t N     threads (default 1)
//   -s SEC   run time in seconds (default 5); -n ITER runs a fixed number of computations per thread instead
//   -m MODE  int8 | bf16 | both (default, alternating) | busy (no AMX: AVX-512 FMA loop, a non-AMX load) |
//            tick (no AMX: wake every -w USEC (default 100), spin ~1 us, sleep again; a timer/poller-style load)
//   -k K     k-steps per computation (default 32, kurn's guarded step length)
//   -p CPU   pin thread i to CPU (CPU + i) % ncpu (default: no pinning); with -S every thread on CPU
//   -S       with -p: all threads share the one CPU
//   -g CYC   timing guard as in kurn: redo a computation whose rdtsc span exceeds CYC + 128 * K cycles
//            (default 0 = off; kurn's default is 4096)
//   -y       sched_yield() halfway through each computation (provokes a context switch; diagnostic)
//   -w USEC  sleep USEC microseconds halfway through each computation (the CPU may go idle; diagnostic)
//   -q       print only the summary line
//   -v       log the rdtsc span of every wrong computation to stderr
//
// Output: one summary line of key=value pairs (bad_span_min/max: rdtsc cycles of the shortest/longest wrong
// computation; slow_ok: correct computations slower than kurn's default guard, i.e. what the guard would redo).
// Exit code: 0 all results correct, 1 wrong results seen, 2 AMX unavailable or setup failed.
#define _GNU_SOURCE
#include <errno.h>
#include <immintrin.h>
#include <pthread.h>
#include <sched.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>
#include <x86intrin.h>

#define ARCH_GET_XCOMP_PERM 0x1022
#define ARCH_REQ_XCOMP_PERM 0x1023
#define XFEATURE_XTILEDATA 18

#define KMAX 64
#define NSETS 4

enum { M_INT8 = 1, M_BF16 = 2, M_BOTH = 3, M_BUSY = 4, M_TICK = 5 };

static int g_threads = 1, g_mode = M_BOTH, g_k = 32, g_pin = -1, g_same = 0, g_yield = 0, g_sleep_us = 0, g_quiet = 0, g_verbose = 0;
static double g_secs = 5;
static long g_iters = 0;
static uint64_t g_guard = 0;
static volatile int g_stop = 0;

typedef struct {
    uint8_t palette, start_row, reserved[14];
    uint16_t colsb[16];
    uint8_t rows[16];
} __attribute__((aligned(64))) tilecfg;

// tmm0 C int32/fp32 16 x 16, tmm1 A 16 x 64 B, tmm2 B 16 x 64 B (int8: 16 rows of 4-byte groups; bf16: 16 pairs)
static void load_config(void) {
    tilecfg c;
    memset(&c, 0, sizeof c);
    c.palette = 1;
    for (int t = 0; t < 3; t++) { c.rows[t] = 16; c.colsb[t] = 64; }
    _tile_loadconfig(&c);
}

static int config_intact(void) {
    tilecfg c;
    _tile_storeconfig(&c);
    return c.palette == 1 && c.rows[0] == 16 && c.colsb[0] == 64 && c.rows[2] == 16 && c.colsb[2] == 64;
}

typedef struct {
    // int8: A[k] 16 x 64 s8, B[k] 16 x 64 s8 (row r = k-group r, 16 columns x 4 bytes)
    int8_t a8[NSETS][KMAX][16 * 64] __attribute__((aligned(64)));
    int8_t b8[NSETS][KMAX][16 * 64] __attribute__((aligned(64)));
    int32_t ref8[NSETS][256] __attribute__((aligned(64)));
    // bf16: A[k] 16 x 32 bf16, B[k] 16 pairs x 16 columns x 2 bf16
    uint16_t a16[NSETS][KMAX][16 * 32] __attribute__((aligned(64)));
    uint16_t b16[NSETS][KMAX][16 * 32] __attribute__((aligned(64)));
    float ref16[NSETS][256] __attribute__((aligned(64)));
} inputs;

typedef struct {
    long id;
    long n8, bad8, n16, bad16, zero, redo, cfg_lost;
    uint64_t bad_span_min, bad_span_max;  // rdtsc span of wrong computations (zero..store)
    long slow_ok;                         // correct computations that took longer than 4096 + 128 K cycles
    double busy_sink;
} result;

static uint64_t rng_next(uint64_t *s) {
    *s ^= *s << 13; *s ^= *s >> 7; *s ^= *s << 17;
    return *s;
}

static uint16_t bf16_of_int(int v) {  // small integers are exact in bf16
    float f = (float)v;
    uint32_t u;
    memcpy(&u, &f, 4);
    return (uint16_t)(u >> 16);
}

static float float_of_bf16(uint16_t h) {
    uint32_t u = (uint32_t)h << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}

static void make_inputs(inputs *in, uint64_t seed) {
    uint64_t s = seed | 1;
    for (int set = 0; set < NSETS; set++) {
        for (int k = 0; k < g_k; k++) {
            for (int i = 0; i < 1024; i++) {
                in->a8[set][k][i] = (int8_t)(rng_next(&s) & 0xff);
                in->b8[set][k][i] = (int8_t)(rng_next(&s) & 0xff);
            }
            for (int i = 0; i < 512; i++) {
                in->a16[set][k][i] = bf16_of_int((int)(rng_next(&s) % 9) - 4);
                in->b16[set][k][i] = bf16_of_int((int)(rng_next(&s) % 9) - 4);
            }
        }
        // C[m][n] = sum_k sum_j A_k[m][j] * B_k[j/4][n*4 + j%4]       (int8, j < 64)
        // C[m][n] = sum_k sum_j A_k[m][j] * B_k[j/2][n*2 + j%2]       (bf16, j < 32)
        for (int m = 0; m < 16; m++) {
            for (int n = 0; n < 16; n++) {
                int32_t acc = 0;
                float facc = 0;
                for (int k = 0; k < g_k; k++) {
                    for (int j = 0; j < 64; j++)
                        acc += (int32_t)in->a8[set][k][m * 64 + j] * in->b8[set][k][(j / 4) * 64 + n * 4 + j % 4];
                    for (int j = 0; j < 32; j++)
                        facc += float_of_bf16(in->a16[set][k][m * 32 + j]) *
                                float_of_bf16(in->b16[set][k][(j / 2) * 32 + n * 2 + j % 2]);
                }
                in->ref8[set][m * 16 + n] = acc;
                in->ref16[set][m * 16 + n] = facc;
            }
        }
    }
}

static int all_zero(const void *p) {
    const uint32_t *u = p;
    for (int i = 0; i < 256; i++)
        if (u[i]) return 0;
    return 1;
}

static void mid_point(void) {
    if (g_yield) sched_yield();
    if (g_sleep_us) {
        struct timespec ts = { 0, g_sleep_us * 1000L };
        nanosleep(&ts, NULL);
    }
}

static inline void compute_int8(const inputs *in, int set, int32_t *c) {
    _tile_zero(0);
    for (int k = 0; k < g_k; k++) {
        if (k == g_k / 2) mid_point();
        _tile_loadd(1, in->a8[set][k], 64);
        _tile_loadd(2, in->b8[set][k], 64);
        _tile_dpbssd(0, 1, 2);
    }
    _tile_stored(0, c, 64);
}

static inline void compute_bf16(const inputs *in, int set, float *c) {
    _tile_zero(0);
    for (int k = 0; k < g_k; k++) {
        if (k == g_k / 2) mid_point();
        _tile_loadd(1, in->a16[set][k], 64);
        _tile_loadd(2, in->b16[set][k], 64);
        _tile_dpbf16ps(0, 1, 2);
    }
    _tile_stored(0, c, 64);
}

static void pin_self(int cpu) {
    cpu_set_t cs;
    CPU_ZERO(&cs);
    CPU_SET(cpu, &cs);
    if (sched_setaffinity(0, sizeof cs, &cs)) {
        perror("sched_setaffinity");
        exit(2);
    }
}

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void *busy_thread(result *r) {
    __m512 a = _mm512_set1_ps(1.0001f), b = _mm512_set1_ps(0.9999f), x0 = a, x1 = b, x2 = a, x3 = b;
    long it = 0;
    while (!g_stop && (!g_iters || it < g_iters)) {
        for (int i = 0; i < 4096; i++) {
            x0 = _mm512_fmadd_ps(x0, a, b); x1 = _mm512_fmadd_ps(x1, b, a);
            x2 = _mm512_fmadd_ps(x2, a, b); x3 = _mm512_fmadd_ps(x3, b, a);
        }
        it++;
    }
    r->busy_sink = _mm512_reduce_add_ps(_mm512_add_ps(_mm512_add_ps(x0, x1), _mm512_add_ps(x2, x3)));
    return NULL;
}

static void *tick_thread(result *r) {
    const long us = g_sleep_us ? g_sleep_us : 100;
    long it = 0;
    while (!g_stop && (!g_iters || it < g_iters)) {
        struct timespec ts = { 0, us * 1000L };
        nanosleep(&ts, NULL);
        const uint64_t t0 = __rdtsc();
        while (__rdtsc() - t0 < 2400) _mm_pause();
        it++;
    }
    r->n8 = it;
    return NULL;
}

static void *run(void *arg) {
    result *r = arg;
    if (g_pin >= 0) pin_self((int)((g_pin + (g_same ? 0 : r->id)) % sysconf(_SC_NPROCESSORS_ONLN)));
    if (g_mode == M_BUSY) return busy_thread(r);
    if (g_mode == M_TICK) return tick_thread(r);

    inputs *in = aligned_alloc(64, sizeof(inputs));
    make_inputs(in, ((uint64_t)getpid() << 20) ^ (uint64_t)(r->id + 1) * 0x9e3779b97f4a7c15ull);
    int32_t c8[256] __attribute__((aligned(64)));
    float c16[256] __attribute__((aligned(64)));
    const uint64_t guard = g_guard ? g_guard + 128u * (uint64_t)g_k : 0;
    load_config();
    for (long it = 0; !g_stop && (!g_iters || it < g_iters); it++) {
        const int set = (int)(it % NSETS);
        const int mode = g_mode == M_BOTH ? ((it & 1) ? M_BF16 : M_INT8) : g_mode;
        if (!config_intact()) {
            r->cfg_lost++;
            load_config();
        }
        uint64_t span;
        for (;;) {
            const uint64_t t0 = __rdtsc();
            if (mode == M_INT8) compute_int8(in, set, c8); else compute_bf16(in, set, c16);
            unsigned aux;
            span = __rdtscp(&aux) - t0;
            if (guard && span > guard && !g_stop) {
                r->redo++;
                continue;
            }
            break;
        }
        if (guard && span > guard) break;  // stopped while redoing
        int bad;
        if (mode == M_INT8) {
            r->n8++;
            bad = memcmp(c8, in->ref8[set], sizeof c8) != 0;
            if (bad) { r->bad8++; r->zero += all_zero(c8); }
        } else {
            r->n16++;
            bad = memcmp(c16, in->ref16[set], sizeof c16) != 0;
            if (bad) { r->bad16++; r->zero += all_zero(c16); }
        }
        if (bad) {
            if (g_verbose) fprintf(stderr, "wrong %s result, span %llu cycles\n", mode == M_INT8 ? "int8" : "bf16",
                                   (unsigned long long)span);
            if (!r->bad_span_min || span < r->bad_span_min) r->bad_span_min = span;
            if (span > r->bad_span_max) r->bad_span_max = span;
        } else if (span > 4096u + 128u * (uint64_t)g_k) {
            r->slow_ok++;
        }
    }
    _tile_release();
    free(in);
    return NULL;
}

static void on_sigill(int sig) {
    (void)sig;
    static const char msg[] = "amx_selftest: SIGILL in a tile instruction (tile configuration lost)\n";
    if (write(2, msg, sizeof msg - 1)) {}
    _exit(3);
}

static int parse_mode(const char *s) {
    if (!strcmp(s, "int8")) return M_INT8;
    if (!strcmp(s, "bf16")) return M_BF16;
    if (!strcmp(s, "both")) return M_BOTH;
    if (!strcmp(s, "busy")) return M_BUSY;
    if (!strcmp(s, "tick")) return M_TICK;
    fprintf(stderr, "unknown mode %s\n", s);
    exit(2);
}

int main(int argc, char **argv) {
    int opt;
    while ((opt = getopt(argc, argv, "t:s:n:m:k:p:g:Syw:qv")) != -1) {
        switch (opt) {
            case 't': g_threads = atoi(optarg); break;
            case 's': g_secs = atof(optarg); break;
            case 'n': g_iters = atol(optarg); break;
            case 'm': g_mode = parse_mode(optarg); break;
            case 'k': g_k = atoi(optarg); break;
            case 'p': g_pin = atoi(optarg); break;
            case 'g': g_guard = strtoull(optarg, NULL, 0); break;
            case 'S': g_same = 1; break;
            case 'y': g_yield = 1; break;
            case 'w': g_sleep_us = atoi(optarg); break;
            case 'q': g_quiet = 1; break;
            case 'v': g_verbose = 1; break;
            default:
                fprintf(stderr, "usage: %s [-t threads] [-s secs | -n iters] [-m int8|bf16|both|busy|tick] [-k ksteps] "
                                "[-p cpu [-S]] [-g guard_cycles] [-y] [-w usec] [-q] [-v]\n", argv[0]);
                return 2;
        }
    }
    if (g_threads < 1 || g_threads > 1024 || g_k < 1 || g_k > KMAX) {
        fprintf(stderr, "bad -t or -k\n");
        return 2;
    }
    if (g_mode != M_BUSY && g_mode != M_TICK) {
        if (!__builtin_cpu_supports("amx-tile") || !__builtin_cpu_supports("amx-int8") ||
            !__builtin_cpu_supports("amx-bf16")) {
            fprintf(stderr, "amx_selftest: CPU does not report AMX-TILE/INT8/BF16\n");
            return 2;
        }
        // must precede thread creation: the larger signal frame / xstate buffer is sized per process
        if (syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)) {
            fprintf(stderr, "amx_selftest: ARCH_REQ_XCOMP_PERM failed: %s\n", strerror(errno));
            return 2;
        }
        unsigned long mask = 0;
        if (syscall(SYS_arch_prctl, ARCH_GET_XCOMP_PERM, &mask) || !(mask & (1ul << XFEATURE_XTILEDATA))) {
            fprintf(stderr, "amx_selftest: XTILEDATA not in the permitted xfeatures (0x%lx)\n", mask);
            return 2;
        }
        signal(SIGILL, on_sigill);
    }

    result *res = calloc((size_t)g_threads, sizeof(result));
    pthread_t *th = calloc((size_t)g_threads, sizeof(pthread_t));
    const double t0 = now();
    for (int i = 0; i < g_threads; i++) {
        res[i].id = i;
        if (pthread_create(&th[i], NULL, run, &res[i])) {
            perror("pthread_create");
            return 2;
        }
    }
    if (!g_iters) {
        struct timespec ts = { (time_t)g_secs, (long)((g_secs - (time_t)g_secs) * 1e9) };
        nanosleep(&ts, NULL);
        g_stop = 1;
    }
    result tot = {0};
    for (int i = 0; i < g_threads; i++) {
        pthread_join(th[i], NULL);
        tot.n8 += res[i].n8; tot.bad8 += res[i].bad8; tot.n16 += res[i].n16; tot.bad16 += res[i].bad16;
        tot.zero += res[i].zero; tot.redo += res[i].redo; tot.cfg_lost += res[i].cfg_lost;
        tot.slow_ok += res[i].slow_ok;
        if (res[i].bad_span_min && (!tot.bad_span_min || res[i].bad_span_min < tot.bad_span_min))
            tot.bad_span_min = res[i].bad_span_min;
        if (res[i].bad_span_max > tot.bad_span_max) tot.bad_span_max = res[i].bad_span_max;
        if (!g_quiet && g_mode != M_BUSY && g_mode != M_TICK)
            printf("thread %d: int8 %ld/%ld wrong, bf16 %ld/%ld wrong, %ld redone\n", i, res[i].bad8, res[i].n8,
                   res[i].bad16, res[i].n16, res[i].redo);
    }
    const double el = now() - t0;
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    const long n = tot.n8 + tot.n16, bad = tot.bad8 + tot.bad16;
    static const char *mname[] = {"", "int8", "bf16", "both", "busy", "tick"};
    printf("amx_selftest pid=%d mode=%s threads=%d k=%d pin=%d same=%d guard=%llu yield=%d sleep_us=%d secs=%.2f "
           "int8_bad=%ld int8_n=%ld bf16_bad=%ld bf16_n=%ld zero_tiles=%ld redo=%ld cfg_lost=%ld nvcsw=%ld nivcsw=%ld "
           "bad_span_min=%llu bad_span_max=%llu slow_ok=%ld rate=%.3g\n",
           getpid(), mname[g_mode], g_threads, g_k, g_pin, g_same, (unsigned long long)g_guard, g_yield, g_sleep_us, el, tot.bad8, tot.n8,
           tot.bad16, tot.n16, tot.zero, tot.redo, tot.cfg_lost, ru.ru_nvcsw, ru.ru_nivcsw,
           (unsigned long long)tot.bad_span_min, (unsigned long long)tot.bad_span_max, tot.slow_ok, n ? (double)bad / n : 0.0);
    return bad ? 1 : 0;
}
