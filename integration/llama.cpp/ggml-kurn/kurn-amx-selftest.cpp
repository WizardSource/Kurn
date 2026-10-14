// AMX permission and startup self-test for the KURN buffer type and kurn attention.
//
// Some KVM guests do not keep AMX tile data across a context switch: a tile computation that a deschedule lands
// in returns wrong results (tools/amx_selftest.c measures how often). kurn's AMX kernels guard against it by
// timing every short step and recomputing steps that spanned a preemption-length gap. The self-test detects
// the condition once at load, so the user is told, and chooses what happens.
//
// Two phases, each in threads of its own so the caller's xstate is untouched; about GGML_KURN_AMX_SELFTEST_MS
// (default 20) ms in all:
//   1. plain: int8 and bf16 tile matmuls compared bit for bit with a scalar reference (small-integer inputs,
//      so the bf16 path is exact). Wrong results here mean AMX itself is unusable: AMX is turned off.
//   2. switch: two threads on one CPU each yield in the middle of every computation, so every computation
//      spans a context switch (the tiles must survive the kernel's XSAVE/XRSTOR). If pinning is refused, 2 x
//      CPUs unpinned threads do the same. Wrong results here are what the guard is for:
//      GGML_KURN_AMX_SELFTEST_ACTION=guard (default) warns and keeps the AMX kernels with the guard forced on;
//      =off warns and runs the non-AMX kernels, as GGML_KURN_AMX=0 does.
// GGML_KURN_AMX_SELFTEST=0, or GGML_KURN_AMX=0 (no AMX use), skips the test. GGML_KURN_VERBOSE=1 also logs
// a pass.
#include "kurn-amx-selftest.h"

#if defined(__AMX_INT8__) && defined(__AMX_BF16__) && defined(__linux__)

#include "ggml-impl.h"

#include <immintrin.h>
#include <sched.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <thread>
#include <vector>

namespace {

constexpr int KS = 8;            // k-steps per computation
constexpr int SETS = 2;          // input sets, alternated so that a stale tile cannot look right
constexpr long MIN_SWITCHES = 16;  // fewer context switches in phase 2: inconclusive

int env_int(const char * name, int def) {
    const char * v = getenv(name);
    return v && *v ? atoi(v) : def;
}

struct alignas(64) inputs {
    int8_t a8[SETS][KS][16 * 64], b8[SETS][KS][16 * 64];        // A 16 x 64 s8; B 16 rows of 16 x 4 s8
    uint16_t a16[SETS][KS][16 * 32], b16[SETS][KS][16 * 32];    // A 16 x 32 bf16; B 16 pairs of 16 x 2 bf16
    int32_t r8[SETS][256];
    float r16[SETS][256];
};

uint16_t bf16_of(int v) {
    const float f = (float) v;
    uint32_t u;
    memcpy(&u, &f, 4);
    return (uint16_t) (u >> 16);
}

float f32_of(uint16_t h) {
    const uint32_t u = (uint32_t) h << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}

void make_inputs(inputs & in, uint64_t s) {
    auto rnd = [&s] { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; };
    for (int set = 0; set < SETS; set++) {
        for (int k = 0; k < KS; k++) {
            for (int i = 0; i < 1024; i++) {
                in.a8[set][k][i] = (int8_t) rnd();
                in.b8[set][k][i] = (int8_t) rnd();
            }
            for (int i = 0; i < 512; i++) {
                in.a16[set][k][i] = bf16_of((int) (rnd() % 9) - 4);
                in.b16[set][k][i] = bf16_of((int) (rnd() % 9) - 4);
            }
        }
        for (int m = 0; m < 16; m++) {
            for (int n = 0; n < 16; n++) {
                int32_t a = 0;
                float f = 0;
                for (int k = 0; k < KS; k++) {
                    for (int j = 0; j < 64; j++) a += in.a8[set][k][m * 64 + j] * in.b8[set][k][(j / 4) * 64 + n * 4 + j % 4];
                    for (int j = 0; j < 32; j++) f += f32_of(in.a16[set][k][m * 32 + j]) * f32_of(in.b16[set][k][(j / 2) * 32 + n * 2 + j % 2]);
                }
                in.r8[set][m * 16 + n] = a;
                in.r16[set][m * 16 + n] = f;
            }
        }
    }
}

void tile_config() {
    struct alignas(64) {
        uint8_t palette, start_row, reserved[14];
        uint16_t colsb[16];
        uint8_t rows[16];
    } c = {};
    c.palette = 1;
    for (int t = 0; t < 3; t++) { c.rows[t] = 16; c.colsb[t] = 64; }
    _tile_loadconfig(&c);
}

// computation `it`: C = sum of KS tile products, int8 for even it, bf16 for odd; true when bit-identical to the reference
bool compute(const inputs & in, long it, bool yield) {
    const int set = (int) (it / 2 % SETS);
    const bool bf = it & 1;
    alignas(64) int32_t c[256];
    _tile_zero(0);
    for (int k = 0; k < KS; k++) {
        if (yield && k == KS / 2) {
            sched_yield();
        }
        if (bf) {
            _tile_loadd(1, in.a16[set][k], 64);
            _tile_loadd(2, in.b16[set][k], 64);
            _tile_dpbf16ps(0, 1, 2);
        } else {
            _tile_loadd(1, in.a8[set][k], 64);
            _tile_loadd(2, in.b8[set][k], 64);
            _tile_dpbssd(0, 1, 2);
        }
    }
    _tile_stored(0, c, 64);
    return memcmp(c, bf ? (const void *) in.r16[set] : (const void *) in.r8[set], sizeof c) == 0;
}

struct counts {
    long n = 0, bad = 0, switches = 0;
};

// nth threads (on `cpu` when >= 0) run up to max_n computations each, until `ms` milliseconds have passed
counts phase(int nth, int cpu, bool yield, long max_n, int ms) {
    std::vector<counts> c((size_t) nth);
    std::vector<std::thread> th;
    const auto end = std::chrono::steady_clock::now() + std::chrono::milliseconds(ms);
    for (int t = 0; t < nth; t++) {
        th.emplace_back([&, t] {
            if (cpu >= 0) {
                cpu_set_t cs;
                CPU_ZERO(&cs);
                CPU_SET(cpu, &cs);
                sched_setaffinity(0, sizeof cs, &cs);
            }
            std::unique_ptr<inputs> in(new inputs);
            make_inputs(*in, 0x9e3779b97f4a7c15ull * (uint64_t) (t + 1) ^ (uint64_t) getpid());
            rusage r0, r1;
            getrusage(RUSAGE_THREAD, &r0);
            tile_config();
            long it = 0;
            for (; it < max_n && ((it & 63) || std::chrono::steady_clock::now() < end); it++) {
                c[t].bad += !compute(*in, it, yield);
            }
            _tile_release();
            getrusage(RUSAGE_THREAD, &r1);
            c[t].n = it;
            c[t].switches = (r1.ru_nvcsw - r0.ru_nvcsw) + (r1.ru_nivcsw - r0.ru_nivcsw);
        });
    }
    counts s;
    for (int t = 0; t < nth; t++) {
        th[t].join();
        s.n += c[t].n;
        s.bad += c[t].bad;
        s.switches += c[t].switches;
    }
    return s;
}

kurn_amx_state run_check() {
    // ARCH_REQ_XCOMP_PERM for XTILEDATA; fails on kernels or VMs without AMX state support
    if (syscall(SYS_arch_prctl, 0x1023, 18) != 0) {
        return KURN_AMX_UNAVAILABLE;
    }
    if (env_int("GGML_KURN_AMX", 1) == 0 || env_int("GGML_KURN_AMX_SELFTEST", 1) == 0) {
        return KURN_AMX_OK;
    }
    const int ms = std::min(10000, std::max(2, env_int("GGML_KURN_AMX_SELFTEST_MS", 20)));
    const char * act = getenv("GGML_KURN_AMX_SELFTEST_ACTION");
    const bool off = act && strcmp(act, "off") == 0;
    const bool verbose = env_int("GGML_KURN_VERBOSE", 0) != 0;

    const counts plain = phase(1, -1, false, 4000, std::max(1, ms / 4));
    cpu_set_t cs;
    int cpu = -1;
    if (sched_getaffinity(0, sizeof cs, &cs) == 0) {
        for (int i = 0; i < CPU_SETSIZE && cpu < 0; i++) {
            if (CPU_ISSET(i, &cs)) cpu = i;
        }
    }
    counts sw = phase(2, cpu, true, 2000, ms);
    if (sw.switches < MIN_SWITCHES) {
        const int ncpu = std::max(1, (int) std::thread::hardware_concurrency());
        sw = phase(std::min(64, 2 * ncpu), -1, true, 2000, ms);
    }

    if (plain.bad) {
        GGML_LOG_WARN("kurn: AMX self-test: %ld of %ld tile matmuls returned wrong results without any context switch; "
                      "AMX kernels off\n", plain.bad, plain.n);
        return KURN_AMX_UNAVAILABLE;
    }
    if (sw.bad) {
        GGML_LOG_WARN("kurn: AMX self-test: %ld of %ld tile matmuls that spanned a context switch returned wrong results "
                      "(this machine does not keep AMX tile data across context switches). %s\n", sw.bad, sw.n,
                      off ? "Using the non-AMX kernels (GGML_KURN_AMX_SELFTEST_ACTION=off)."
                          : "AMX kernels stay on with the preemption guard forced on; GGML_KURN_AMX_SELFTEST_ACTION=off "
                            "uses the non-AMX kernels instead.");
        return off ? KURN_AMX_UNAVAILABLE : KURN_AMX_GUARDED;
    }
    if (verbose) {
        GGML_LOG_INFO("kurn: AMX self-test %s: %ld plain tile matmuls, %ld across %ld context switches, all correct\n",
                      sw.switches < MIN_SWITCHES ? "inconclusive (too few context switches)" : "passed", plain.n, sw.n,
                      sw.switches);
    }
    return KURN_AMX_OK;
}

}  // namespace

kurn_amx_state kurn_amx_check() {
    static const kurn_amx_state s = run_check();
    return s;
}

#else

#if defined(__linux__)
#include <sys/syscall.h>
#include <unistd.h>
#endif

kurn_amx_state kurn_amx_check() {
#if defined(__linux__) && defined(__x86_64__)
    // no AMX kernels compiled in: only the permission (ggml's AMX buffer, which kurn's per-tensor fallback mirrors)
    static const kurn_amx_state s = syscall(SYS_arch_prctl, 0x1023, 18) == 0 ? KURN_AMX_OK : KURN_AMX_UNAVAILABLE;
    return s;
#else
    return KURN_AMX_UNAVAILABLE;
#endif
}

#endif
