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
//
// The guard cannot see a direct switch between two AMX threads on one vCPU (one blocks or yields, the other runs
// briefly): it can complete in ~2.3 us, under any useful threshold. So after a failed test kurn_amx_thread_cap()
// keeps compute threads within this process's vCPUs: the affinity set, a cgroup v2 cpu.max quota, and a fair share
// against other kurn processes using AMX on the same CPUs (each registers its requested threads and CPUs in
// /dev/shm/kurn-amx/<pid>). Linux does not report other processes' AMX use, so running threads of other processes
// on these CPUs that used AVX-512 within the last second are reported as possible AMX users, without capping.
#include "kurn-amx-selftest.h"

#if defined(__AMX_INT8__) && defined(__AMX_BF16__) && defined(__linux__)

#include "ggml-impl.h"

#include <dirent.h>
#include <immintrin.h>
#include <sched.h>
#include <signal.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
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

int affinity_cpus(cpu_set_t & cs) {
    CPU_ZERO(&cs);
    if (sched_getaffinity(0, sizeof cs, &cs) != 0) {
        const int n = std::max(1, (int) std::thread::hardware_concurrency());
        for (int i = 0; i < n && i < CPU_SETSIZE; i++) CPU_SET(i, &cs);
    }
    return CPU_COUNT(&cs);
}

// cgroup v2 cpu.max of this process and its ancestors, as whole CPUs (rounded up); 0 without a quota
int cgroup_quota_cpus() {
    FILE * f = fopen("/proc/self/cgroup", "r");
    if (!f) return 0;
    char line[4096];
    std::string path;
    while (fgets(line, sizeof line, f)) {
        if (strncmp(line, "0::", 3) == 0) {
            path = line + 3;
            while (!path.empty() && (path.back() == '\n' || path.back() == '/')) path.pop_back();
        }
    }
    fclose(f);
    int best = 0;
    for (;;) {
        FILE * q = fopen(("/sys/fs/cgroup" + path + "/cpu.max").c_str(), "r");
        if (q) {
            char quota[32] = {0};
            long long period = 0;
            if (fscanf(q, "%31s %lld", quota, &period) == 2 && strcmp(quota, "max") != 0 && period > 0) {
                const int c = (int) std::max(1LL, (atoll(quota) + period - 1) / period);
                best = best ? std::min(best, c) : c;
            }
            fclose(q);
        }
        if (path.empty()) break;
        path.resize(path.rfind('/') == std::string::npos ? 0 : path.rfind('/'));
    }
    return best;
}

std::string registry_dir() {
    static const std::string d = [] {
        for (const char * base : {"/dev/shm", "/tmp"}) {
            const std::string p = std::string(base) + "/kurn-amx";
            const mode_t old = umask(0);
            mkdir(p.c_str(), 01777);
            umask(old);
            if (access(p.c_str(), W_OK) == 0) return p;
        }
        return std::string();
    }();
    return d;
}

void registry_remove() {
    const std::string d = registry_dir();
    if (!d.empty()) unlink((d + "/" + std::to_string(getpid())).c_str());
}

void registry_write(int n, const cpu_set_t & cs) {
    const std::string d = registry_dir();
    if (d.empty()) return;
    static bool at_exit = false;
    if (!at_exit) {
        atexit(registry_remove);
        at_exit = true;
    }
    std::string s = "threads=" + std::to_string(n) + " cpus=";
    for (int i = 0; i < CPU_SETSIZE; i++) {
        if (CPU_ISSET(i, &cs)) s += std::to_string(i) + ",";
    }
    s += "\n";
    const std::string tmp = d + "/." + std::to_string(getpid()), dst = d + "/" + std::to_string(getpid());
    FILE * f = fopen(tmp.c_str(), "w");
    if (!f) return;
    fputs(s.c_str(), f);
    fclose(f);
    rename(tmp.c_str(), dst.c_str());
}

// other live kurn processes on these CPUs: count, and sum of min(their threads, CPUs shared with us)
void registry_others(const cpu_set_t & ours, int & procs, int & threads) {
    procs = threads = 0;
    const std::string d = registry_dir();
    DIR * dir = d.empty() ? nullptr : opendir(d.c_str());
    if (!dir) return;
    for (struct dirent * e; (e = readdir(dir));) {
        const long pid = strtol(e->d_name, nullptr, 10);
        if (pid <= 0 || pid == (long) getpid() || strspn(e->d_name, "0123456789") != strlen(e->d_name)) continue;
        if (kill((pid_t) pid, 0) != 0 && errno == ESRCH) {
            unlink((d + "/" + e->d_name).c_str());
            continue;
        }
        FILE * f = fopen((d + "/" + e->d_name).c_str(), "r");
        if (!f) continue;
        char buf[8192] = {0};
        const size_t len = fread(buf, 1, sizeof buf - 1, f);
        fclose(f);
        buf[len] = 0;
        const char * t = strstr(buf, "threads="), * c = strstr(buf, "cpus=");
        if (!t || !c) continue;
        int shared = 0;
        for (const char * p = c + 5; *p >= '0' && *p <= '9';) {
            char * end;
            const long cpu = strtol(p, &end, 10);
            if (cpu >= 0 && cpu < CPU_SETSIZE && CPU_ISSET(cpu, &ours)) shared++;
            p = *end == ',' ? end + 1 : end;
        }
        if (shared) {
            procs++;
            threads += std::min(atoi(t + 8), shared);
        }
    }
    closedir(dir);
}

// running threads of other processes (not registered kurn ones) allowed on our CPUs that used AVX-512 within the
// last second
int vector_threads_on(const cpu_set_t & ours) {
    int n = 0;
    DIR * proc = opendir("/proc");
    if (!proc) return 0;
    const long self = (long) getpid();
    const std::string reg = registry_dir();
    for (struct dirent * e; (e = readdir(proc));) {
        const long pid = strtol(e->d_name, nullptr, 10);
        if (pid <= 0 || pid == self || (!reg.empty() && access((reg + "/" + e->d_name).c_str(), F_OK) == 0)) continue;
        const std::string tdir = std::string("/proc/") + e->d_name + "/task";
        DIR * tasks = opendir(tdir.c_str());
        if (!tasks) continue;
        for (struct dirent * t; (t = readdir(tasks));) {
            const long tid = strtol(t->d_name, nullptr, 10);
            if (tid <= 0) continue;
            const std::string base = tdir + "/" + t->d_name;
            char buf[512] = {0};
            FILE * f = fopen((base + "/stat").c_str(), "r");
            if (!f) continue;
            const size_t len = fread(buf, 1, sizeof buf - 1, f);
            fclose(f);
            buf[len] = 0;
            const char * rp = strrchr(buf, ')');
            if (!rp || rp[1] != ' ' || rp[2] != 'R') continue;
            f = fopen((base + "/arch_status").c_str(), "r");
            long ms = -1;
            if (f) {
                if (fscanf(f, "AVX512_elapsed_ms: %ld", &ms) != 1) ms = -1;
                fclose(f);
            }
            if (ms < 0 || ms > 1000) continue;
            cpu_set_t cs;
            if (sched_getaffinity((pid_t) tid, sizeof cs, &cs) != 0) continue;
            CPU_AND(&cs, &cs, &ours);
            n += CPU_COUNT(&cs) > 0;
        }
        closedir(tasks);
    }
    closedir(proc);
    return n;
}

}  // namespace

kurn_amx_state kurn_amx_check() {
    static const kurn_amx_state s = run_check();
    return s;
}

int kurn_amx_thread_cap(int n) {
    if (n < 1 || kurn_amx_check() != KURN_AMX_GUARDED) {
        return n;
    }
    static std::mutex mu;
    static int last_n = 0, last_cap = 0, warned_n = 0, warned_cap = 0, scanned_n = 0;
    static std::chrono::steady_clock::time_point last;
    std::lock_guard<std::mutex> lock(mu);
    const auto now = std::chrono::steady_clock::now();
    if (n == last_n && now - last < std::chrono::seconds(2)) {
        return last_cap;
    }
    cpu_set_t cs;
    const int aff = affinity_cpus(cs), quota = cgroup_quota_cpus();
    const int vcpus = std::max(1, quota ? std::min(aff, quota) : aff);
    registry_write(n, cs);
    int procs, others;
    registry_others(cs, procs, others);
    const int cap = n + others > vcpus ? std::max(1, (int) ((int64_t) vcpus * n / (n + others))) : n;
    const char * pol = getenv("GGML_KURN_AMX_OVERSUB");
    const bool warn_only = pol && strcmp(pol, "warn") == 0;
    if (cap < n && (n != warned_n || cap != warned_cap)) {
        char who[160] = "";
        if (procs) {
            snprintf(who, sizeof who, ", shared with %d other kurn process%s using AMX (%d thread%s)", procs,
                     procs > 1 ? "es" : "", others, others > 1 ? "s" : "");
        }
        GGML_LOG_WARN("kurn: AMX: %d threads on %d available vCPUs (affinity %d%s%s%s). After the failed AMX self-test, "
                      "two AMX threads switching on one vCPU can lose tile data faster than the preemption guard detects. "
                      "%s\n", n, vcpus, aff, quota ? ", cgroup quota " : "", quota ? std::to_string(quota).c_str() : "",
                      who,
                      warn_only ? ("Keeping " + std::to_string(n) + " threads (GGML_KURN_AMX_OVERSUB=warn); run with -t " +
                                   std::to_string(cap) + ".").c_str()
                                : ("Using " + std::to_string(cap) + " threads; run with -t " + std::to_string(cap) +
                                   " (GGML_KURN_AMX_OVERSUB=warn keeps " + std::to_string(n) + ").").c_str());
        warned_n = n;
        warned_cap = cap;
    }
    if (n != scanned_n) {
        scanned_n = n;
        const int vec = vector_threads_on(cs);
        if (vec && std::min(n, cap) + vec > vcpus) {
            GGML_LOG_WARN("kurn: AMX: %d running thread%s of other processes on these %d vCPUs used AVX-512 in the last "
                          "second. If they run AMX too, their switches with kurn's threads can bypass the preemption guard: "
                          "give kurn and them separate CPUs, or use GGML_KURN_AMX_SELFTEST_ACTION=off.\n", vec,
                          vec > 1 ? "s" : "", vcpus);
        }
    }
    last_n = n;
    last_cap = warn_only ? n : cap;
    last = now;
    return last_cap;
}

#else

#if defined(__linux__)
#include <sys/syscall.h>
#include <unistd.h>
#endif

int kurn_amx_thread_cap(int n_threads) {
    return n_threads;
}

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
