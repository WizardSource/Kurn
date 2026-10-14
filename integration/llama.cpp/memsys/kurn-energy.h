#pragma once
// kurn energy meter for llama-bench --energy (kurn/integration/llama.cpp/memsys).
//
// CPU: RAPL package (+ DRAM where the platform has it) energy, from /sys/class/powercap/intel-rapl* (energy_uj,
// root-only on kernels since 5.10 unless chmod'ed) or, if that is unreadable, from perf's `power` PMU
// (energy-pkg / energy-ram, system-wide: perf_event_paranoid <= 0 or CAP_PERFMON). AMD Zen exposes the same
// `power/energy-pkg` event. GPU: NVML's total energy counter (Volta and later) via dlopen("libnvidia-ml.so.1").
// Cloud VMs usually have no RAPL (no powercap, or MSRs that read 0): a counter that does not advance during a
// test is reported as unavailable, never as 0 J.
// Joules are whole-package (all cores, uncore, idle siblings), not per process: run on an otherwise idle host.

#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#if defined(__linux__)
#include <dlfcn.h>
#include <linux/perf_event.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

struct kurn_energy_meter {
    struct rapl_file {
        std::string path, name;
        double max_uj = 0;
        double last_uj = 0, acc_j = 0;
    };
    struct perf_ctr {
        int fd = -1;
        double scale = 0;  // joules per count
        std::string name;
        uint64_t last = 0;
        double acc_j = 0;
    };

    std::vector<rapl_file> rapl;
    std::vector<perf_ctr> perf;
    std::string cpu_src = "none", cpu_why, gpu_src = "none", gpu_why;

    // NVML
    void * nvml = nullptr;
    int (*nvmlDeviceGetTotalEnergyConsumption_)(void *, unsigned long long *) = nullptr;
    std::vector<void *> gpus;

    static bool read_num(const std::string & p, double & v) {
        FILE * f = fopen(p.c_str(), "r");
        if (!f) {
            return false;
        }
        const bool ok = fscanf(f, "%lf", &v) == 1;
        fclose(f);
        return ok;
    }

    static std::string read_str(const std::string & p) {
        char buf[256] = { 0 };
        FILE * f = fopen(p.c_str(), "r");
        if (!f) {
            return "";
        }
        if (!fgets(buf, sizeof buf, f)) {
            buf[0] = 0;
        }
        fclose(f);
        std::string s(buf);
        while (!s.empty() && (s.back() == '\n' || s.back() == ' ')) {
            s.pop_back();
        }
        return s;
    }

    void init() {
#if defined(__linux__)
        init_cpu();
        init_gpu();
#else
        cpu_why = "not Linux";
        gpu_why = "not Linux";
#endif
    }

#if defined(__linux__)
    void init_cpu() {
        // powercap: top-level package zones and their dram sub-zones (intel-rapl:N, intel-rapl:N:M)
        bool denied = false, any = false;
        for (int n = 0; n < 64; n++) {
            for (int m = -1; m < 8; m++) {
                char z[96];
                if (m < 0) snprintf(z, sizeof z, "/sys/class/powercap/intel-rapl:%d", n);
                else snprintf(z, sizeof z, "/sys/class/powercap/intel-rapl:%d:%d", n, m);
                const std::string name = read_str(std::string(z) + "/name");
                if (name.empty()) {
                    continue;
                }
                any = true;
                if (m >= 0 && name != "dram") {  // core / uncore are inside the package
                    continue;
                }
                rapl_file r;
                r.path = std::string(z) + "/energy_uj";
                r.name = name;
                read_num(std::string(z) + "/max_energy_range_uj", r.max_uj);
                if (!read_num(r.path, r.last_uj)) {
                    denied = denied || errno == EACCES;
                    continue;
                }
                rapl.push_back(r);
            }
        }
        if (!rapl.empty()) {
            cpu_src = "rapl-powercap";
            for (auto & r : rapl) cpu_src += ":" + r.name;
            return;
        }
        // perf power PMU
        double type = -1;
        if (read_num("/sys/bus/event_source/devices/power/type", type)) {
            const char * evs[] = { "energy-pkg", "energy-ram", "energy-psys" };
            std::vector<int> pkg_cpus = package_cpus();
            for (const char * ev : evs) {
                if (!strcmp(ev, "energy-psys") && !perf.empty()) {
                    break;  // psys only when there is nothing better (it covers the whole platform)
                }
                const std::string base = std::string("/sys/bus/event_source/devices/power/events/") + ev;
                const std::string cfg = read_str(base);
                double scale = 0;
                if (cfg.empty() || !read_num(base + ".scale", scale)) {
                    continue;
                }
                unsigned long long config = 0;
                if (sscanf(cfg.c_str(), "event=0x%llx", &config) != 1) {
                    continue;
                }
                for (int cpu : pkg_cpus) {
                    perf_event_attr a;
                    memset(&a, 0, sizeof a);
                    a.type = (uint32_t) type;
                    a.size = sizeof a;
                    a.config = config;
                    const int fd = (int) syscall(SYS_perf_event_open, &a, -1, cpu, -1, 0);
                    if (fd < 0) {
                        denied = denied || errno == EACCES || errno == EPERM;
                        continue;
                    }
                    perf_ctr c;
                    c.fd = fd;
                    c.scale = scale;
                    c.name = ev;
                    if (read(fd, &c.last, sizeof c.last) != sizeof c.last) {
                        close(fd);
                        continue;
                    }
                    perf.push_back(c);
                }
                if (!strcmp(ev, "energy-ram") && perf.empty()) {
                    break;
                }
            }
            if (!perf.empty()) {
                cpu_src = "rapl-perf";
                std::string last;
                for (auto & c : perf) {
                    if (c.name != last) cpu_src += ":" + c.name;
                    last = c.name;
                }
                return;
            }
        }
        if (denied) {
            cpu_why = std::string(any ? "powercap RAPL" : "perf power PMU") +
                      " present but not readable by this user (run as root, sudo chmod a+r "
                      "/sys/class/powercap/intel-rapl:*/energy_uj, or sysctl kernel.perf_event_paranoid=0)";
            if (hypervisor()) {
                cpu_why += "; this is a VM, whose RAPL counters usually read 0";
            }
        } else {
            cpu_why = any ? "no RAPL package domain"
                          : "no RAPL interface (no /sys/class/powercap/intel-rapl*, no perf power PMU): typical of cloud VMs";
        }
    }

    static bool hypervisor() {
        FILE * f = fopen("/proc/cpuinfo", "r");
        if (!f) {
            return false;
        }
        char line[4096];
        bool hv = false;
        while (!hv && fgets(line, sizeof line, f)) {
            hv = !strncmp(line, "flags", 5) && strstr(line, " hypervisor");
        }
        fclose(f);
        return hv;
    }

    static std::vector<int> package_cpus() {  // first online CPU of each physical package
        std::vector<int> cpus, seen;
        for (int c = 0; c < 4096; c++) {
            double pk;
            char p[128];
            snprintf(p, sizeof p, "/sys/devices/system/cpu/cpu%d/topology/physical_package_id", c);
            if (!read_num(p, pk)) {
                if (c > 0 && access(("/sys/devices/system/cpu/cpu" + std::to_string(c)).c_str(), F_OK) != 0) break;
                continue;
            }
            bool have = false;
            for (int s : seen) have = have || s == (int) pk;
            if (!have) {
                seen.push_back((int) pk);
                cpus.push_back(c);
            }
        }
        if (cpus.empty()) cpus.push_back(0);
        return cpus;
    }

    void init_gpu() {
        nvml = dlopen("libnvidia-ml.so.1", RTLD_LAZY | RTLD_LOCAL);
        if (!nvml) {
            gpu_why = "no NVML (libnvidia-ml.so.1)";
            return;
        }
        auto init = (int (*)()) dlsym(nvml, "nvmlInit_v2");
        auto count = (int (*)(unsigned *)) dlsym(nvml, "nvmlDeviceGetCount_v2");
        auto handle = (int (*)(unsigned, void **)) dlsym(nvml, "nvmlDeviceGetHandleByIndex_v2");
        nvmlDeviceGetTotalEnergyConsumption_ =
            (int (*)(void *, unsigned long long *)) dlsym(nvml, "nvmlDeviceGetTotalEnergyConsumption");
        unsigned n = 0;
        if (!init || !count || !handle || !nvmlDeviceGetTotalEnergyConsumption_ || init() != 0 || count(&n) != 0 || n == 0) {
            gpu_why = "NVML present, no usable GPU";
            return;
        }
        for (unsigned i = 0; i < n; i++) {
            void * h = nullptr;
            unsigned long long mj;
            if (handle(i, &h) == 0 && nvmlDeviceGetTotalEnergyConsumption_(h, &mj) == 0) {
                gpus.push_back(h);
            }
        }
        if (gpus.empty()) {
            gpu_why = "GPU without an energy counter (pre-Volta?)";
            return;
        }
        gpu_src = "nvml:" + std::to_string(gpus.size()) + " GPU(s)";
    }
#endif

    bool cpu_ok() const { return !rapl.empty() || !perf.empty(); }
    bool gpu_ok() const { return !gpus.empty(); }

    // cumulative joules since init, wraparound-safe if read at least once per counter range
    double cpu_joules() {
#if defined(__linux__)
        double total = 0;
        for (auto & r : rapl) {
            double v;
            if (read_num(r.path, v)) {
                double d = v - r.last_uj;
                if (d < 0) d += r.max_uj > 0 ? r.max_uj : 0;
                r.acc_j += d * 1e-6;
                r.last_uj = v;
            }
            total += r.acc_j;
        }
        for (auto & c : perf) {
            uint64_t v;
            if (read(c.fd, &v, sizeof v) == sizeof v) {
                c.acc_j += (double) (v - c.last) * c.scale;
                c.last = v;
            }
            total += c.acc_j;
        }
        return total;
#else
        return 0;
#endif
    }

    double gpu_joules() {
        double total = 0;
        for (void * h : gpus) {
            unsigned long long mj = 0;
            if (nvmlDeviceGetTotalEnergyConsumption_(h, &mj) == 0) {
                total += mj * 1e-3;
            }
        }
        return total;
    }

    std::string describe() const {
        std::string s = "cpu " + (cpu_ok() ? cpu_src : "unavailable (" + cpu_why + ")");
        s += ", gpu " + (gpu_ok() ? gpu_src : "unavailable (" + gpu_why + ")");
        return s;
    }
};
