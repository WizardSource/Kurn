"""`kurn energy`: joules and tokens per joule from hardware counters, or a clean "unavailable".

    kurn energy probe [--json]                       which counters this host has (CPU RAPL, GPU NVML), checked to advance
    kurn energy run [--tokens N] [--json OUT] -- CMD ...
                                                     energy of one command (CPU package + DRAM, every GPU); with
                                                     --tokens: tokens per joule and joules per token
    kurn energy sample --out FILE [--interval S] [--pid PID]
                                                     background sampler: JSON lines {t, cpu_j, gpu_j, gpu_w} until killed
                                                     or PID exits (the GPU kit runs one for the whole run)
    kurn energy between FILE T0 T1                   joules and mean watts between two epoch times of a sample file

Sources, in order:
  CPU  /sys/class/powercap/intel-rapl:N (package) and intel-rapl:N:M named dram (energy_uj; root-only on kernels >= 5.10
       unless chmod'ed), else perf's `power` PMU (energy-pkg, energy-ram; system-wide, so perf_event_paranoid <= 0 or
       CAP_PERFMON). AMD Zen has the same power/energy-pkg event.
  GPU  NVML nvmlDeviceGetTotalEnergyConsumption (mJ, Volta+) via ctypes on libnvidia-ml.so.1 (what nvidia-smi uses).
Cloud VMs usually have no RAPL, or RAPL MSRs that read 0: a counter that does not advance is reported as unavailable
with the reason, never as 0 J. Package energy is whole-socket (all cores, uncore), not per process.

Bare metal, CPU: `sudo llama-bench -m MODEL -p 512 -n 128 --energy` (llama.cpp with kurn's memsys patch), or
`sudo kurn energy run --tokens 128 -- llama-cli ...`, or plain `sudo perf stat -a -e power/energy-pkg/,power/energy-ram/ -- CMD`.
"""

import ctypes
import ctypes.util
import json
import os
import struct
import subprocess
import sys
import time

POWERCAP = "/sys/class/powercap"
PERF_PMU = "/sys/bus/event_source/devices/power"


def _read(path):
    try:
        with open(path) as fh:
            return fh.read().strip()
    except OSError:
        return None


def _hypervisor():
    try:
        with open("/proc/cpuinfo") as fh:
            return any(ln.startswith("flags") and " hypervisor" in ln for ln in fh)
    except OSError:
        return False


class _Powercap:
    def __init__(self):
        self.zones, self.denied, self.present = [], False, False
        if not os.path.isdir(POWERCAP):
            return
        for z in sorted(os.listdir(POWERCAP)):
            if not z.startswith("intel-rapl:"):
                continue
            name = _read(os.path.join(POWERCAP, z, "name"))
            if name is None:
                continue
            self.present = True
            sub = z.count(":") == 2
            if sub and name != "dram":  # core / uncore are inside the package
                continue
            path = os.path.join(POWERCAP, z, "energy_uj")
            v = _read(path)
            if v is None:
                self.denied = self.denied or (os.path.exists(path) and not os.access(path, os.R_OK))
                continue
            rng = float(_read(os.path.join(POWERCAP, z, "max_energy_range_uj")) or 0)
            self.zones.append({"name": name, "path": path, "max": rng, "last": float(v), "acc": 0.0})

    def joules(self):
        tot = 0.0
        for z in self.zones:
            v = _read(z["path"])
            if v is not None:
                d = float(v) - z["last"]
                if d < 0:
                    d += z["max"]
                z["acc"] += d * 1e-6
                z["last"] = float(v)
            tot += z["acc"]
        return tot


class _PerfPower:
    """perf_event_open on the power PMU, system-wide, one fd per package and event."""

    NR = {"x86_64": 298, "aarch64": 241}

    def __init__(self):
        self.fds, self.denied, self.events = [], False, []
        typ = _read(os.path.join(PERF_PMU, "type"))
        nr = self.NR.get(os.uname().machine)
        if typ is None or nr is None:
            return
        libc = ctypes.CDLL(None, use_errno=True)
        cpus = self._package_cpus()
        for ev in ("energy-pkg", "energy-ram", "energy-psys"):
            if ev == "energy-psys" and self.fds:
                break  # the platform domain only when nothing better exists
            cfg = _read(os.path.join(PERF_PMU, "events", ev))
            scale = _read(os.path.join(PERF_PMU, "events", ev + ".scale"))
            if not cfg or not scale or not cfg.startswith("event="):
                continue
            config = int(cfg.split("=")[1], 16)
            attr = struct.pack("<IIQ", int(typ), 128, config) + bytes(112)
            buf = ctypes.create_string_buffer(attr, len(attr))
            for cpu in cpus:
                fd = libc.syscall(nr, buf, -1, cpu, -1, 0)
                if fd < 0:
                    self.denied = self.denied or ctypes.get_errno() in (1, 13)
                    continue
                v = struct.unpack("<Q", os.read(fd, 8))[0]
                self.fds.append({"fd": fd, "scale": float(scale), "last": v, "acc": 0.0, "event": ev})
            if self.fds and ev not in self.events and any(f["event"] == ev for f in self.fds):
                self.events.append(ev)

    @staticmethod
    def _package_cpus():
        seen, cpus = set(), []
        base = "/sys/devices/system/cpu"
        for d in sorted((d for d in os.listdir(base) if d[3:].isdigit() and d.startswith("cpu")), key=lambda d: int(d[3:])):
            pk = _read(os.path.join(base, d, "topology", "physical_package_id"))
            if pk is not None and pk not in seen:
                seen.add(pk)
                cpus.append(int(d[3:]))
        return cpus or [0]

    def joules(self):
        tot = 0.0
        for f in self.fds:
            v = struct.unpack("<Q", os.read(f["fd"], 8))[0]
            f["acc"] += (v - f["last"]) * f["scale"]
            f["last"] = v
            tot += f["acc"]
        return tot


class _Nvml:
    def __init__(self):
        self.devs, self.names, self.why, self.lib, self.smi = [], [], "", None, False
        name = ctypes.util.find_library("nvidia-ml") or "libnvidia-ml.so.1"
        try:
            self.lib = ctypes.CDLL(name)
        except OSError:
            self.lib = None
        if self.lib is None:
            self.why = "no NVML (libnvidia-ml.so.1)"
            return
        L = self.lib
        n = ctypes.c_uint(0)
        if L.nvmlInit_v2() != 0 or L.nvmlDeviceGetCount_v2(ctypes.byref(n)) != 0 or n.value == 0:
            self.why = "NVML present, no usable GPU"
            return
        for i in range(n.value):
            h = ctypes.c_void_p()
            mj = ctypes.c_ulonglong(0)
            if L.nvmlDeviceGetHandleByIndex_v2(i, ctypes.byref(h)) != 0:
                continue
            nm = ctypes.create_string_buffer(96)
            L.nvmlDeviceGetName(h, nm, 96)
            self.names.append(nm.value.decode(errors="replace"))
            if L.nvmlDeviceGetTotalEnergyConsumption(h, ctypes.byref(mj)) == 0:
                self.devs.append(h)
        if not self.devs:
            self.why = "GPU without an NVML energy counter (pre-Volta?)"

    def joules(self):
        out = []
        for h in self.devs:
            mj = ctypes.c_ulonglong(0)
            self.lib.nvmlDeviceGetTotalEnergyConsumption(h, ctypes.byref(mj))
            out.append(mj.value * 1e-3)
        return out

    def watts(self):
        out = []
        for h in self.devs:
            mw = ctypes.c_uint(0)
            self.lib.nvmlDeviceGetPowerUsage(h, ctypes.byref(mw))
            out.append(mw.value * 1e-3)
        return out


class Meter:
    """Cumulative CPU joules (RAPL) and per-GPU joules (NVML) since construction."""

    def __init__(self, cpu=True, gpu=True):
        self.cpu = self.cpu_src = None
        self.cpu_why = ""
        if cpu:
            pc = _Powercap()
            if pc.zones:
                self.cpu, self.cpu_src = pc, "rapl-powercap:" + ",".join(z["name"] for z in pc.zones)
            else:
                pp = _PerfPower()
                if pp.fds:
                    self.cpu, self.cpu_src = pp, "rapl-perf:" + ",".join(pp.events)
                elif pc.denied or pp.denied:
                    self.cpu_why = (("powercap RAPL" if pc.denied else "perf power PMU") + " present but not readable by this user "
                                    "(run as root, sudo chmod a+r /sys/class/powercap/intel-rapl:*/energy_uj, or "
                                    "sysctl kernel.perf_event_paranoid=0)") + ("; this is a VM, whose RAPL usually reads 0" if _hypervisor() else "")  # fmt: skip
                else:
                    self.cpu_why = "no RAPL interface (no /sys/class/powercap/intel-rapl*, no perf power PMU): typical of cloud VMs"
        self.gpu = _Nvml() if gpu else None
        if self.gpu is not None and not self.gpu.devs:
            self.gpu_why, self.gpu = self.gpu.why, None
        else:
            self.gpu_why = "" if gpu else "not requested"
        self.cpu_stuck = False  # set by check(): the counter exists but did not advance

    def cpu_joules(self):
        return self.cpu.joules() if self.cpu else float("nan")

    def gpu_joules(self):
        return self.gpu.joules() if self.gpu else []

    def check(self, secs=0.3):
        """Spin one core for `secs` and drop counters that did not move (virtualized RAPL reads 0)."""
        if self.cpu is None:
            return
        j0, t = self.cpu_joules(), time.perf_counter()
        x = 0
        while time.perf_counter() - t < secs:
            x += 1
        if not self.cpu_joules() > j0:
            self.cpu_stuck = True
            self.cpu_why = f"{self.cpu_src} counter did not advance over {secs:.1f} s of load (virtualized RAPL reads 0)"
            self.cpu = None

    def describe(self):
        return {
            "cpu": {"available": self.cpu is not None, "source": self.cpu_src if self.cpu else None, "reason": self.cpu_why or None},
            "gpu": {"available": self.gpu is not None, "source": f"nvml:{len(self.gpu.devs)} GPU(s)" if self.gpu else None,
                    "names": self.gpu.names if self.gpu else [], "reason": self.gpu_why or None},
        }  # fmt: skip


def _fmt(d):
    c, g = d["cpu"], d["gpu"]
    return (f"cpu: {c['source'] if c['available'] else 'unavailable (' + str(c['reason']) + ')'}\n"
            f"gpu: {g['source'] + ' ' + ', '.join(g['names']) if g['available'] else 'unavailable (' + str(g['reason']) + ')'}")  # fmt: skip


def cmd_probe(argv):
    m = Meter()
    m.check()
    d = m.describe()
    if "--json" in argv:
        print(json.dumps(d, indent=1))
    else:
        print(_fmt(d))
        if not d["cpu"]["available"]:
            print("bare metal: sudo llama-bench -m MODEL -p 512 -n 128 --energy   |   "
                  "sudo perf stat -a -e power/energy-pkg/,power/energy-ram/ -- CMD")
    return 0


def cmd_run(argv):
    if "--" not in argv:
        print("usage: kurn energy run [--tokens N] [--json OUT] -- CMD ...", file=sys.stderr)
        return 2
    i = argv.index("--")
    opts, cmd = argv[:i], argv[i + 1:]
    tokens = int(opts[opts.index("--tokens") + 1]) if "--tokens" in opts else None
    out = opts[opts.index("--json") + 1] if "--json" in opts else None
    m = Meter()
    m.check(0.2)
    c0, g0, t0 = m.cpu_joules(), m.gpu_joules(), time.time()
    rc = subprocess.call(cmd)
    c1, g1, t1 = m.cpu_joules(), m.gpu_joules(), time.time()
    cj = c1 - c0 if m.cpu else None
    gj = sum(b - a for a, b in zip(g0, g1)) if m.gpu else None
    if cj is not None and not cj > 0:
        cj = None
    tot = (cj or 0.0) + (gj or 0.0)
    res = {"cmd": cmd, "exit": rc, "seconds": t1 - t0, "cpu_joules": cj, "gpu_joules": gj, "tokens": tokens,
           "tokens_per_joule": tokens / tot if tokens and tot > 0 else None,
           "joules_per_token": tot / tokens if tokens and tot > 0 else None, **{"sources": m.describe()}}  # fmt: skip
    def j(x):
        return f"{x:.3f} J" if x is not None else "unavailable"
    print(f"kurn energy: {res['seconds']:.2f} s, cpu {j(cj)}, gpu {j(gj)}"
          + (f", {res['tokens_per_joule']:.3f} tok/J ({res['joules_per_token']:.4f} J/tok)" if res["tokens_per_joule"] else
             (", tokens per joule unavailable" if tokens else "")), file=sys.stderr)  # fmt: skip
    if cj is None:
        print(f"kurn energy: cpu {m.cpu_why or 'unavailable'}", file=sys.stderr)
    if out:
        with open(out, "w") as fh:
            json.dump(res, fh, indent=1)
    return rc


def cmd_sample(argv):
    out = argv[argv.index("--out") + 1] if "--out" in argv else None
    interval = float(argv[argv.index("--interval") + 1]) if "--interval" in argv else 0.2
    pid = int(argv[argv.index("--pid") + 1]) if "--pid" in argv else None
    if not out:
        print("usage: kurn energy sample --out FILE [--interval S] [--pid PID]", file=sys.stderr)
        return 2
    m = Meter()
    m.check(0.2)
    with open(out, "w") as fh:
        fh.write(json.dumps({"meta": m.describe(), "interval": interval}) + "\n")
        fh.flush()
        if m.cpu is None and m.gpu is None:
            return 0
        while True:
            row = {"t": time.time()}
            if m.cpu:
                row["cpu_j"] = m.cpu_joules()
            if m.gpu:
                row["gpu_j"] = m.gpu_joules()
                row["gpu_w"] = m.gpu.watts()
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            if pid is not None:
                try:
                    os.kill(pid, 0)
                except OSError:
                    return 0
            time.sleep(interval)


def load_samples(path):
    meta, rows = {}, []
    if not os.path.exists(path):
        return meta, rows
    with open(path) as fh:
        for ln in fh:
            if not ln.strip():
                continue
            r = json.loads(ln)
            if "meta" in r:
                meta = r
            else:
                rows.append(r)
    return meta, rows


def between(rows, t0, t1):
    """{cpu_j, gpu_j, gpu_w, seconds} between epoch times t0 and t1 (counter differences of the nearest samples)."""
    inside = [r for r in rows if t0 <= r["t"] <= t1]
    if len(inside) < 2:
        return None
    a, b = inside[0], inside[-1]
    dt = b["t"] - a["t"]
    res = {"seconds": dt}
    if "cpu_j" in a:
        res["cpu_j"] = b["cpu_j"] - a["cpu_j"]
    if "gpu_j" in a:
        res["gpu_j"] = sum(y - x for x, y in zip(a["gpu_j"], b["gpu_j"]))
        res["gpu_w"] = res["gpu_j"] / dt if dt > 0 else float("nan")
    return res


def cmd_between(argv):
    if len(argv) != 3:
        print("usage: kurn energy between FILE T0 T1", file=sys.stderr)
        return 2
    _, rows = load_samples(argv[0])
    print(json.dumps(between(rows, float(argv[1]), float(argv[2]))))
    return 0


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    cmds = {"probe": cmd_probe, "run": cmd_run, "sample": cmd_sample, "between": cmd_between}
    if not argv or argv[0] not in cmds:
        print(__doc__.split("\n\n")[1] if argv and argv[0] in ("-h", "--help") else __doc__, file=sys.stderr)
        return 0 if argv and argv[0] in ("-h", "--help") else 2
    return cmds[argv[0]](argv[1:])


if __name__ == "__main__":
    sys.exit(main())
