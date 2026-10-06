"""Compilers for `target cuda`: the CPU emulator build (host C++) and nvcc builds with ptxas
resource reports and a static occupancy estimate.

Environment overrides:
    KURN_NVCC   nvcc to use (default: nvcc on PATH, then /usr/local/cuda*/bin/nvcc)
    KURN_CXX    host C++ compiler for the emulator (default: $CXX, then g++, clang++)
"""

import functools
import glob
import hashlib
import os
import re
import shlex
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from importlib import resources

from ..toolchain import cache_dir
from .codegen import generate
from .spec import threads


class GpuBuildError(Exception):
    pass


class GpuToolchainError(GpuBuildError):
    """The host C++ compiler can't build the emulator at all (reported once; callers skip, not fail)."""


def data_path(name):
    return str(resources.files("kurn.gpu") / "data" / name)


def _sha(*parts):
    h = hashlib.sha1()
    for p in parts:
        h.update(p if isinstance(p, bytes) else str(p).encode())
        h.update(b"\0")
    return h.hexdigest()[:14]


def _data_hash(*names):
    return _sha(*(open(data_path(n), "rb").read() for n in names))


@functools.cache
def nvcc():
    for cand in (os.environ.get("KURN_NVCC"), shutil.which("nvcc"), "/usr/local/cuda/bin/nvcc",
                 *sorted(glob.glob("/usr/local/cuda-*/bin/nvcc"), reverse=True)):  # fmt: skip
        if cand and os.path.exists(cand):
            return cand
    return None


@functools.cache
def nvcc_version():
    n = nvcc()
    if not n:
        return None
    out = subprocess.run([n, "--version"], capture_output=True, text=True).stdout
    m = re.search(r"release (\d+\.\d+)", out)
    return m.group(1) if m else out.strip().splitlines()[-1]


@functools.cache
def cxx():
    for cand in (os.environ.get("KURN_CXX"), os.environ.get("CXX"), "g++", "clang++"):
        if cand and shutil.which(shlex.split(cand)[0]):
            return tuple(shlex.split(cand))
    raise GpuBuildError("no host C++ compiler for the emulator (set KURN_CXX)")


_PROBE_CXX = """#include <ucontext.h>
#include <cmath>
#include <cstdio>
#include <functional>
#include <vector>
int main() { std::vector<double> v{2.0}; std::function<double()> f = [&] { return std::sqrt(v[0]); }; std::printf("%g", f()); }
"""


@functools.cache
def cxx_problem():
    """None if the host C++ compiler builds a C++17 program with the headers the emulator uses, else why not.
    (Ubuntu's clang++ often selects a GCC installation whose libstdc++ headers are not installed.)"""
    try:
        cc = cxx()
    except GpuBuildError as e:
        return str(e)
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "probe.cpp")
        with open(src, "w") as fh:
            fh.write(_PROBE_CXX)
        r = subprocess.run([*cc, "-std=c++17", src, "-o", os.path.join(d, "probe")], capture_output=True, text=True)
    if r.returncode == 0:
        return None
    first = next((ln.strip() for ln in r.stderr.splitlines() if "error" in ln), r.stderr.strip()[:200])
    return (f"host C++ compiler `{shlex.join(cc)}` can't build a C++17 program ({first}); install its C++ standard "
            "library headers (for clang++ on Ubuntu: libstdc++-<gcc version>-dev) or set KURN_CXX")  # fmt: skip


@functools.cache
def cxx_is_clang():
    r = subprocess.run([*cxx(), "--version"], capture_output=True, text=True)
    return "clang" in r.stdout


def _run(args, what):
    r = subprocess.run(args, capture_output=True, text=True)
    if r.returncode:
        raise GpuBuildError(f"{what} failed:\n$ {shlex.join(args)}\n{r.stderr[-6000:]}")
    return r


def _out_dir(sub):
    d = os.path.join(cache_dir(), "gpu", sub)
    os.makedirs(d, exist_ok=True)
    return d


def emu_build(c, src=None):
    """Compile a config's CUDA source with the CPU emulator into an `emu_run` executable."""
    problem = cxx_problem()
    if problem:
        raise GpuToolchainError(problem)
    src = src or generate(c)
    flags = ["-std=c++17", "-O1", "-g0", "-Wall", "-Wextra", "-Wno-unknown-pragmas", "-Wno-unused-parameter",
             "-Wno-unused-function", "-Wno-unused-variable", "-Werror", "-DKURN_EMU"]  # fmt: skip
    if cxx_is_clang():  # clang honours `#pragma unroll` and reports loops it can't unroll at -O1; nvcc does not care
        flags.append("-Wno-pass-failed")
    h = _sha(src, _data_hash("kurn_cuemu.h", "kurn_gpu.h", "emu_main.cpp"), shlex.join(cxx()), shlex.join(flags))
    d = _out_dir("emu")
    exe = os.path.join(d, f"{c['weights']}_{c['op']}_{h}")
    if os.path.exists(exe):
        return exe
    cu = exe + ".cu"
    with open(cu, "w") as fh:
        fh.write(src)
    tmp = f"{exe}.tmp{os.getpid()}"
    _run([*cxx(), *flags, "-I", os.path.dirname(data_path("kurn_gpu.h")), "-x", "c++", cu, "-x", "c++", data_path("emu_main.cpp"),
          "-o", tmp], "emulator build")  # fmt: skip
    os.replace(tmp, exe)
    return exe


def arch_flags(archs):
    out = []
    for a in archs:
        n = a.split("_")[1]
        out += ["-gencode", f"arch=compute_{n},code=sm_{n}"]
    return out


def nvcc_build(c, archs=None, src=None, out_dir=None, extra=()):
    """nvcc -> shared library implementing kurn_gpu.h. Returns (path, resources per arch)."""
    n = nvcc()
    if not n:
        raise GpuBuildError("nvcc not found (install the CUDA toolkit or set KURN_NVCC)")
    src = src or generate(c)
    archs = tuple(archs or (c["arch"],))
    flags = ["-O3", "-std=c++17", "-shared", "-Xcompiler", "-fPIC", "-Xptxas", "-v", "-lineinfo", *arch_flags(archs), *extra]
    h = _sha(src, n, nvcc_version(), shlex.join(flags))
    d = out_dir or _out_dir("cuda")
    os.makedirs(d, exist_ok=True)
    so = os.path.join(d, f"{c['weights']}_{c['op']}_{h}.so")
    rep = so + ".ptxas.txt"
    if not (os.path.exists(so) and os.path.exists(rep)):
        cu = so[:-3] + ".cu"
        with open(cu, "w") as fh:
            fh.write(src)
        tmp = f"{so}.tmp{os.getpid()}"
        r = _run([n, *flags, cu, "-o", tmp], "nvcc build")
        with open(rep, "w") as fh:
            fh.write(r.stderr)
        os.replace(tmp, so)
    with open(rep) as fh:
        return so, parse_ptxas(fh.read())


def ptxas_report(c, archs=("sm_80", "sm_90", "sm_100")):
    """Compile to cubin for each arch (no GPU needed); returns {arch: {kernel: resources}}."""
    n = nvcc()
    if not n:
        raise GpuBuildError("nvcc not found")
    src = generate(c)
    out = {}
    for a in archs:
        h = _sha(src, n, nvcc_version(), a)
        d = _out_dir("ptxas")
        rep = os.path.join(d, f"{c['weights']}_{c['op']}_{a}_{h}.txt")
        if not os.path.exists(rep):
            cu = rep[:-4] + ".cu"
            with open(cu, "w") as fh:
                fh.write(src)
            r = _run([n, "-O3", "-std=c++17", "-cubin", f"-arch={a}", "-Xptxas", "-v", cu, "-o", rep[:-4] + ".cubin"], f"ptxas {a}")
            with open(rep, "w") as fh:
                fh.write(r.stderr)
        with open(rep) as fh:
            out[a] = parse_ptxas(fh.read())
    return out


_ENTRY = re.compile(r"Compiling entry function '(\w+)' for '(\w+)'")
_USED = re.compile(r"Used (\d+) registers")
_SMEM = re.compile(r"(\d+) bytes smem")
_SPILL = re.compile(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads")


def parse_ptxas(text):
    """ptxas -v output -> {'arch:kernel': {regs, smem, stack, spill_st, spill_ld}} (kernel names demangled loosely)."""
    out, cur = {}, None
    for line in text.splitlines():
        m = _ENTRY.search(line)
        if m:
            name = m.group(1)
            short = next((k for k in ("kg_gemv", "kg_gemm", "kg_quant_q8_0", "kg_quant_q8_K", "kg_repack") if k in name), name)
            cur = f"{m.group(2)}:{short}"
            out[cur] = {"regs": 0, "smem": 0, "stack": 0, "spill_st": 0, "spill_ld": 0}
            continue
        if cur is None:
            continue
        m = _SPILL.search(line)
        if m:
            out[cur].update(stack=int(m.group(1)), spill_st=int(m.group(2)), spill_ld=int(m.group(3)))
        m = _USED.search(line)
        if m:
            sm = _SMEM.search(line)
            out[cur].update(regs=int(m.group(1)), smem=int(sm.group(1)) if sm else 0)
    return out


# Per-SM limits used for the static occupancy estimate (CUDA occupancy rules, simplified).
SM_LIMITS = {
    "sm_80": dict(threads=2048, blocks=32, regs=65536, smem=167936, name="A100"),
    "sm_86": dict(threads=1536, blocks=16, regs=65536, smem=102400, name="RTX 30 / A10 / A40"),
    "sm_89": dict(threads=1536, blocks=24, regs=65536, smem=102400, name="L4 / L40 / RTX 40"),
    "sm_90": dict(threads=2048, blocks=32, regs=65536, smem=233472, name="H100 / H200 / GH200"),
    "sm_100": dict(threads=2048, blocks=32, regs=65536, smem=233472, name="B200 / GB200"),
    "sm_120": dict(threads=1536, blocks=32, regs=65536, smem=102400, name="RTX 50 / RTX PRO 6000"),
}


def occupancy(regs, smem, nthreads, arch):
    """Resident warps per SM / max warps (register, shared-memory and thread limits)."""
    lim = SM_LIMITS[arch]
    warps = (nthreads + 31) // 32
    regs_per_warp = ((max(regs, 1) * 32 + 255) // 256) * 256
    by_regs = lim["regs"] // (regs_per_warp * warps)
    by_smem = lim["smem"] // (smem + 1024) if smem else lim["blocks"]
    by_thr = lim["threads"] // (warps * 32)
    blocks = max(0, min(by_regs, by_smem, by_thr, lim["blocks"]))
    return blocks, blocks * warps / (lim["threads"] // 32)


def resource_rows(c, report):
    """Flatten a ptxas report for the main kernel into rows with occupancy."""
    rows = []
    main = "kg_gemm" if c["op"] == "gemm" else "kg_gemv"
    for arch, kernels in report.items():
        for key, r in kernels.items():
            a, k = key.split(":")
            if k != main:
                continue
            blocks, occ = occupancy(r["regs"], r["smem"], threads(c), arch)
            rows.append({"arch": arch, "regs": r["regs"], "smem": r["smem"], "spill": r["spill_st"] + r["spill_ld"] + r["stack"],
                         "stack": r["stack"], "blocks_per_sm": blocks, "occupancy": round(occ, 3)})  # fmt: skip
    return rows


def build_many(configs, fn, jobs=None):
    """Run fn(config) in parallel (builds); returns list of (config, result or exception)."""
    jobs = jobs or max(1, (os.cpu_count() or 2))

    def one(c):
        try:
            return c, fn(c)
        except Exception as e:  # noqa: BLE001  (report every failure, keep going)
            return c, e

    with ThreadPoolExecutor(jobs) as ex:
        return list(ex.map(one, configs))
