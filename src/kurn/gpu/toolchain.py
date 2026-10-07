"""Compilers for `target cuda`: the CPU emulator build (host C++) and nvcc builds with ptxas
resource reports and a static occupancy estimate.

Environment overrides:
    KURN_NVCC   nvcc to use (default: CUDA_HOME/bin, CUDA_PATH/bin, PATH, /usr/local/cuda/bin, newest /usr/local/cuda-*/bin)
    KURN_CXX    host C++ compiler for the emulator (default: $CXX, then g++, clang++)
The CUDA runtime headers and libraries every nvcc build needs are located by kurn.gpu.cudaenv (KURN_CUDA_INCLUDE,
KURN_CUDA_LIB, CUDA_HOME, CUDA_PATH, NVCC_PREPEND_FLAGS / NVCC_APPEND_FLAGS, then probing), checked once by its preflight.
"""

import functools
import glob
import hashlib
import os
import re
import shlex
import shutil
import subprocess
import sys
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


def _nvcc_candidates():
    yield os.environ.get("KURN_NVCC")
    for var in ("CUDA_HOME", "CUDA_PATH"):
        for root in re.split(rf"[{re.escape(os.pathsep)},]", os.environ.get(var) or ""):
            if root.strip():
                yield os.path.join(root, "bin", "nvcc")
    yield shutil.which("nvcc")
    yield "/usr/local/cuda/bin/nvcc"
    yield from sorted(glob.glob("/usr/local/cuda-*/bin/nvcc"), reverse=True)


@functools.cache
def nvcc():
    for cand in _nvcc_candidates():
        if cand and os.path.exists(cand):
            return cand
    return None


def parse_nvcc_version(text):
    """'major.minor' from `nvcc --version` output ('release 12.9, V12.9.86' or just 'V12.9.86'), else None."""
    m = re.search(r"release (\d+)\.(\d+)", text or "") or re.search(r"\bV(\d+)\.(\d+)", text or "")
    return f"{int(m.group(1))}.{int(m.group(2))}" if m else None


def version_tuple(v):
    """'12.10' -> (12, 10) (numeric, so 12.10 > 12.9); None or junk -> None."""
    m = re.fullmatch(r"(\d+)\.(\d+)", (v or "").strip())
    return (int(m.group(1)), int(m.group(2))) if m else None


@functools.cache
def nvcc_version():
    """'major.minor' of the nvcc kurn builds with, or None (no nvcc, or output it can't parse)."""
    n = nvcc()
    if not n:
        return None
    r = subprocess.run([n, "--version"], capture_output=True, text=True)
    return parse_nvcc_version(r.stdout + r.stderr)


# First CUDA release whose nvcc targets each arch (fallback when `nvcc --list-gpu-code` is unavailable).
MIN_NVCC = {"sm_70": (9, 0), "sm_75": (10, 0), "sm_80": (11, 0), "sm_86": (11, 1), "sm_87": (11, 4), "sm_89": (11, 8),
            "sm_90": (11, 8), "sm_100": (12, 8), "sm_101": (12, 8), "sm_103": (12, 9), "sm_120": (12, 8), "sm_121": (12, 9)}  # fmt: skip


@functools.cache
def nvcc_gpu_codes():
    """The sm_XX targets this nvcc can build, from `nvcc --list-gpu-code` (CUDA 11+); None if it can't say."""
    n = nvcc()
    if not n:
        return None
    r = subprocess.run([n, "--list-gpu-code"], capture_output=True, text=True)
    codes = set(re.findall(r"\bsm_\d+[af]?\b", r.stdout)) if r.returncode == 0 else set()
    return frozenset(codes) or None


def nvcc_supports(arch):
    """True / False if this nvcc can (not) build SASS for `arch`; None if neither its target list nor its version is
    known (the build then tries and nvcc's own error decides)."""
    codes = nvcc_gpu_codes()
    if codes is not None:
        return arch in codes
    have = version_tuple(nvcc_version())
    if have is None or arch not in MIN_NVCC:
        return None
    return have >= MIN_NVCC[arch]


def _arch_num(a):
    return int(re.sub(r"\D", "", a.split("_")[1]))


WARNED = set()  # messages already printed (once per process)


def _warn(msg):
    if msg not in WARNED:
        WARNED.add(msg)
        print(f"kurn: {msg}", file=sys.stderr)


def nvcc_name():
    return f"nvcc {nvcc_version() or '(unknown version)'} ({nvcc() or 'not found'})"


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


def cant_build(arch):
    """Why this nvcc can't build `arch` (for error and warning messages)."""
    need = f" (needs CUDA {'.'.join(map(str, MIN_NVCC[arch]))}+)" if arch in MIN_NVCC else ""
    return f"{nvcc_name()} can't build {arch} SASS{need}"


def ptx_below(arch):
    """The newest arch at or below `arch` whose SASS/PTX this nvcc can build, or None."""
    cands = [c for c in (nvcc_gpu_codes() or MIN_NVCC) if c[-1].isdigit() and _arch_num(c) <= _arch_num(arch) and nvcc_supports(c)]
    return max(cands, key=_arch_num) if cands else None


def arch_flags(archs, fallback=False):
    """-gencode flags for SASS of each arch. An arch this nvcc can't build is an error, unless `fallback` (the arch was
    detected from the local GPU rather than requested): then it gets PTX of the newest arch the nvcc can build below
    it, which the driver JIT-compiles for the GPU, and a warning says so."""
    out = []
    for a in archs:
        n = a.split("_")[1]
        if nvcc_supports(a) is False:
            below = ptx_below(a)
            if not fallback or not below:
                raise GpuBuildError(f"{cant_build(a)}; install a newer CUDA toolkit (or set KURN_NVCC), or choose other archs")
            p = below.split("_")[1]
            _warn(f"{cant_build(a)}; building compute_{p} PTX instead, which the driver JIT-compiles for the {a} GPU")
            out += ["-gencode", f"arch=compute_{p},code=compute_{p}"]
            continue
        out += ["-gencode", f"arch=compute_{n},code=sm_{n}"]
    return out


def nvcc_build(c, archs=None, src=None, out_dir=None, extra=(), fallback=False):
    """nvcc -> shared library implementing kurn_gpu.h. Returns (path, resources per arch). `fallback`: archs this
    nvcc can't build get older PTX with a warning instead of an error (for archs detected from the local GPU)."""
    n = nvcc()
    if not n:
        raise GpuBuildError("nvcc not found (install the CUDA toolkit or set KURN_NVCC)")
    from .cudaenv import build_flags

    src = src or generate(c)
    archs = tuple(archs or (c["arch"],))
    flags = ["-O3", "-std=c++17", "-shared", "-Xcompiler", "-fPIC", "-Xptxas", "-v", "-lineinfo", *arch_flags(archs, fallback), *extra,
             *build_flags()]  # fmt: skip
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
    from .cudaenv import build_flags

    inc = build_flags(link=False)
    src = generate(c)
    out = {}
    for a in archs:
        h = _sha(src, n, nvcc_version(), a, shlex.join(inc))
        d = _out_dir("ptxas")
        rep = os.path.join(d, f"{c['weights']}_{c['op']}_{a}_{h}.txt")
        if not os.path.exists(rep):
            cu = rep[:-4] + ".cu"
            with open(cu, "w") as fh:
                fh.write(src)
            r = _run([n, "-O3", "-std=c++17", "-cubin", f"-arch={a}", "-Xptxas", "-v", *inc, cu, "-o", rep[:-4] + ".cubin"], f"ptxas {a}")
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
    """Run fn(config) in parallel (builds); returns list of (config, result or exception). A failed CUDA preflight
    (cudaenv.CudaToolchainError: no build can work) is raised once instead of being collected per config."""
    from .cudaenv import CudaToolchainError

    jobs = jobs or max(1, (os.cpu_count() or 2))

    def one(c):
        try:
            return c, fn(c)
        except CudaToolchainError:
            raise
        except Exception as e:  # noqa: BLE001  (report every failure, keep going)
            return c, e

    with ThreadPoolExecutor(jobs) as ex:
        return list(ex.map(one, configs))
