"""Where the CUDA runtime lives, and a fail-fast preflight before any GPU build.

nvcc alone is not enough: the generated kernels include cuda_runtime.h / cuda_fp16.h and link libcudart (the
harnesses also cuBLAS). Some installs split them: nvcc in /usr/local/cuda-12.8, headers and libraries in a monorepo
third-party dir with `include_no_implicit/` and `lib/`. This module finds them, decides which -I/-L flags kurn must add,
and checks the result once with a tiny program before hundreds of builds can fail the same way.

Resolution (each result records how it was found):
    nvcc         KURN_NVCC, CUDA_HOME/bin, CUDA_PATH/bin, PATH, /usr/local/cuda/bin, newest /usr/local/cuda-*/bin
    headers      KURN_CUDA_INCLUDE (paths separated by ':' or ','), -I/-isystem in NVCC_PREPEND_FLAGS / NVCC_APPEND_FLAGS,
    libraries    KURN_CUDA_LIB, -L in those flags; then under CUDA_HOME, CUDA_PATH, nvcc's own toolkit, /usr/local/cuda and
                 sibling cuda-* installs (same version as nvcc first): include/, targets/<arch>-linux/include/,
                 include_no_implicit/ and lib64/, lib/, targets/<arch>-linux/lib/ (KURN_CUDA_NO_PROBE=1: skip
                 /usr/local/cuda and sibling installs - explicit settings and nvcc's own toolkit only)
NVCC_PREPEND_FLAGS / NVCC_APPEND_FLAGS are never modified: nvcc applies them itself, kurn only reads them.

preflight: compile and link a minimal .cu (cuda_runtime.h, cuda_fp16.h, -lcudart) with the exact nvcc and flags the
builds use - first as nvcc is configured (implicit paths plus the explicit settings above), then with the discovered
directories - and run it when a GPU is present. If that fails, CudaToolchainError says what is missing, where kurn
looked and which variables fix it; every later build raises the same error at once.
"""

import functools
import glob
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import tempfile
import threading

from .toolchain import GpuToolchainError, _warn, nvcc, nvcc_version, version_tuple

SYSTEM_ROOTS = ["/usr/local/cuda"]  # probed after the environment and nvcc's own toolkit, with their sibling cuda-* installs
MAX_ATTEMPTS = 4  # discovered (headers, library) pairs the preflight tries
ENV_VARS = (
    "KURN_NVCC",
    "CUDA_HOME",
    "CUDA_PATH",
    "KURN_CUDA_INCLUDE",
    "KURN_CUDA_LIB",
    "NVCC_PREPEND_FLAGS",
    "NVCC_APPEND_FLAGS",
    "KURN_CUDA_NO_PROBE",
)
HEADERS = ("cuda_runtime.h", "cuda_fp16.h")


class CudaToolchainError(GpuToolchainError):
    """nvcc can't compile and link a minimal CUDA program with the resolved runtime: every GPU build would fail."""


def _paths(value):
    return [p for p in re.split(rf"[{re.escape(os.pathsep)},]", value or "") if p.strip()]


def flag_dirs(text):
    """(-I / -isystem dirs, -L dirs) in an NVCC_*_FLAGS value."""
    incs, libs = [], []
    toks = shlex.split(text or "")
    i = 0
    while i < len(toks):
        t = toks[i]
        for pre, out in (("-I", incs), ("-isystem", incs), ("--include-path", incs), ("-L", libs), ("--library-path", libs)):
            if t == pre and i + 1 < len(toks):
                out.append(toks[i + 1])
                i += 1
                break
            if t.startswith(pre) and len(t) > len(pre):
                out.append(t[len(pre) :].lstrip("="))
                break
        i += 1
    return incs, libs


def _target():
    return "sbsa-linux" if platform.machine() in ("aarch64", "arm64") else "x86_64-linux"


def _toolkit_roots(nv):
    """Roots of nvcc's own toolkit (through symlinks as well)."""
    out = []
    for p in (nv, os.path.realpath(nv)) if nv else ():
        r = os.path.dirname(os.path.dirname(p))
        if r not in out:
            out.append(r)
    return out


def _roots(nv, version):
    """(root, how) in search order."""
    out = []
    for var in ("CUDA_HOME", "CUDA_PATH"):
        for p in _paths(os.environ.get(var)):
            out.append((p, var))
    for r in _toolkit_roots(nv):
        out.append((r, "nvcc's toolkit"))
    if os.environ.get("KURN_CUDA_NO_PROBE", "") not in ("", "0"):  # explicit settings and nvcc's own toolkit only
        return _uniq(out)
    sib = set()
    for r in _toolkit_roots(nv) + list(SYSTEM_ROOTS):
        sib.update(glob.glob(os.path.join(os.path.dirname(r), "cuda-*")))
    out += [(r, r) for r in SYSTEM_ROOTS]
    vt = version or ""
    for r in sorted(sib, key=lambda d: (not d.endswith(f"-{vt}"), -_num(d))):  # same version as nvcc first, then newest
        out.append((r, "sibling install"))
    return _uniq(out)


def _uniq(roots):
    seen, out = set(), []
    for r, how in roots:
        if r not in seen:
            seen.add(r)
            out.append((r, how))
    return out


def _num(d):
    m = re.search(r"(\d+)\.(\d+)$", d)
    return int(m.group(1)) * 1000 + int(m.group(2)) if m else 0


def _has(d, names):
    return all(os.path.isfile(os.path.join(d, n)) for n in names)


def _has_lib(d, stem):
    return bool(glob.glob(os.path.join(d, f"lib{stem}.so*")) or glob.glob(os.path.join(d, f"lib{stem}_static.a")))


def header_version(inc):
    """CUDART_VERSION (e.g. 12080) from cuda_runtime_api.h in `inc`, or None."""
    try:
        with open(os.path.join(inc, "cuda_runtime_api.h"), errors="ignore") as fh:
            m = re.search(r"#define\s+CUDART_VERSION\s+(\d+)", fh.read())
        return int(m.group(1)) if m else None
    except OSError:
        return None


def cuda_version_str(v):
    return f"{v // 1000}.{v % 1000 // 10}" if v else None


@functools.cache
def resolve():
    """The resolved toolchain: nvcc, include and library candidates (with how each was found), what was found."""
    nv = nvcc()
    ver = nvcc_version()
    env_inc, env_lib = [], []
    for var in ("NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS"):
        i, li = flag_dirs(os.environ.get(var))
        env_inc += [(d, var) for d in i]
        env_lib += [(d, var) for d in li]
    t = _target()
    inc_c = [(d, "KURN_CUDA_INCLUDE") for d in _paths(os.environ.get("KURN_CUDA_INCLUDE"))] + env_inc
    lib_c = [(d, "KURN_CUDA_LIB") for d in _paths(os.environ.get("KURN_CUDA_LIB"))] + env_lib
    for d, how in list(lib_c):  # a library dir's sibling include dirs (third-party layouts keep them side by side)
        parent = os.path.dirname(os.path.normpath(d))
        inc_c += [(os.path.join(parent, s), f"next to {how}") for s in ("include", "include_no_implicit")]
    for d, how in list(inc_c):
        parent = os.path.dirname(os.path.normpath(d))
        lib_c += [(os.path.join(parent, s), f"next to {how}") for s in ("lib64", "lib")]
    for r, how in _roots(nv, ver):
        subs = ("include", f"targets/{t}/include", "include_no_implicit", f"targets/{t}/include_no_implicit")
        inc_c += [(os.path.join(r, s), how) for s in subs]
        lib_c += [(os.path.join(r, s), how) for s in ("lib64", "lib", f"targets/{t}/lib")]
    inc_c, lib_c = _dedup(inc_c), _dedup(lib_c)

    def first(cands, ok):
        return next(((d, how) for d, how in cands if os.path.isdir(d) and ok(d)), (None, None))

    pairs = _pairs(inc_c, lib_c, version_tuple(ver))
    inc, inc_how = first(inc_c, lambda d: _has(d, HEADERS))
    if pairs:
        inc, inc_how = pairs[0]["include"], pairs[0]["include_how"]
    rt_only = None if inc else first(inc_c, lambda d: _has(d, HEADERS[:1]))[0]
    lib, lib_how = first(lib_c, lambda d: _has_lib(d, "cudart"))
    if pairs and pairs[0]["lib"]:
        lib, lib_how = pairs[0]["lib"], pairs[0]["lib_how"]
    cublas_inc, _ = first(inc_c, lambda d: _has(d, ("cublas_v2.h",)))
    cublas_lib, _ = first(lib_c, lambda d: _has_lib(d, "cublas"))
    cuobjdump, cuobj_how = None, None
    for d, how in [(os.path.dirname(nv), "next to nvcc")] * bool(nv) + [(os.path.join(r, "bin"), h) for r, h in _roots(nv, ver)]:
        if os.access(os.path.join(d, "cuobjdump"), os.X_OK):
            cuobjdump, cuobj_how = os.path.join(d, "cuobjdump"), how
            break
    if not cuobjdump and shutil.which("cuobjdump"):
        cuobjdump, cuobj_how = shutil.which("cuobjdump"), "PATH"
    return {
        "nvcc": nv, "nvcc_version": ver, "nvcc_how": _nvcc_how(nv),
        "include": inc, "include_how": inc_how, "cuda_runtime_h_without_fp16": rt_only,
        "lib": lib, "lib_how": lib_how, "header_cudart": header_version(inc) if inc else None,
        "candidates": pairs, "cublas_include": cublas_inc, "cublas_lib": cublas_lib, "cuobjdump": cuobjdump, "cuobjdump_how": cuobj_how,
        "env": {v: os.environ[v] for v in ENV_VARS if os.environ.get(v)},
        "searched_include": [d for d, _ in inc_c], "searched_lib": [d for d, _ in lib_c],
        "explicit_include": [d for d, h in inc_c if h == "KURN_CUDA_INCLUDE"],
        "explicit_lib": [d for d, h in lib_c if h == "KURN_CUDA_LIB"],
        "env_flag_include": [d for d, _ in env_inc], "env_flag_lib": [d for d, _ in env_lib],
    }  # fmt: skip


def _pairs(inc_c, lib_c, nv):
    """Every directory with both headers, each paired with the libcudart directory of the same root (else the first one
    found). Explicit settings first, then installs whose CUDART_VERSION matches nvcc, then the rest."""
    libs = [(d, how) for d, how in lib_c if os.path.isdir(d) and _has_lib(d, "cudart")]
    out = []
    for i, (d, how) in enumerate(inc_c):
        if not (os.path.isdir(d) and _has(d, HEADERS)):
            continue
        root = os.path.dirname(os.path.normpath(d))
        near = [(ld, lh) for ld, lh in libs if os.path.normpath(ld).startswith(root + os.sep)]
        if os.path.basename(root) == "include" or os.path.basename(os.path.dirname(root)) == "targets":
            near = near or [(ld, lh) for ld, lh in libs if os.path.normpath(ld).startswith(os.path.dirname(root) + os.sep)]
        ld, lh = (near or libs or [(None, None)])[0]
        v = header_version(d)
        match = nv is not None and v is not None and (v // 1000, v % 1000 // 10) == nv
        rank = 0 if how in ("KURN_CUDA_INCLUDE", "NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS") else 1 if match or v is None else 2
        out.append({"include": d, "include_how": how, "lib": ld, "lib_how": lh, "cudart": v, "rank": rank, "order": i})
    return sorted(out, key=lambda p: (p["rank"], p["order"]))


def _dedup(c):
    seen, out = set(), []
    for d, how in c:
        k = os.path.normpath(d)
        if k not in seen:
            seen.add(k)
            out.append((d, how))
    return out


def _nvcc_how(nv):
    if not nv:
        return None
    if os.environ.get("KURN_NVCC") == nv:
        return "KURN_NVCC"
    for var in ("CUDA_HOME", "CUDA_PATH"):
        for p in _paths(os.environ.get(var)):
            if os.path.normpath(os.path.join(p, "bin", "nvcc")) == os.path.normpath(nv):
                return var
    return "PATH" if shutil.which("nvcc") == nv else "default install path"


def _dir_flags(incs, libs):
    out = []
    for d in incs:
        out += ["-I", d]
    for d in libs:
        out += ["-L", d, f"-Xlinker=-rpath={d}"]  # the harnesses load libcudart.so / libcublas.so at run time
    return out


PREFLIGHT_CU = r"""#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
__global__ void kurn_preflight(__half *x) { x[threadIdx.x] = __float2half(1.0f + threadIdx.x); }
int main(int argc, char **argv) {
  int rt = 0, drv = 0;
  cudaRuntimeGetVersion(&rt);
  cudaDriverGetVersion(&drv);
  int ok = 1;
  const char *err = "";
  if (argc > 1) {  // run a kernel too
    __half *d = nullptr;
    cudaError_t e = cudaMalloc(&d, 32 * sizeof(__half));
    if (e == cudaSuccess) { kurn_preflight<<<1, 32>>>(d); e = cudaDeviceSynchronize(); }
    if (e != cudaSuccess) { ok = 0; err = cudaGetErrorString(e); }
    if (d) cudaFree(d);
  }
  printf("{\"runtime\": %d, \"driver\": %d, \"header\": %d, \"ok\": %d, \"run_error\": \"%s\"}\n", rt, drv, CUDART_VERSION, ok, err);
  return ok ? 0 : 1;
}
"""

PREFLIGHT_CUBLAS_CU = r"""#include <cublas_v2.h>
#include <cstdio>
int main(int argc, char **argv) {
  int ok = 1;
  if (argc > 1) { cublasHandle_t h; ok = cublasCreate(&h) == CUBLAS_STATUS_SUCCESS; if (ok) cublasDestroy(h); }
  printf("{\"cublas_ok\": %d}\n", ok);
  return ok ? 0 : 1;
}
"""


def _gpu_present():
    from .harness import gpu_present

    return gpu_present()


def _compile(nv, src, flags, link, d, name):
    cu = os.path.join(d, name + ".cu")
    exe = os.path.join(d, name)
    with open(cu, "w") as fh:
        fh.write(src)
    args = [nv, "-std=c++17", *flags, cu, "-o", exe, *link]
    r = subprocess.run(args, capture_output=True, text=True)
    return r.returncode == 0, exe, args, (r.stderr or r.stdout).strip()


def _first_error(text):
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    return next((ln for ln in lines if re.search(r"error|fatal|cannot find|undefined", ln, re.I)), lines[0] if lines else "")


_STATE = {}
_LOCK = threading.Lock()  # parallel builds (build_many) must share one preflight, not each run their own


def reset():
    """Forget every resolution and preflight result (tests; after changing the environment)."""
    from . import toolchain

    _STATE.clear()
    resolve.cache_clear()
    for f in (toolchain.nvcc, toolchain.nvcc_version, toolchain.nvcc_gpu_codes):
        f.cache_clear()


def preflight(arch_flags=None, run=None, cublas=False):
    """Check that nvcc compiles and links a minimal CUDA program (and run it on a GPU). Returns a report dict whose
    "flags" are the -I/-L flags every build must add; raises CudaToolchainError with the fix otherwise. The result is
    cached per (arch flags, cublas): later builds pay nothing and fail at once with the same message."""
    key = (tuple(arch_flags or ()), cublas)
    with _LOCK:
        if key not in _STATE:
            try:
                _STATE[key] = _preflight(list(arch_flags or ["-gencode", "arch=compute_80,code=sm_80"]),
                                         _gpu_present() if run is None else run, cublas)  # fmt: skip
            except CudaToolchainError as e:
                _STATE[key] = e
        st = _STATE[key]
    if isinstance(st, Exception):
        raise st
    return st


def _preflight(arch, run, cublas):
    res = resolve()
    nv = res["nvcc"]
    if not nv:
        raise CudaToolchainError("nvcc not found: install the CUDA toolkit, put its bin/ on PATH, or set KURN_NVCC / CUDA_HOME")
    for var, dirs in (("KURN_CUDA_INCLUDE", res["explicit_include"]), ("KURN_CUDA_LIB", res["explicit_lib"])):
        for d in dirs:
            if not os.path.isdir(d):
                _warn(f"{var}: {d} is not a directory (ignored)")
    ex_inc = [d for d in res["explicit_include"] if os.path.isdir(d)]
    ex_lib = [d for d in res["explicit_lib"] if os.path.isdir(d)]
    explicit = _dir_flags(ex_inc, ex_lib) + [f"-Xlinker=-rpath={d}" for d in res["env_flag_lib"]]
    used = [v for v in ("KURN_CUDA_INCLUDE", "KURN_CUDA_LIB", "NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS") if v in res["env"]]
    extra = f" + {', '.join(used)}" if used else ""
    attempts = [(f"as configured (nvcc's implicit paths{extra})", explicit, None)]
    known = set(res["explicit_include"] + res["env_flag_include"])
    for p in res["candidates"][:MAX_ATTEMPTS]:
        if p["include"] in known:
            continue
        v = cuda_version_str(p["cudart"])
        attempts.append((f"with discovered {p['include']} ({p['include_how']}{', CUDA ' + v if v else ''}) and {p['lib'] or '-'}",
                         explicit + _dir_flags([p["include"]], [p["lib"]] if p["lib"] and p["lib"] not in res["explicit_lib"] else []),
                         p))  # fmt: skip
    errors = []
    with tempfile.TemporaryDirectory(prefix="kurn-preflight-") as d:
        for how, flags, pair in attempts:
            ok, exe, args, err = _compile(nv, PREFLIGHT_CU, arch + flags, ["-lcudart"], d, "preflight")
            if not ok:
                errors.append((how, args, err))
                continue
            rep = {"flags": flags, "how": how, "command": shlex.join(args), "ran": False}
            r = subprocess.run([exe] + (["run"] if run else []), capture_output=True, text=True, timeout=120)
            line = next((ln for ln in r.stdout.splitlines() if ln.startswith("{")), None)
            info = json.loads(line) if line else {}
            rep.update(info, ran=bool(run))
            if run and (r.returncode or not info.get("ok")):
                why = info.get("run_error") or _first_error(r.stderr) or f"exit {r.returncode}"
                raise CudaToolchainError(
                    f"CUDA preflight: the test program built with {nv} but failed on the GPU: {why}\n"
                    + _version_lines(res, info) + "  If the driver is older than the CUDA runtime, update the driver or "
                    "build with an older toolkit (KURN_NVCC / CUDA_HOME).")  # fmt: skip
            if pair is not None:
                res = dict(res, include=pair["include"], include_how=pair["include_how"], lib=pair["lib"], lib_how=pair["lib_how"],
                           header_cudart=pair["cudart"])  # fmt: skip
                rep["include"], rep["lib"] = pair["include"], pair["lib"]
                _warn(f"CUDA runtime not on nvcc's default paths; using {pair['include']} ({pair['include_how']}) and {pair['lib']} "
                      f"({pair['lib_how']}). Set KURN_CUDA_INCLUDE / KURN_CUDA_LIB to make this explicit.")  # fmt: skip
            _version_warnings(res, info)
            if cublas:
                ok2, exe2, args2, err2 = _compile(nv, PREFLIGHT_CUBLAS_CU, arch + flags, ["-lcublas"], d, "preflight_cublas")
                if not ok2:
                    raise CudaToolchainError(_message(res, [("cuBLAS (cublas_v2.h, -lcublas)", args2, err2)], what="cuBLAS"))
                r2 = subprocess.run([exe2] + (["run"] if run else []), capture_output=True, text=True, timeout=120)
                if run and r2.returncode:
                    raise CudaToolchainError(f"CUDA preflight: cuBLAS built but cublasCreate failed on the GPU ({_first_error(r2.stderr)})")
                rep["cublas"] = True
            return rep
    raise CudaToolchainError(_message(res, errors))


def _version_lines(res, info):
    out = f"  nvcc: {res['nvcc']} (CUDA {res['nvcc_version'] or '?'}, found via {res['nvcc_how']})\n"
    hdr = info.get("header") or res["header_cudart"]
    if hdr:
        out += f"  headers: CUDA {cuda_version_str(hdr)}"
        out += f" ({res['include']}, {res['include_how']})\n" if res["include"] else "\n"
    if info.get("runtime"):
        out += f"  runtime: CUDA {cuda_version_str(info['runtime'])}, driver supports CUDA {cuda_version_str(info.get('driver'))}\n"
    return out


def _version_warnings(res, info):
    nv = version_tuple(res["nvcc_version"])
    hdr = info.get("header") or res["header_cudart"]
    if nv and hdr and (hdr // 1000, hdr % 1000 // 10) != nv:
        _warn(f"CUDA headers are {cuda_version_str(hdr)} but nvcc is {res['nvcc_version']} ({res['nvcc']}): mixing toolkit versions; "
              "point KURN_CUDA_INCLUDE / KURN_CUDA_LIB (or CUDA_HOME) at the runtime that matches nvcc")  # fmt: skip
    rt, drv = info.get("runtime"), info.get("driver")
    if rt and drv and drv < rt:
        _warn(f"the NVIDIA driver supports CUDA {cuda_version_str(drv)} but the runtime is CUDA {cuda_version_str(rt)}: kernels will "
              "not load; update the driver or build with an older toolkit")  # fmt: skip


def _message(res, errors, what="the CUDA runtime"):
    missing = []
    if not res["include"]:
        missing.append("cuda_runtime.h + cuda_fp16.h" if not res["cuda_runtime_h_without_fp16"] else "cuda_fp16.h")
    if not res["lib"]:
        missing.append("libcudart")
    if what == "cuBLAS":
        missing = [x for x, ok in (("cublas_v2.h", res["cublas_include"]), ("libcublas", res["cublas_lib"])) if not ok] or ["cuBLAS"]
    lines = [f"CUDA preflight failed: {res['nvcc']} (CUDA {res['nvcc_version'] or '?'}, via {res['nvcc_how']}) can't compile and link a "
             f"minimal CUDA program, so every GPU build would fail."]  # fmt: skip
    if missing:
        lines.append(f"  missing: {', '.join(missing)} (not on nvcc's default paths or anywhere kurn looked)")
    for how, _args, err in errors:
        lines.append(f"  tried {how}: {_first_error(err)}")
    lines.append("  searched for headers: " + ", ".join(res["searched_include"]))
    lines.append("  searched for libcudart: " + ", ".join(res["searched_lib"]))
    if res["env"]:
        lines.append("  environment: " + ", ".join(f"{k}={v}" for k, v in res["env"].items()))
    lines.append("  fix: point kurn at the directories that hold them, e.g.")
    lines.append("    KURN_CUDA_INCLUDE=/path/to/include KURN_CUDA_LIB=/path/to/lib ./run_gpu_check.sh")
    lines.append("  (several paths: separate with ':'), or CUDA_HOME=/path/to/root (with include/ or include_no_implicit/ and")
    lines.append('  lib/ or lib64/), or NVCC_APPEND_FLAGS="-I/path/to/include -L/path/to/lib" (passed to nvcc unchanged).')
    return "\n".join(lines)


def build_flags(arch_flags=None, link=True, cublas=False):
    """The -I/-L flags every nvcc build adds (runs the preflight once; raises CudaToolchainError if it fails).
    link=False: only the -I flags (compile-only builds such as -cubin)."""
    flags = list(preflight(arch_flags, cublas=cublas)["flags"])
    if link:
        return flags
    out = []
    for i, f in enumerate(flags):
        if f == "-I":
            out += ["-I", flags[i + 1]]
    return out


def cuobjdump():
    """Path of cuobjdump from the same toolkit resolution (next to nvcc first)."""
    p = resolve()["cuobjdump"]
    if not p:
        raise CudaToolchainError(f"cuobjdump not found next to {resolve()['nvcc']} or in CUDA_HOME / CUDA_PATH / PATH")
    return p


def report(arch_flags=None, run=None, cublas=False):
    """Everything for run.log / archs.json: the resolution plus the preflight result (or its error)."""
    out = {k: v for k, v in resolve().items() if not k.startswith("searched_")}
    out["searched_include"], out["searched_lib"] = resolve()["searched_include"], resolve()["searched_lib"]
    try:
        out["preflight"] = preflight(arch_flags, run, cublas)
    except CudaToolchainError as e:
        out["preflight"] = {"error": str(e)}
    return out


def _ver(v):
    return f", CUDA {cuda_version_str(v)}" if v else ""


def summary(rep):
    """Human-readable lines of report() for run.log."""
    pf = rep.get("preflight", {})
    via = lambda d, how: f"{d} (via {how})" if d else "-"  # noqa: E731
    lines = [
        f"nvcc:       {rep['nvcc'] or 'not found'} (CUDA {rep['nvcc_version'] or '?'}, via {rep['nvcc_how']})",
        f"headers:    {via(rep['include'], rep['include_how'])}" + _ver(rep["header_cudart"]),
        f"libcudart:  {via(rep['lib'], rep['lib_how'])}",
        f"cuBLAS:     cublas_v2.h {rep['cublas_include'] or 'not found'}, libcublas {rep['cublas_lib'] or 'not found'}",
        f"cuobjdump:  {via(rep['cuobjdump'], rep['cuobjdump_how'])}",
        "environment: " + (", ".join(f"{k}={v}" for k, v in rep["env"].items()) or "(none of " + ", ".join(ENV_VARS) + ")"),
    ]  # fmt: skip
    if "error" in pf:
        lines.append("preflight:  FAILED")
        lines += pf["error"].splitlines()
    else:
        lines.append(f"preflight:  ok, {pf.get('how')}; kurn adds: {' '.join(pf.get('flags') or []) or 'nothing'}")
        if pf.get("include"):
            lines.append(f"            runtime used: {pf['include']} and {pf.get('lib')}")
        rt, drv = pf.get("runtime"), pf.get("driver")
        if pf.get("ran"):
            lines.append(f"            ran on the GPU: runtime CUDA {cuda_version_str(rt)}, driver supports CUDA {cuda_version_str(drv)}")
        if pf.get("cublas"):
            lines.append("            cuBLAS: compiles and links" + (" and runs" if pf.get("ran") else ""))
    return lines
