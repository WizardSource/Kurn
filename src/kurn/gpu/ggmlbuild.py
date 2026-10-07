"""Build llama.cpp's ggml with CUDA - the kit's llama.cpp competitor (attention decode baseline, matmul ggml-cuda
line) - in a directory of the caller's choosing, with kurn's CUDA toolchain resolution (cudaenv).

cmake's own CUDA discovery assumes headers and libraries inside nvcc's toolkit; on a split install (runtime in e.g. a
monorepo third_party dir with include_no_implicit/ and lib/) it fails. So the directories the kurn preflight found
(KURN_CUDA_INCLUDE / KURN_CUDA_LIB, CUDA_HOME, NVCC_*_FLAGS, probing) are handed to cmake explicitly: CUDA compile and
link flags, find_library / find_path search paths, and a build rpath so libggml-cuda finds libcublas at run time.
Nothing is downloaded and nothing is written outside the source and build directories (sources: a llama.cpp checkout,
or the ggml subset the kit bundles with a two-line wrapper CMakeLists.txt).
"""

import json
import os
import re
import shutil
import subprocess
import sys
import time

from . import cudaenv
from .toolchain import nvcc, nvcc_version


def cuda_dirs():
    """(include dirs, lib dirs) cmake must see: the preflight's -I / -L, NVCC_*_FLAGS dirs, cuBLAS's dirs."""
    pf = cudaenv.preflight(cublas=True)
    incs, libs = [], []
    fl = list(pf.get("flags") or [])
    for i, f in enumerate(fl[:-1]):
        if f == "-I":
            incs.append(fl[i + 1])
        elif f == "-L":
            libs.append(fl[i + 1])
    ei, el = cudaenv.flag_dirs(" ".join(os.environ.get(k, "") for k in ("NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS")))
    res = cudaenv.resolve()
    incs += ei + [d for d in (pf.get("include"), res.get("cublas_include"), res.get("include")) if d]
    libs += el + [d for d in (pf.get("lib"), res.get("cublas_lib"), res.get("lib")) if d]
    uniq = lambda xs: [x for i, x in enumerate(xs) if x and os.path.isdir(x) and x not in xs[:i]]  # noqa: E731
    return uniq(incs), uniq(libs)


def _cmake():
    exe = shutil.which("cmake")
    if exe:
        return [exe]
    try:
        import cmake  # noqa: F401  (the pip `cmake` package, if present)

        return [sys.executable, "-m", "cmake"]
    except ImportError:
        return None


def _commit(src):
    for d in (src, os.path.dirname(src)):
        p = os.path.join(d, "COMMIT")
        if os.path.exists(p):
            return open(p).read().strip()
    r = subprocess.run(["git", "-C", src, "log", "--oneline", "-1"], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else "unknown"


def build(src, build_dir, arch="sm_80", jobs=None, log=None):
    """Configure and build ggml + ggml-cuda from `src` (a llama.cpp checkout or the kit's ggml bundle) into
    `build_dir` (libraries in build_dir/bin). Returns a status dict (also the content of the kit's build.json)."""
    t0 = time.time()
    src, build_dir = os.path.abspath(src), os.path.abspath(build_dir)
    out = {"source": src, "dir": build_dir, "arch": arch, "commit": _commit(src)}
    cm = _cmake()
    if not cm:
        return {**out, "status": "skipped: cmake not found (install cmake, or pip's cmake package, then re-run)"}
    n = nvcc()
    if not n:
        return {**out, "status": "skipped: nvcc not found"}
    if glob_libs(build_dir):
        return {**out, "status": "ok (already built)", "min": 0}
    incs, libs = cuda_dirs()
    num = re.sub(r"\D", "", arch.split("_")[1])
    rp = ":".join(libs)
    ldf = " ".join([f"-L{d}" for d in libs] + ([f"-Wl,-rpath,{rp}"] if rp else []))
    cuf = " ".join([f"-I{d}" for d in incs] + [f"-L{d}" for d in libs])
    full = os.path.exists(os.path.join(src, "src", "llama.cpp")) or os.path.exists(os.path.join(src, "include", "llama.h"))
    defs = {
        "GGML_CUDA": "ON", "GGML_NATIVE": "OFF", "BUILD_SHARED_LIBS": "ON", "CMAKE_BUILD_TYPE": "Release",
        "CMAKE_CUDA_ARCHITECTURES": num, "CMAKE_CUDA_COMPILER": n, "CMAKE_CUDA_FLAGS": cuf,
        "CMAKE_INCLUDE_PATH": ";".join(incs), "CMAKE_LIBRARY_PATH": ";".join(libs), "CMAKE_BUILD_RPATH": ";".join(libs),
        "CMAKE_SHARED_LINKER_FLAGS": ldf, "CMAKE_EXE_LINKER_FLAGS": ldf,
        "CMAKE_LIBRARY_OUTPUT_DIRECTORY": os.path.join(build_dir, "bin"), "CMAKE_RUNTIME_OUTPUT_DIRECTORY": os.path.join(build_dir, "bin"),
        "GGML_BUILD_TESTS": "OFF", "GGML_BUILD_EXAMPLES": "OFF",
        "GGML_CCACHE": "OFF",  # ccache would write its cache under $HOME
    }  # fmt: skip
    if full:
        defs.update(
            LLAMA_CURL="OFF", LLAMA_BUILD_TESTS="OFF", LLAMA_BUILD_EXAMPLES="OFF", LLAMA_BUILD_SERVER="OFF", LLAMA_BUILD_TOOLS="OFF"
        )
    env = dict(os.environ)
    if shutil.which("g++") and shutil.which("gcc"):
        env.update(CC="gcc", CXX="g++")
        defs["CMAKE_CUDA_HOST_COMPILER"] = shutil.which("g++")
    out.update(nvcc=n, nvcc_version=nvcc_version(), include=incs, lib=libs)
    logf = open(log, "w") if log else subprocess.DEVNULL
    try:
        cfg = [*cm, "-S", src, "-B", build_dir, *[f"-D{k}={v}" for k, v in defs.items()]]
        if subprocess.run(cfg, stdout=logf, stderr=subprocess.STDOUT, env=env).returncode:
            return {**out, "status": "FAILED at cmake configure (see the log)", "min": round((time.time() - t0) / 60, 1)}
        j = str(jobs or os.cpu_count() or 4)
        b = [*cm, "--build", build_dir, "-j", j, "--target", "ggml", "ggml-cuda"]
        if subprocess.run(b, stdout=logf, stderr=subprocess.STDOUT, env=env).returncode:
            return {**out, "status": "FAILED at build (see the log)", "min": round((time.time() - t0) / 60, 1)}
    finally:
        if log:
            logf.close()
    ok = glob_libs(build_dir)
    return {**out, "status": "ok" if ok else "FAILED: no libggml-cuda.so produced", "min": round((time.time() - t0) / 60, 1)}


def glob_libs(build_dir):
    return os.path.exists(os.path.join(build_dir, "bin", "libggml-cuda.so"))


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(prog="kurn gpu ggml-build", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="llama.cpp checkout, or the kit's bundled ggml sources")
    ap.add_argument("--build", required=True, help="build directory (libraries land in BUILD/bin)")
    ap.add_argument("--arch", default="sm_80")
    ap.add_argument("--jobs", type=int)
    ap.add_argument("--log")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    try:
        r = build(a.src, a.build, a.arch, a.jobs, a.log)
    except cudaenv.CudaToolchainError as e:
        r = {"source": a.src, "dir": a.build, "arch": a.arch, "status": f"FAILED: CUDA toolchain: {str(e).splitlines()[0]}"}
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(r, fh, indent=1)
    print(json.dumps(r, indent=1))
    return 0 if r["status"].startswith("ok") else 1
