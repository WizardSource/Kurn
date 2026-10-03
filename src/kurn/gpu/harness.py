"""Build and drive the GPU harness (data/bench_gpu.cu) and detect the local GPU."""

import glob
import json
import os
import shutil
import subprocess

from .toolchain import GpuBuildError, _out_dir, _run, _sha, arch_flags, data_path, nvcc, nvcc_version


class HarnessError(Exception):
    pass


def nvidia_smi(*query):
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    r = subprocess.run([exe, *query], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None


def gpu_present():
    out = nvidia_smi("-L")
    return bool(out and "GPU" in out)


def detect_arch():
    """sm_XX of GPU 0 (respecting CUDA_VISIBLE_DEVICES ordering), or None."""
    out = nvidia_smi("--query-gpu=compute_cap", "--format=csv,noheader")
    if not out:
        return None
    vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
    idx = int(vis.split(",")[0]) if vis and vis.split(",")[0].isdigit() and int(vis.split(",")[0]) < len(lines) else 0
    major, _, minor = lines[idx].partition(".")
    return f"sm_{major}{minor}"


def ggml_paths(llama_dir):
    """(include dirs, lib dir) of a llama.cpp checkout built with -DGGML_CUDA=ON, or None."""
    if not llama_dir:
        return None
    inc = os.path.join(llama_dir, "ggml", "include")
    libs = glob.glob(os.path.join(llama_dir, "build*", "bin", "libggml-cuda.so")) + \
        glob.glob(os.path.join(llama_dir, "build*", "ggml", "src", "ggml-cuda", "libggml-cuda.so"))  # fmt: skip
    if not (os.path.isdir(inc) and libs):
        return None
    return [inc], os.path.dirname(sorted(libs)[0])


def build_harness(arch, llama_dir=None, out_dir=None):
    """nvcc the harness for `arch` (optionally linked with llama.cpp's ggml-cuda). Returns its path."""
    n = nvcc()
    if not n:
        raise GpuBuildError("nvcc not found (install the CUDA toolkit or set KURN_NVCC)")
    names = ("bench_gpu.cu", "kurn_gpu.h", "kurn_gpu_ref.h")
    src = b"".join(open(data_path(x), "rb").read() for x in names)
    args = [n, "-O3", "-std=c++17", *arch_flags([arch]), "-I", os.path.dirname(data_path("kurn_gpu.h"))]
    link = ["-lcublas", "-ldl"]
    gp = ggml_paths(llama_dir)
    if llama_dir and not gp:
        raise GpuBuildError(f"{llama_dir}: no ggml/include or build*/bin/libggml-cuda.so (build llama.cpp with -DGGML_CUDA=ON)")
    if gp:
        incs, libdir = gp
        for i in incs:
            args += ["-I", i]
        args += ["-DKURN_GGML"]
        libs = [f"-l{os.path.basename(p)[3:-3]}" for p in sorted(glob.glob(os.path.join(libdir, "libggml*.so")))]
        link += ["-L", libdir, *libs, f"-Xlinker=-rpath={libdir}", "-Xlinker=--allow-shlib-undefined"]
    h = _sha(src, n, nvcc_version(), " ".join(args + link))
    d = out_dir or _out_dir("harness")
    os.makedirs(d, exist_ok=True)
    exe = os.path.join(d, f"bench_gpu_{arch}{'_ggml' if gp else ''}_{h}")
    if not os.path.exists(exe):
        tmp = f"{exe}.tmp{os.getpid()}"
        _run([*args, data_path("bench_gpu.cu"), "-o", tmp, *link], "harness build")
        os.replace(tmp, exe)
    return exe


def run_json(cmd, timeout=None):
    """Run the harness; returns its JSON lines (errors raise HarnessError)."""
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    rows = []
    for ln in r.stdout.splitlines():
        ln = ln.strip()
        if ln.startswith("{"):
            rows.append(json.loads(ln))
    err = next((x for x in rows if x.get("kind") == "error"), None)
    if err or (r.returncode and not rows):
        raise HarnessError((err or {}).get("error") or r.stderr.strip()[-2000:] or f"exit {r.returncode}")
    return rows


def info(harness):
    return run_json([harness, "info"])[0]


def roofline(harness, secs=3.0):
    return run_json([harness, "roofline", "--secs", str(secs)])[0]


def run_kernel(harness, lib, fmt, n, k, m, reps=3, secs=0.3, quant=0, cold=1, seed=1):
    rows = run_json([harness, "run", "--lib", lib, "--fmt", fmt, "--N", str(n), "--K", str(k), "--M", str(m), "--reps", str(reps),
                     "--secs", str(secs), "--quant", str(quant), "--cold", str(cold), "--seed", str(seed)], timeout=600)  # fmt: skip
    check = next(x for x in rows if x["kind"] == "check")
    samples = [x for x in rows if x["kind"] == "sample"]
    return check, samples
