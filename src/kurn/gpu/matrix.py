"""The benchmark matrix: every format x batch size, KURN against every installed competitor,
interleaved in one harness process (see data/bench_gpu.cu, `matrix` command).

Workload: the weight matmuls of one Llama-3-8B decoder layer (fused QKV, O, fused gate+up, down),
times 32 layers, so tokens/s = batch / (32 x the sum of the four matmul times).
"""

import os
import subprocess

from .spec import FORMATS, resolve
from .toolchain import nvcc_build

MODEL = {"name": "Llama-3-8B (matmuls only)", "layers": 32,
         "shapes": ((6144, 4096), (4096, 4096), (28672, 4096), (4096, 14336))}  # fmt: skip
BATCHES = (1, 4, 16, 64, 256)
COMPETITORS = ("ggml", "cublas-fp16", "cublas-int8")


def default_kernels(fmt, arch):
    """KURN kernels for a format before (or without) tuning: {name: (config, mmin, mmax)}."""
    gemm = FORMATS[fmt]["gemm"]
    out = {
        "gemv": (resolve({"op": "gemv", "weights": fmt, "arch": arch}), 1, 1),
        "cols": (resolve({"op": "gemv", "weights": fmt, "arch": arch, "cols": 8, "rpb": 2}), 2, 16 if gemm else 256),
    }
    if gemm:
        out["gemm"] = (resolve({"op": "gemm", "weights": fmt, "arch": arch}), 16, 256)
    return out


def write_plan(path, kernels, formats, shapes=MODEL["shapes"], batches=BATCHES, competitors=COMPETITORS, reps=5, secs=0.5,
               cold=1, quant=1, seed=1):  # fmt: skip
    """kernels: {fmt: {name: (lib path, mmin, mmax)}}."""
    lines = ["# kurn gpu matrix plan", f"reps {reps}", f"secs {secs}", f"cold {cold}", f"quant {quant}", f"seed {seed}"]
    for f in formats:
        lines.append(f"fmt {f}")
        for name, (lib, lo, hi) in kernels.get(f, {}).items():
            lines.append(f"kurn {name} {lib} {lo} {hi}")
        lines.append("shapes " + " ".join(f"{n}x{k}" for n, k in shapes))
        lines.append("batches " + " ".join(str(b) for b in batches))
        lines.append("competitors " + " ".join(competitors))
        lines.append("end")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


def build_kernels(configs, log=print):
    """{fmt: {name: (config, lo, hi)}} -> {fmt: {name: (lib, lo, hi)}} (nvcc, parallel)."""
    from .toolchain import build_many

    flat = [(f, n, c, lo, hi) for f, d in configs.items() for n, (c, lo, hi) in d.items()]
    built = build_many([c for _, _, c, _, _ in flat], lambda c: nvcc_build(c)[0])
    out = {}
    for (f, n, _c, lo, hi), (_, lib) in zip(flat, built):
        if isinstance(lib, Exception):
            log(f"FAIL build {f} {n}: {str(lib).splitlines()[0]}")
            continue
        out.setdefault(f, {})[n] = (lib, lo, hi)
    return out


def run(harness, plan, out, log_path=None):
    """Run the harness on a plan, appending JSON lines to `out`. Returns the exit code."""
    if os.path.exists(out):
        os.remove(out)
    with open(log_path or os.devnull, "w") as lg:
        r = subprocess.run([harness, "matrix", "--plan", plan, "--out", out], stdout=lg, stderr=subprocess.STDOUT)
    return r.returncode
