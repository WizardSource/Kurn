"""The benchmark matrix: every format x batch size, KURN against every installed competitor,
interleaved in one harness process (see data/bench_gpu.cu, `matrix` command).

Workload: the weight matmuls of one Llama-3-8B decoder layer (fused QKV, O, fused gate+up, down),
times 32 layers, so tokens/s = batch / (32 x the sum of the four matmul times).
"""

import os
import subprocess

from .spec import resolve
from .toolchain import nvcc_build

MODEL = {"name": "Llama-3-8B (matmuls only)", "layers": 32,
         "shapes": ((6144, 4096), (4096, 4096), (28672, 4096), (4096, 14336))}  # fmt: skip
BATCHES = (1, 4, 16, 64, 256)
COMPETITORS = ("ggml", "cublas-fp16", "cublas-int8")


# Tensor-core engine tiles per batch range: (name, batch range, preferred overrides, fallbacks if illegal for the format)
ENGINE_TILES = (
    ("mma8", (1, 8), [{"bn": 8, "bm": 64, "wm": 4, "wn": 1, "xin": "f32", "stages": 4}]),
    ("mma16", (9, 16), [{"bn": 16, "bm": 64, "wm": 4, "wn": 1, "xin": "f32", "stages": 4},
                        {"bn": 16, "bm": 64, "wm": 4, "wn": 1, "xin": "f16", "stages": 3}]),
    ("mma64", (17, 64), [{"bn": 64, "bm": 128, "wm": 4, "wn": 2, "xin": "f16", "bk": 128, "stages": 3},
                         {"bn": 64, "bm": 64, "wm": 2, "wn": 2, "xin": "f16", "bk": 128, "stages": 3},
                         {"bn": 64, "bm": 64, "wm": 2, "wn": 2, "xin": "f16", "stages": 2}]),
    ("mma256", (65, 256), [{"bn": 128, "bm": 128, "wm": 2, "wn": 4, "xin": "f16", "bk": 128, "stages": 3},
                           {"bn": 128, "bm": 128, "wm": 2, "wn": 4, "xin": "f16", "bk": 64, "stages": 3},
                           {"bn": 128, "bm": 128, "wm": 2, "wn": 4, "xin": "f16", "stages": 2},
                           {"bn": 64, "bm": 128, "wm": 4, "wn": 2, "xin": "f16", "bk": 128, "stages": 3},
                           {"bn": 64, "bm": 64, "wm": 2, "wn": 2, "xin": "f16", "stages": 2}]),
)  # fmt: skip


def engine_tile(fmt, arch, options):
    from ..spec import SpecError

    for ov in options:
        try:
            return resolve({"op": "gemm", "weights": fmt, "arch": arch, **ov})
        except SpecError:
            continue
    return None


def default_kernels(fmt, arch):
    """KURN's default kernels for a format: {name: (config, mmin, mmax)}; names start with `default-`."""
    out = {"default-gemv": (resolve({"op": "gemv", "weights": fmt, "arch": arch}), 1, 1)}
    for name, (lo, hi), options in ENGINE_TILES:
        c = engine_tile(fmt, arch, options)
        if c is not None:
            out[f"default-{name}"] = (c, lo, hi)
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


def build_kernels(configs, log=print, fallback=False):
    """{fmt: {name: (config, lo, hi)}} -> {fmt: {name: (lib, lo, hi)}} (nvcc, parallel)."""
    from .toolchain import build_many

    flat = [(f, n, c, lo, hi) for f, d in configs.items() for n, (c, lo, hi) in d.items()]
    built = build_many([c for _, _, c, _, _ in flat], lambda c: nvcc_build(c, fallback=fallback)[0])
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
