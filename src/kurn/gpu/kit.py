"""Brief on-box tuning for the hand-run kit: per format, pick the KURN kernels the benchmark matrix
will race against the competitors.

    tuned-gemv     dp4a GEMV, batch 1, fastest config          (matrix batch 1)
    tuned-energy   dp4a GEMV, batch 1, lowest NVML joules/call (matrix batch 1; the energy-tuned niche)
    tuned-mma8     tensor-core engine, bn=8, tuned at batch 1  (matrix batches 1-8)
    tuned-mma16    tensor-core engine, bn=16, batch 16          (9-16)
    tuned-mma64    tensor-core engine, bn=64, batch 64          (17-64)
    tuned-mma256   tensor-core engine, bn=64/128, batch 256     (65-256)
The matrix also runs every `default-*` kernel, so the report shows tuned and default side by side.

Output: {fmt: {name: [kg_config string, mmin, mmax]}} for `kurn gpu matrix --kernels`.
"""

import json

from .codegen import codegen_keys
from .spec import FORMATS
from .tune import BRIEF, tune

TUNE_SHAPE = (4096, 14336)  # the down projection of the matrix model: K large, N moderate


def config_string(c):
    return " ".join(f"{k}={c[k]}" for k in codegen_keys(c["op"]))


# Engine tune spaces per batch range (the matrix name, the batch it is tuned at, the space)
ENGINE_SPACES = {
    "mma8": (1, {"bn": [8], "bm": [32, 64, 128], "wm": [1, 2, 4], "wn": [1], "bk": [128, 256], "stages": [3, 4, 5],
                 "splitk": [0], "xin": ["f32"]}),
    "mma16": (16, {"bn": [16], "bm": [32, 64, 128], "wm": [1, 2, 4], "wn": [1, 2], "bk": [128, 256], "stages": [3, 4],
                   "xin": ["f32", "f16"]}),
    "mma64": (64, {"bn": [64], "bm": [64, 128], "wm": [2, 4], "wn": [1, 2], "bk": [64, 128, 256], "stages": [2, 3, 4],
                   "xin": ["f16"]}),
    "mma256": (256, {"bn": [64, 128], "bm": [128, 256], "wm": [2, 4], "wn": [2, 4], "bk": [64, 128, 256], "stages": [2, 3],
                     "xin": ["f16"]}),
}  # fmt: skip
RANGES = {"mma8": (1, 8), "mma16": (9, 16), "mma64": (17, 64), "mma256": (65, 256)}


def tune_format(fmt, harness, arch, quick=False, log=print):
    """Brief tuning of every kernel the matrix races for this format; names start with `tuned-`."""
    out = {}
    n, k = TUNE_SHAPE
    sample = 8 if quick else 20
    base = {"op": "gemv", "weights": fmt, "target": "cuda", "arch": arch}
    res, _ = tune(base, dict(BRIEF["gemv"]), harness, (n, k, 1), "speed", 0.2, 3, sample=sample, log=log, arch=arch)
    if res:
        out["tuned-gemv"] = [config_string(res[0]["config"]), 1, 1]
        e = min((r for r in res if r["J"] == r["J"]), key=lambda r: r["J"], default=None)
        if e is not None and e is not res[0]:
            out["tuned-energy"] = [config_string(e["config"]), 1, 1]
    gb = {"op": "gemm", "weights": fmt, "target": "cuda", "arch": arch}
    for name, (m, space) in ENGINE_SPACES.items():
        if quick and name in ("mma16", "mma64"):
            continue
        res, _ = tune(gb, dict(space), harness, (n, k, m), "speed", 0.2, 3, sample=sample, log=log, arch=arch)
        if res:
            lo, hi = RANGES[name]
            if quick and name == "mma8":
                hi = 16
            if quick and name == "mma256":
                lo = 17
            out[f"tuned-{name}"] = [config_string(res[0]["config"]), lo, hi]
    return out


def tune_all(harness, arch, formats=None, quick=False, out=None, log=print):
    tuned = {}
    for f in formats or list(FORMATS):
        log(f"== tune {f}")
        tuned[f] = tune_format(f, harness, arch, quick, log)
        if out:
            with open(out, "w") as fh:
                json.dump(tuned, fh, indent=1)
    return tuned


def main(argv=None):
    import argparse

    from .harness import build_harness, detect_arch

    ap = argparse.ArgumentParser(prog="kurn gpu kit-tune")
    ap.add_argument("--out", default="tuned.json")
    ap.add_argument("--formats")
    ap.add_argument("--harness")
    ap.add_argument("--arch")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args(argv)
    arch = a.arch or detect_arch() or "sm_80"
    h = a.harness or build_harness(arch, explicit=bool(a.arch))
    tune_all(h, arch, a.formats.split(",") if a.formats else None, a.quick, a.out)
    return 0
