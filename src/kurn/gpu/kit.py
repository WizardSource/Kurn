"""Brief on-box tuning for the hand-run kit: per format, pick the KURN kernels the benchmark matrix
will race against the competitors.

    gemv    batch 1, fastest config             (matrix batches 1)
    energy  batch 1, lowest NVML joules/call    (matrix batches 1; the energy-tuned niche)
    cols    multi-column GEMV, batch 8          (matrix batches 2-16, or 2-256 without a GEMM)
    gemm    int8 tensor-core GEMM, batch 64     (matrix batches 16-256; Q8_0, Q4_0, IQ4_NL)

Output: {fmt: {name: [kg_config string, mmin, mmax]}} for `kurn gpu matrix --kernels`.
"""

import json

from .codegen import codegen_keys
from .spec import FORMATS
from .tune import BRIEF, tune

TUNE_SHAPE = (4096, 14336)  # the down projection of the matrix model: K large, N moderate


def config_string(c):
    return " ".join(f"{k}={c[k]}" for k in codegen_keys(c["op"]))


def tune_format(fmt, harness, arch, quick=False, log=print):
    out = {}
    n, k = TUNE_SHAPE
    sample = 10 if quick else 28
    base = {"op": "gemv", "weights": fmt, "target": "cuda", "arch": arch}
    res, _ = tune(base, dict(BRIEF["gemv"]), harness, (n, k, 1), "speed", 0.2, 3, sample=sample, log=log, arch=arch)
    if res:
        out["gemv"] = [config_string(res[0]["config"]), 1, 1]
        e = min((r for r in res[: max(4, len(res))] if r["J"] == r["J"]), key=lambda r: r["J"], default=None)
        if e is not None and e is not res[0]:
            out["energy"] = [config_string(e["config"]), 1, 1]
    cols_space = {**BRIEF["gemv"], "cols": [4, 8], "unroll": [1, 2]}
    res, _ = tune(base, cols_space, harness, (n, k, 8), "speed", 0.2, 3, sample=sample // 2, log=log, arch=arch)
    gemm = FORMATS[fmt]["gemm"]
    if res:
        out["cols"] = [config_string(res[0]["config"]), 2, 16 if gemm else 256]
    if gemm:
        gb = {"op": "gemm", "weights": fmt, "target": "cuda", "arch": arch}
        res, _ = tune(gb, dict(BRIEF["gemm"]), harness, (n, k, 64), "speed", 0.2, 3, sample=sample, log=log, arch=arch)
        if res:
            out["gemm"] = [config_string(res[0]["config"]), 16, 256]
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
    h = a.harness or build_harness(arch)
    tune_all(h, arch, a.formats.split(",") if a.formats else None, a.quick, a.out)
    return 0
