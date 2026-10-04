"""Measure the dp4a GEMV register limits in src/kurn/gpu/data/gemv_threads.json (needs nvcc; no GPU).

`__launch_bounds__(tpr * rpb, max(1, minb))` caps registers at 64K / (tpr * rpb * max(1, minb)): 255 up to 256 threads,
128 at 512 and 64 at 1024. Whether a kernel fits under a cap is a ptxas decision that a register estimate does not
predict (uncapped counts swing by +-90 with tpr alone), so the limits are measured: for every (weights, xlayout, sub,
cols, unroll), the largest of 256/512/1024 at which every layout/mins/unpack variant compiles without spill or stack
for every tpr and minb (1, 2, 4) giving that block size on every arch, and 0 when it spills even at 256.

    PATH=/usr/local/cuda/bin:$PATH python tools/gemv_threads.py [--jobs N]
"""

import argparse
import itertools
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from kurn.gpu import spec, toolchain  # noqa: E402

ARCHS = ("sm_80", "sm_90", "sm_100")
TPRS = (8, 16, 32, 64, 128)
# (threads, minb) per effective block size: ptxas schedules __launch_bounds__(128, 4) differently from (512, 1)
TIERS = {t: tuple((t // m, m) for m in (1, 2, 4) if t // m >= 32) for t in (256, 512, 1024)}
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "kurn", "gpu", "data", "gemv_threads.json")


def variants(weights, nt, minb, tpr):
    keys = spec.GEMV_KEYS
    base = {"op": "gemv", "weights": weights, "target": "cuda", "tpr": tpr, "rpb": nt // tpr, "minb": minb}
    names = [k for k in keys if k not in ("tpr", "rpb", "minb")]
    for combo in itertools.product(*(keys[k](base) for k in names)):
        try:
            yield spec.resolve(base, dict(zip(names, combo)))
        except spec.SpecError:
            pass


def key(c):
    return c["weights"], c["xlayout"], c["sub"], c["cols"], c["unroll"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--jobs", type=int, default=None)
    a = ap.parse_args()
    spec.INVALID[:] = [r for r in spec.INVALID if r not in spec.GEMV_SPILL_RULES]
    lim, alive = {}, None  # a tier is measured only for the keys that compiled clean at every smaller one
    for tier, pairs in sorted(TIERS.items()):
        cs = [
            c
            for f in spec.FORMATS
            for nt, m in pairs
            for t in TPRS
            if 1 <= nt // t <= 32
            for c in variants(f, nt, m if m > 1 else 0, t)
            if alive is None or key(c) in alive
        ]
        print(f"block size {tier}: {len(cs)} configurations x {len(ARCHS)} archs", file=sys.stderr)
        spills = {}
        for c, rep in toolchain.build_many(cs, lambda c: toolchain.ptxas_report(c, ARCHS), a.jobs):
            if isinstance(rep, Exception):
                raise SystemExit(f"{spec.label(c)}: {rep}")
            spills[key(c)] = spills.get(key(c), False) or any(r["spill"] for r in toolchain.resource_rows(c, rep))
        for k, sp in spills.items():
            lim.setdefault(k, 0)
            if not sp:
                lim[k] = tier
        alive = {k for k, sp in spills.items() if not sp}
    limits, gk = {}, spec.GEMV_KEYS
    for w in spec.FORMATS:
        p = {"op": "gemv", "weights": w}
        for xl, sub, cols, unroll in itertools.product(gk["xlayout"](p), gk["sub"](p), gk["cols"](p), gk["unroll"](p)):
            row = limits.setdefault(spec.gemv_threads_key(w, xl, sub, cols), {})
            row[str(unroll)] = lim.get((w, xl, sub, cols, unroll), 0)
    meta = {
        "nvcc": toolchain.nvcc_version(),
        "archs": list(ARCHS),
        "tpr": list(TPRS),
        "threads_minb": {str(k): [list(p) for p in v] for k, v in TIERS.items()},
    }
    lines = [f"  {json.dumps(k)}: {json.dumps(v, sort_keys=True)}" for k, v in sorted(limits.items())]
    with open(OUT, "w") as fh:
        fh.write('{\n "measured_with": ' + json.dumps(meta) + ',\n "limits": {\n' + ",\n".join(lines) + "\n }\n}\n")
    print(f"wrote {os.path.normpath(OUT)}", file=sys.stderr)


if __name__ == "__main__":
    main()
