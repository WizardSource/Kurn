"""Sparse SwiGLU FFN (kurn.sparse) vs dense: time per FFN at b = 1 on Qwen3-1.7B shapes.

    benchlock.sh python bench_sparse.py [--threads 8] [--reps 3] [--out results/sparse_ffn.csv]

Rows: kurn.sparse dense / gate / pred(r=128) at densities 1.0 .. 0.1 (random active rows,
cold weights), and kurn's dense vnni16 Q8_0 GEMV for gate+up (6144x2048) and down (2048x6144),
the dense baseline (sum of the three GEMVs; no SwiGLU / barrier cost added to it).
"""

import argparse
import csv
import os
import statistics

from kurn import sparse
from kurn.harness import bench as kbench
from kurn.spec import load, resolve
from kurn.toolchain import build

HERE = os.path.dirname(os.path.abspath(__file__))
SPEC = os.path.join(HERE, "..", "..", "..", "..", "examples", "q8_0_gemv_vnni16.kurn")


def vnni16_ffn_us(threads, secs):
    spec, _ = load(SPEC)
    c = resolve(spec, {"threads": threads})
    so = build(c)
    up = kbench(so, c, "cold", secs, ["--K", "2048", "--N", "6144"])
    down = kbench(so, c, "cold", secs, ["--K", "6144", "--N", "2048"])
    return 2 * up["us"] + down["us"], 2 * up["cpu_us"] + down["cpu_us"], up, down


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--secs", type=float, default=0.5)
    ap.add_argument("--out", default=os.path.join(HERE, "..", "results", "sparse_ffn.csv"))
    a = ap.parse_args()
    cases = [("dense", None, 0)] + [(m, dens, r) for dens in (1.0, 0.7, 0.5, 0.3, 0.1) for m, r in (("gate", 0), ("pred", 128))]
    raw = []
    for rep in range(a.reps):
        us, cpu, up, down = vnni16_ffn_us(a.threads, a.secs)
        raw.append({"impl": "vnni16-dense", "density": 1.0, "rep": rep, "us": us, "cpu_us": cpu, "GBps": 3 * 13.37e6 / (us * 1e3)})
        print(f"{rep} vnni16 dense FFN {us:9.1f} us  (up {up['us']:.1f}, down {down['us']:.1f} us, {up['GBps']:.1f} GB/s)", flush=True)
        for mode, dens, r in cases:
            row = sparse.bench(mode, density=dens, rank=r or 128, threads=a.threads, regime="cold", secs=a.secs)
            name = f"sparse-{mode}" + (f"-r{r}" if r else "")
            raw.append({"impl": name, "density": row["density"], "rep": rep, "us": row["us_per_call"], "cpu_us": row["cpu_us"],
                        "GBps": row["GBps"], "check": row["check"]})  # fmt: skip
            us = row["us_per_call"]
            print(f"{rep} {name:16} density {row['density']:.3f} {us:9.1f} us {row['GBps']:7.1f} GB/s {row['check']}", flush=True)
    keys = sorted({(r["impl"], round(r["density"], 1)) for r in raw})
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["impl", "density", "us_med", "us_min", "us_max", "GBps_med", "uJ_med", "load"])
        load_avg = open("/proc/loadavg").read().split()[0]
        for impl, dens in keys:
            rs = [r for r in raw if r["impl"] == impl and round(r["density"], 1) == dens]
            us = [r["us"] for r in rs]
            w.writerow([impl, dens, round(statistics.median(us), 1), round(min(us), 1), round(max(us), 1),
                        round(statistics.median(r["GBps"] for r in rs), 1),
                        round(statistics.median(r["cpu_us"] for r in rs) * 5.47, 1), load_avg])  # fmt: skip
    print("wrote", a.out)


if __name__ == "__main__":
    main()
