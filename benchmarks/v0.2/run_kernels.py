#!/usr/bin/env python3
"""kurn v0.2 kernel experiments (results -> benchmarks/v0.2/results/*.csv).

  run_kernels.py tune        per format: tune the algorithm/schedule space, hot 1T and cold 8T, energy objective
  run_kernels.py q4k         Q4_K algorithm space: native v1/v2 keys x rows x prefetch, plus i16 options (hot 1T)
  run_kernels.py lut         <= 2-bit: lookup tables (l32) vs unpack + VNNI (i16), hot 1T and cold 8T
  run_kernels.py verify      multi-token verify: time per pass vs M = 1..8 (cold 8T)
  run_kernels.py cliff       cache-cliff sweep: effective GB/s vs total weight footprint (8T, static rows)
Every configuration is checked against the exact reference by the harness; failures abort.
"""
import csv
import os
import sys

from kurn import spec, toolchain
from kurn.harness import bench

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)
W = 5.47
GEMV_FORMATS = ("q8_0", "q4_0", "iq4_nl", "q4_K", "q2_0", "tq2_0", "q1_0")
KEYS = ("layout", "align", "rows", "cols", "act", "prefetch", "unpack", "correction", "scales", "accum", "threads")


def run(c, regime, secs=0.6, extra=()):
    so = toolchain.build(c)
    r = bench(so, c, regime, secs, list(extra))
    if not r or r.get("check") != "ok":
        raise SystemExit(f"FAILED correctness: {c} -> {r}")
    us = float(r["us_per_call"])
    cpu_us = float(r["cpu_s"]) / float(r["calls"]) * 1e6
    return {"us": us, "cpu_us": cpu_us, "energy_uJ": cpu_us * W, "GBps": float(r["GBps"]), "relerr": float(r["relerr"])}


def write(name, rows):
    if not rows:
        return
    p = os.path.join(OUT, name)
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote", p)


def space(fmt, target="avx512_vnni"):
    """Every legal GEMV codegen config for one format on one target (prefetch 0/4/8)."""
    return [c for c in spec.legal_configs(op="gemv", weights=fmt, target=target, prefetch=(0, 4, 8))]


def tune():
    rows = []
    for fmt in GEMV_FORMATS:
        for regime, T in (("hot", 1), ("cold", 8)):
            for c in space(fmt):
                c = dict(c, threads=T)
                m = run(c, regime)
                rows.append({"format": fmt, "regime": regime, "threads": T, **{k: c[k] for k in KEYS}, **m})
                print(fmt, regime, {k: c[k] for k in KEYS if k != "threads"}, f"{m['us']:.1f} us {m['energy_uJ']:.0f} uJ", flush=True)
    write("tune_gemv.csv", rows)


def q4k():
    rows = []
    for c in spec.legal_configs(op="gemv", weights="q4_K", target="avx512_vnni", prefetch=(0, 4)):
        c = dict(c, threads=1)
        m = run(c, "hot", 0.8)
        rows.append({**{k: c[k] for k in KEYS}, **m})
        print({k: c[k] for k in KEYS}, f"{m['us']:.2f} us", flush=True)
    write("q4k_algorithm_space.csv", rows)


def lut():
    rows = []
    for fmt in ("q1_0", "q2_0", "tq2_0"):
        for regime, T in (("hot", 1), ("cold", 8)):
            for c in space(fmt):
                if c["layout"] not in ("i16", "l32"):
                    continue
                c = dict(c, threads=T)
                m = run(c, regime)
                rows.append({"format": fmt, "regime": regime, **{k: c[k] for k in KEYS}, **m})
                print(fmt, regime, c["layout"], c["accum"], c["rows"], c["prefetch"], f"{m['us']:.1f} us", flush=True)
    write("lut_vs_vnni.csv", rows)


def verify():
    rows = []
    for fmt in ("q8_0", "q4_0", "q4_K", "q1_0"):
        g = spec.resolve({"op": "gemv", "weights": fmt, "target": "avx512_vnni", "layout": "i16", "rows": 1, "threads": 8})
        m1 = run(g, "cold", 1.0)
        rows.append({"format": fmt, "M": 1, "kernel": "gemv", **m1})
        for cols, M in ((2, 2), (4, 3), (4, 4), (8, 6), (8, 8)):
            c = spec.resolve({"op": "verify", "weights": fmt, "target": "avx512_vnni", "cols": cols, "rows": 1, "threads": 8})
            m = run(c, "cold", 1.0, ["--M", str(M)])
            rows.append({"format": fmt, "M": M, "kernel": f"verify cols={cols}", **m})
            print(fmt, M, f"{m['us']:.1f} us vs gemv {m1['us']:.1f}", flush=True)
    write("verify_vs_M.csv", rows)


def cliff():
    rows = []
    c = spec.resolve({"op": "gemv", "weights": "q8_0", "target": "avx512_vnni", "layout": "vnni16", "rows": 8, "threads": 8})
    for mb in (4, 8, 16, 32, 64, 128, 192, 256, 320, 384, 512, 768, 1200):
        m = run(c, "cold", 1.0, ["--N", "1024", "--footprint", str(mb)])
        rows.append({"footprint_MB": mb, **m})
        print(mb, f"{m['GBps']:.0f} GB/s", flush=True)
    write("cache_cliff.csv", rows)


if __name__ == "__main__":
    {"tune": tune, "q4k": q4k, "lut": lut, "verify": verify, "cliff": cliff}[sys.argv[1]]()
