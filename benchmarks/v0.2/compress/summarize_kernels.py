"""Median / min / max over the interleaved reps of bench_kernels.sh (harness CSV rows), with
weights/s, % of the DRAM roofline (cold only: 18.3 GB/s 1T, 128.7 GB/s 8T) and the energy proxy.
    summarize_kernels.py results/kernels.csv"""

import csv
import statistics as st
import sys
from collections import defaultdict

ROOF = {1: 18.3, 8: 128.7}
F = ["label", "kernel", "regime", "threads", "K", "N", "M", "calls", "wall", "cpu", "us", "GBps", "GOPs", "uJ",
     "pJ_MAC", "relerr", "status", "drift"]  # fmt: skip
g = defaultdict(list)
for r in csv.DictReader(open(sys.argv[1]), fieldnames=F):
    g[(r["regime"], int(r["threads"]), r["label"])].append(r)
print("| regime | T | kernel | us/call median (min-max) | G weights/s | GB/s | % roofline | uJ/call (proxy) | reps |")
print("|---|---|---|---|---|---|---|---|---|")
for (reg, t, lab), rs in sorted(g.items(), key=lambda kv: (kv[0][0], kv[0][1], st.median(float(r["us"]) for r in kv[1]))):
    us = [float(r["us"]) for r in rs]
    gb = st.median(float(r["GBps"]) for r in rs)
    k, n = int(rs[0]["K"]), int(rs[0]["N"])
    roof = f"{100 * gb / ROOF[t]:.0f}%" if reg == "cold" and t in ROOF else "-"
    uj = st.median(float(r["uJ"]) for r in rs)
    print(f"| {reg} | {t} | {lab} | {st.median(us):.1f} ({min(us):.1f}-{max(us):.1f}) | {k * n / st.median(us) / 1e3:.1f} | "
          f"{gb:.1f} | {roof} | {uj:.0f} | {len(rs)} |")  # fmt: skip
