#!/usr/bin/env python3
"""Median / min-max of results/e2e_bench.csv per (model, config, n_gen), plus the ratio to ggml plain.
summarize_e2e.py [--md]"""

import csv
import os
import statistics
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
rows = list(csv.DictReader(open(os.path.join(HERE, "results", "e2e_bench.csv"))))
g = defaultdict(list)
for r in rows:
    try:
        g[(r["model"], r["n_gen"], r["config"])].append(float(r["avg_ts"]))
    except ValueError:
        pass
md = "--md" in sys.argv
if md:
    print("| model | n_gen x reps | config | tok/s median | min–max | n | vs ggml plain |")
    print("|---|---|---|---|---|---|---|")
for (m, n, c), v in sorted(g.items()):
    base = g.get((m, n, "ggml plain (repack=0)"))
    med = statistics.median(v)
    ratio = med / statistics.median(base) if base else float("nan")
    if md:
        print(f"| {m} | {n} | {c} | {med:.1f} | {min(v):.1f}–{max(v):.1f} | {len(v)} | {ratio:.2f}× |")
    else:
        print(f"{m:24s} {n:12s} {c:52s} {med:7.1f}  [{min(v):6.1f}-{max(v):6.1f}] n={len(v)}  x{ratio:.2f}")
