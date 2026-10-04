#!/usr/bin/env python3
"""Median-of-reps tables (markdown) from rtbench CSVs written by run_matrix.py.

    summarize.py results/partition-cold.csv [--base v0.1-static-sync] [--ggml ggml-graph+sched] [--platform-w 10]

Labels look like `T8:olmoe-moe:balanced#r0`; rows are grouped by the part before `#`
and compared with the baseline config of the same `T8:workload` prefix.
"""

import argparse
import csv
import statistics
from collections import defaultdict


def load(paths):
    rows = []
    for p in paths:
        with open(p) as fh:
            rows += list(csv.DictReader(fh))
    return rows


def groups(rows):
    g = defaultdict(list)
    for r in rows:
        g[r["label"].split("#")[0]].append(r)
    return g


def med(rs, k):
    return statistics.median(float(r[k]) for r in rs)


def sigma_rel(rs, k="us_med"):
    v = [float(r[k]) for r in rs]
    return statistics.stdev(v) / statistics.mean(v) if len(v) > 1 else float("nan")


def verdict(ratio, s_a, s_b):
    """Brief rule: a win needs ratio > 1 + 2 sigma_rel (both configs' sigmas combined) and > 5 %."""
    s = (s_a**2 + s_b**2) ** 0.5
    if ratio > 1 + max(2 * s, 0.05):
        return "win"
    if ratio < 1 - max(2 * s, 0.05):
        return "loss"
    return "n.r."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--base", default="v0.1-static-sync,v0.1-static")
    ap.add_argument("--ggml", default="ggml-graph+sched,ggml")
    ap.add_argument("--platform-w", type=float, default=10.0)
    ap.add_argument("--roofline", type=float, default=128.7, help="GB/s, 8-core read roofline")
    a = ap.parse_args()
    g = groups(load(a.csv))
    bases, ggmls = a.base.split(","), a.ggml.split(",")
    print(
        f"| config | reps | µs/layer (med) | σ | MB/layer | GB/s | % roof | spin share | cpu/wall | mJ/layer proxy | +{a.platform_w:g} W "
        "| model tok/s | vs v0.1 | vs ggml | max relerr | max load | max drift |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for label in g:
        rs = g[label]
        prefix = label.rsplit(":", 1)[0]

        def ref(names, prefix=prefix):
            for n in names:
                if f"{prefix}:{n}" in g:
                    return g[f"{prefix}:{n}"]
            return None

        us, s = med(rs, "us_med"), sigma_rel(rs)

        def cmp(other, us=us, s=s):
            if not other:
                return "–"
            r = med(other, "us_med") / us
            return f"{r:.3f}x {verdict(r, s, sigma_rel(other))}"

        mj = med(rs, "proxy_mJ_tok")
        gbps = med(rs, "GBps")
        print(
            f"| {label} | {len(rs)} | {us:.1f} | {100 * s:.1f}% | {med(rs, 'bytes_per_tok') / 1e6:.2f} | {gbps:.1f} "
            f"| {100 * gbps / a.roofline:.0f}% | {100 * med(rs, 'spin_share'):.1f}% "
            f"| {med(rs, 'cpu_wall'):.2f} | {mj:.3f} | {mj + a.platform_w * us * 1e-3:.3f} "
            f"| {med(rs, 'model_tok_s'):.1f} | {cmp(ref(bases))} | {cmp(ref(ggmls))} "
            f"| {max(float(r['relerr']) for r in rs):.1e} | {max(float(r['load']) for r in rs):.1f} "
            f"| {max(abs(float(r['drift_s'])) for r in rs):.3f} |"
        )


if __name__ == "__main__":
    main()
