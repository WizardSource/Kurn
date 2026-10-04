#!/usr/bin/env python3
"""Summarize bench.py CSVs: per config the median and [min-max] over reps of decode tok/s (mean
over the run), tok/s from the median and p10 per-token latency, J/token (5.47 W x CPU s) and
J/token + 10 W platform, barriers/token, wait share (all tokens / faster half), runq share, foreign CPU.

  summarize.py results/gen_qwen3.csv [...] [--md]
"""

import csv
import statistics
import sys


def rng(vals, fmt):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "-"
    return f"{fmt.format(statistics.median(vals))} [{fmt.format(min(vals))}-{fmt.format(max(vals))}]"


def f(r, k):
    v = r.get(k, "")
    return float(v) if v not in ("", None) else None


def main():
    md = "--md" in sys.argv
    for path in [a for a in sys.argv[1:] if not a.startswith("--")]:
        rows = list(csv.DictReader(open(path)))
        by = {}
        for r in rows:
            by.setdefault(r["config"], []).append(r)
        hdr = [
            "config",
            "n",
            "tok/s (mean)",
            "tok/s (median lat.)",
            "tok/s (p10 lat.)",
            "J/tok",
            "J/tok +10W",
            "barriers/tok",
            "wait",
            "quiet wait",
            "runq",
            "foreign cores",
        ]
        print(f"\n{path}")
        if md:
            print("| " + " | ".join(hdr) + " |")
            print("|" + "---|" * len(hdr))
        for name, rs in by.items():
            inv = lambda k, rs=rs: [1000.0 / f(r, k) if f(r, k) else None for r in rs]  # noqa: E731
            cells = [
                name,
                str(len(rs)),
                rng([f(r, "decode_tok_s") for r in rs], "{:.1f}"),
                rng(inv("med_ms"), "{:.1f}"),
                rng(inv("p10_ms"), "{:.1f}"),
                rng([f(r, "J_per_tok") for r in rs], "{:.3f}"),
                rng([f(r, "J_per_tok_10W") for r in rs], "{:.3f}"),
                rng([f(r, "barriers_per_tok") for r in rs], "{:.0f}"),
                rng([f(r, "wait_share") for r in rs], "{:.3f}"),
                rng([f(r, "quiet_wait_share") for r in rs], "{:.3f}"),
                rng([f(r, "runq_share") for r in rs], "{:.3f}"),
                rng([f(r, "foreign_cpu") for r in rs], "{:.2f}"),
            ]
            print(("| " + " | ".join(cells) + " |") if md else "  ".join(cells))


if __name__ == "__main__":
    main()
