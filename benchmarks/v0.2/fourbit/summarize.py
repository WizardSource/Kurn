#!/usr/bin/env python3
"""Markdown tables from results/ (bench sets, AMX, quality, end-to-end, tuner) for fourbit.md.
summarize.py [results-dir]
"""

import csv
import glob
import os
import re
import statistics
import sys

W_PER_CORE = 5.47
ROOF = {1: 18.3, 8: 128.7}


def bench_table(path):
    rows = list(csv.DictReader(open(path)))
    if not rows:
        return ""
    order, by = [], {}
    sroof = [float(r["GBps"]) for r in rows if r["name"].startswith("read roofline")]
    sroof = statistics.median(sroof) if sroof else None
    rows = [r for r in rows if not r["name"].startswith("read roofline")]
    for r in rows:
        if r["name"] not in by:
            order.append(r["name"])
        by.setdefault(r["name"], {}).setdefault((r["regime"], int(r["threads"])), []).append(r)
    q8 = by.get("q8_0 vnni16 a64 r8", {})
    out = ([f"In-session read roofline (8T, median): {sroof:.1f} GB/s; '% sess' is relative to it.", ""] if sroof else []) + [
        "| config | hot 1T us | hot 1T uJ | DRAM 8T us | GB/s | % roof | % sess | uJ/call | spread | vs Q8_0 (8T) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name in order:
        cells = []
        for key in (("hot", 1), ("cold", 8)):
            rs = by[name].get(key, [])
            if not rs:
                cells.append(None)
                continue
            us = statistics.median(float(r["us"]) for r in rs)
            gb = statistics.median(float(r["GBps"]) for r in rs)
            cpu = statistics.median(float(r["cpu_us"]) for r in rs)
            spread = (max(float(r["us"]) for r in rs) - min(float(r["us"]) for r in rs)) / us
            cells.append((us, gb, cpu * W_PER_CORE, spread, len(rs)))
        h, c = cells
        ref = q8.get(("cold", 8))
        ratio = ""
        if c and ref:
            ratio = f"{statistics.median(float(r['us']) for r in ref) / c[0]:.2f}x"
        out.append(
            f"| {name} | {f'{h[0]:.2f}' if h else '-'} | {f'{h[2]:.1f}' if h else '-'} | "
            + (
                f"{c[0]:.1f} | {c[1]:.1f} | {100 * c[1] / ROOF[8]:.0f}% | "
                f"{f'{100 * c[1] / sroof:.0f}%' if sroof else '-'} | {c[2]:.0f} | {100 * c[3]:.0f}% | {ratio} |"
                if c
                else "- | - | - | - | - | - | |"
            )  # fmt: skip
        )
    return "\n".join(out)


def amx_table(path):
    pat = re.compile(r"rep\d (\w+)\s+M=\s*(\d+)\s+([\d.]+) us/call\s+([\d.]+) GMAC/s.*relerr (\S+) (\w+)")
    res = {}
    for line in open(path):
        m = pat.search(line)
        if m:
            res.setdefault((int(m[2]), m[1]), []).append((float(m[3]), float(m[4]), m[6]))
    out = ["| M | AMX us | VNNI us | AMX / VNNI time | AMX GMAC/s | checks |", "|---|---|---|---|---|---|"]
    for M in sorted({k[0] for k in res}):
        a, v = res.get((M, "amx"), []), res.get((M, "vnni"), [])
        if not a or not v:
            continue
        au, vu = statistics.median(x[0] for x in a), statistics.median(x[0] for x in v)
        ok = all(x[2] == "ok" for x in a + v)
        out.append(f"| {M} | {au:.1f} | {vu:.1f} | {au / vu:.2f} | {statistics.median(x[1] for x in a):.0f} | "
                   f"{'ok' if ok else 'FAIL'} |")  # fmt: skip
    return "\n".join(out)


def csv_table(path, cols=None):
    rows = list(csv.DictReader(open(path)))
    if not rows:
        return ""
    cols = cols or list(rows[0])
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(r.get(c, "") for c in cols) + " |" for r in rows]
    return "\n".join(out)


def tune_table(path, n=5):
    rows = list(csv.DictReader(open(path)))
    if not rows:
        return ""
    keys = [k for k in rows[0] if k not in ("us", "cpu_us", "energy_uJ", "edp", "GBps", "GOPs", "relerr")]
    best_speed = min(rows, key=lambda r: float(r["us"]))
    out = [f"{len(rows)} configs passed; top {n} by energy (fastest marked *)", "",
           "| " + " ".join(keys) + " | us | uJ | GB/s |", "|---|---|---|---|"]  # fmt: skip
    for r in rows[:n] + ([best_speed] if best_speed not in rows[:n] else []):
        star = "*" if r is best_speed else ""
        out.append(f"| {' '.join(f'{k}={r[k]}' for k in keys)}{star} | {float(r['us']):.1f} | "
                   f"{float(r['energy_uJ']):.0f} | {float(r['GBps']):.1f} |")  # fmt: skip
    return "\n".join(out)


def main(d):
    for s in ("headline", "headline_cold", "compare", "nibble", "fp4", "ggml"):
        p = os.path.join(d, f"{s}.csv")
        if os.path.exists(p):
            print(f"## {s}\n\n{bench_table(p)}\n")
    p = os.path.join(d, "amx.txt")
    if os.path.exists(p):
        print(f"## amx\n\n{amx_table(p)}\n")
    for s, cols in (("quality", ["label", "size_MiB", "ppl", "ppl_err", "kld", "same_top", "ppl_ratio"]), ("e2e", None)):
        p = os.path.join(d, f"{s}.csv")
        if os.path.exists(p):
            print(f"## {s}\n\n{csv_table(p, cols)}\n")
    for p in sorted(glob.glob(os.path.join(d, "tune_*.csv"))):
        print(f"## {os.path.basename(p)}\n\n{tune_table(p)}\n")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
