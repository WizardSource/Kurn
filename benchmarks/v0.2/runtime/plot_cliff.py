#!/usr/bin/env python3
"""Cache-cliff chart: effective bandwidth vs weights touched per token (rtbench `sweep`).

plot_cliff.py results/cliff.csv [results/cliff-t1.csv] -o cliff.png --summary cliff_summary.csv
"""

import argparse
import csv
import statistics
from collections import defaultdict

L2_MB, L3_MB, ROOF = 16, 320, 128.7


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+")
    ap.add_argument("-o", "--out", default="cliff.png")
    ap.add_argument("--summary", default="cliff_summary.csv")
    a = ap.parse_args()
    series = defaultdict(lambda: defaultdict(list))
    for p in a.csv:
        for r in csv.DictReader(open(p)):
            lab = r["label"].split("#")[0]  # T8:read:64MB
            t, kind, mb = lab.split(":")
            series[f"{kind} {t}"][float(mb[:-2])].append(float(r["GBps"]))
    rows = []
    for name, pts in sorted(series.items()):
        for mb in sorted(pts):
            v = pts[mb]
            m = statistics.median(v)
            rows.append({"series": name, "footprint_mb": mb, "GBps_med": round(m, 1), "GBps_min": round(min(v), 1),
                         "GBps_max": round(max(v), 1), "reps": len(v), "x_dram_roofline": round(m / ROOF, 2)})  # fmt: skip
    with open(a.summary, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.8))
    styles = {"read": "-o", "q8gemv": "-s", "q8gemv-rotate": "--s", "read-rotate": "--o", "q8gemv-steal": ":^"}
    for name, pts in sorted(series.items()):
        xs = sorted(pts)
        ys = [statistics.median(pts[x]) for x in xs]
        ax.plot(xs, ys, styles.get(name.split()[0], "-x"), label=name, markersize=4)
    ax.axvline(L2_MB, color="grey", lw=0.8, ls=":")
    ax.axvline(L3_MB, color="grey", lw=0.8, ls=":")
    ax.axhline(ROOF, color="red", lw=0.8, ls="--")
    ax.text(L2_MB * 1.05, ax.get_ylim()[1] * 0.92, "L2 total 16 MB", fontsize=8)
    ax.text(L3_MB * 1.05, ax.get_ylim()[1] * 0.92, "L3 320 MB", fontsize=8)
    ax.text(1.1, ROOF * 1.04, "DRAM roofline 128.7 GB/s (8 cores)", fontsize=8, color="red")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("weights touched per token (MB), static per-core ownership, repeated tokens")
    ax.set_ylabel("effective bandwidth (GB/s)")
    ax.set_title("Cache cliffs, 8-vCPU Emerald Rapids KVM guest (rtbench sweep)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(a.out, dpi=130)
    print(f"wrote {a.out} and {a.summary}")


if __name__ == "__main__":
    main()
