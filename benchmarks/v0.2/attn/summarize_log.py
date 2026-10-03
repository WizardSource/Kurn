"""Rebuild a matrix summary (median, min-max over reps) from a matrix.py log.

summarize_log.py results/prefill17.log [out.csv]
"""

import statistics
import sys
from collections import defaultdict


def main(path, out=None):
    rows = defaultdict(list)
    order = []
    for ln in open(path):
        f = ln.split()
        if len(f) < 12 or not f[0].isdigit():
            continue
        key = (f[1], f[2])
        if key not in rows:
            order.append(key)
        rows[key].append({"us": float(f[3]), "GFLOPs": float(f[5]), "GBps": float(f[7]), "relerr": float(f[10]), "check": f[11]})
    lines = ["case,impl,reps,us_med,us_min,us_max,GFLOPs_med,GBps_med,relerr_max,check"]
    for key in order:
        rs = rows[key]
        us = [r["us"] for r in rs]
        lines.append(",".join(map(str, [*key, len(rs), round(statistics.median(us), 1), round(min(us), 1), round(max(us), 1),
                                         round(statistics.median(r["GFLOPs"] for r in rs), 1),
                                         round(statistics.median(r["GBps"] for r in rs), 1),
                                         f"{max(r['relerr'] for r in rs):.1e}",
                                         "ok" if all(r["check"] == "ok" for r in rs) else "FAIL"])))  # fmt: skip
    text = "\n".join(lines) + "\n"
    if out:
        open(out, "w").write(text)
    print(text, end="")


if __name__ == "__main__":
    main(*sys.argv[1:])
