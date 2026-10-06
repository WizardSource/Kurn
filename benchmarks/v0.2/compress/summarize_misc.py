"""Medians (min-max) over reps of the misc_bench.sh logs (entropy_*.log, lut.log, lowrank.log).
summarize_misc.py results/"""

import os
import re
import statistics as st
import sys
from collections import defaultdict

d = sys.argv[1]


def table(path, key_re, val="per_call_us"):
    g, order, hdr = defaultdict(list), [], ""
    for line in open(os.path.join(d, path)):
        if not re.search(r"\brep=", line):
            hdr = line.strip()
            continue
        f = dict(kv.split("=", 1) for kv in re.findall(r"\S+=\S+", line))
        k = (hdr, key_re(line, f))
        if k not in g:
            order.append(k)
        g[k].append(float(re.search(rf"{val} (\S+)", line).group(1)) if " " + val + " " in line else float(f[val]))
    print(f"## {path}")
    last = None
    for h, k in order:
        if h != last:
            print(f"  [{h}]")
            last = h
        v = g[(h, k)]
        print(f"    {k:28s} {st.median(v):10.1f} us  ({min(v):.1f}-{max(v):.1f}, n={len(v)})")


table("entropy_cold.log", lambda s, f: f"{s.split()[0]} T={f['threads']}")
table("entropy_hot.log", lambda s, f: f"{s.split()[0]} T={f['threads']}")
table("lut.log", lambda s, f: s.split()[0])
table("lowrank.log", lambda s, f: f"r={f['rank']} {f['mode']} +{f['extra_bpw']}bpw")
