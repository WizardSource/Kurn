#!/usr/bin/env python3
"""Paired A/B statistics: speedup of B over A with 95% confidence intervals.

Input lines: `NAME PAIR CONFIG VALUE [ms|tps]` (tsv or spaces). Within a pair, A and B ran back to back (order
alternated between pairs), so slow phases of a noisy host hit both. Per pair r = A/B for times (ms) or B/A for rates
(tps); reported: geometric mean of r, a Student-t 95% CI on mean(log r), a bootstrap (10,000 resamples) percentile
CI, and how many pairs favoured B.

    paired_ci.py A_CONFIG B_CONFIG FILE...
"""

import math
import random
import sys
from collections import defaultdict

T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201,
        12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086, 25: 2.060,
        30: 2.042}  # fmt: skip


def t975(df):
    return T975.get(df) or T975[max(k for k in T975 if k <= df)] if df <= 30 else 1.96


def stats(logs, seed=1):
    n = len(logs)
    m = sum(logs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in logs) / (n - 1)) if n > 1 else float("nan")
    h = t975(n - 1) * sd / math.sqrt(n) if n > 1 else float("nan")
    rng = random.Random(seed)
    boots = sorted(sum(rng.choice(logs) for _ in range(n)) / n for _ in range(10000))
    return math.exp(m), math.exp(m - h), math.exp(m + h), math.exp(boots[249]), math.exp(boots[9749])


def main(argv):
    a, b, files = argv[1], argv[2], argv[3:]
    data = defaultdict(dict)
    unit = {}
    for f in files:
        for line in open(f):
            p = line.split()
            if len(p) < 4:
                continue
            name, pair, cfg, val = p[0], p[1], p[2], float(p[3])
            unit[name] = p[4] if len(p) > 4 else "ms"
            data[(name, pair)][cfg] = val
    names = sorted({k[0] for k in data}, key=lambda n: list(dict.fromkeys(k[0] for k in data)).index(n))
    print(f"{'benchmark':34} {'pairs':>5} {'B/A speedup':>11}  {'t 95% CI':>15}  {'bootstrap 95%':>15}  B wins")
    for name in names:
        logs = []
        for (nm, _), v in data.items():
            if nm == name and a in v and b in v:
                r = v[a] / v[b] if unit[name] == "ms" else v[b] / v[a]
                logs.append(math.log(r))
        if not logs:
            continue
        g, lo, hi, blo, bhi = stats(logs)
        wins = sum(x > 0 for x in logs)
        print(f"{name:34} {len(logs):5d} {g:10.3f}x  [{lo:.3f}, {hi:.3f}]  [{blo:.3f}, {bhi:.3f}]  {wins}/{len(logs)}")


if __name__ == "__main__":
    main(sys.argv)
