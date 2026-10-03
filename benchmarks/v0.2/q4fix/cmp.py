#!/usr/bin/env python3
"""Interleaved A/B rounds for opbench / llama-bench style commands.

    cmp.py --reps 5 'label|ENV=V ENV2=V|command args' ...  [--csv out.csv]

Every round runs every variant once, in order (rotated each round so no variant always runs
first). The number reported per run is the first match of --pattern (default: opbench's
"<ms> ms (min"), or llama-bench JSON avg_ts with --llama. Prints median, mean, sd and each
variant's ratio to the first one with a 2-sigma bound on the ratio (from the per-round ratios).
"""

import argparse
import csv
import json
import os
import re
import shlex
import statistics
import subprocess
import sys
import time


def run(env_s, cmd, pattern, llama):
    env = dict(os.environ)
    for kv in shlex.split(env_s):
        k, v = kv.split("=", 1)
        env[k] = v
    t0, c0 = time.time(), os.times()
    p = subprocess.run(cmd, shell=True, env=env, capture_output=True, text=True)
    c1 = os.times()
    cpu = (c1.children_user - c0.children_user) + (c1.children_system - c0.children_system)
    wall = time.time() - t0
    if llama:
        try:
            js = json.loads(p.stdout)
        except json.JSONDecodeError:
            sys.stderr.write(p.stdout[-2000:] + p.stderr[-2000:])
            raise
        return [r["avg_ts"] for r in js], cpu, wall
    m = re.search(pattern, p.stdout + p.stderr)
    if not m:
        sys.stderr.write(p.stdout[-2000:] + p.stderr[-2000:])
        raise RuntimeError(f"no match for {pattern!r} in output of {cmd}")
    return [float(m.group(1))], cpu, wall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("variants", nargs="+")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--pattern", default=r": ([0-9.]+) ms \(min")
    ap.add_argument("--llama", action="store_true", help="parse llama-bench -o json (avg_ts; higher is better)")
    ap.add_argument("--csv")
    a = ap.parse_args()
    vs = [v.split("|", 2) for v in a.variants]
    res = {lab: [] for lab, _, _ in vs}
    cpus = {lab: [] for lab, _, _ in vs}
    rows = []
    for i in range(a.reps):
        order = vs[i % len(vs):] + vs[: i % len(vs)]
        for lab, env_s, cmd in order:
            vals, cpu, wall = run(env_s, cmd, a.pattern, a.llama)
            res[lab].append(vals)
            cpus[lab].append(cpu)
            rows.append({"round": i, "label": lab, "values": " ".join(f"{v:.4f}" for v in vals),
                         "cpu_s": f"{cpu:.3f}", "wall_s": f"{wall:.3f}", "load1": f"{os.getloadavg()[0]:.2f}"})
            print(f"round {i} {lab}: {' '.join(f'{v:.3f}' for v in vals)} (cpu {cpu:.1f}s wall {wall:.1f}s)", flush=True)
    nmet = len(res[vs[0][0]][0])
    base = vs[0][0]
    for j in range(nmet):
        print(f"--- metric {j} ({'tok/s, higher is better' if a.llama else 'ms, lower is better'})")
        for lab, _, _ in vs:
            x = [r[j] for r in res[lab]]
            med, mu = statistics.median(x), statistics.mean(x)
            sd = statistics.stdev(x) if len(x) > 1 else 0.0
            cpu = statistics.median(cpus[lab])
            line = f"{lab:<24} median {med:9.3f}  mean {mu:9.3f}  sd {sd:7.3f} ({100 * sd / mu:4.1f}%)  cpu {cpu:.2f}s"
            if lab != base:
                b = [r[j] for r in res[base]]
                ratios = [(bb / xx) if not a.llama else (xx / bb) for bb, xx in zip(b, x)]
                rm = statistics.median(ratios)
                rs = statistics.stdev(ratios) if len(ratios) > 1 else 0.0
                verdict = "WIN" if rm - 2 * rs > 1 and rm > 1.05 else ("LOSS" if rm + 2 * rs < 1 else "not resolved")
                line += f"  speedup vs {base}: {rm:.3f} (2sd {2 * rs:.3f}) {verdict}"
            print(line)
    if a.csv:
        with open(a.csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            if f.tell() == 0:
                w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
