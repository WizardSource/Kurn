#!/usr/bin/env python3
"""Paired A/B benchmark with alternating order and kurn's paired-CI gate.
usage: paired.py --pairs N --min-gain 0.02 --metric {tps,ms} -- 'cmdA' -- 'cmdB'
Each command prints a number on its last stdout line matching the metric (parsed by --regex).
Ratio is B/A of cost (ms) or A/B of throughput (tps), so < 1 means B is faster."""
import argparse, re, subprocess, sys, statistics, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'src'))
from kurn.tune import paired_gate, bootstrap_ci
ap = argparse.ArgumentParser()
ap.add_argument('--pairs', type=int, default=8)
ap.add_argument('--min-gain', type=float, default=0.0)
ap.add_argument('--metric', default='tps')
ap.add_argument('--regex', default=r'([0-9.]+)\s*$')
ap.add_argument('--label', default='')
ap.add_argument('cmds', nargs='+')
a = ap.parse_args()
cmds = ' '.join(a.cmds).split(' -- ')
cmds = [c.strip().lstrip('-').strip() for c in cmds if c.strip().strip('-').strip()]
assert len(cmds) == 2, cmds
rx = re.compile(a.regex, re.M)
def run(c):
    out = subprocess.run(c, shell=True, capture_output=True, text=True).stdout + ''
    m = rx.findall(out)
    if not m:
        print('no match:', c, out[-500:], file=sys.stderr); sys.exit(1)
    v = m[-1]; v = v[-1] if isinstance(v, tuple) else v
    return float(v)
A, B, R = [], [], []
for i in range(a.pairs):
    order = (0, 1) if i % 2 == 0 else (1, 0)
    v = [None, None]
    for j in order:
        v[j] = run(cmds[j])
    A.append(v[0]); B.append(v[1])
    R.append(v[1] / v[0] if a.metric == 'ms' else v[0] / v[1])
    print(f'pair {i}: A={v[0]:.3f} B={v[1]:.3f} cost ratio B/A={R[-1]:.4f}', file=sys.stderr, flush=True)
g, lo, hi = paired_gate(R, a.min_gain)
res = dict(label=a.label, metric=a.metric, A_med=statistics.median(A), B_med=statistics.median(B), ratio_med=statistics.median(R), ci=[lo, hi], gate=g, pairs=a.pairs, A=A, B=B)
print(json.dumps(res))
