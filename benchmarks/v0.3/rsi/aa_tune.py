#!/usr/bin/env python3
"""A/A test of kurn's tuner confirmation on the real harness: the challenger is the incumbent's own .so,
so every admission is a false win. Runs the new paired-CI gate (tune._confirm) and the old rule it replaced
(median of 3 interleaved repetitions beating the incumbent by > 2%) on the same kernel and regime."""
import json
import os
import statistics
import sys

from kurn import tune
from kurn.spec import load, resolve
from kurn.toolchain import build

SPEC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "examples", "q8_0_gemv_vnni16.kurn")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 10
EXTRA = ["--K", "4096", "--N", "4096"]

spec, _ = load(SPEC)
c = resolve(spec, {"threads": 8, "wait": "spin", "rows": 8, "prefetch": 0, "align": "packed"})
so = build(c)
out = {"trials": N, "new_admits": 0, "old_admits": 0, "new": [], "old": []}
for i in range(N):
    a, b = {"c": c, "so": so}, {"c": c, "so": so}
    tune._confirm(a, b, "cold", "us", 0.25, EXTRA, None, 0.0, 0.02)
    out["new_admits"] += b["_gate"] == 1
    out["new"].append([b["_gate"], round(b["_ratio"], 4), [round(x, 4) for x in b["_ci"]], b["_pairs"]])
    ra, rb = [], []
    for _ in range(3):  # the old rule: a, b interleaved per repetition, medians compared
        ra.append(tune._measure(c, so, "cold", 0.25, EXTRA, None, 0.0)["us"])
        rb.append(tune._measure(c, so, "cold", 0.25, EXTRA, None, 0.0)["us"])
    ok = statistics.median(rb) < statistics.median(ra) * 0.98
    out["old_admits"] += ok
    out["old"].append([ok, round(statistics.median(rb) / statistics.median(ra), 4)])
    print(f"A/A {i}: new gate {b['_gate']} ratio {b['_ratio']:.3f} CI [{b['_ci'][0]:.3f}, {b['_ci'][1]:.3f}] "
          f"pairs {b['_pairs']}; old rule {'ADMIT' if ok else 'reject'} ({out['old'][-1][1]:.3f})", flush=True)
json.dump(out, open("aa_tune.json", "w"), indent=1)
print(json.dumps({k: out[k] for k in ("trials", "new_admits", "old_admits")}))
