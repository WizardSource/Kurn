"""Per-format staged search vs the hand-picked fixed layouts (hot, 1 thread).

    python format_search.py --formats q4_0,q8_0,iq4_nl,q2_0,q1_0 --n0 81 --out results/format_search.json

For each format: `tune.search` over every legal value of every codegen key (layout
included, so the fixed layouts compete with composed ones), then an interleaved
re-measurement (median of --reps) of the search winner against every fixed layout x rows.
"""

import argparse
import json
import statistics
import time

from kurn import spec, tune
from kurn.harness import bench
from kurn.toolchain import build

KEYS = (
    "layout",
    "rows",
    "prefetch",
    "unpack",
    "correction",
    "scales",
    "accum",
    "align",
    "lanes",
    "plane",
    "kblock",
    "rgroup",
    "meta",
    "rgpad",
    "swizzle",
    "chains",
    "pfhint",
    "pfgran",
    "stages",
    "kpanel",
    "rpanel",
)


def fixed(op, f, t):
    out = {}
    for lay in spec.SCHEDULE["layout"](op, f, t):
        if lay == "composed":
            continue
        for rows in spec.SCHEDULE["rows"](op, f, t):
            try:
                c = spec.resolve({"op": op, "weights": f, "target": t, "layout": lay, "rows": rows, "threads": 1})
            except spec.SpecError:
                continue
            out[f"{lay}_r{rows}"] = c
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--formats", default="q4_0,q8_0,iq4_nl,q2_0,q1_0")
    ap.add_argument("--target", default="avx512_vnni")
    ap.add_argument("--n0", type=int, default=81)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--secs", type=float, default=0.3)
    ap.add_argument("--out")
    a = ap.parse_args()
    report = {}
    for f in a.formats.split(","):
        t0 = time.time()
        sp = {"op": "gemv", "weights": f, "target": a.target, "threads": 1}
        res, st = tune.search(sp, None, "hot", "energy", n0=a.n0, log=lambda m: None)
        ts = time.time() - t0
        win = spec.resolve({**res[0]["config"], "threads": 1})
        cands = {"search_winner": win, **fixed("gemv", f, a.target)}
        sos = {n: build(c) for n, c in cands.items()}
        meas = {n: [] for n in cands}
        for _ in range(a.reps):
            for n, c in cands.items():
                r = bench(sos[n], c, "hot", a.secs)
                if r["check"] == "ok":
                    meas[n].append(r["us"])
        med = {n: round(statistics.median(v), 2) for n, v in meas.items() if v}
        best_fixed = min((n for n in med if n != "search_winner"), key=med.get)
        report[f] = {
            "search_s": round(ts, 1),
            "stats": {k: st[k] for k in ("space_active", "legal_est", "builds", "measurements")},
            "winner": {k: win[k] for k in KEYS},
            "us": med,
            "best_fixed": best_fixed,
            "us_min_max": {n: [round(min(v), 2), round(max(v), 2)] for n, v in meas.items() if v},
            "speedup_vs_best_fixed": round(med[best_fixed] / med["search_winner"], 3),
        }
        print(
            f"{f}: search {ts:.0f}s, winner {med['search_winner']} us vs best fixed {best_fixed} {med[best_fixed]} us "
            f"({report[f]['speedup_vs_best_fixed']}x)  winner={ {k: win[k] for k in KEYS} }",
            flush=True,
        )
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(report, fh, indent=1)


if __name__ == "__main__":
    main()
