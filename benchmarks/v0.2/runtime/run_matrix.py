#!/usr/bin/env python3
"""Run rtbench experiment plans: every config `--reps` times, interleaved, in batches of
<= --budget seconds, each batch under one benchmarks/v0.2/benchlock.sh acquisition.
Appends rows to results/<plan>.csv.

    PYTHONPATH=kurn/src KURN_CACHE_DIR=/tmp/kurn-cache-runtime \
      /opt/kenv/bin/python kurn/benchmarks/v0.2/runtime/run_matrix.py partition-cold --reps 3

Plans: see PLANS below (`--list`). `--only SUBSTR` keeps configs whose label contains SUBSTR.
"""

import argparse
import os
import shlex
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
LOCK = os.path.join(HERE, "..", "benchlock.sh")


def moe_set(w, extra=()):
    """kurn v0.1 (barrier per matrix), engine-style static, balanced, steal, stream-K, ggml graph + schedule."""
    e = list(extra)
    return [
        (f"{w}:v0.1-static-sync", [w, "static", "--sync-each", *e]),
        (f"{w}:static", [w, "static", *e]),
        (f"{w}:balanced", [w, "balanced", *e]),
        (f"{w}:steal", [w, "steal", *e]),
        (f"{w}:streamk-ks2", [w, "streamk", "--ksplit", "2", *e]),
        (f"{w}:ggml-graph+sched", [f"{w}-sep", "ggml", "--op-barriers", "2", "--tile", "64", *e]),
        (f"{w}:ggml-graph+balanced", [f"{w}-sep", "balanced", "--op-barriers", "2", *e]),
    ]


def dense_set(w, extra=()):
    e = list(extra)
    return [
        (f"{w}:v0.1-static", [w, "static", *e]),
        (f"{w}:balanced", [w, "balanced", *e]),
        (f"{w}:steal", [w, "steal", *e]),
        (f"{w}:streamk-ks2", [w, "streamk", "--ksplit", "2", *e]),
        (f"{w}:ggml-graph+sched", [f"{w}-sep", "ggml", "--op-barriers", "2", "--tile", "64", *e]),
        (f"{w}:ggml-graph+balanced", [f"{w}-sep", "balanced", "--op-barriers", "2", *e]),
    ]


def skinny_set(w, extra=()):
    e = list(extra)
    return [
        (f"{w}:v0.1-static", [w, "static", *e]),
        (f"{w}:balanced", [w, "balanced", *e]),
        (f"{w}:steal", [w, "steal", "--tile", "32", *e]),
        (f"{w}:ggml", [w, "ggml", "--tile", "64", *e]),
        (f"{w}:streamk-ks2", [w, "streamk", "--ksplit", "2", *e]),
        (f"{w}:streamk-ks4", [w, "streamk", "--ksplit", "4", *e]),
        (f"{w}:streamk-ks8", [w, "streamk", "--ksplit", "8", *e]),
    ]


def T(n, cfgs):
    return [(f"T{n}:{lab}", [*args, "--threads", str(n)]) for lab, args in cfgs]


CLIFF_MB = [8, 16, 32, 64, 160, 256, 320, 512, 1200]

PLANS = {
    "partition-cold": T(8, moe_set("olmoe-moe") + moe_set("qwen3moe") + dense_set("dense17")),
    "partition-cold-r5": T(8, moe_set("olmoe-moe") + moe_set("qwen3moe") + dense_set("dense17")),  # rerun, per-thread spin column
    # T7 leaves one vCPU to the other workers' nice-19 jobs: at nice 0, T8 threads get preempted mid-token and the
    # whole pool waits at the next barrier (bimodal reps, spin share ~55 %)
    "partition-cold-t7": T(7, moe_set("olmoe-moe") + moe_set("qwen3moe") + dense_set("dense17")),
    "partition-cold-t6": T(6, moe_set("olmoe-moe") + moe_set("qwen3moe")),
    "partition-hot": T(8, moe_set("olmoe-moe", ["--regime", "hot"]) + moe_set("qwen3moe", ["--regime", "hot"])),
    "skinny": T(
        8,
        [
            (lab.replace("skinny1:", f"skinny1-{reg}:"), args)
            for reg in ("hot", "cold")
            for lab, args in skinny_set("skinny1", ["--regime", reg])
        ],
    ),
    "cliff": T(
        8,
        [(f"read:{mb}MB", ["sweep", "balanced", "--impl", "read", "--footprint", str(mb)]) for mb in CLIFF_MB]
        + [(f"q8gemv:{mb}MB", ["sweep", "balanced", "--footprint", str(mb)]) for mb in CLIFF_MB]
        + [(f"q8gemv-rotate:{mb}MB", ["sweep", "balanced", "--rotate", "--footprint", str(mb)]) for mb in (8, 16, 64, 160, 256)]
        + [
            (f"read-rotate:{mb}MB", ["sweep", "balanced", "--impl", "read", "--rotate", "--footprint", str(mb)])
            for mb in (8, 16, 64, 160, 256)
        ],
    ),
    "waits": T(
        8,
        [
            (f"olmoe-moe:{reg}:gap{u}:{w}", ["olmoe-moe", "balanced", "--regime", reg, "--serial-us", str(u), "--wait", w])
            for reg in ("cold", "hot")
            for u in (0, 5, 20, 50, 200)
            for w in ("spin", "hybrid:500", "hybrid:5000", "futex")
        ],
    ),
    "prefetch": T(
        8,
        [
            (f"{w}:gaps{g.replace(',', '/')}:{lab}", [w, "balanced", "--gaps", g, *pf])
            for w, gs in (("olmoe-layer", ("0", "20,5,1,2")), ("dense17", ("0", "20,0,1,2")))
            for g in gs
            for lab, pf in (
                ("none", []),
                ("pf256k-t0", ["--prefetch", "wait", "--pf-kb", "256", "--pf-hint", "0"]),
                ("pf1m-t1", ["--prefetch", "wait", "--pf-kb", "1024", "--pf-hint", "1"]),
            )
        ],
    ),
    "expert-predict": T(
        8,
        [
            (f"olmoe-moe:gaps0/{g}:{lab}", ["olmoe-moe", "balanced", "--gaps", f"0,{g}", *pf])
            for g in (10, 30)
            for lab, pf in [("none", [])]
            + [
                (f"pred{p}", ["--prefetch", "wait", "--pf-kb", "1024", "--pf-hint", "1", "--pf-predict", str(p)])
                for p in (1.0, 0.7, 0.4, 0.0)
            ]
        ],
    ),
    "producers": [
        (f"T{t}:{w}:consumers{t - p}+producers{p}", [w, "balanced", "--threads", str(t), "--producers", str(p)])
        for w in ("olmoe-moe", "dense17")
        for t, p in ((8, 0), (8, 1), (8, 2), (8, 4), (6, 0), (4, 0))
    ],
    # ggml's chunked schedule vs ours on the same graph, at both chunk sizes (separates schedule from 64-row kernel calls)
    "ggml-iso": T(
        8,
        [
            (f"{w}:{part}-c{c}", [f"{w}-sep", part, "--op-barriers", "2", "--tile", str(c)])
            for w in ("qwen3moe", "dense17")
            for part in ("ggml", "balanced", "steal")
            for c in (64, 128)
        ],
    ),
    # 4 KB pages (THP is madvise-only here) vs glibc-requested THP, sweep vs model-shaped streaming at the same footprint
    "dram-check": T(
        8,
        [
            (f"{lab}:{pg}", [*args, *(["--thp"] if pg == "thp" else [])])
            for lab, args in (
                ("read-sweep1200", ["sweep", "balanced", "--impl", "read", "--footprint", "1200"]),
                ("q8gemv-sweep1200", ["sweep", "balanced", "--footprint", "1200"]),
                ("read-sweep160", ["sweep", "balanced", "--impl", "read", "--footprint", "160"]),
                ("read-dense17-cold", ["dense17", "balanced", "--impl", "read"]),
                ("dense17-cold-steal", ["dense17", "steal"]),
                ("olmoe-moe-cold-steal", ["olmoe-moe", "steal"]),
            )
            for pg in ("4k", "thp")
        ],
    ),
    "cliff-thp": T(
        8,
        [(f"read-thp:{mb}MB", ["sweep", "balanced", "--impl", "read", "--thp", "--footprint", str(mb)]) for mb in CLIFF_MB]
        + [(f"q8gemv-thp:{mb}MB", ["sweep", "balanced", "--thp", "--footprint", str(mb)]) for mb in CLIFF_MB],
    ),
    "waits-t7": T(
        7,
        [
            (f"olmoe-moe:cold:gap{u}:{w}", ["olmoe-moe", "balanced", "--serial-us", str(u), "--wait", w])
            for u in (0, 20, 50, 200)
            for w in ("spin", "hybrid:2000", "futex")
        ],
    ),
    "producers-t7": [
        (f"T{t}:{w}:consumers{t - p}+producers{p}", [w, "balanced", "--threads", str(t), "--producers", str(p)])
        for w in ("olmoe-moe", "dense17")
        for t, p in ((7, 0), (7, 1), (7, 2), (7, 3), (5, 0), (4, 0))
    ],
    # T7 MoE: flattened ranges (balanced) ran 1.6x slower than static; T8 did not. Alignment? DRAM-only?
    "t7-diag": [
        (f"T{t}:olmoe-moe:{reg}:{lab}", ["olmoe-moe", part, "--threads", str(t), "--regime", reg, *x])
        for t, reg in ((7, "cold"), (8, "cold"), (7, "hot"))
        for lab, part, x in (("static", "static", []), ("balanced", "balanced", []), ("balanced-a128", "balanced", ["--align", "128"]))
        if not (reg == "hot" and lab == "balanced-a128")
    ],
    "cliff-t1": T(
        1,
        [(f"read:{mb}MB", ["sweep", "balanced", "--impl", "read", "--footprint", str(mb)]) for mb in (1, 2, 4, 8, 32, 160, 1200)]
        + [(f"q8gemv:{mb}MB", ["sweep", "balanced", "--footprint", str(mb)]) for mb in (1, 2, 4, 8, 32, 160, 1200)],
    ),
}


for _p in ("expert-predict", "prefetch"):
    PLANS[f"{_p}-t7"] = [(lab.replace("T8:", "T7:", 1), [*args[:-2], "--threads", "7"]) for lab, args in PLANS[_p]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("plan", nargs="*", help="one or more plans; their runs share lock acquisitions, in the order given")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--secs", type=float, default=1.5)
    ap.add_argument("--only", default="")
    ap.add_argument("--out")
    ap.add_argument("--budget", type=float, default=420, help="estimated seconds of runs per lock acquisition")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    for k in a.plan:
        if k not in PLANS:
            ap.error(f"unknown plan {k}")
    if a.list or not a.plan:
        for k, v in PLANS.items():
            print(f"{k:20s} {len(v)} configs")
        return
    from kurn import runtime

    rt, so = runtime.build_rtbench(), runtime.default_kernel()
    if a.out and len(a.plan) > 1:
        ap.error("--out needs a single plan")
    runs = []
    for plan in a.plan:
        out = a.out or os.path.join(HERE, "results", f"{plan}.csv")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        cfgs = [c for c in PLANS[plan] if a.only in c[0]]
        n0 = len(runs)
        for rep in range(a.reps):  # interleaved repetitions
            for label, args in cfgs:
                w, part, *rest = args
                impl = so
                if "--impl" in rest:
                    i = rest.index("--impl")
                    impl = rest[i + 1]
                    del rest[i : i + 2]
                pre = []
                if "--thp" in rest:  # glibc madvises its mmap'd chunks (ours and the kernel's packed weights)
                    rest.remove("--thp")
                    pre = ["env", "GLIBC_TUNABLES=glibc.malloc.hugetlb=1"]
                cmd = [*pre, rt, "--impl", impl, "--workload", w, "--part", part, "--secs", str(a.secs), "--csv", out,
                       "--label", f"{label}#r{rep}", *rest]  # fmt: skip
                est = a.secs + (1.0 if "hot" in rest or w == "sweep" else 3.0)
                runs.append((est, cmd))
        print(f"{plan}: {len(cfgs)} configs x {a.reps} reps = {len(runs) - n0} runs -> {out}", flush=True)
    batches, cur, t = [], [], 0.0
    for est, cmd in runs:
        if cur and t + est > a.budget:
            batches.append(cur)
            cur, t = [], 0.0
        cur.append(cmd)
        t += est
    if cur:
        batches.append(cur)
    print(f"{len(runs)} runs in {len(batches)} locked batches", flush=True)
    for bi, batch in enumerate(batches):
        # one lock acquisition per batch, at the lock holder's priority (the coordinator renices lock holders to 0)
        script = "".join(shlex.join(c) + " | tail -1\n" for c in batch)
        if a.dry:
            print(script)
            continue
        t0 = time.time()
        r = subprocess.run([LOCK, "bash", "-c", script], capture_output=True, text=True)
        print(f"--- batch {bi + 1}/{len(batches)}: {len(batch)} runs, {time.time() - t0:.0f}s (incl. lock wait)", flush=True)
        for line in r.stdout.strip().splitlines():
            print(line[:260], flush=True)
        if r.returncode:
            print(r.stderr[-2000:], file=sys.stderr)


if __name__ == "__main__":
    main()
