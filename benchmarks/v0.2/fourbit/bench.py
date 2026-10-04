#!/usr/bin/env python3
"""WS-B (fourbit) kernel measurements: 4-bit GEMV variants vs Q8_0 vnni16 and kurn v0.1.

  bench.py SET [--reps 3] [--secs 1.0] [--out results/<SET>.csv]

Every configuration is built, checked against the exact reference by the harness
(a failed check aborts), and timed in the L2-hot 1-thread and DRAM 8-thread regimes.
Repetitions are interleaved (rep 1 of every config, then rep 2, ...); the CSV records
the load average at each run and the harness clock drift. Run under benchlock.sh.
Sets are defined in SETS below.
"""

import argparse
import csv
import os
import statistics
import subprocess
import sys
import time

from kurn import spec, toolchain
from kurn.harness import bandwidth, bench

HERE = os.path.dirname(os.path.abspath(__file__))
W_PER_CORE = 5.47
ROOF = {1: 18.3, 8: 128.7}  # measured read roofline, GB/s
V01 = "/opt/kenv01/bin/kurn"
GGML_HARNESS = os.environ.get("GGML_HARNESS", "/tmp/fb_ggml/bench_ggml")  # build_ggml_harness.sh
IDLE_PATIENCE = float(os.environ.get("FB_IDLE_PATIENCE", "10"))
ROOF_STREAMS = 4
os.environ.setdefault("BENCH_CACHE", "/tmp/fb_ggml/cache")  # ggml-quantized cold data, reused across runs


def C(weights, op="gemv", **kw):
    return dict(op=op, weights=weights, target="avx512_vnni", **kw)


SETS = {
    "baseline": [
        ("q8_0 vnni16 a64 r8", C("q8_0", layout="vnni16", align=64, rows=8)),
        ("q8_0 vnni16 r4 pf8", C("q8_0", layout="vnni16", rows=4, prefetch=8)),
        ("q4_0 i16 r2", C("q4_0", layout="i16", rows=2)),
        ("q4_0 i16 r4 pf8", C("q4_0", layout="i16", rows=4, prefetch=8)),
        ("iq4_nl i16 r4", C("iq4_nl", layout="i16", rows=4)),
        ("q4_K i16 r2", C("q4_K", layout="i16", rows=2)),
        ("q4_K i16 r4 pf4", C("q4_K", layout="i16", rows=4, prefetch=4)),
        ("q4_K native r4 pf4", C("q4_K", layout="native", rows=4, prefetch=4)),
    ],
    "nibble": [
        ("q8_0 vnni16 a64 r8", C("q8_0", layout="vnni16", align=64, rows=8)),
        ("q4_0 i16 r2", C("q4_0", layout="i16", rows=2)),
        ("q4_0 mask16 r1", C("q4_0", layout="i16", unpack="mask16", rows=1)),
        ("q4_0 mask16 r2", C("q4_0", layout="i16", unpack="mask16", rows=2)),
        ("q4_0 pair r1", C("q4_0", layout="i16", unpack="pair", rows=1)),
        ("q4_0 pair r2", C("q4_0", layout="i16", unpack="pair", rows=2)),
        ("iq4_nl i16 r4", C("iq4_nl", layout="i16", rows=4)),
        ("iq4_nl perm r1", C("iq4_nl", layout="i16", unpack="perm", rows=1)),
        ("iq4_nl perm r2", C("iq4_nl", layout="i16", unpack="perm", rows=2)),
        ("q4_K i16 r2", C("q4_K", layout="i16", rows=2)),
        ("q4_K mask16 act r2", C("q4_K", layout="i16", unpack="mask16", correction="act", rows=2)),
        ("q4_K mask16 dpmin r1", C("q4_K", layout="i16", unpack="mask16", correction="dpmin", rows=1)),
        ("q4_K mask16 dpmin r2", C("q4_K", layout="i16", unpack="mask16", correction="dpmin", rows=2)),
        ("q4_K pair dpmin r1", C("q4_K", layout="i16", unpack="pair", correction="dpmin", rows=1)),
        ("q4_K pair dpmin r2", C("q4_K", layout="i16", unpack="pair", correction="dpmin", rows=2)),
        ("q4_K native r4 pf4", C("q4_K", layout="native", rows=4, prefetch=4)),
        ("v0.1 q4_K r1", C("q4_K", rows=1, prefetch=0, v01="rows=1 prefetch=0")),
        ("v0.1 q4_K r4 pf4", C("q4_K", rows=4, prefetch=4, v01="rows=4 prefetch=4")),
    ],
    "headline": [
        ("q8_0 vnni16 a64 r8", C("q8_0", layout="vnni16", align=64, rows=8)),
        ("q4_0 i16 r2 (v0.2 base)", C("q4_0", layout="i16", rows=2)),
        ("q4_0 pair r2", C("q4_0", layout="i16", unpack="pair", rows=2)),
        ("q4_K i16 r4 pf4 (v0.2 base)", C("q4_K", layout="i16", rows=4, prefetch=4)),
        ("q4_K mask16 dpmin r2", C("q4_K", layout="i16", unpack="mask16", correction="dpmin", rows=2)),
        ("q4_K pair dpmin r2", C("q4_K", layout="i16", unpack="pair", correction="dpmin", rows=2)),
        ("v0.1 q4_K r4 pf4", C("q4_K", rows=4, prefetch=4, v01="rows=4 prefetch=4")),
        ("iq4_nl i16 r4 (v0.2 base)", C("iq4_nl", layout="i16", rows=4)),
        ("iq4_nl perm r2", C("iq4_nl", layout="i16", unpack="perm", rows=2)),
        ("mxfp4 perm r2", C("mxfp4", layout="i16", unpack="perm", rows=2)),
        ("nvfp4 perm r2", C("nvfp4", layout="i16", unpack="perm", rows=2)),
    ],
    "fp4": [
        ("mxfp4 lut r1", C("mxfp4", layout="i16", unpack="lut", rows=1)),
        ("mxfp4 lut r2", C("mxfp4", layout="i16", unpack="lut", rows=2)),
        ("mxfp4 perm r1", C("mxfp4", layout="i16", unpack="perm", rows=1)),
        ("mxfp4 perm r2", C("mxfp4", layout="i16", unpack="perm", rows=2)),
        ("nvfp4 lut r2", C("nvfp4", layout="i16", unpack="lut", rows=2)),
        ("nvfp4 perm r1", C("nvfp4", layout="i16", unpack="perm", rows=1)),
        ("nvfp4 perm r2", C("nvfp4", layout="i16", unpack="perm", rows=2)),
        ("nvfp4 perm packed r2", C("nvfp4", layout="i16", unpack="perm", scales="packed", rows=2)),
    ],
    "ggml": [
        (f"ggml {impl[5:] or 'vec_dot'} {w}", C(w, ggml=impl))
        for w in ("q8_0", "q4_0", "q4_K", "iq4_nl", "mxfp4", "nvfp4")
        for impl in ("ggml", "ggml-graph-cpu", "ggml-graph-repack", "ggml-graph-amx")
        if not (impl == "ggml-graph-amx" and w in ("iq4_nl", "mxfp4", "nvfp4"))
        and not (impl == "ggml-graph-repack" and w in ("q8_0", "nvfp4"))
    ],
}
# One lock session's worth (< 10 min) next to `headline`: the FP4 variants not in it, and ggml's
# raw vec_dot / repack / AMX paths (graph-cpu only adds graph overhead to vec_dot).
# The q4_0 8-column verify kernels are the kurn side of the AMX comparison (amx/amx4.c, M = 8).
SETS["compare"] = (
    [r for r in SETS["fp4"] if r[0] not in dict(SETS["headline"])]
    + [r for r in SETS["ggml"] if "graph-cpu" not in r[0]]
    + [
        ("q4_0 verify mask16 c8", C("q4_0", op="verify", layout="i16", unpack="mask16", rows=1, cols=8)),
        ("q4_0 verify i16 c8", C("q4_0", op="verify", layout="i16", rows=1, cols=8)),
    ]
)
# NVFP4 scale storage: fp16 per 16 values (5.0 bpw in the record) vs raw UE4M3 (4.5 bpw, decoded in-kernel).
SETS["nvfp4_scales"] = [
    (f"nvfp4 perm {s} r{n}", C("nvfp4", layout="i16", unpack="perm", scales=s, rows=n)) for s in ("unpacked", "packed") for n in (1, 2)
]
REGIMES = (("hot", 1), ("cold", 8))


def busy_cores(dt=0.5):
    """CPU cores busy (all processes) over dt seconds, from /proc/stat."""

    def snap():
        with open("/proc/stat") as fh:
            v = [int(x) for x in fh.readline().split()[1:]]
        return sum(v), v[3] + v[4]

    t0, i0 = snap()
    time.sleep(dt)
    t1, i1 = snap()
    return (os.cpu_count() or 1) * (1 - (i1 - i0) / max(1, t1 - t0))


def wait_idle(limit=0.7, patience=90):
    """Wait (up to `patience` s) until other processes use < `limit` cores; returns the busy count."""
    t_end = time.time() + patience
    while True:
        b = busy_cores()
        if b < limit or time.time() > t_end:
            return b
        time.sleep(2)


def build(c):
    """kurn v0.2 build; with `v01` set, the same spec built by kurn v0.1 (same ABI); with
    `ggml` set, one of the ggml baselines of the ggml-linked harness (no build)."""
    v01, impl = c.pop("v01", None), c.pop("ggml", None)
    c = spec.resolve(c)
    if impl:
        return c, impl
    if v01 is None:
        return c, toolchain.build(c)
    cmd = [V01, "build", os.path.join(HERE, "../../../examples/q4_K_gemv.kurn"), *v01.split(), f"threads={c['threads']}"]
    env = dict(os.environ, KURN_CACHE_DIR="/tmp/kurn-cache-fourbit-v01", PYTHONPATH="")
    out = subprocess.run(cmd, capture_output=True, text=True, check=True, env=env).stdout.strip().splitlines()[-1]
    return c, out


def run(name, c, regime, T, secs, extra=()):
    c, so = build(dict(c, threads=T))
    # Other workers' untimed jobs run at nice 19 while the lock holder is reniced up: an L2-hot
    # single thread only needs a core, so it does not wait; DRAM runs wait (bounded) for quiet.
    load = busy_cores() if regime == "hot" and T == 1 else wait_idle(limit=1.0, patience=IDLE_PATIENCE)
    harness = GGML_HARNESS if so.startswith("ggml") else None
    if c["op"] == "verify":
        extra = [*extra, "--M", str(c["cols"])]
    r = bench(so, c, regime, secs, list(extra), harness=harness)
    if r["check"] != "ok":
        raise SystemExit(f"FAILED correctness: {name} {regime} -> relerr {r['relerr']}")
    return {
        "name": name,
        "regime": regime,
        "threads": T,
        "us": r["us"],
        "cpu_us": r["cpu_us"],
        "GBps": r["GBps"],
        "relerr": r["relerr"],
        "drift_s": float(r["drift_s"]),
        "busy_cores": load,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("set", choices=sorted(SETS))
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--secs", type=float, default=1.0)
    ap.add_argument("--regimes", default="hot:1,cold:8")
    ap.add_argument("--out")
    ap.add_argument("--build-only", action="store_true", help="compile every config (run outside the lock)")
    a = ap.parse_args()
    regimes = [(r.split(":")[0], int(r.split(":")[1])) for r in a.regimes.split(",")]
    if a.build_only:
        for _name, c in SETS[a.set]:
            for _regime, T in regimes:
                build(dict(c, threads=T))
        return 0
    rows = []
    out = a.out or os.path.join(HERE, "results", f"{a.set}.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as fh:  # written row by row: a cut-short session keeps what it measured
        w = None
        for rep in range(a.reps):
            if any(r == "cold" for r, _ in regimes):  # the read roofline available right now (shared VM)
                bw = bandwidth(8, ROOF_STREAMS)
                rows.append({"name": f"read roofline s{ROOF_STREAMS}", "regime": "cold", "threads": 8, "us": 0.0,
                             "cpu_us": 0.0, "GBps": bw["dram"], "relerr": 0.0, "drift_s": 0.0,
                             "busy_cores": busy_cores(), "rep": rep})  # fmt: skip
                if w is None:
                    w = csv.DictWriter(fh, fieldnames=list(rows[-1]))
                    w.writeheader()
                w.writerow(rows[-1])
                print(f"rep{rep} read roofline 8T: {bw['dram']:.1f} GB/s DRAM, {bw['l2']:.1f} GB/s L2", flush=True)
            for name, c in SETS[a.set]:
                for regime, T in regimes:
                    row = run(name, c, regime, T, a.secs)
                    row["rep"] = rep
                    rows.append(row)
                    if w is None:
                        w = csv.DictWriter(fh, fieldnames=list(row))
                        w.writeheader()
                    w.writerow(row)
                    fh.flush()
                    print(
                        f"rep{rep} {name:28s} {regime:4s} T={T} {row['us']:9.2f} us {row['GBps']:7.1f} GB/s "
                        f"busy {row['busy_cores']:.2f} drift {row['drift_s']:+.3f}",
                        flush=True,
                    )
    print(f"\nmedian of {a.reps} ({time.strftime('%Y-%m-%d %H:%M')}), wrote {out}")
    print(f"{'config':28s} {'regime':8s} {'us':>9s} {'GB/s':>7s} {'%roof':>6s} {'uJ/call':>8s} {'spread':>7s}")
    for name, _ in SETS[a.set]:
        for regime, T in regimes:
            rs = [r for r in rows if r["name"] == name and r["regime"] == regime and r["threads"] == T]
            us = statistics.median(r["us"] for r in rs)
            gb = statistics.median(r["GBps"] for r in rs)
            cpu = statistics.median(r["cpu_us"] for r in rs)
            spread = (max(r["us"] for r in rs) - min(r["us"] for r in rs)) / us
            roof = f"{100 * gb / ROOF[T]:5.0f}%" if regime == "cold" else "     -"
            print(f"{name:28s} {regime}/{T:<3d} {us:9.2f} {gb:7.1f} {roof} {cpu * W_PER_CORE:8.1f} {100 * spread:6.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
