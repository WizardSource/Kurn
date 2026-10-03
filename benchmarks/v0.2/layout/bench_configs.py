"""Interleaved repeated measurement of named kurn configs (median of reps).

    python bench_configs.py CONFIGS.json --regime hot --threads 1 --reps 3 --secs 0.5 [--bench-args '--K 4096']

CONFIGS.json: {"name": {spec dict}, ...}. Prints one line per config and writes CSV to --out.
A spec dict may carry "_so": a prebuilt library (e.g. built by kurn v0.1) or a ggml impl name
(ggml, ggml-graph-amx, ... with --harness bench_ggml); its other keys only select the kernel.
"""
import argparse
import csv
import json
import shlex
import statistics
import sys
import time

from kurn.harness import bench
from kurn.spec import resolve
from kurn.toolchain import build


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("configs")
    ap.add_argument("--regime", default="hot")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--secs", type=float, default=0.5)
    ap.add_argument("--bench-args", default="")
    ap.add_argument("--harness")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    with open(a.configs) as fh:
        cfgs = json.load(fh)
    built = {}
    for name, spec in cfgs.items():
        spec = dict(spec)
        so = spec.pop("_so", None)
        c = resolve({**spec, "threads": a.threads})
        built[name] = (c, so or build(c))
    rows = {n: [] for n in cfgs}
    for _ in range(a.reps):
        for name, (c, so) in built.items():
            extra = shlex.split(a.bench_args)
            if c["op"] == "verify" and "--M" not in extra:
                extra += ["--M", str(c["cols"])]
            r = bench(so, c, a.regime, a.secs, extra, a.harness)
            rows[name].append(r)
    out = []
    for name, rs in rows.items():
        us = statistics.median(r["us"] for r in rs)
        cpu = statistics.median(r["cpu_us"] for r in rs)
        gb = statistics.median(r["GBps"] for r in rs)
        drift = max(abs(float(r["drift_s"])) for r in rs)
        ok = all(r["check"] == "ok" for r in rs)
        rec = {"name": name, "regime": a.regime, "threads": a.threads, "us": round(us, 2), "cpu_wall": round(cpu / us, 2),
               "GBps": round(gb, 1), "energy_uJ": round(cpu * 5.47, 1), "drift_s": drift, "check": "ok" if ok else "FAIL",
               "reps": ";".join(f"{r['us']:.1f}" for r in rs)}
        out.append(rec)
        print(f"{name:40s} {us:9.2f} us  {gb:7.1f} GB/s  cpu/wall {cpu / us:.2f}  drift {drift:.3f}  {rec['check']}  [{rec['reps']}]")
        sys.stdout.flush()
    if a.out:
        with open(a.out, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(out[0]))
            if fh.tell() == 0:
                w.writeheader()
            w.writerows(out)
    print(f"loadavg {open('/proc/loadavg').read().split()[:3]}  time {time.strftime('%H:%M:%S')}")


if __name__ == "__main__":
    main()
