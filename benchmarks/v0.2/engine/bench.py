#!/usr/bin/env python3
"""Interleaved decode benchmark: kurn engine variants vs llama.cpp (lcdrive), same protocol.

Each rep runs every configuration once (interleaved), after waiting for the machine to be
quiet. Per run it records decode tok/s (mean over N tokens), the median per-token latency,
CPU s/token (J/token proxy = 5.47 W x busy-core seconds; +10 W x wall for the platform
variant), barriers/token and barrier-wait share (engine only, rdtsc), the CPU time other
processes used during the run (`foreign_cpu`, cores) and the wall/monotonic drift.
Run it under benchlock.sh. Writes CSV rows to --csv and prints a median summary.

  bench.py --csv out.csv --reps 3 --ngen 128 --threads 8 --prompt 785,2326,... \
      NAME=CMD... (CMD gets `gen T N PROMPT` appended; a THREADS=n token overrides T for that config)
"""

import argparse
import csv
import os
import re
import statistics
import subprocess
import time

W_CORE, W_PLATFORM = 5.47, 10.0


def busy_jiffies():
    v = [int(x) for x in open("/proc/stat").readline().split()[1:]]
    return sum(v) - v[3] - v[4]  # minus idle, iowait


def procs_running():
    for line in open("/proc/stat"):
        if line.startswith("procs_running"):
            return int(line.split()[1])
    return 0


def wait_quiet(max_wait=90.0):
    """Wait until no other runnable task shows up in 5 samples over 1 s (or give up)."""
    t0 = time.monotonic()
    while time.monotonic() - t0 < max_wait:
        quiet = True
        for _ in range(5):
            if procs_running() > 1:
                quiet = False
                break
            time.sleep(0.2)
        if quiet:
            return time.monotonic() - t0
        time.sleep(1.0)
    return -1.0


def run_one(cmd, timeout=900):
    hz = os.sysconf("SC_CLK_TCK")
    d0, j0, w0 = time.time() - time.monotonic(), busy_jiffies(), time.monotonic()
    r0 = os.times()
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    r1 = os.times()
    wall, j1, d1 = time.monotonic() - w0, busy_jiffies(), time.time() - time.monotonic()
    child = (r1.children_user + r1.children_system) - (r0.children_user + r0.children_system)
    foreign = max(0.0, (j1 - j0) / hz - child) / wall
    return p.stdout, p.stderr, foreign, d1 - d0


def parse(out):
    line = next((ln for ln in out.splitlines() if "decode_tok_s" in ln), "")
    m = {k: float(v) for k, v in re.findall(r"(\w+) (-?[\d.]+(?:e-?\d+)?)", line)}
    g = re.search(r"gen:([ \d]*)", out)
    m["tokens"] = g.group(1).strip() if g else ""
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--ngen", type=int, default=128)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--model", default="")
    ap.add_argument("configs", nargs="+", help="NAME=CMD (CMD split on spaces; env vars as VAR=x prefix allowed)")
    ap.add_argument("--quiet-wait", type=float, default=20.0, help="max seconds to wait for an idle machine per run")
    ap.add_argument("--log", default="", help="append every run's stderr (e.g. KURN_PROF per-thread waits) here")
    ap.add_argument(
        "--renice",
        action="store_true",
        help="sudo renice this runner (and so every timed child) to -20, so untimed nice>=0 work on a shared "
        "VM cannot preempt spin-waiting decode threads",
    )
    a = ap.parse_args()
    if a.renice:
        subprocess.run(["sudo", "-n", "renice", "-n", "-20", "-p", str(os.getpid())], check=True, capture_output=True)
    cfgs = []
    for c in a.configs:
        name, _, cmd = c.partition("=")
        parts = cmd.split()
        thr = next((int(p[8:]) for p in parts if p.startswith("THREADS=")), a.threads)
        parts = [p for p in parts if not p.startswith("THREADS=")]
        env = [p for p in parts if re.match(r"^[A-Z_]+=", p)]
        cmd = ["env", *env, *[p for p in parts if p not in env]] if env else parts
        cfgs.append((name, cmd, thr))
    new = not os.path.exists(a.csv)
    fh = open(a.csv, "a", newline="")
    w = csv.writer(fh)
    cols = [
        "model",
        "config",
        "rep",
        "threads",
        "ngen",
        "decode_tok_s",
        "med_ms",
        "p10_ms",
        "p90_ms",
        "cpu_s_per_tok",
        "J_per_tok",
        "J_per_tok_10W",
        "barriers_per_tok",
        "wait_share",
        "quiet_wait_share",
        "foreign_cpu",
        "quiet_wait_s",
        "drift_s",
        "load1",
        "tokens",
        "runq_share",
    ]
    if new:
        w.writerow(cols)
    res = {n: [] for n, _, _ in cfgs}
    toks = {}
    for rep in range(a.reps):
        for name, cmd, thr in cfgs:
            qw = wait_quiet(a.quiet_wait)
            load1 = open("/proc/loadavg").read().split()[0]
            out, err, foreign, drift = run_one(cmd + ["gen", str(thr), str(a.ngen), a.prompt])
            m = parse(out)
            if a.log:
                with open(a.log, "a") as lf:
                    lf.write(f"== rep {rep} {name}\n{err}")
            if "decode_tok_s" not in m:
                print(f"{name}: no result\n{out}\n{err[-2000:]}")
                continue
            spt = 1.0 / m["decode_tok_s"]
            row = dict(
                model=a.model,
                config=name,
                rep=rep,
                threads=thr,
                ngen=a.ngen,
                decode_tok_s=m["decode_tok_s"],
                med_ms=m.get("med_ms", ""),
                p10_ms=m.get("p10_ms", ""),
                p90_ms=m.get("p90_ms", ""),
                cpu_s_per_tok=m["cpu_s_per_tok"],
                J_per_tok=round(m["cpu_s_per_tok"] * W_CORE, 4),
                J_per_tok_10W=round(m["cpu_s_per_tok"] * W_CORE + spt * W_PLATFORM, 4),
                barriers_per_tok=m.get("barriers_per_tok", ""),
                wait_share=m.get("wait_share", ""),
                quiet_wait_share=m.get("quiet_wait_share", ""),
                foreign_cpu=round(foreign, 3),
                quiet_wait_s=round(qw, 1),
                drift_s=round(drift, 4),
                load1=load1,
                tokens=m["tokens"],
                runq_share=m.get("runq_share", ""),
            )
            w.writerow([row[c] for c in cols])
            fh.flush()
            res[name].append(row)
            toks.setdefault(name, m["tokens"])
            print(
                f"rep {rep} {name:14s} {m['decode_tok_s']:7.2f} tok/s  med {row['med_ms']} ms  J/tok {row['J_per_tok']}  "
                f"wait {row['wait_share']}  foreign {row['foreign_cpu']}  drift {row['drift_s']}",
                flush=True,
            )
    print("\nmedian over reps (tok/s from mean; from median / p10 per-token latency; J/token; J/token +10W):")
    ref = next(iter(toks.values()), "")
    for name, rows in res.items():
        if not rows:
            continue

        def md(k, rows=rows):
            return statistics.median(float(r[k]) for r in rows if r[k] != "")

        medtok = 1000.0 / md("med_ms") if rows[0]["med_ms"] != "" else float("nan")
        p10tok = 1000.0 / md("p10_ms") if rows[0]["p10_ms"] != "" else float("nan")
        p10tok = 1000.0 / md("p10_ms") if rows[0]["p10_ms"] != "" else float("nan")
        a, b = toks[name].split(), ref.split()
        same = "identical" if a == b else f"differs at {next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))}"
        extra = f"  barriers/tok {md('barriers_per_tok'):.0f}  wait {md('wait_share'):.3f}" if rows[0]["wait_share"] != "" else ""
        if rows[0]["quiet_wait_share"] != "":
            extra += f" (quiet {md('quiet_wait_share'):.3f})"
        if rows[0]["runq_share"] != "":
            extra += f"  runq {md('runq_share'):.4f}"
        print(
            f"  {name:14s} {md('decode_tok_s'):7.2f}  {medtok:7.2f} / {p10tok:6.2f}  {md('J_per_tok'):.3f}  {md('J_per_tok_10W'):.3f}"
            f"  foreign {md('foreign_cpu'):.2f}{extra}  tokens vs first: {same}"
        )


if __name__ == "__main__":
    main()
