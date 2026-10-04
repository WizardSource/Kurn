"""Run kernels through the benchmark harness (bundled `data/bench.c`, or any
harness with the same command line and CSV output, e.g. the ggml-linked one in
`contrib/ggml-harness`)."""

import os
import subprocess

from .kernels import kernel
from .toolchain import harness_command, scratch_dir

CSV_COLUMNS = (
    "impl,kernel,regime,threads,K,N,M,calls,wall_s,cpu_s,us_per_call,GBps,GOPs,proxy_uJ_per_call,proxy_pJ_per_MAC,relerr,check,drift_s"
).split(",")


class HarnessError(Exception):
    pass


def bench(so, c, regime="hot", secs=1.0, extra=(), harness=None, timeout=900):
    """Run one kernel. Returns a dict of the harness CSV row plus `us` and `cpu_us`
    (per call). Raises HarnessError if the harness produced no result."""
    tmp = os.path.join(scratch_dir(), "row.csv")
    cmd = harness_command(c["target"], harness) + [
        "--impl", so, "--kernel", kernel(c).bench, "--regime", regime,
        "--threads", str(c["threads"]), "--wait", c["wait"], "--secs", str(secs), "--csv", tmp, *extra,
    ]  # fmt: skip
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if not os.path.exists(tmp):
        raise HarnessError(f"harness failed (exit {r.returncode}): {(r.stderr or r.stdout).strip()[:2000]}")
    with open(tmp) as fh:
        fields = fh.read().strip().split(",")
    os.remove(tmp)
    os.rmdir(os.path.dirname(tmp))
    row = dict(zip(CSV_COLUMNS, fields))
    calls = float(row["calls"])
    row["us"] = float(row["us_per_call"])
    row["cpu_us"] = float(row["cpu_s"]) / calls * 1e6
    row["relerr"] = float(row["relerr"])
    row["GBps"], row["GOPs"] = float(row["GBps"]), float(row["GOPs"])
    return row


def check_extra(c):
    """Default shapes for a correctness check: 200 rows (thread slices with tails) and, for
    verify kernels, one column fewer than compiled (exercises the column padding)."""
    extra = ["--N", "200"]
    if c["op"] == "verify":
        extra += ["--M", str(max(2, c["cols"] - 1))]
    return extra


def check(so, c, extra=None, harness=None):
    """Numerical check on awkward sizes (row tails, uneven thread slices). Returns the row."""
    if extra is None:
        extra = ["--M", "40", "--N", "100"] if c["op"] == "gemm" else check_extra(c)
    c = {**c, "threads": min(3, c["threads"]), "wait": "spin"}
    return bench(so, c, "hot", 0, extra, harness)


def bandwidth(threads, streams=0, harness=None, detail=False):
    """Peak read bandwidth (GB/s) from DRAM and L2, via the harness `--bw` mode: widest vector loads, all `threads`
    pinned, best of several barrier-timed groups; streams=0 measures 1, 2, 4 and 8 streams per thread and keeps the best.
    Returns {"dram": GB/s, "l2": GB/s}; with detail=True also "<level>_median" and "<level>_streams"."""
    cmd = harness_command("scalar", harness) + ["--bw", "--threads", str(threads), "--streams", str(streams)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    if r.returncode:
        raise HarnessError(r.stderr.strip())
    out = {}
    for line in r.stdout.splitlines():
        if line.startswith("bandwidth"):
            f = line.split()
            out[f[1]] = float(line.split(":")[1].split()[0])
            if detail:
                out[f"{f[1]}_streams"] = int(next(x for x in f if x.startswith("streams=")).split("=")[1])
                out[f"{f[1]}_median"] = float(line.split("median")[1].split()[0])
    return out
