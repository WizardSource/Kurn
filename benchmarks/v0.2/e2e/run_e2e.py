#!/usr/bin/env python3
"""End-to-end llama.cpp comparison per model (format) and execution mode.

Metrics per (model, mode), each the median of --reps interleaved rounds (every round runs all
modes back to back while holding the shared bench lock):
  decode tok/s     llama-bench -p 0 -n NG (tg), reported avg_ts
  prefill tok/s    llama-bench -p NP -n 0 (pp)
  J/token (proxy)  busy CPU seconds x 5.47 W per generated (or prompt) token, from the CPU time
                   of the tg / pp run minus a load-only baseline run (-p 0 -n 1)
  perplexity       llama-perplexity on a fixed text, ubatch 1 (every token through the decode
                   path, -c 128) and batched (-c 512 -b 512 -ub 512, prefill path)
  memory           peak RSS of the tg run, split into anonymous and file-backed (mmap) pages
plus load average at lock time, CPU/wall ratio and wall-clock drift (VM freezes) per run.

Modes (NAME=BIN_DIR[:ENV=VAL,...][:extra llama flags]); defaults:
  default     stock build, extra buffers on (AMX / CPU_REPACK): ggml's best path
  plain       stock build, --repack 0
  kurn        kurn build (KURN extra buffer type), GGML_KURN_AMX=0: prefill through the verify kernels
  kurn-amx    kurn build, GGML_KURN_AMX=1: Q8_0 prefill through kurn's AMX kernel (not reliable on
              VMs that lose AMX tile state; run under the bench lock only)
  kurn-noamx  alias of kurn (older result files)

    run_e2e.py --models ~/models/Qwen3-1.7B-Q8_0.gguf --out e2e.csv
    run_e2e.py --models M.gguf --modes default,kurn --skip ppl_ub1 --reps 5
    run_e2e.py --summary e2e.csv          # markdown table of a results file
"""

import argparse
import contextlib
import csv
import fcntl
import json
import os
import re
import statistics
import subprocess
import sys
import threading
import time

WATTS_PER_CORE = 5.47
HERE = os.path.dirname(os.path.abspath(__file__))
LOCK = "/tmp/kurn-bench.lock"
LOCK_LOG = "/tmp/kurn-bench-lock.log"
METRICS = ("decode", "prefill", "ppl_ub1", "ppl_batched")
FIELDS = [
    "model",
    "mode",
    "threads",
    "decode_tok_s",
    "decode_spread",
    "decode_J_tok",
    "prefill_tok_s",
    "prefill_spread",
    "prefill_J_tok",
    "ppl_ub1",
    "ppl_ub1_err",
    "ppl_batched",
    "ppl_batched_err",
    "rss_mb",
    "anon_mb",
    "file_mb",
    "max_load",
    "min_cpu_wall",
    "max_drift_s",
    "reps",
    "commit",
    "timestamp",
]


def default_modes(stock, kurn):
    return {
        "default": (stock, {}, []),
        "plain": (stock, {}, ["--repack", "0"]),
        "kurn": (kurn, {"GGML_KURN_AMX": "0"}, []),
        "kurn-amx": (kurn, {"GGML_KURN_AMX": "1"}, []),
        "kurn-noamx": (kurn, {"GGML_KURN_AMX": "0"}, []),
        "kurn-off": (kurn, {"GGML_KURN": "0"}, []),
    }


def parse_mode(spec):
    name, rest = spec.split("=", 1)
    parts = rest.split(":")
    env = dict(kv.split("=", 1) for kv in parts[1].split(",") if kv) if len(parts) > 1 else {}
    flags = parts[2].split() if len(parts) > 2 else []
    return name, (os.path.expanduser(parts[0]), env, flags)


def set_autogroup_nice(n):
    subprocess.run(["sudo", "-n", "sh", "-c", f"echo {n} > /proc/{os.getpid()}/autogroup"], capture_output=True)


class BenchLock:
    """Exclusive lock shared with benchlock.sh (same file, same log format).

    KURN_NO_LOCK=1 skips it (smoke tests of the harnesses only: timings are then meaningless)."""

    def __enter__(self):
        self.load = float(open("/proc/loadavg").read().split()[0])
        self.fd = None
        if os.environ.get("KURN_NO_LOCK") == "1":
            return self
        self.fd = open(LOCK, "a+")
        fcntl.flock(self.fd, fcntl.LOCK_EX)
        self.load = float(open("/proc/loadavg").read().split()[0])
        # timed children inherit nice -20; the session autogroup also gets -20, because with
        # sched_autogroup_enabled nice only ranks processes within one session
        subprocess.run(["sudo", "-n", "renice", "-n", "-20", "-p", str(os.getpid())], capture_output=True)
        set_autogroup_nice(-20)
        with open(LOCK_LOG, "a") as f:
            f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime())} start load={self.load} pid={os.getpid()} cmd=run_e2e.py\n")
        return self

    def __exit__(self, *exc):
        if self.fd is None:
            return
        os.setpriority(os.PRIO_PROCESS, 0, max(os.getpriority(os.PRIO_PROCESS, 0), 0))
        set_autogroup_nice(0)
        with open(LOCK_LOG, "a") as f:
            f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime())} end   rc=0 pid={os.getpid()}\n")
        fcntl.flock(self.fd, fcntl.LOCK_UN)
        self.fd.close()


def run(cmd, env_extra, timeout=1800, split=False):
    """Run cmd; returns dict(out, err, cpu_s, wall_s, rss/anon/file MB peaks, drift_s).

    out is stdout+stderr, or stdout only with split=True (stderr then in err)."""
    env = dict(os.environ, **env_extra)
    d0 = time.time() - time.monotonic()
    t0 = time.monotonic()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE if split else subprocess.STDOUT, env=env, text=True)
    peak = {"VmRSS": 0, "RssAnon": 0, "RssFile": 0}

    def sample():
        while p.poll() is None:
            try:
                for line in open(f"/proc/{p.pid}/status"):
                    k = line.split(":")[0]
                    if k in peak:
                        peak[k] = max(peak[k], int(line.split()[1]))
            except OSError:
                pass
            time.sleep(0.05)

    th = threading.Thread(target=sample, daemon=True)
    th.start()
    out_chunks = []
    reader = threading.Thread(target=lambda: out_chunks.append(p.stdout.read()), daemon=True)
    reader.start()
    err_chunks = []
    if split:
        ereader = threading.Thread(target=lambda: err_chunks.append(p.stderr.read()), daemon=True)
        ereader.start()
    _, status, ru = os.wait4(p.pid, 0)
    p.returncode = os.waitstatus_to_exitcode(status)
    reader.join(5)
    if split:
        ereader.join(5)
    th.join(1)
    wall = time.monotonic() - t0
    out = "".join(out_chunks)
    err = "".join(err_chunks)
    if p.returncode != 0:
        raise RuntimeError(f"command failed ({p.returncode}): {' '.join(cmd)}\n{(out + err)[-3000:]}")
    return {
        "out": out,
        "err": err,
        "cpu_s": ru.ru_utime + ru.ru_stime,
        "wall_s": wall,
        "rss_mb": peak["VmRSS"] / 1024,
        "anon_mb": peak["RssAnon"] / 1024,
        "file_mb": peak["RssFile"] / 1024,
        "drift_s": (time.time() - time.monotonic()) - d0,
    }


def pin(threads):
    """Threads pinned 1:1 to vCPUs 0..threads-1 (required for AMX on this VM, see README)."""
    return ["--cpu-mask", f"{(1 << threads) - 1:x}", "--cpu-strict", "1"]


def bench_json(out):
    i = out.index("[")
    return json.loads(out[i : out.rindex("]") + 1])


def ppl_value(out):
    m = re.search(r"Final estimate: PPL = ([0-9.]+) \+/- ([0-9.]+)", out)
    if not m:
        raise RuntimeError("no PPL in output:\n" + out[-2000:])
    return float(m.group(1)), float(m.group(2))


def git_commit(path):
    try:
        return subprocess.run(["git", "-C", path, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    except OSError:
        return ""


def measure(a, models, modes):
    """All models x modes; each speed round runs every model and mode back to back under one lock
    hold (released and re-taken between models once a hold exceeds --max-hold seconds)."""
    t = str(a.threads)
    res = {
        (mo, m): {"decode": [], "prefill": [], "dJ": [], "pJ": [], "load": [], "cw": [], "drift": [], "mem": None}
        for mo in models
        for m in modes
    }

    def note(x, r):
        x["cw"].append(r["cpu_s"] / max(r["wall_s"], 1e-9))
        x["drift"].append(abs(r["drift_s"]))

    def speed(model, rep, lk):
        for m, (bindir, env, flags) in modes.items():
            x = res[(model, m)]
            x["load"].append(lk.load)
            bench = [os.path.join(bindir, "llama-bench"), "-m", model, "-t", t, "-r", "1", "-o", "json", *pin(a.threads), *flags]
            base = run(bench + ["-p", "0", "-n", "1"], env)
            note(x, base)
            if "decode" in a.metrics:
                r = run(bench + ["-p", "0", "-n", str(a.n_gen)], env)
                note(x, r)
                x["decode"].append(bench_json(r["out"])[0]["avg_ts"])
                x["dJ"].append((r["cpu_s"] - base["cpu_s"]) * WATTS_PER_CORE / (a.n_gen - 1))
                if x["mem"] is None or r["rss_mb"] > x["mem"]["rss_mb"]:
                    x["mem"] = r
            if "prefill" in a.metrics:
                r = run(bench + ["-p", str(a.n_prompt), "-n", "0"], env)
                note(x, r)
                x["prefill"].append(bench_json(r["out"])[0]["avg_ts"])
                x["pJ"].append((r["cpu_s"] - base["cpu_s"]) * WATTS_PER_CORE / (2 * a.n_prompt))
            name = os.path.basename(model)
            print(f"  rep {rep + 1} {name} {m}: decode {x['decode'][-1:]} prefill {x['prefill'][-1:]} load {lk.load}", flush=True)

    def held(items, fn, rep):
        todo = list(items)
        while todo:
            with BenchLock() as lk:
                t0 = time.monotonic()
                while todo and (time.monotonic() - t0 < a.max_hold or not lk.fd):
                    fn(todo.pop(0), rep, lk)

    if "decode" in a.metrics or "prefill" in a.metrics:
        for rep in range(a.reps):
            held(models, speed, rep)
    ppl = {(mo, m): {} for mo in models for m in modes}

    def perplexity(model, unlocked):
        for kind in ("ppl_ub1", "ppl_batched"):
            if kind not in a.metrics:
                continue
            for m, (bindir, env, flags) in modes.items():
                pflags = [
                    "--no-repack" if f == "--repack" and flags[i + 1] == "0" else f
                    for i, f in enumerate(flags)
                    if not (i > 0 and flags[i - 1] == "--repack")
                ]
                pt = str(a.ppl_threads or t)
                cmd = [os.path.join(bindir, "llama-perplexity"), "-m", model, "-f", a.ppl_text, "-t", pt, *pin(int(pt)), *pflags]
                if unlocked:
                    cmd = ["nice", "-n", "19", *cmd]
                if kind == "ppl_ub1":
                    cmd += ["-c", "128", "-b", "128", "-ub", "1", "--chunks", str(a.ppl_ub1_chunks)]
                else:
                    cmd += ["-c", "512", "-b", "512", "-ub", "512"] + (["--chunks", str(a.ppl_chunks)] if a.ppl_chunks else [])
                r = run(cmd, env)
                ppl[(model, m)][kind] = ppl_value(r["out"])
                print(f"  {kind} {os.path.basename(model)} {m}: {ppl[(model, m)][kind]}", flush=True)

    if any(k.startswith("ppl") for k in a.metrics):
        if a.ppl_unlocked_small:
            # perplexity is not timed: the lock is only needed for the RAM (models above ~1.5 GB)
            for model in models:
                small = os.path.getsize(model) < 1.5e9
                with contextlib.nullcontext() if small else BenchLock():
                    perplexity(model, small)
        else:
            held(models, lambda model, rep, lk: perplexity(model, False), 0)
    rows = []
    for model in models:
        for m in modes:
            x = res[(model, m)]
            p = ppl[(model, m)]

            def med(v):
                return round(statistics.median(v), 3) if v else ""

            def spread(v):
                return round((max(v) - min(v)) / statistics.median(v), 3) if len(v) > 1 else ""

            mem = x["mem"] or {}
            rows.append(
                {
                    "model": os.path.basename(model).removesuffix(".gguf"),
                    "mode": m,
                    "threads": a.threads,
                    "decode_tok_s": med(x["decode"]),
                    "decode_spread": spread(x["decode"]),
                    "decode_J_tok": med(x["dJ"]),
                    "prefill_tok_s": med(x["prefill"]),
                    "prefill_spread": spread(x["prefill"]),
                    "prefill_J_tok": med(x["pJ"]),
                    "ppl_ub1": p.get("ppl_ub1", ("", ""))[0],
                    "ppl_ub1_err": p.get("ppl_ub1", ("", ""))[1],
                    "ppl_batched": p.get("ppl_batched", ("", ""))[0],
                    "ppl_batched_err": p.get("ppl_batched", ("", ""))[1],
                    "rss_mb": round(mem.get("rss_mb", 0)) or "",
                    "anon_mb": round(mem.get("anon_mb", 0)) or "",
                    "file_mb": round(mem.get("file_mb", 0)) or "",
                    "max_load": max(x["load"]) if x["load"] else "",
                    "min_cpu_wall": round(min(x["cw"]), 2) if x["cw"] else "",
                    "max_drift_s": round(max(x["drift"]), 3) if x["drift"] else "",
                    "reps": a.reps,
                    "commit": git_commit(modes[m][0]),
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
            )
    return rows


def summary(path):
    rows = list(csv.DictReader(open(path)))
    cols = [
        "model",
        "mode",
        "decode_tok_s",
        "decode_J_tok",
        "prefill_tok_s",
        "prefill_J_tok",
        "ppl_ub1",
        "ppl_batched",
        "rss_mb",
        "anon_mb",
        "max_load",
    ]
    print("| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for r in rows:
        print("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+")
    ap.add_argument("--out", default="e2e.csv")
    ap.add_argument("--stock-bin", default=os.path.expanduser("~/src/llama.cpp/build/bin"))
    ap.add_argument("--kurn-bin", default=os.path.expanduser("~/src/llama-kurn/build/bin"))
    ap.add_argument("--modes", default="default,plain,kurn", help="comma list of predefined mode names")
    ap.add_argument("--mode", action="append", default=[], help="custom NAME=BIN_DIR[:ENV=V,...][:flags]")
    ap.add_argument("--ppl-unlocked-small", action="store_true", help="perplexity of models < 1.5 GB at nice 19 without the bench lock")
    ap.add_argument("--ppl-threads", type=int, default=0, help="threads for perplexity runs (default --threads)")
    ap.add_argument("--skip", default="", help=f"comma list of {','.join(METRICS)}")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--max-hold", type=float, default=480, help="seconds per bench-lock hold before re-queueing")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--n-gen", type=int, default=128)
    ap.add_argument("--n-prompt", type=int, default=512)
    ap.add_argument("--ppl-text", default="/tmp/kurn-e2e-ppl.txt")
    ap.add_argument("--ppl-ub1-chunks", type=int, default=8, help="128-token chunks for the ubatch-1 perplexity")
    ap.add_argument("--ppl-chunks", type=int, default=0, help="512-token chunks for the batched perplexity (0: all)")
    ap.add_argument("--summary", help="print a markdown table of a results CSV and exit")
    a = ap.parse_args(argv)
    if a.summary:
        summary(a.summary)
        return
    if not a.models:
        ap.error("--models is required")
    a.metrics = [m for m in METRICS if m not in a.skip.split(",")]
    known = default_modes(a.stock_bin, a.kurn_bin)
    modes = {m: known[m] for m in a.modes.split(",") if m}
    modes.update(parse_mode(s) for s in a.mode)
    for name, (bindir, _, _) in modes.items():
        if not os.path.exists(os.path.join(bindir, "llama-bench")):
            sys.exit(f"mode {name}: no llama-bench in {bindir}")
    if any(k.startswith("ppl") for k in a.metrics) and not os.path.exists(a.ppl_text):
        subprocess.run([os.path.join(HERE, "make_ppl_text.sh"), a.ppl_text], check=True)
    new = not os.path.exists(a.out) or os.path.getsize(a.out) == 0
    with open(a.out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for row in measure(a, [os.path.expanduser(m) for m in a.models], modes):
            w.writerow(row)
            f.flush()
            print("  " + json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
