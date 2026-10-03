#!/usr/bin/env python3
"""Mixed prefill/decode scheduling in llama-server (parallel slots, continuous batching).

Scenario per run: --decoders clients each send a short prompt and stream --n-gen tokens; --arrive seconds
later a long-prompt client (--long-tokens prompt tokens, 32 generated) arrives, so its prefill shares
ubatches with the running decodes. Measured per configuration (median over --reps interleaved rounds,
each round under the shared bench lock):
  ttft_long      time to first token of the long-prompt request
  itl_p50/p95    inter-token latency of the decode streams over the whole run
  itl_max        worst decode stall (usually while the long prompt is prefilled)
  stall_ms       sum of decode inter-token gaps above 3x the median gap
  tok_s          generated tokens (all requests) / wall time
Configurations: NAME=BIN_DIR[:ENV=VAL,...][:extra llama-server flags], e.g.
  default=~/src/llama.cpp/build/bin   kurn=~/src/llama-kurn/build/bin   kurn-ub128=~/src/llama-kurn/build/bin::-ub 128

    run_server.py --model ~/models/Qwen3-1.7B-Q8_0.gguf --out server.csv
"""

import argparse
import csv
import json
import os
import socket
import statistics
import subprocess
import sys
import threading
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_e2e import BenchLock, parse_mode, pin  # noqa: E402

FIELDS = ["config", "ttft_long_ms", "itl_p50_ms", "itl_p95_ms", "itl_max_ms", "stall_ms", "tok_s", "max_load", "reps"]
DEFAULT_CONFIGS = [
    "default=~/src/llama.cpp/build/bin",
    "plain=~/src/llama.cpp/build/bin::--no-repack",
    "kurn=~/src/llama-kurn/build/bin:GGML_KURN_AMX=0",
    "kurn-amx=~/src/llama-kurn/build/bin:GGML_KURN_AMX=1",
]
TEXT = os.environ.get("KURN_PPL_TEXT", "/tmp/kurn-e2e-ppl.txt")


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def stream(port, prompt, n, out, t_start):
    body = json.dumps({"prompt": prompt, "n_predict": n, "stream": True, "temperature": 0, "ignore_eos": True,
                       "cache_prompt": False}).encode()  # fmt: skip
    req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", body, {"Content-Type": "application/json"})
    t0 = time.monotonic()
    stamps = []
    with urllib.request.urlopen(req, timeout=600) as r:
        for line in r:
            if line.startswith(b"data: "):
                d = json.loads(line[6:])
                if d.get("content") or d.get("tokens"):
                    stamps.append(time.monotonic())
                if d.get("stop"):
                    break
    out.update(t0=t0 - t_start, stamps=[s - t_start for s in stamps])


def scenario(port, a, long_prompt):
    t_start = time.monotonic()
    res = [{} for _ in range(a.decoders + 1)]
    th = []
    for i in range(a.decoders):
        p = f"Write a long story about a lighthouse keeper number {i}."
        th.append(threading.Thread(target=stream, args=(port, p, a.n_gen, res[i], t_start)))
        th[-1].start()
    time.sleep(a.arrive)
    th.append(threading.Thread(target=stream, args=(port, long_prompt, 32, res[-1], t_start)))
    th[-1].start()
    for t in th:
        t.join()
    wall = time.monotonic() - t_start
    gaps = []
    for r in res[:-1]:
        s = r["stamps"]
        gaps += [(b - c) * 1000 for c, b in zip(s, s[1:])]
    gaps.sort()
    med = statistics.median(gaps)
    long = res[-1]
    return {
        "ttft_long_ms": (long["stamps"][0] - long["t0"]) * 1000,
        "itl_p50_ms": med,
        "itl_p95_ms": gaps[int(0.95 * (len(gaps) - 1))],
        "itl_max_ms": gaps[-1],
        "stall_ms": sum(g for g in gaps if g > 3 * med),
        "tok_s": sum(len(r["stamps"]) for r in res) / wall,
    }


def start_server(bindir, env, flags, a, port):
    cmd = [os.path.join(bindir, "llama-server"), "-m", a.model, "-t", str(a.threads), "-np", str(a.decoders + 1),
           "-c", str(a.ctx), "--port", str(port), "--no-webui", "-cb", *pin(a.threads), *flags]  # fmt: skip
    p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=dict(os.environ, **env))
    for _ in range(600):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                if r.status == 200:
                    return p
        except OSError:
            pass
        if p.poll() is not None:
            raise RuntimeError(f"llama-server exited: {' '.join(cmd)}")
        time.sleep(0.5)
    p.kill()
    raise RuntimeError("llama-server did not become healthy")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--configs", nargs="*", default=DEFAULT_CONFIGS)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--decoders", type=int, default=3)
    ap.add_argument("--n-gen", type=int, default=160)
    ap.add_argument("--long-tokens", type=int, default=1500, help="approximate prompt length of the late request")
    ap.add_argument("--arrive", type=float, default=3.0)
    ap.add_argument("--ctx", type=int, default=16384)
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args(argv)
    a.model = os.path.expanduser(a.model)
    text = open(TEXT).read()
    long_prompt = "Summarize this text:\n" + text[: a.long_tokens * 4]
    configs = dict(parse_mode(c) for c in a.configs)
    res = {c: [] for c in configs}
    loads = {c: [] for c in configs}
    for rep in range(a.reps):
        for name, (bindir, env, flags) in configs.items():
            with BenchLock() as lk:
                port = free_port()
                srv = start_server(bindir, env, flags, a, port)
                try:
                    scenario(port, a, long_prompt[: len(long_prompt) // 4])  # warm-up
                    r = scenario(port, a, long_prompt)
                finally:
                    srv.terminate()
                    srv.wait(30)
            res[name].append(r)
            loads[name].append(lk.load)
            print(f"rep {rep + 1} {name}: " + " ".join(f"{k}={v:.1f}" for k, v in r.items()) + f" load {lk.load}", flush=True)
    rows = []
    for name, rs in res.items():
        row = {"config": name, "max_load": max(loads[name]), "reps": a.reps}
        for k in FIELDS[1:7]:
            row[k] = round(statistics.median(r[k] for r in rs), 1)
        rows.append(row)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print("| " + " | ".join(FIELDS[:7]) + " |")
    print("|" + "---|" * 7)
    for r in rows:
        print("| " + " | ".join(str(r[k]) for k in FIELDS[:7]) + " |")


if __name__ == "__main__":
    main()
