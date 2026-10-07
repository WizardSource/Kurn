#!/usr/bin/env python3
"""llama-server speculative decoding bench: one server per config, 8 prompts x N tokens (temp 0, ignore_eos),
tok/s from the server's own timings. Appends rows to OUT (jsonl).

    server_bench.py OUT CONFIG [CONFIG ...]     CONFIG = name:bin_dir:mode[:extra env as K=V,K=V]
        mode: nodraft | kN (fixed draft length N) | policy=TABLE (--spec-width TABLE, k <= 15)
"""
import json
import os
import statistics
import subprocess
import sys
import time
import urllib.request

sys.path.insert(0, os.path.expanduser("~/work/Kurn-gpu/benchmarks/v0.2/specwidth"))
sys.path.insert(0, os.path.expanduser("~/work/Kurn-gpu/benchmarks/v0.2/e2e"))
from run_spec import PROMPTS, chat  # noqa: E402
from run_width import HELDOUT  # noqa: E402

W = os.path.expanduser("~/work")
TARGET, DRAFT = f"{W}/models/Qwen3-8B-Q8_0.gguf", f"{W}/models/Qwen3-0.6B-Q8_0.gguf"
N = int(os.environ.get("N_PREDICT", 256))
PORT = 8091


def post(path, body, timeout=900):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def wait_up(proc, t=300):
    t0 = time.time()
    while time.time() - t0 < t:
        if proc.poll() is not None:
            raise RuntimeError("server exited")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2) as r:
                if r.status == 200:
                    return
        except Exception:
            pass
        time.sleep(1)
    raise RuntimeError("server did not come up")


def run_config(spec, out):
    parts = spec.split(":")
    name, bindir, mode = parts[0], parts[1], parts[2]
    env = dict(os.environ)
    if len(parts) > 3 and parts[3]:
        env.update(dict(kv.split("=", 1) for kv in parts[3].split(",")))
    args = [f"{bindir}/llama-server", "-m", TARGET, "-t", "8", "-c", "4096", "--port", str(PORT), "-fa", "on", "-np", "1",
            "-lm", "none", "--cpu-mask", "ff", "--cpu-strict", "1"]
    if mode != "nodraft":
        args += ["-md", DRAFT, "--spec-type", "draft-simple", "-td", "8", "--spec-draft-p-min", "0"]
        if mode.startswith("policy="):
            args += ["--spec-width", mode.split("=", 1)[1], "--spec-draft-n-max", "15"]
        else:
            args += ["--spec-draft-n-max", mode[1:]]
    log = open(f"{W}/logs/srv-{name}.log", "w")
    proc = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT, env=env)
    try:
        wait_up(proc)
        post("/completion", {"prompt": chat("Say hi."), "n_predict": 8, "temperature": 0})  # warm-up
        rates = []
        for i, p in enumerate(PROMPTS + HELDOUT):
            r = post("/completion", {"prompt": chat(p), "n_predict": N, "temperature": 0, "ignore_eos": True, "cache_prompt": False})
            t = r["timings"]
            row = {"config": name, "prompt": i, "tok_s": t["predicted_per_second"], "n": t["predicted_n"],
                   "draft_n": t.get("draft_n"), "draft_acc": t.get("draft_n_accepted"), "load": os.getloadavg()[0]}
            rates.append(row["tok_s"])
            out.write(json.dumps(row) + "\n")
            out.flush()
        print(f"{name}: mean {statistics.mean(rates):.2f} tok/s  " + " ".join(f"{x:.1f}" for x in rates), flush=True)
    finally:
        proc.terminate()
        try:
            proc.wait(30)
        except subprocess.TimeoutExpired:
            proc.kill()
        time.sleep(2)


def main():
    out = open(sys.argv[1], "a")
    for spec in sys.argv[2:]:
        run_config(spec, out)


if __name__ == "__main__":
    main()
