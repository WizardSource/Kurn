#!/usr/bin/env python3
"""llama-server benchmark for the kurn retrieval-arm selector (LLAMA_SERVER=.../llama-server MODELS=dir).

  bench.py CONFIG WORKLOAD [REPS]     -> appends JSON lines to results.jsonl

Each run starts its own server, sends the workload's prompts one at a time (temperature 0, exact mode
unless KURN_FAST=1), and records tokens/s from the server's own timings, the text, and the server log's
kurn arm statistics.
"""
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.request

B = os.environ.get("LLAMA_SERVER", "llama-server")
M = os.environ.get("MODELS", "models")  # Qwen3-1.7B/8B/0.6B Q8_0 GGUFs and the q17.cost / q8b.cost tables
PORT = int(os.environ.get("PORT", "8097"))
HERE = os.path.dirname(os.path.abspath(__file__))

CODE = open(os.path.join(HERE, "buggy.py")).read()


def chat(u, prefill=""):
    return f"<|im_start|>user\n{u}<|im_end|>\n<|im_start|>assistant\n{prefill}"


WORKLOADS = {
    "think": ("1.7B", 384, [chat("How many positive integers below 1000 are divisible by 3 or 5 but not by 15? Think carefully.")]),
    "refine": ("1.7B", 384, [chat("The function `median` in this file crashes on an empty list and returns the wrong value for even-length "
                                  "lists. Fix both bugs and output the complete corrected file, unchanged otherwise.\n\n```python\n"
                                  + CODE + "\n```", "<think>\n\n</think>\n\n```python\n")]),
    "eight": ("8B", 256, [chat(p + " /no_think") for p in [
        "Write a Python function that checks whether a string is a palindrome, with tests.",
        "Explain how a hash map handles collisions, and compare chaining with open addressing.",
        "Prove that the sum of the first n odd numbers is n squared.",
        "Write a C function that reverses a singly linked list in place, and explain its complexity.",
        "Summarize the causes of the French Revolution in five bullet points.",
        "Find all real x such that x^2 - 5x + 6 = 0, then check each root by substitution.",
        "Write a SQL query that returns the three highest-paid employees per department, and explain it.",
        "Describe how TCP congestion control works, including slow start and congestion avoidance.",
    ]]),
}

MODELS = {"1.7B": ("Qwen3-1.7B-Q8_0.gguf", "q17.cost"), "8B": ("Qwen3-8B-Q8_0.gguf", "q8b.cost")}

NG = ["--spec-ngram-simple-size-n", "3", "--spec-ngram-simple-size-m", "16"]


def configs(size):
    table = f"{M}/{MODELS[size][1]}"
    md = ["-md", f"{M}/Qwen3-0.6B-Q8_0.gguf", "-td", "8", "--spec-draft-n-max", "15"]
    return {
        "none": ([], {}),
        "draft_selector": (md + ["--spec-type", "draft-simple", "--spec-width", table], {}),
        "ngram": (["--spec-type", "ngram-simple"] + NG, {}),
        "ngram_selector": (["--spec-type", "ngram-simple", "--spec-width", table] + NG, {}),
        "union_naive": (md + ["--spec-type", "ngram-simple,draft-simple"] + NG, {}),
        "arms": (md + ["--spec-type", "ngram-simple,draft-simple", "--spec-width", table] + NG, {}),
        "arms_eager": (md + ["--spec-type", "ngram-simple,draft-simple", "--spec-width", table] + NG, {"KURN_SPEC_LAZY": "0"}),
    }


def start(model, args, env_extra, log):
    env = {**os.environ, **env_extra}
    if os.environ.get("KURN_FAST") != "1":
        env["GGML_KURN_FA_MODE"] = "exact"
    cmd = [B, "-m", f"{M}/{model}", "-t", "8", "-c", "8192", "-np", "1", "--port", str(PORT), "--no-webui", *args]
    p = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, preexec_fn=os.setsid)
    for _ in range(600):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=1) as r:
                if r.status == 200:
                    return p
        except Exception:
            if p.poll() is not None:
                raise RuntimeError("server exited")
            time.sleep(0.5)
    raise RuntimeError("server did not start")


def stop(p):
    os.killpg(p.pid, signal.SIGTERM)
    try:
        p.wait(timeout=60)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)


def post(body):
    r = urllib.request.Request(f"http://127.0.0.1:{PORT}/completion", data=json.dumps(body).encode(),
                               headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=3600) as fh:
        return json.loads(fh.read())


ARM_RE = re.compile(r"kurn spec arms: steps none (\d+), retrieval (\d+) \((\d+)/(\d+) accepted\), model (\d+) \((\d+)/(\d+) accepted\)")


def main():
    cfg, wl = sys.argv[1], sys.argv[2]
    reps = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    size, n, prompts = WORKLOADS[wl]
    args, env = configs(size)[cfg]
    logpath = os.path.join(HERE, "logs", f"{wl}-{cfg}.log")
    os.makedirs(os.path.dirname(logpath), exist_ok=True)
    with open(logpath, "w") as log:
        p = start(MODELS[size][0], args, env, log)
        try:
            for rep in range(reps):
                for i, pr in enumerate(prompts):
                    out = post({"prompt": pr, "n_predict": n, "temperature": 0, "cache_prompt": False, "seed": 1})
                    t = out["timings"]
                    rec = {"workload": wl, "config": cfg, "rep": rep, "prompt": i, "fast": os.environ.get("KURN_FAST") == "1",
                           "tok_s": t["predicted_per_second"], "n": t["predicted_n"], "prompt_ms": t["prompt_ms"], "predicted_ms": t["predicted_ms"], "draft_n": t.get("draft_n"),
                           "draft_accepted": t.get("draft_n_accepted"), "text": out["content"]}
                    with open(os.path.join(HERE, "results.jsonl"), "a") as fh:
                        fh.write(json.dumps(rec) + "\n")
                    print(f"{wl} {cfg} rep{rep} p{i}: {rec['tok_s']:.1f} tok/s, n {rec['n']}, draft {rec['draft_n']} acc {rec['draft_accepted']}", flush=True)
        finally:
            stop(p)
    arms = ARM_RE.findall(open(logpath).read())
    if arms:
        print("arm stats (per task):", arms[-len(prompts) * reps:])


if __name__ == "__main__":
    main()
