#!/usr/bin/env python3
"""Speculative decoding (draft model) vs plain greedy decode, stock ggml vs kurn builds.

Per build, prompt and repetition (each (rep, build) pair runs under the shared bench lock):
  greedy   llama-completion --temp 0 (non-speculative reference; output text kept for identity)
  spec-tdN llama-speculative-simple -md DRAFT --temp 0, draft model on N threads
Reported: decode tok/s (tool's own decode timer), acceptance, J/token proxy (busy CPU seconds
x 5.47 W per generated token, minus a -n 1 run of the same command to remove load/prompt cost),
and whether the generated text equals the non-speculative greedy text of the same build.

    run_spec.py --target ~/models/Qwen3-8B-Q8_0.gguf --draft ~/models/Qwen3-0.6B-Q8_0.gguf --out spec.csv
    run_spec.py --summary spec.csv
"""

import argparse
import csv
import os
import re
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_e2e import WATTS_PER_CORE, BenchLock, bench_json, git_commit, pin, run  # noqa: E402

PROMPTS = [
    "Write a Python function that parses an ISO 8601 date string and returns a datetime object. Include error handling.",
    "Explain step by step how a CPU cache works, including cache lines, associativity and eviction policies.",
    "Summarize the plot of Pride and Prejudice by Jane Austen in a few paragraphs.",
]
FIELDS = [
    "build",
    "config",
    "prompt",
    "tok_s",
    "tok_s_spread",
    "J_tok",
    "accept_pct",
    "n_drafted",
    "n_accept",
    "identical",
    "first_diff",
    "max_load",
    "reps",
    "commit",
]


def chat(p):
    return f"<|im_start|>user\n{p} /no_think<|im_end|>\n<|im_start|>assistant\n"


def greedy_cmd(bindir, a, prompt, n):
    return [os.path.join(bindir, "llama-completion"), "-m", a.target, "-p", chat(prompt), "-n", str(n), "--temp", "0",
            "-t", str(a.threads), "-c", "4096", "-no-cnv", "--no-warmup", "--ignore-eos", "--no-display-prompt",
            "--log-verbosity", "3", *pin(a.threads)]  # fmt: skip


def spec_cmd(bindir, a, prompt, n, td):
    return [os.path.join(bindir, "llama-speculative-simple"), "-m", a.target, "-md", a.draft, "-p", chat(prompt),
            "--spec-type", "draft-simple", "-n", str(n), "--temp", "0", "-t", str(a.threads), "-td", str(td),
            "-c", "4096", "--ignore-eos",
            "--spec-draft-n-max", str(a.draft_max), "--spec-draft-p-min", str(a.p_min), *pin(a.threads),
            "--spec-draft-cpu-mask", f"{(1 << td) - 1:x}", "--spec-draft-cpu-strict", "1"]  # fmt: skip


def parse(out, kind):
    if kind == "greedy":
        m = re.search(r"eval time =\s*([0-9.]+) ms /\s*(\d+) runs", out.split("prompt eval time")[-1])
        tok_s = int(m.group(2)) / (float(m.group(1)) / 1000) if m and float(m.group(1)) > 0 else float("nan")
        return {"tok_s": tok_s, "n": int(m.group(2)) + 1 if m else 0}
    m = re.search(r"decoded\s+(\d+) tokens in\s+([0-9.]+) seconds", out)
    d = re.search(r"n_drafted = (\d+)", out)
    c = re.search(r"n_accept  = (\d+)", out)
    return {
        "tok_s": int(m.group(1)) / float(m.group(2)) if m else float("nan"),
        "n": int(m.group(1)) if m else 0,
        "n_drafted": int(d.group(1)) if d else 0,
        "n_accept": int(c.group(1)) if c else 0,
    }


def gen_text(out, kind, prompt):
    """Generated text only (stdout minus the prompt echo of llama-speculative-simple)."""
    if kind == "spec":
        out = out.split(chat(prompt), 1)[-1]
    return out.strip()


def measure(a):
    builds = dict(b.split("=", 1) for b in a.builds.split(","))
    builds = {k: os.path.expanduser(v) for k, v in builds.items()}
    configs = ["greedy"] + [f"spec-td{td}" for td in a.td]
    prompts = PROMPTS[: a.n_prompts]
    res = {}
    for rep in range(a.reps):
        for b, bindir in builds.items():
            with BenchLock() as lk:
                for pi, p in enumerate(prompts):
                    for cfg in configs:
                        kind = "greedy" if cfg == "greedy" else "spec"

                        def cmd(n, cfg=cfg, p=p, bindir=bindir):
                            if cfg == "greedy":
                                return greedy_cmd(bindir, a, p, n)
                            return spec_cmd(bindir, a, p, n, int(cfg[7:]))

                        env = {"GGML_KURN_AMX": "1" if a.amx else "0"}
                        r = run(cmd(a.n), env, split=True)
                        r1 = run(cmd(1), env, split=True) if rep == 0 or a.energy_every_rep else None
                        out, err = r["out"], r["err"]
                        x = res.setdefault((b, cfg, pi), {"tok_s": [], "J": [], "load": [], "text": None, "d": 0, "c": 0})
                        v = parse(err, kind)
                        x["tok_s"].append(v["tok_s"])
                        if r1:
                            n1 = parse(r1["err"], kind)["n"]
                            x["J"].append((r["cpu_s"] - r1["cpu_s"]) * WATTS_PER_CORE / max(v["n"] - n1, 1))
                        x["load"].append(lk.load)
                        x["d"] += v.get("n_drafted", 0)
                        x["c"] += v.get("n_accept", 0)
                        if x["text"] is None:
                            x["text"] = gen_text(out, kind, p)
                        print(f"rep {rep + 1} {b} {cfg} p{pi}: {v} load {lk.load}", flush=True)
    rows = []
    for (b, cfg, pi), x in res.items():
        ref = res[(b, "greedy", pi)]["text"]
        txt = x["text"]
        # speculative runs overshoot -n by up to draft_max tokens: compare the common prefix length
        n = min(len(ref), len(txt))
        diff = next((i for i, (u, w) in enumerate(zip(ref, txt)) if u != w), n)
        same = diff == n
        rows.append({
            "build": b,
            "config": cfg,
            "prompt": pi,
            "tok_s": round(statistics.median(x["tok_s"]), 2),
            "tok_s_spread": round((max(x["tok_s"]) - min(x["tok_s"])) / statistics.median(x["tok_s"]), 3),
            "J_tok": round(statistics.median(x["J"]), 3) if x["J"] else "",
            "accept_pct": round(100 * x["c"] / x["d"], 1) if x["d"] else "",
            "n_drafted": x["d"],
            "n_accept": x["c"],
            "identical": same,
            "first_diff": "" if same else diff,
            "max_load": max(x["load"]),
            "reps": a.reps,
            "commit": git_commit(builds[b]),
        })  # fmt: skip
        if not same:
            print(f"  {b} {cfg} p{pi} differs at char {diff}: ref ...{ref[max(0, diff - 40) : diff + 40]!r}")
            print(f"  {' ' * len(b)} {' ' * len(cfg)}          got ...{txt[max(0, diff - 40) : diff + 40]!r}")
    return rows


def verify_m(a):
    """Time per target forward pass vs number of tokens M (the verify batch), per build."""
    builds = {k: os.path.expanduser(v) for k, v in (b.split("=", 1) for b in a.builds.split(","))}
    ms = ",".join(str(m) for m in (1, 2, 3, 4, 5, 6, 7, 8, 16))
    rows = []
    for rep in range(a.reps):
        with BenchLock() as lk:
            for b, bindir in builds.items():
                for name, flags in (("", []), ("-plain", ["--repack", "0"])) if b == "stock" else (("", []),):
                    cmd = [os.path.join(bindir, "llama-bench"), "-m", a.target, "-t", str(a.threads), "-p", ms, "-n", "0",
                           "-r", "3", "-o", "json", *pin(a.threads), *flags]  # fmt: skip
                    for x in bench_json(run(cmd, {})["out"]):
                        rows.append({"build": b + name, "M": x["n_prompt"], "ms_per_pass": 1000 * x["n_prompt"] / x["avg_ts"],
                                     "rep": rep, "load": lk.load})  # fmt: skip
                        print(rows[-1], flush=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["build", "M", "ms_per_pass", "rep", "load"])
        w.writeheader()
        w.writerows(rows)


def summary(path):
    rows = list(csv.DictReader(open(path)))
    print("| build | config | prompt | tok/s | J/token | accept % | identical to greedy |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['build']} | {r['config']} | {r['prompt']} | {r['tok_s']} | {r['J_tok']} | {r['accept_pct']} | {r['identical']} |")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target")
    ap.add_argument("--draft")
    ap.add_argument("--out")
    ap.add_argument("--builds", default="stock=~/src/llama.cpp/build/bin,kurn=~/src/llama-kurn/build/bin")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--td", type=lambda s: [int(x) for x in s.split(",")], default=[8, 2])
    ap.add_argument("--n", type=int, default=192)
    ap.add_argument("--draft-max", type=int, default=8)
    ap.add_argument("--p-min", type=float, default=0.0)
    ap.add_argument("--n-prompts", type=int, default=3)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--amx", action="store_true", help="GGML_KURN_AMX=1 (kurn AMX prompt prefill) for every run")
    ap.add_argument("--energy-every-rep", action="store_true")
    ap.add_argument("--verify-m", action="store_true", help="only time target passes vs M = 1..8, 16 (llama-bench)")
    ap.add_argument("--summary")
    a = ap.parse_args(argv)
    if a.summary:
        return summary(a.summary)
    if a.verify_m:
        a.target = os.path.expanduser(a.target)
        return verify_m(a)
    a.target, a.draft = os.path.expanduser(a.target), os.path.expanduser(a.draft)
    rows = measure(a)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {a.out} ({time.strftime('%H:%M:%S')})")
    summary(a.out)


if __name__ == "__main__":
    main()
