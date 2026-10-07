#!/usr/bin/env python3
"""Cost-aware verify width vs fixed draft lengths, end to end in llama-speculative-simple on the KURN buffer type.

Needs a llama.cpp checkout with integration/llama.cpp/apply.sh and spec-width/apply.sh applied and built.

    run_width.py calib  --target T.gguf --draft D.gguf --out DIR [--prompt-set main|heldout|all]
        cost table (first prompt, unless DIR/calib.cost exists) + one greedy trace per prompt
    run_width.py run    --target T.gguf --draft D.gguf --out DIR [--configs k1,k4,policy,...] [--reps 2]
    run_width.py summary DIR [--csv e2e.csv]

Configs: greedy = plain decoding (llama-completion, the identity reference, always run), kN = fixed draft length N
(p_min 0), kN-pP = draft length N with llama.cpp's p_min cutoff P, policy = KURN_SPEC_WIDTH (cap + per-token stop +
truncation), cap = KURN_SPEC_WIDTH_MODE=cap (rate-only cap).
Each (rep, prompt) runs every config back to back, configs in a rotated order per rep. Output text of every
speculative run is compared with plain greedy decoding (llama-completion --temp 0) of the same build.
"""

import argparse
import csv
import os
import re
import statistics
import subprocess
import sys
import time

HERE = os.path.expanduser("~/work/Kurn-gpu/benchmarks/v0.2/specwidth")
sys.path.insert(0, os.path.join(HERE, "..", "e2e"))
from run_spec import PROMPTS, chat  # noqa: E402
from run_spec import parse as parse_greedy  # noqa: E402

# Held out from the policy's development (its defaults were fixed on PROMPTS only).
HELDOUT = [
    "Rewrite this Python function to use a dictionary comprehension and add type hints:\n\n"
    "def squares(nums):\n    out = {}\n    for n in nums:\n        if n % 2 == 0:\n            out[n] = n * n\n    return out",
    "Extract every person, organization and date from the following text and return them as a JSON object with keys "
    "people, organizations, dates: 'On March 3, 2021, Maria Lopez joined Acme Robotics as CTO, after ten years at "
    "Globex. The board, chaired by Kenji Watanabe, approved the hire on February 27.'",
    "Translate into French: 'The meeting has been moved to Thursday afternoon because the conference room is being "
    "renovated. Please bring your laptop and the quarterly figures.'",
    "A train leaves at 9:40 and travels 210 km at 84 km/h, then stops for 12 minutes, then travels 95 km at 76 km/h. "
    "At what time does it arrive? Show your reasoning step by step.",
    "List twenty common kitchen tools with a one-sentence description of each.",
]
FIELDS = ["config", "prompt", "rep", "tok_s", "n_predict", "n_drafted", "n_accept", "accept_pct", "widths", "identical", "first_diff",
          "load"]  # fmt: skip


def pin(threads):
    return ["--cpu-mask", f"{(1 << threads) - 1:x}", "--cpu-strict", "1"]


def base_args(a, prompt):
    return ["-m", a.target, "-p", chat(prompt), "-n", str(a.n), "--temp", "0", "-t", str(a.threads), "-c", "4096", *pin(a.threads)]


def spec_args(a, prompt):
    return [*base_args(a, prompt), "-md", a.draft, "--spec-type", "draft-simple", "-td", str(a.threads), "--ignore-eos",
            "--spec-draft-cpu-mask", f"{(1 << a.threads) - 1:x}", "--spec-draft-cpu-strict", "1"]  # fmt: skip


def run(cmd, env):
    # errors="replace": past EOS (--ignore-eos) a token boundary can split a UTF-8 sequence
    r = subprocess.run(cmd, capture_output=True, text=True, errors="replace", env={**os.environ, **env}, timeout=3600)
    if r.returncode:
        raise RuntimeError(f"{' '.join(cmd[:3])} ... exit {r.returncode}: {r.stderr[-2000:]}")
    return r


def config_cmd(a, cfg, prompt):
    exe = os.path.join(a.bin, "llama-speculative-simple")
    env = {}
    if cfg in ("policy", "cap"):
        env["KURN_SPEC_WIDTH"] = a.table or os.path.join(a.out, "calib.cost")
        if cfg == "cap":
            env["KURN_SPEC_WIDTH_MODE"] = "cap"
        n_max, p_min = a.k_max, 0.0
    else:
        m = re.fullmatch(r"k(\d+)(?:-p([0-9.]+))?", cfg)
        if not m:
            raise ValueError(f"unknown config {cfg}")
        n_max, p_min = int(m.group(1)), float(m.group(2) or 0)
    return [exe, *spec_args(a, prompt), "--spec-draft-n-max", str(n_max), "--spec-draft-p-min", str(p_min)], env


def parse(err):
    g = lambda pat, f=float: (lambda m: f(m.group(1)) if m else None)(re.search(pat, err))  # noqa: E731
    m = re.search(r"decoded\s+(\d+) tokens in\s+([0-9.]+) seconds", err)
    return {
        "tok_s": int(m.group(1)) / float(m.group(2)) if m else float("nan"),
        "n_predict": g(r"n_predict = (\d+)", int),
        "n_drafted": g(r"n_drafted = (\d+)", int),
        "n_accept": g(r"n_accept  = (\d+)", int),
        "widths": (lambda m: m.group(1).strip() if m else "")(re.search(r"verify widths \(M:steps\) =(.*)", err)),
    }


def gen_text(out, prompt):
    return out.split(chat(prompt), 1)[-1].strip()


def prompt_set(a):
    """[(name, prompt)]: p0.. for the development prompts, h0.. for the held-out ones."""
    main = [(f"p{i}", p) for i, p in enumerate(PROMPTS)]
    held = [(f"h{i}", p) for i, p in enumerate(HELDOUT)]
    return {"main": main, "heldout": held, "all": main + held}[a.prompt_set]


def cmd_calib(a):
    os.makedirs(a.out, exist_ok=True)
    exe = os.path.join(a.bin, "kurn-spec-calib")
    table = os.path.join(a.out, "calib.cost")
    for name, p in prompt_set(a):
        first = not os.path.exists(table)
        env = {"GGML_KURN_AMX": "0", "KURN_CALIB_OUT": os.path.join(a.out, name), "KURN_CALIB_MMAX": str(a.m_max),
               "KURN_CALIB_REPS": str(a.calib_reps), "KURN_CALIB_TABLE": "1" if first else "0"}  # fmt: skip
        r = run([exe, *spec_args(a, p)], env)
        print(r.stderr[-3000:] if first else r.stderr.strip().splitlines()[-1], flush=True)
        if first:
            os.replace(os.path.join(a.out, f"{name}.cost"), table)


def cmd_run(a):
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, a.csv)
    new = not os.path.exists(path)
    fh = open(path, "a", newline="")
    w = csv.DictWriter(fh, fieldnames=FIELDS)
    if new:
        w.writeheader()
    prompts = prompt_set(a)
    refs = {}
    for name, p in prompts:
        r = run([os.path.join(a.bin, "llama-completion"), *base_args(a, p), "-no-cnv", "--no-warmup", "--ignore-eos",
                 "--no-display-prompt", "--log-verbosity", "3"], {})  # fmt: skip
        refs[name] = r.stdout.strip()
        g = parse_greedy(r.stderr, "greedy")
        w.writerow({"config": "greedy", "prompt": name, "rep": 0, "tok_s": g["tok_s"], "n_predict": g["n"], "identical": True,
                    "load": float(open("/proc/loadavg").read().split()[0])})  # fmt: skip
        print(f"greedy {name}: {g['tok_s']:.2f} tok/s", flush=True)
    cfgs = a.configs.split(",")
    for rep in range(a.reps):
        order = cfgs[rep % len(cfgs) :] + cfgs[: rep % len(cfgs)]
        for name, p in prompts:
            for cfg in order:
                cmd, env = config_cmd(a, cfg, p)
                load = float(open("/proc/loadavg").read().split()[0])
                r = run(cmd, env)
                v = parse(r.stderr)
                txt, ref = gen_text(r.stdout, p), refs[name]
                n = min(len(txt), len(ref))
                diff = next((i for i, (x, y) in enumerate(zip(ref, txt)) if x != y), n)
                row = {"config": cfg, "prompt": name, "rep": rep, **v, "load": load, "identical": diff == n,
                       "first_diff": "" if diff == n else diff,
                       "accept_pct": round(100 * v["n_accept"] / v["n_drafted"], 1) if v["n_drafted"] else ""}  # fmt: skip
                w.writerow(row)
                fh.flush()
                print(f"rep {rep} {name} {cfg:10s} {v['tok_s']:6.2f} tok/s  accept {row['accept_pct']}  widths {v['widths']}  "
                      f"identical {row['identical']}  load {load}  ({time.strftime('%H:%M:%S')})", flush=True)  # fmt: skip


def cmd_summary(a):
    rows = list(csv.DictReader(open(os.path.join(a.dir, a.csv))))
    prompts = list(dict.fromkeys(r["prompt"] for r in rows))
    cfgs = list(dict.fromkeys(r["config"] for r in rows))
    print("| config | " + " | ".join(f"{p} tok/s" for p in prompts) + " | mean | accept % | identical |")
    print("|---|" + "---|" * (len(prompts) + 3))
    for c in cfgs:
        med = []
        for p in prompts:
            xs = [float(r["tok_s"]) for r in rows if r["config"] == c and r["prompt"] == p]
            med.append(statistics.median(xs) if xs else float("nan"))
        rr = [r for r in rows if r["config"] == c]
        d = sum(int(r["n_drafted"] or 0) for r in rr)
        acc = sum(int(r["n_accept"] or 0) for r in rr)
        ident = all(r["identical"] == "True" for r in rr)
        rate = 100 * acc / d if d else 0
        print(f"| {c} | " + " | ".join(f"{x:.2f}" for x in med) + f" | {statistics.mean(med):.2f} | {rate:.1f} | {ident} |")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["calib", "run", "summary"])
    ap.add_argument("dir", nargs="?")
    ap.add_argument("--target")
    ap.add_argument("--draft")
    ap.add_argument("--out")
    ap.add_argument("--csv", default="e2e.csv")
    ap.add_argument("--table", help="cost table for policy / cap (default DIR/calib.cost of --out)")
    ap.add_argument("--bin", default=os.path.expanduser("~/src/llama-kurn/build/bin"))
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--prompt-set", default="main", choices=["main", "heldout", "all"])
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--k-max", type=int, default=15)
    ap.add_argument("--m-max", type=int, default=24)
    ap.add_argument("--calib-reps", type=int, default=5)
    ap.add_argument("--configs", default="k1,k2,k3,k4,k7,k8,k11,k15,k15-p0.7,cap,policy")
    a = ap.parse_args(argv)
    if a.cmd == "summary":
        return cmd_summary(a)
    a.target, a.draft = os.path.expanduser(a.target), os.path.expanduser(a.draft)
    return {"calib": cmd_calib, "run": cmd_run}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
