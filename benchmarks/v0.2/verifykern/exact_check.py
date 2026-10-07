#!/usr/bin/env python3
"""Speculative output == no-draft output in llama-server with GGML_KURN_FA_MODE=exact (batch-invariant attention plus
the buffer type's batch-invariant matmuls). exact_check.py BIN_DIR TABLE OUT.json"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import server_bench as sb  # noqa: E402
import subprocess  # noqa: E402

N = 128


def texts(bindir, mode):
    env = dict(os.environ, GGML_KURN_FA_MODE="exact")
    args = [f"{bindir}/llama-server", "-m", sb.TARGET, "-t", "8", "-c", "4096", "--port", str(sb.PORT), "-fa", "on", "-np", "1",
            "-lm", "none", "--cpu-mask", "ff", "--cpu-strict", "1"]
    if mode != "nodraft":
        args += ["-md", sb.DRAFT, "--spec-type", "draft-simple", "-td", "8", "--spec-draft-p-min", "0"]
        args += ["--spec-width", mode.split("=", 1)[1], "--spec-draft-n-max", "15"] if mode.startswith("policy=") else ["--spec-draft-n-max", mode[1:]]
    proc = subprocess.Popen(args, stdout=open(f"{sb.W}/logs/srv-exact-{mode[:6]}.log", "w"), stderr=subprocess.STDOUT, env=env)
    try:
        sb.wait_up(proc)
        out = []
        for p in sb.PROMPTS + sb.HELDOUT:
            r = sb.post("/completion", {"prompt": sb.chat(p), "n_predict": N, "temperature": 0, "ignore_eos": True, "cache_prompt": False})
            out.append({"content": r["content"], "tokens": r.get("tokens"), "draft_n": r["timings"].get("draft_n")})
        return out
    finally:
        proc.terminate()
        proc.wait(30)


def main():
    bindir, table, outp = sys.argv[1:4]
    ref = texts(bindir, "nodraft")
    res = {"nodraft": ref}
    for mode in (f"policy={table}", "k7", "k3"):
        got = texts(bindir, mode)
        res[mode] = got
        same = [a["content"] == b["content"] for a, b in zip(ref, got)]
        print(f"{mode}: {sum(same)}/{len(same)} identical to no-draft; drafted {[g['draft_n'] for g in got]}", flush=True)
    json.dump(res, open(outp, "w"), indent=1)


if __name__ == "__main__":
    main()
