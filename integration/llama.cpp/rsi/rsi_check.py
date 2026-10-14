#!/usr/bin/env python3
"""End-to-end replay checks for a llama-server built with rsi/apply.sh.

  rsi_check.py --server BUILD/bin/llama-server --model TARGET.gguf --draft DRAFT.gguf --table TARGET.cost [--out result.json]

Every check starts its own server (exact mode: GGML_KURN_FA_MODE=exact) and sends seeded temperature-0.8 requests:
  probs     speculative runs (draft model, retrieval arms) return the same completion_probabilities as plain
            decoding, pre- and post-sampling, and no accepted draft token comes back as logprob 0 / empty top
  chain     the kurn_replay hash chain is equal across plain, draft model, arms, 4 concurrent slots and 4 threads,
            and differs in fast mode (it is sensitive to the numerics)
  resume    with kurn_rng_position, a request resumed from its own first 32 tokens continues with the same tokens;
            the sequential RNG does not
  manifest  the run manifest is present and stable across identical servers; with KURN_MANIFEST_SHA256=1 the
            model hash equals sha256 of the file
Exit status 0 when every check passes.
"""
import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request

PORT = int(os.environ.get("PORT", "8096"))
PROMPTS = [
    "<|im_start|>user\nGive three creative names for a bakery and explain each in one sentence.<|im_end|>\n<|im_start|>assistant\n",
    "<|im_start|>user\nWrite a short Python function that returns the n-th Fibonacci number, then show it for n = 10.<|im_end|>\n<|im_start|>assistant\n",
    "<|im_start|>user\nSuggest a strategy to solve: find two numbers whose sum is 20 and product is 96.<|im_end|>\n<|im_start|>assistant\n",
]
FILLER = "<|im_start|>user\nList ten animals that live in the ocean.<|im_end|>\n<|im_start|>assistant\n"


class Server:
    def __init__(self, a, args=(), env=None, exact=True):
        e = {**os.environ, **(env or {})}
        if exact:
            e["GGML_KURN_FA_MODE"] = "exact"
        else:
            e.pop("GGML_KURN_FA_MODE", None)
        cmd = [a.server, "-m", a.model, "-t", "8", "-c", "8192", "--port", str(PORT), "--no-webui", *args]
        self.p = subprocess.Popen(cmd, env=e, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, preexec_fn=os.setsid)
        for _ in range(600):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=1) as r:
                    if r.status == 200:
                        return
            except Exception:
                if self.p.poll() is not None:
                    raise RuntimeError(f"server exited: {cmd}")
                time.sleep(0.5)
        raise RuntimeError("server did not start")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        os.killpg(self.p.pid, signal.SIGTERM)
        self.p.wait(timeout=60)

    @staticmethod
    def post(path, body):
        r = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", data=json.dumps(body).encode(),
                                   headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(r, timeout=1800) as fh:
            return json.loads(fh.read())

    def complete(self, prompt, seed, n=64, **kw):
        body = {"prompt": prompt, "n_predict": n, "temperature": 0.8, "top_k": 40, "top_p": 0.95, "seed": seed,
                "cache_prompt": False, "return_tokens": True, **kw}
        return self.post("/completion", body)


def spec_args(a, kind):
    md = ["-md", a.draft, "-td", "8", "--spec-draft-n-max", "8"]
    ng = ["--spec-ngram-simple-size-n", "3", "--spec-ngram-simple-size-m", "16"]
    return {
        "plain": [],
        "draft": md + ["--spec-type", "draft-simple", "--spec-width", a.table],
        "arms": md + ["--spec-type", "ngram-simple,draft-simple", "--spec-width", a.table] + ng,
    }[kind]


def run_set(srv, concurrent=False, **kw):
    out = []
    for i, p in enumerate(PROMPTS):
        if concurrent:
            with cf.ThreadPoolExecutor(4) as ex:
                fill = [ex.submit(srv.complete, FILLER, 1000 + j, 96) for j in range(3)]
                time.sleep(0.3)
                out.append(ex.submit(srv.complete, p, 7 + i, **kw).result())
                [f.result() for f in fill]
        else:
            out.append(srv.complete(p, 7 + i, **kw))
    return out


def check_probs(a, res):
    ok = True
    for post in (False, True):
        runs = {}
        for kind in ("plain", "draft", "arms"):
            with Server(a, spec_args(a, kind)) as s:
                runs[kind] = run_set(s, n_probs=5, post_sampling_probs=post)
        tag = "post" if post else "pre"
        for kind in ("draft", "arms"):
            same = [x["completion_probabilities"] == y["completion_probabilities"] and x["content"] == y["content"]
                    for x, y in zip(runs[kind], runs["plain"])]
            res[f"probs_{tag}_{kind}_equal_plain"] = same
            ok &= all(same)
        drafted = [r["timings"].get("draft_n_accepted", 0) for k in ("draft", "arms") for r in runs[k]]
        res[f"probs_{tag}_accepted_draft_tokens"] = drafted
        key = "prob" if post else "logprob"
        empty = sum(1 for k in ("draft", "arms") for r in runs[k] for t in r["completion_probabilities"]
                    if not t[f"top_{key}s"] or (not post and t["logprob"] == 0.0 and len(t["top_logprobs"]) == 0))
        res[f"probs_{tag}_empty_entries"] = empty
        ok &= empty == 0 and sum(drafted) > 0
    return ok


def check_chain(a, res):
    chains = {}
    for name, args, conc, exact in [
        ("plain", spec_args(a, "plain"), False, True),
        ("draft", spec_args(a, "draft"), False, True),
        ("arms", spec_args(a, "arms"), False, True),
        ("concurrent_np4", ["-np", "4"], True, True),
        ("threads4", ["-t", "4"], False, True),
        ("fast_mode", [], False, False),
    ]:
        with Server(a, args, exact=exact) as s:
            chains[name] = [r["kurn_replay"]["hash_chain"] for r in run_set(s, concurrent=conc, kurn_replay=2)]
    res["chains"] = chains
    ok = True
    for k in ("draft", "arms", "concurrent_np4", "threads4"):
        res[f"chain_{k}_equal_plain"] = [x == y for x, y in zip(chains[k], chains["plain"])]
        ok &= all(res[f"chain_{k}_equal_plain"])
    res["chain_fast_mode_differs"] = [x != y for x, y in zip(chains["fast_mode"], chains["plain"])]
    ok &= any(res["chain_fast_mode_differs"])
    return ok


def check_resume(a, res):
    ok = True
    for rng in ("position", "sequential"):
        kw = {"kurn_rng_position": rng == "position"}
        for kind in ("plain", "arms"):
            same = []
            with Server(a, spec_args(a, kind)) as s:
                for i, p in enumerate(PROMPTS):
                    ids = s.post("/tokenize", {"content": p, "add_special": False})["tokens"]
                    full = s.complete(ids, 7 + i, n=64, **kw)
                    toks = full["tokens"]
                    if len(toks) < 64:
                        same.append(None)
                        continue
                    rest = s.complete(ids + toks[:32], 7 + i, n=32, **kw)
                    same.append(rest["tokens"] == toks[32:64])
            res[f"resume_{rng}_{kind}"] = same
            if rng == "position":
                ok &= all(x is not False for x in same) and any(same)
    res["resume_sequential_note"] = "expected to differ: the sequential stream restarts at the resumed request"
    return ok


def check_manifest(a, res):
    ms = []
    for _ in range(2):
        with Server(a, spec_args(a, "arms"), env={"KURN_MANIFEST_SHA256": "1"}) as s:
            ms.append(s.complete(PROMPTS[0], 7, n=8, kurn_replay=1)["kurn_replay"])
    m = ms[0]["manifest"]
    h = hashlib.sha256()
    with open(a.model, "rb") as fh:
        for chunk in iter(lambda: fh.read(8 << 20), b""):
            h.update(chunk)
    res["manifest"] = m
    res["manifest_stable"] = ms[0]["manifest_sha256"] == ms[1]["manifest_sha256"]
    res["manifest_model_sha256_matches"] = m["model"]["sha256"] == h.hexdigest()
    need = ("llama_build", "model", "draft_model", "env", "cpu", "threads", "cache_type_k", "spec_types", "sampling")
    res["manifest_fields_present"] = all(k in m for k in need)
    return res["manifest_stable"] and res["manifest_model_sha256_matches"] and res["manifest_fields_present"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--draft", required=True)
    ap.add_argument("--table", required=True)
    ap.add_argument("--out")
    ap.add_argument("--only", help="comma-separated subset of probs,chain,resume,manifest")
    a = ap.parse_args()
    res, status = {}, {}
    for name, fn in [("probs", check_probs), ("chain", check_chain), ("resume", check_resume), ("manifest", check_manifest)]:
        if a.only and name not in a.only.split(","):
            continue
        t0 = time.time()
        status[name] = bool(fn(a, res))
        print(f"{name}: {'PASS' if status[name] else 'FAIL'} ({time.time() - t0:.0f} s)", flush=True)
    res["status"] = status
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(res, fh, indent=1)
    return 0 if all(status.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
