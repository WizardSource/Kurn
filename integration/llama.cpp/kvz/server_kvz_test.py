#!/usr/bin/env python3
"""llama-server with --cache-kvz: slot save / erase / restore and the RAM prompt cache give the same greedy text as the
first run, hit the cache, and kvz slot files are smaller than plain ones.

    server_kvz_test.py LLAMA_SERVER MODEL [CACHE_TYPES...]     (default: k4c_q4 f16; exit code = failures)

Per cache type it runs one server with --cache-kvz and one without (GGML_KURN_FA_MODE=exact, one slot):
  1. prompt A -> text, 2. save slot, erase, restore, prompt A again, 3. prompt B (A's state goes to the RAM prompt
  cache), prompt A again (restored from the cache)."""

import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request

FAIL = []


def check(ok, what):
    print(("ok   " if ok else "FAIL ") + what, flush=True)
    if not ok:
        FAIL.append(what)


def post(port, path, body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read())


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def text_a():
    parts = []
    for i in range(60):
        parts.append(f"Record {i}: the session parked in host memory holds key and value rows for {i * 37 % 101} layers; "
                     f"restoring it must give back every byte, and the next token must not change. ")
    return "".join(parts) + "\nSummarize the records above in one sentence:"


def text_b():
    return " ".join(f"Unrelated line {i} about tomatoes, gardens and the weather in {1990 + i}." for i in range(40)) + "\nContinue:"


def run(server, model, ctk, kvz, workdir):
    port = free_port()
    slots = os.path.join(workdir, f"slots-{ctk}-{int(kvz)}")
    os.makedirs(slots, exist_ok=True)
    args = [server, "-m", model, "-c", "4096", "-np", "1", "-t", "8", "--port", str(port), "--slot-save-path", slots,
            "--cache-ram", "2048", "-ctk", ctk] + (["--cache-kvz"] if kvz else [])
    log = open(os.path.join(workdir, f"server-{ctk}-{int(kvz)}.log"), "w")
    p = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT, env=dict(os.environ, GGML_KURN_FA_MODE="exact"))
    try:
        for _ in range(600):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                    if r.status == 200:
                        break
            except Exception:
                time.sleep(0.5)
        else:
            raise RuntimeError("server did not start")
        tag = f"[{ctk}{' kvz' if kvz else ''}]"

        def complete(prompt):
            return post(port, "/completion", {"prompt": prompt, "n_predict": 24, "temperature": 0, "cache_prompt": True, "seed": 1})

        r1 = complete(text_a())
        n_prompt = r1["timings"]["prompt_n"]
        saved = post(port, "/slots/0?action=save", {"filename": "a.bin"})
        post(port, "/slots/0?action=erase", {})
        restored = post(port, "/slots/0?action=restore", {"filename": "a.bin"})
        r2 = complete(text_a())
        check(r2["content"] == r1["content"], f"{tag} restored slot: same text")
        check(r2["timings"]["prompt_n"] <= 2, f"{tag} restored slot: prompt cache hit (prompt_n {r2['timings']['prompt_n']} of {n_prompt})")
        check(restored.get("n_restored") == saved.get("n_saved"), f"{tag} restore read the saved tokens")
        complete(text_b())
        r3 = complete(text_a())
        check(r3["content"] == r1["content"], f"{tag} RAM prompt cache: same text")
        check(r3["timings"]["prompt_n"] <= 2, f"{tag} RAM prompt cache: hit (prompt_n {r3['timings']['prompt_n']} of {n_prompt})")
        size = os.path.getsize(os.path.join(slots, "a.bin"))
        print(f"{tag} prompt {n_prompt} tokens, slot file {size} bytes, save {saved['timings']['save_ms']:.0f} ms, "
              f"restore {restored['timings']['restore_ms']:.0f} ms", flush=True)
        return r1["content"], size
    finally:
        p.terminate()
        p.wait(timeout=60)
        log.close()


def main():
    server, model = sys.argv[1], sys.argv[2]
    types = sys.argv[3:] or ["k4c_q4", "f16"]
    with tempfile.TemporaryDirectory() as d:
        for ctk in types:
            t_plain, s_plain = run(server, model, ctk, False, d)
            t_kvz, s_kvz = run(server, model, ctk, True, d)
            check(t_kvz == t_plain, f"[{ctk}] same text with and without --cache-kvz")
            check(s_kvz < s_plain, f"[{ctk}] kvz slot file smaller ({s_kvz} vs {s_plain} bytes, {s_plain / s_kvz:.3f}x)")
    print(f"server kvz: {len(FAIL)} failed")
    return len(FAIL)


if __name__ == "__main__":
    sys.exit(main())
