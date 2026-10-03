"""Attention benchmark matrix: kurn attn vs ggml flash_attn_ext (FA) and the non-FA path.

    benchlock.sh python matrix.py SUITE [--reps 3] [--secs 0.5] [--threads 8] [--out DIR]

SUITE: prefill (llama.cpp ubatch n_q=512 at depth), decode (n_q=1, cold KV), mla, d64,
or a comma list. Every (case, impl) runs once per rep, reps interleaved; the summary is the
median with min-max spread, CPU/wall ratio and the largest clock drift. Raw rows go to
DIR/SUITE_raw.csv, the summary to DIR/SUITE.csv.
"""

import argparse
import csv
import os
import statistics
import subprocess
import sys
import tempfile
import time

from kurn import attention as A

HERE = os.path.dirname(os.path.abspath(__file__))
LLAMA = os.path.expanduser(os.environ.get("LLAMA_SRC", "~/src/llama.cpp"))
GGML_BIN = os.environ.get("GGML_ATTN_BIN", "/tmp/attn-res/bench_ggml_attn")
COLS = A.CSV_COLUMNS

QWEN17 = {"heads": 16, "kv_heads": 8}  # Qwen3-1.7B (and 0.6B): 16 q heads, 8 kv heads, d=128
QWEN8 = {"heads": 32, "kv_heads": 8}  # Qwen3-8B: 32 q heads, 8 kv heads, d=128


def ggml_bin():
    src = f"{HERE}/bench_ggml_attn.c"
    if not os.path.exists(GGML_BIN) or os.path.getmtime(GGML_BIN) < os.path.getmtime(src):
        os.makedirs(os.path.dirname(GGML_BIN), exist_ok=True)
        b = f"{LLAMA}/build/bin"
        subprocess.run(["gcc", "-O3", "-march=native", "-I", f"{LLAMA}/ggml/include", src, "-o",
                        GGML_BIN, f"-L{b}", "-lggml", "-lggml-base", "-lggml-cpu", f"-Wl,-rpath,{b}", "-lm"], check=True)  # fmt: skip
    return GGML_BIN


def kurn_impl(name, target, kv, **sched):
    return {"name": name, "kind": "kurn", "target": target, "kv": kv, "sched": sched}


def ggml_impl(name, path, kv):
    return {"name": name, "kind": "ggml", "path": path, "kv": kv}


def suite(name):
    cases = []
    if name == "prefill":
        for shape_name, shape in (("1.7B", QWEN17), ("8B", QWEN8)):
            for depth in (512, 2048, 8192, 16384, 32768):
                prob = {"nq": 512, "nkv": depth, "causal": 1, **shape, "dk": 128}
                impls = [kurn_impl("kurn-amx-f16", "amx_bf16", "f16"), kurn_impl("kurn-amx-q8_0", "amx_bf16", "q8_0"),
                         kurn_impl("kurn-amx-f16-t128x256", "amx_bf16", "f16", tile_q=128, tile_kv=256),
                         kurn_impl("kurn-fma-f16", "avx512", "f16"), ggml_impl("ggml-fa-f16", "fa", "f16")]  # fmt: skip
                if depth <= 16384:  # the non-FA KQ matrix is n_q * n_kv * heads floats
                    impls.append(ggml_impl("ggml-nofa-f16", "nofa", "f16"))
                if depth <= 8192:
                    impls.append(ggml_impl("ggml-fa-q8_0", "fa", "q8_0"))
                cases.append((f"prefill-{shape_name}-d{depth}", prob, impls))
    elif name == "decode":
        for shape_name, shape in (("1.7B", QWEN17), ("8B", QWEN8)):
            for depth in (512, 2048, 8192, 32768):
                prob = {"nq": 1, "nkv": depth, "causal": 1, **shape, "dk": 128}
                impls = [kurn_impl("kurn-f16", "avx512", "f16"), kurn_impl("kurn-q8_0", "avx512", "q8_0"),
                         ggml_impl("ggml-fa-f16", "fa", "f16"), ggml_impl("ggml-fa-q8_0", "fa", "q8_0"),
                         ggml_impl("ggml-nofa-f16", "nofa", "f16")]  # fmt: skip
                cases.append((f"decode-{shape_name}-d{depth}", prob, impls))
    elif name == "d64":
        for nq, depth in ((512, 4096), (1, 8192)):
            prob = {"nq": nq, "nkv": depth, "causal": 1, "heads": 16, "kv_heads": 4, "dk": 64}
            impls = [kurn_impl("kurn-amx-f16" if nq > 1 else "kurn-f16", "amx_bf16" if nq > 1 else "avx512", "f16"),
                     ggml_impl("ggml-fa-f16", "fa", "f16"), ggml_impl("ggml-nofa-f16", "nofa", "f16")]  # fmt: skip
            cases.append((f"{'prefill' if nq > 1 else 'decode'}-d64-g4-d{depth}", prob, impls))
    elif name == "mla":
        # absorbed MLA (DeepSeek-V2/V3 style): one latent KV head of 512 + 64 rope dims, V = first 512
        # dims of the same row; vs GQA 1.7B-like f16 KV at the same context
        for depth in (2048, 8192, 32768):
            for heads in (16, 128):
                cases.append((f"decode-mla-h{heads}-d{depth}", {"nq": 1, "nkv": depth, "causal": 1, "heads": heads,
                              "kv_heads": 1, "dk": 576, "mla": 1}, [kurn_impl("kurn-mla-amx-f16", "amx_bf16", "f16"),
                              kurn_impl("kurn-mla-fma-f16", "avx512", "f16"), ggml_impl("ggml-mla-fa-f16", "fa", "f16")]))  # fmt: skip
            cases.append((f"decode-gqa-1.7B-d{depth}", {"nq": 1, "nkv": depth, "causal": 1, **QWEN17, "dk": 128},
                          [kurn_impl("kurn-f16", "avx512", "f16"), ggml_impl("ggml-fa-f16", "fa", "f16")]))  # fmt: skip
    else:
        raise SystemExit(f"unknown suite {name}")
    return cases


def run_kurn(impl, prob, threads, regime, secs):
    c = A.resolve({"target": impl["target"], "kv": impl["kv"], "dk": prob["dk"], **impl["sched"],
                   **{k: v for k, v in prob.items() if k != "dk"}, "threads": threads})  # fmt: skip
    return A.bench(A.build(c), c, regime, secs, ("--check-toks", "4"))


def run_ggml(impl, prob, threads, regime, secs):
    fd, tmp = tempfile.mkstemp(suffix=".csv")
    os.close(fd)
    os.remove(tmp)
    cmd = [ggml_bin(), "--path", impl["path"], "--kv", impl["kv"], "--dk", str(prob["dk"]), "--threads", str(threads),
           "--regime", regime, "--secs", str(secs), "--csv", tmp, "--check-toks", "4", "--tol", "5e-2",
           *A.problem_args({**A.PROBLEM, **prob, "pos0": -1})]  # fmt: skip
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
    if not os.path.exists(tmp):
        raise RuntimeError(f"ggml harness failed: {(r.stderr or r.stdout)[-800:]}")
    with open(tmp) as fh:
        row = dict(zip(COLS, fh.read().strip().split(",")))
    os.remove(tmp)
    row["us"] = float(row["us_per_call"])
    row["cpu_us"] = float(row["cpu_s"]) / float(row["calls"]) * 1e6
    return row


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("suites")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--secs", type=float, default=0.5)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--only", default="", help="substring filter on case names")
    ap.add_argument("--impls", default="", help="comma list of impl names to keep")
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    ap.add_argument("--tag", default="", help="suffix for the output file names")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    keep = set(a.impls.split(",")) - {""}
    for sname in a.suites.split(","):
        cases = [(n, p, [i for i in imps if not keep or i["name"] in keep]) for n, p, imps in suite(sname) if a.only in n]
        load = open("/proc/loadavg").read().split()[0]
        print(f"# suite {sname}: {len(cases)} cases, reps {a.reps}, threads {a.threads}, load {load}", flush=True)
        raw = []
        for rep in range(a.reps):
            for cname, prob, impls in cases:
                regime = "cold" if prob["nq"] == 1 else "hot"
                for impl in impls:
                    t0 = time.time()
                    try:
                        run = run_kurn if impl["kind"] == "kurn" else run_ggml
                        row = run(impl, prob, a.threads, regime, a.secs)
                    except Exception as e:  # noqa: BLE001 - one failing impl must not kill a locked run
                        print(f"FAIL {cname} {impl['name']}: {str(e)[:300]}", flush=True)
                        continue
                    raw.append({**{k: row[k] for k in COLS}, "lib": row["impl"], "case": cname, "impl": impl["name"], "rep": rep,
                                "regime": regime, "cpu_us": row["cpu_us"], "us": row["us"]})  # fmt: skip
                    print(f"{rep} {cname:24} {impl['name']:16} {row['us']:10.1f} us  {float(row['GFLOPs']):8.1f} GFLOP/s "
                          f"{float(row['GBps']):7.1f} GB/s  relerr {float(row['relerr']):.1e} {row['check']} "
                          f"({time.time() - t0:.1f}s)", flush=True)  # fmt: skip
        with open(os.path.join(a.out, f"{sname}{a.tag}_raw.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(raw[0]) if raw else ["case"])
            w.writeheader()
            w.writerows(raw)
        summary = []
        for cname, _prob, impls in cases:
            for impl in impls:
                rows = [r for r in raw if r["case"] == cname and r["impl"] == impl["name"]]
                if not rows:
                    continue
                us = [r["us"] for r in rows]
                med = statistics.median(us)
                mid = min(rows, key=lambda r: abs(r["us"] - med))
                summary.append({
                    "case": cname, "impl": impl["name"], "reps": len(rows), "us_med": round(med, 2),
                    "us_min": round(min(us), 2), "us_max": round(max(us), 2),
                    "GFLOPs": mid["GFLOPs"], "GBps": mid["GBps"],
                    "cpu_wall": round(statistics.median(r["cpu_us"] / r["us"] for r in rows), 2),
                    "uJ_per_call": round(statistics.median(r["cpu_us"] for r in rows) * A.PROXY_W_PER_CORE, 1),
                    "relerr": max(float(r["relerr"]) for r in rows), "check": mid["check"],
                    "drift_s": max(abs(float(r["drift_s"])) for r in rows), "load": load,
                })  # fmt: skip
        with open(os.path.join(a.out, f"{sname}{a.tag}.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(summary[0]) if summary else ["case"])
            w.writeheader()
            w.writerows(summary)
        print(f"# wrote {a.out}/{sname}{a.tag}.csv", flush=True)


if __name__ == "__main__":
    sys.exit(main())
