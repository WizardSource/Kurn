"""Reports of the GPU kit v2: the attention section (kurn vs llama.cpp / FlashInfer / FlashAttention-2 decode, Q8_0
vs F16, ablations, against the stated target) and the single report.md that wraps every step of run_kit.sh."""

import json
import os

NOMINAL_BW = {"A100-SXM4-80GB": 2039, "A100 80GB": 2039, "PG509-210": 2039, "A100-SXM4-40GB": 1555, "A100-PCIE-40GB": 1555,
              "A100 80GB PCIe": 1935}  # fmt: skip
# Decode target, stated before the measurements: decode attention is a stream over the KV cache, so the yardstick is
# the HBM read bandwidth the box actually reaches (roofline.json, a pure-read kernel), not FLOPs.
TARGET = {
    "f16_frac": 0.75,  # F16 / BF16 KV at >= 4K context: >= 75% of measured HBM read bandwidth (run 1: ~1.1 TB/s, ~55% of nominal)
    "q8_speedup": 1.5,  # Q8_0 KV >= 1.5x kurn's own F16 at >= 8K (1.88x fewer bytes; run 1: 0.46x)
    "q8_vs_best": 1.3,  # the plan's stop-or-continue rule: Q8_0 >= 1.3x the best F16 baseline decode at >= 8K
    "min_ctx": 4096,
    "q8_ctx": 8192,
}
UNTESTABLE_A100 = [
    "FP8 (e4m3) KV attention: needs FP8 mma.sync (sm_89+); on an A100 it is covered by compile checks (sm_100 / sm_120 "
    "SASS, ptxas spills) and the CPU warp emulator only, never run",
    "sm_100 (B200 / GB200) and sm_120 (RTX 50) runtime: their SASS is built and checked (ptxas registers / spills, "
    "cuobjdump contents) but cannot execute on an A100; correctness there rests on the emulator plus the shared code path",
    "Native Blackwell paths (block-scaled FP4 MMA, tcgen05 / TMEM / TMA): plan only, no code",
]


def _rows(path):
    if not os.path.exists(path):
        return []
    with open(path) as fh:
        return [json.loads(ln) for ln in fh if ln.strip()]


def _f(x, fmt="{:.1f}"):
    return fmt.format(x) if isinstance(x, (int, float)) and x == x else "-"


def _hbm(results):
    for p in (os.path.join(results, "roofline.json"), os.path.join(results, "..", "roofline.json")):
        if os.path.exists(p):
            try:
                with open(p) as fh:
                    r = json.load(fh)
                if r.get("hbm_read_gbs"):
                    return float(r["hbm_read_gbs"]), "measured (roofline.json, pure-read kernel)"
            except (OSError, ValueError):
                pass
    return None, None


def attn_report(results):
    """Markdown for an attention results dir (attn_matrix.jsonl + optional attn_baselines.jsonl / attn_ablation.jsonl)."""
    rows = _rows(os.path.join(results, "attn_matrix.jsonl"))
    base = _rows(os.path.join(results, "attn_baselines.jsonl"))
    abl = _rows(os.path.join(results, "attn_ablation.jsonl"))
    L = ["## Attention decode: kurn vs baselines\n"]
    if not rows:
        return "\n".join(L + ["No attention matrix in this run.\n"])
    ok = [r for r in rows if r.get("status") == "ok"]
    dev = next((r["device"] for r in rows if r.get("device")), "?")
    sm = next((r["sm"] for r in rows if r.get("sm")), None)
    native = next((r["native"] for r in rows if "native" in r), None)
    hbm, how = _hbm(results)
    nominal = next((v for k, v in NOMINAL_BW.items() if k in dev), None)
    if hbm is None and nominal:
        hbm, how = 0.9 * nominal, f"assumed 90% of nominal {nominal} GB/s (no roofline.json)"
    L.append(f"Ran on: {dev} (sm_{sm}, {'native SASS' if native else 'JIT from PTX' if native is not None else '?'}). "
             f"{len(rows)} kurn cells, {len(ok)} correct vs the float64 reference.")  # fmt: skip
    if hbm:
        L.append(f"HBM read bandwidth: {hbm:.0f} GB/s, {how}{f'; nominal {nominal} GB/s' if nominal else ''}.")
    L.append(
        "\nAll lines run the same problem (same data, same cold-KV rotation: the cache is replicated past L2 and one call "
        "per layer is replayed as a CUDA graph) and count the same bytes (KV read once + q/out). Times are the mean per "
        "call; GB/s from the best round.\n"
    )
    tgt = TARGET
    L.append(
        f"**Target** (stated before measuring): F16/BF16 decode at >= {tgt['min_ctx'] // 1024}K context reaches "
        f">= {tgt['f16_frac']:.0%} of the measured HBM read bandwidth (A100 run 1: ~1.1 TB/s = ~55% of the 80 GB part's "
        f"2.04 TB/s); Q8_0 KV is >= {tgt['q8_speedup']}x kurn's own F16 at >= {tgt['q8_ctx'] // 1024}K (it reads 1.88x fewer "
        f"bytes); and the plan's stop-or-continue rule, Q8_0 >= {tgt['q8_vs_best']}x the best F16 baseline (llama.cpp, "
        "FlashAttention-2, FlashInfer) at >= 8K.\n"
    )
    bidx = {}
    for b in base:
        if "model" in b:
            bidx[(b["impl"], b["model"], b["kv"], b["nkv"], b.get("nq", 1))] = b
    impls = sorted({b["impl"] for b in base if "model" in b})
    missing = [b for b in base if b.get("status") == "not installed"]
    hdr = "| model | kv | context | kurn us | kurn GB/s | % HBM | splits | relerr |"
    sep = "|---|---|---|---|---|---|---|---|"
    for i in impls:
        hdr += f" {i} us | kurn speedup vs {i} |"
        sep += "---|---|"
    L += [hdr, sep]
    for r in rows:
        key = (r["model"], r["kv"], r["nkv"], r.get("nq", 1))
        if r.get("status") == "error":
            L.append(f"| {r['model']} | {r['kv']} | {r['nkv']} | error: {r.get('error', '')[:60]} | | | | |" + " | |" * len(impls))
            continue
        pct = r["GBps"] / hbm * 100 if hbm else float("nan")
        line = (f"| {r['model']} | {r['kv']} | {r['nkv']} | {_f(r['us'])} ± {_f(r['us_sd'])} | {_f(r['GBps'], '{:.0f}')} | "
                f"{_f(pct, '{:.0f}')}% | "
                f"{r['splits']} | {r['relerr']:.1e}{'' if r['status'] == 'ok' else ' ' + r['status']} |")  # fmt: skip
        for i in impls:
            b = bidx.get((i, *key))
            if not b:
                line += " n/a | |"
            elif b.get("status") == "error":
                line += f" {b.get('error', 'error')[:40]} | |"
            else:
                flag = "" if b["status"] == "ok" else f" ({b['status']} {b['relerr']:.0e})"
                line += f" {_f(b['us'])}{flag} | {b['us'] / r['us']:.2f}x |"
        L.append(line)
    if missing:
        L.append("\nNot measured: " + "; ".join(f"{b['impl']} ({b.get('error', 'not installed')})" for b in missing) + ".")
    # Q8_0 vs F16
    by = {(r["impl"] if "impl" in r else "kurn", r["model"], r["kv"], r["nkv"]): r for r in ok}
    by.update({(b["impl"], b["model"], b["kv"], b["nkv"]): b for b in base if b.get("status") == "ok" and "model" in b})
    L += ["\n### Q8_0 vs F16 KV (time ratio F16 / Q8_0: > 1 means Q8_0 is faster)\n",
          "| model | context | kurn F16 us | kurn Q8_0 us | kurn Q8_0 speedup | ggml-cuda Q8_0 speedup | Q8_0 vs best F16 baseline |",
          "|---|---|---|---|---|---|---|"]  # fmt: skip
    verdicts = {"f16": [], "q8": [], "best": []}
    for model in dict.fromkeys(r["model"] for r in rows):
        for ctx in sorted({r["nkv"] for r in rows if r["model"] == model}):
            f, q = by.get(("kurn", model, "f16", ctx)), by.get(("kurn", model, "q8_0", ctx))
            gf, gq = by.get(("ggml-cuda", model, "f16", ctx)), by.get(("ggml-cuda", model, "q8_0", ctx))
            bests = [by[(i, model, "f16", ctx)]["us"] for i in impls if (i, model, "f16", ctx) in by]
            sp = f["us"] / q["us"] if f and q else None
            gsp = gf["us"] / gq["us"] if gf and gq else None
            vb = min(bests) / q["us"] if q and bests else None
            L.append(
                f"| {model} | {ctx} | {_f(f and f['us'])} | {_f(q and q['us'])} | {_f(sp, '{:.2f}x')} | {_f(gsp, '{:.2f}x')} | "
                f"{_f(vb, '{:.2f}x')} |"
            )
            if ctx >= tgt["q8_ctx"] and sp is not None:
                verdicts["q8"].append(sp >= tgt["q8_speedup"])
            if ctx >= tgt["q8_ctx"] and vb is not None:
                verdicts["best"].append(vb >= tgt["q8_vs_best"])
            for kv in ("f16", "bf16"):
                r = by.get(("kurn", model, kv, ctx))
                if r and hbm and ctx >= tgt["min_ctx"] and model != "mla-dsv2-lite":
                    verdicts["f16"].append(r["GBps"] >= tgt["f16_frac"] * hbm)
    L.append("\n**Against the target:**")
    for k, name in (("f16", f"F16/BF16 >= {tgt['f16_frac']:.0%} of HBM (GQA models, >= {tgt['min_ctx'] // 1024}K)"),
                    ("q8", f"Q8_0 >= {tgt['q8_speedup']}x kurn F16 (>= {tgt['q8_ctx'] // 1024}K)"),
                    ("best", f"Q8_0 >= {tgt['q8_vs_best']}x best F16 baseline (>= {tgt['q8_ctx'] // 1024}K)")):  # fmt: skip
        v = verdicts[k]
        L.append(f"- {name}: " + (f"{sum(v)}/{len(v)} cells meet it" + (" - MET" if all(v) else " - not met") if v else "no data"))
    if abl:
        L += ["\n### Ablations (each Phase-2 change set back to its A100 run-1 value; + fused merge)\n",
              "| model | kv | context | variant | us | vs default | splits | relerr |", "|---|---|---|---|---|---|---|---|"]  # fmt: skip
        for a in abl:
            d = by.get(("kurn", a["model"], a["kv"], a["nkv"]))
            if a.get("status") == "error":
                L.append(f"| {a['model']} | {a['kv']} | {a['nkv']} | {a['variant']} | error: {a.get('error', '')[:50]} | | | |")
                continue
            rel = f"{(a['us'] / d['us'] - 1) * 100:+.0f}%" if d else "-"
            L.append(
                f"| {a['model']} | {a['kv']} | {a['nkv']} | {a['variant']} | {_f(a['us'])} | {rel} | {a['splits']} | {a['relerr']:.1e} |"
            )
        L.append("\n(+x% = the variant is slower than the new default by x%.)")
    return "\n".join(L) + "\n"


def _section(path, start, stop="\n## "):
    if not os.path.exists(path):
        return None
    txt = open(path).read()
    i = txt.find(start)
    if i < 0:
        return None
    j = txt.find(stop, i + len(start))
    return txt[i : j if j > 0 else None].strip()


def _tail(path, n=1):
    if not os.path.exists(path):
        return None
    with open(path, errors="replace") as fh:
        lines = [ln.rstrip() for ln in fh if ln.strip()]
    return "\n".join(lines[-n:]) if lines else None


def kit_report(out):
    """The single report.md of a kit v2 run directory."""
    kit = {}
    if os.path.exists(os.path.join(out, "kit.json")):
        kit = json.load(open(os.path.join(out, "kit.json")))
    steps = _rows(os.path.join(out, "steps.jsonl"))
    mode = {"dryrun": "dry run, no GPU", "quick": "QUICK run"}.get(kit.get("mode"), f"{kit.get('mode', '?')} run")
    L = [f"# kurn GPU kit v2 report ({mode})\n"]
    arch = {}
    if os.path.exists(os.path.join(out, "arch.json")):
        arch = json.load(open(os.path.join(out, "arch.json")))
    L.append(f"Host: {kit.get('host', '?')}. GPU: {kit.get('gpu', '?')}. Ran on {arch.get('ran_on', '?')} (tier {arch.get('tier', '?')}, "
             f"{arch.get('how', '?')}); fatbin SASS: {arch.get('fatbin', '?')}. Version: {kit.get('kurn', '?')}.\n")  # fmt: skip
    tc = _tail(os.path.join(out, "toolchain.txt"), 12)
    if tc:
        L += ["## Toolchain\n", "```", tc, "```\n"]
    if steps:
        L += ["## Steps\n", "| step | result | minutes | log |", "|---|---|---|---|"]
        for s in steps:
            L.append(f"| {s['step']} | {s['result']} | {s.get('min', '-')} | {s.get('log', '')} |")
        L.append("")
    ran = str(arch.get("ran_on", ""))
    if ran.startswith("sm_8"):
        untestable = UNTESTABLE_A100
    elif ran.startswith("sm_"):
        untestable = UNTESTABLE_A100[2:]
    else:
        untestable = ["No GPU (dry run): nothing ran on a GPU. Every timing is unmeasured; correctness rests on the compile checks "
                      "(ptxas, cuobjdump) and the CPU warp emulator"] + UNTESTABLE_A100[2:]  # fmt: skip
    L += ["## Not testable on this box\n"] + [f"- {u}" for u in untestable]
    L += [f"- {s['step']}: {s['result']}" for s in steps if s["result"].startswith("skipped")]
    L += ["", "## Correctness and build checks\n"]
    for name, f in (("attention builds, 3 tiers (ptxas: registers, spills)", "attn/attn_ptxas.txt"),
                    ("attention on the GPU (covering set x awkward shapes vs float64)", "attn/attn_verify_gpu.txt"),
                    ("attention on the CPU emulator", "attn/attn_verify_emu.txt"),
                    ("matmul builds (ptxas)", "matmul/ptxas.txt"),
                    ("matmul on the GPU (covering set vs the exact reference)", "matmul/verify_gpu.txt"),
                    ("matmul on the CPU emulator", "matmul/verify_emu.txt")):  # fmt: skip
        p = os.path.join(out, f)
        if os.path.exists(p):  # every "N configurations, M failures" summary in the log (some steps run two passes)
            sums = [ln.strip() for ln in open(p, errors="replace") if " configurations, " in ln and "failures" in ln]
            if sums or _tail(p):
                L.append(f"- {name}: {'; '.join(sums) or _tail(p)}")
    mmc = _rows(os.path.join(out, "matmul", "ggml_mm_check.jsonl"))
    if mmc:
        ok = sum(r.get("status") == "ok" for r in mmc)
        L.append(f"- ggml comparison path on ggml's CPU backend: {ok}/{len(mmc)} formats ok ({', '.join(r.get('fmt', '?') for r in mmc)})")
    L.append("")
    L.append(attn_report(os.path.join(out, "attn")))
    mm = os.path.join(out, "matmul", "report.md")
    L.append("## GEMM / GEMV matmul kit (incl. MXFP4 / NVFP4 dequant on sm_80)\n")
    if os.path.exists(mm):
        w = _section(mm, "## Wins")
        L.append(w.replace("## ", "### ", 1) if w else "(no wins table)")
        for fmt in ("mxfp4", "nvfp4"):
            d = _section(mm, f"### {fmt}", "\n### ")
            if d:
                L += ["", d.replace("### ", "#### ", 1)]
        L.append("\nFull matmul report: matmul/report.md.\n")
    else:
        L.append("Not run or no report (see steps).\n")
    L.append("## Tests (GPU-relevant pytest subset)\n")
    pt = _tail(os.path.join(out, "pytest", "pytest.txt"))
    L.append((pt or "Not run (see steps).") + "\n")
    fails = []
    if os.path.exists(os.path.join(out, "pytest", "pytest.txt")):
        fails = [
            ln.strip()
            for ln in open(os.path.join(out, "pytest", "pytest.txt"), errors="replace")
            if ln.startswith("FAILED") or ln.startswith("ERROR")
        ]
    if fails:
        L += ["```"] + fails[:30] + ["```\n"]
    L.append("## llama.cpp (ggml-cuda) build\n")
    lb = {}
    if os.path.exists(os.path.join(out, "llama", "build.json")):
        lb = json.load(open(os.path.join(out, "llama", "build.json")))
    L.append(f"{lb.get('status', 'not run')}: sources {lb.get('source', '?')} (commit {lb.get('commit', '?')}), built in "
             f"{lb.get('dir', '?')} for {lb.get('arch', '?')} in {lb.get('min', '?')} min. Used by the attention baseline and the "
             "matmul kit's ggml-cuda competitor.\n")  # fmt: skip
    return "\n".join(L)


def write(out):
    txt = kit_report(out)
    with open(os.path.join(out, "report.md"), "w") as fh:
        fh.write(txt)
    return txt
