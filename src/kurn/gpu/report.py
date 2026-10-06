"""Turn benchmark-matrix results into the wins/ties/losses report and the dispatch table.

Win rule, per (format, batch), on the per-round step times T_r (sum over the model's matmul shapes):
    ratio = mean(T_competitor) / mean(T_kurn),  sigma = sqrt(cv_kurn^2 + cv_competitor^2)
    win   if ratio > 1 + 2 sigma;  loss if 1 / ratio > 1 + 2 sigma;  tie otherwise
(fewer than 3 rounds on either side: "unresolved"). Applied against the best same-format competitor
(which decides dispatch: KURN only on a win, the competitor's own kernel otherwise) and against the
best competitor of any format (the honest "better than the competition" column).
Anything without a measurement on this GPU is reported as "unmeasured".
"""

import json
import math
import os
import statistics

from .matrix import BATCHES, MODEL

SAME_FORMAT = ("ggml",)  # competitors that read the same ggml bytes (valid fallbacks)
CROSS_FORMAT = {"cublas-fp16": None, "marlin": ("q4_0", "q4_K", "iq4_nl")}  # None: every format
REFERENCE_ONLY = ("cublas-int8",)  # not the same math: shown, never counted as a competitor
MIN_ROUNDS = 3


def load(results_dir):
    rows, meta = [], {}
    for name in sorted(os.listdir(results_dir)):
        p = os.path.join(results_dir, name)
        if name.endswith(".jsonl") and name.startswith(("matrix", "marlin")):
            with open(p) as fh:
                rows += [json.loads(ln) for ln in fh if ln.strip().startswith("{")]
        elif name in ("info.json", "roofline.json", "kit.json"):
            with open(p) as fh:
                meta[name[:-5]] = json.load(fh)
    return rows, meta


def _num(v):
    return float("nan") if v is None else v


def _family(impl):
    if impl.startswith("kurn:"):
        return "kurn"
    return impl.split(":")[0].split("-int4")[0] if impl.startswith("marlin") else impl


def verdict(tk, cvk, tc, cvc, nk, nc):
    if nk < MIN_ROUNDS or nc < MIN_ROUNDS:
        return "unresolved", tc / tk
    ratio = tc / tk
    thr = 2 * math.sqrt(cvk**2 + cvc**2)
    if ratio > 1 + thr:
        return "win", ratio
    if 1 / ratio > 1 + thr:
        return "loss", ratio
    return "tie", ratio


def aggregate(rows, model=MODEL):
    """-> {(fmt, M, impl): stats}. A model step needs every shape; marlin rows (fmt int4-g128) attach to 4-bit cells."""
    shapes = set(model["shapes"])
    L = model["layers"]
    samples, checks = {}, {}
    for r in rows:
        fmts = [r["fmt"]]
        if _family(r.get("impl", "")) == "marlin":
            fmts = list(CROSS_FORMAT["marlin"])
        for f in fmts:
            key = (f, r["M"], r["impl"])
            if r["kind"] == "sample":
                samples.setdefault(key, {}).setdefault((r["N"], r["K"]), {})[r["round"]] = r
            elif r["kind"] in ("check", "skip"):
                checks.setdefault(key, []).append(r)
    out = {}
    for key in set(samples) | set(checks):
        f, m, impl = key
        st = {"fmt": f, "M": m, "impl": impl, "family": _family(impl), "checks": checks.get(key, [])}
        bad = [c for c in st["checks"] if c["kind"] == "skip" or not str(c.get("status", "ok")).startswith("ok")]
        st["status"] = "ok" if not bad else (bad[0].get("status") or bad[0].get("reason") or "skipped")
        st["relerr_exact"] = max((c.get("relerr_exact") or 0) for c in st["checks"] if c["kind"] == "check") if st["checks"] else None
        st["relerr_model"] = max((c.get("relerr_model") or 0) for c in st["checks"] if c["kind"] == "check") if st["checks"] else None
        per_shape = samples.get(key, {})
        if shapes <= set(per_shape) and not bad:
            rounds = sorted(set.intersection(*(set(per_shape[s]) for s in shapes)))
            T = [sum(per_shape[s][r]["us"] for s in shapes) * L for r in rounds]
            J = [sum(_num(per_shape[s][r].get("joules")) for s in shapes) * L / m for r in rounds]
            B = sum(per_shape[s][rounds[0]]["bytes"] for s in shapes) * L if rounds else 0
            OPS = sum(per_shape[s][rounds[0]]["ops"] for s in shapes) * L if rounds else 0
            if rounds:
                tok = [m / (t * 1e-6) for t in T]
                sd = statistics.stdev if len(T) > 1 else (lambda _v: 0.0)
                st.update(rounds=len(rounds), T_us=statistics.mean(T), T_sd=sd(T), tok_s=statistics.mean(tok), tok_sd=sd(tok),
                          J_tok=statistics.mean(J) if all(j == j for j in J) else float("nan"), bytes=B, ops=OPS)  # fmt: skip
                st["cv"] = st["T_sd"] / st["T_us"] if st["T_us"] else 0.0
        elif per_shape and not bad:
            st["status"] = "incomplete (not every model shape measured)"
        out[key] = st
    return out


def decide(agg, formats=None, batches=BATCHES):
    """Per (fmt, M): best KURN, best same-format and overall competitor, verdicts and the dispatch choice."""
    cells = {}
    fmts = sorted(formats or {k[0] for k in agg})
    for f in fmts:
        for m in sorted(batches or {k[1] for k in agg if k[0] == f}):
            here = [s for (ff, mm, _), s in agg.items() if ff == f and mm == m]
            timed = [s for s in here if "T_us" in s]
            kurn = min((s for s in timed if s["family"] == "kurn"), key=lambda s: s["T_us"], default=None)
            same = min((s for s in timed if s["family"] in SAME_FORMAT), key=lambda s: s["T_us"], default=None)
            cross = [
                s for s in timed if s["family"] in CROSS_FORMAT and (CROSS_FORMAT[s["family"]] is None or f in CROSS_FORMAT[s["family"]])
            ]
            overall = min([s for s in [same] if s] + cross, key=lambda s: s["T_us"], default=None)
            cell = {"fmt": f, "M": m, "kurn": kurn, "same": same, "overall": overall, "all": here}
            for tag, comp in (("v_same", same), ("v_overall", overall)):
                if kurn is None:
                    cell[tag], cell[tag + "_ratio"] = "unmeasured", None
                elif comp is None:
                    cell[tag], cell[tag + "_ratio"] = "no competitor measured", None
                else:
                    cell[tag], cell[tag + "_ratio"] = verdict(
                        kurn["T_us"], kurn["cv"], comp["T_us"], comp["cv"], kurn["rounds"], comp["rounds"]
                    )
            ggml_skip = next((s for s in here if s["family"] == "ggml" and "T_us" not in s), None)
            if cell["v_same"] == "win":
                cell["dispatch"] = "kurn"
            elif (
                cell["v_same"] == "no competitor measured"
                and kurn is not None
                and ggml_skip is not None
                and "not supported" in ggml_skip["status"]
            ):
                cell["dispatch"] = "kurn"  # stock has no GPU kernel for this format at all
                cell["v_same"] = "only GPU kernel (ggml-cuda: " + ggml_skip["status"] + ")"
            else:
                cell["dispatch"] = "stock"
            cells[(f, m)] = cell
    return cells


def _kurn_config(st):
    c = next((c for c in st["checks"] if c.get("kind") == "check"), None)
    return c.get("config") if c else None


def dispatch_table(cells, meta):
    info = meta.get("info", {})
    arch = f"sm_{info['cc']}" if "cc" in info else None
    table = {"device": info.get("name"), "arch": arch, "rule": "kurn only where it beats the best same-format competitor by > 2 sigma",
             "cells": {}}  # fmt: skip
    for (f, m), c in cells.items():
        e = {"impl": c["dispatch"], "verdict": c["v_same"], "ratio": c.get("v_same_ratio")}
        if c["dispatch"] == "kurn":
            e["kurn_impl"] = c["kurn"]["impl"]
            e["config"] = _kurn_config(c["kurn"])
        table["cells"].setdefault(f, {})[str(m)] = e
    return table


def _fmt_tok(s):
    if not s or "T_us" not in s:
        return "—"
    return f"{s['tok_s']:,.0f} ± {s['tok_sd']:,.0f}"


def _pct(x):
    return f"{100 * x:.0f}%" if x == x and x is not None else "—"


def render(cells, agg, meta, results_dir=None, dry=False):
    info, roof, kit = meta.get("info", {}), meta.get("roofline", {}), meta.get("kit", {})
    bw = roof.get("hbm_read_gbs") or info.get("nominal_bw_gbs")
    peak = roof.get("int8_mma_sync_tops")
    L = []
    title = info.get("name") or "no GPU (dry run)"
    L.append(f"# KURN GPU benchmark matrix: {title}\n")
    if dry or not agg:
        L.append("**No GPU measurements in this result set. Every cell below is unmeasured.** The dry run checked code "
                 "generation, compilation and emulator numerics only; none of that is a performance result.\n")  # fmt: skip
    L.append("## Run")
    l2 = (info.get("l2_bytes") or 0) / 2**20
    L.append(f"- Device: {info.get('name', '—')} (cc {info.get('cc', '—')}, {info.get('sms', '—')} SMs, L2 {l2:.0f} MB), "
             f"driver {info.get('driver', '—')}, NVML energy: {info.get('nvml_energy', '—')}")  # fmt: skip
    L.append(f"- HBM read bandwidth: measured {roof.get('hbm_read_gbs', '—')} GB/s, nominal {info.get('nominal_bw_gbs', '—')} GB/s; "
             f"int8 mma.sync peak measured {peak or '—'} TOPS; idle board power {info.get('idle_w', '—')} W")  # fmt: skip
    L.append(f"- Workload: {MODEL['name']}, {MODEL['layers']} layers x shapes {', '.join(f'{n}x{k}' for n, k in MODEL['shapes'])}; "
             "tokens/s = batch / step time of those matmuls (activation quantization included for KURN and ggml)")  # fmt: skip
    if kit:
        L.append(f"- Kit: {json.dumps(kit)}")
    L.append("- Win rule: ratio = competitor time / KURN time; win if ratio > 1 + 2·sqrt(cv_KURN² + cv_comp²), loss if the "
             f"mirror holds, else tie; fewer than {MIN_ROUNDS} rounds = unresolved. Dispatch (`kernel_for()`) uses KURN only on a "
             "same-format win, or where ggml-cuda has no kernel for the format.\n")  # fmt: skip
    # summary
    L.append("## Wins, ties and losses\n")
    L.append(
        "| format | batch | KURN tok/s | best same-format competitor | vs same-format | best competitor (any format) | vs any | dispatch |"
    )
    L.append("|---|---|---|---|---|---|---|---|")
    counts = {}
    for (f, m), c in sorted(cells.items()):
        same = f"{c['same']['impl']}: {_fmt_tok(c['same'])}" if c["same"] else "—"
        ov = f"{c['overall']['impl']}: {_fmt_tok(c['overall'])}" if c["overall"] else "—"
        rs = f" ({c['v_same_ratio']:.2f}x)" if c.get("v_same_ratio") else ""
        ro = f" ({c['v_overall_ratio']:.2f}x)" if c.get("v_overall_ratio") else ""
        L.append(f"| {f} | {m} | {_fmt_tok(c['kurn'])} | {same} | {c['v_same']}{rs} | {ov} | {c['v_overall']}{ro} | {c['dispatch']} |")
        counts[c["v_overall"].split(" (")[0]] = counts.get(c["v_overall"].split(" (")[0], 0) + 1
    L.append("\nTotals vs the best competitor of any format: " + ", ".join(f"{k}: {v}" for k, v in sorted(counts.items())) + "\n")
    L.append(default_vs_tuned(cells))
    # detail
    L.append("## Detail per format\n")
    L.append("Columns: tokens/s (mean ± sd over rounds), % of measured HBM bandwidth, % of the measured int8 mma.sync peak, "
             "NVML joules per token, max relative error vs the exact reference on the same quantized inputs (KURN) and vs the "
             "f32-activation reference (everyone), status.\n")  # fmt: skip
    for f in sorted({k[0] for k in cells}):
        L.append(f"### {f}\n")
        L.append("| batch | implementation | tok/s | % HBM | % int8 TC | J/token | exact relerr | model relerr | status |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for m in sorted({k[1] for k in cells if k[0] == f}):
            for s in sorted(cells[(f, m)]["all"], key=lambda s: (s["family"] != "kurn", s["impl"])):
                hbm = s["bytes"] / (s["T_us"] * 1e-6) / 1e9 / bw if "T_us" in s and bw else float("nan")
                tc = s["ops"] / (s["T_us"] * 1e-6) / 1e12 / peak if "T_us" in s and peak else float("nan")
                ex = f"{s['relerr_exact']:.1e}" if s.get("relerr_exact") and s["family"] == "kurn" else "—"
                mo = f"{s['relerr_model']:.1e}" if s.get("relerr_model") else "—"
                jt = f"{s['J_tok']:.3f}" if s.get("J_tok") == s.get("J_tok") and "J_tok" in s else "—"
                ref = " (reference only)" if s["family"] in REFERENCE_ONLY else ""
                L.append(f"| {m} | {s['impl']}{ref} | {_fmt_tok(s)} | {_pct(hbm)} | {_pct(tc)} | {jt} | {ex} | {mo} | {s['status']} |")
        L.append("")
    return "\n".join(L) + "\n"


def default_vs_tuned(cells):
    """Markdown table: best `kurn:default-*` vs best `kurn:tuned-*` per (format, batch)."""
    out = ["## Default vs tuned KURN kernels\n",
           "Best default kernel and best tuned kernel per cell (tokens/s, mean ± sd); ratio = tuned / default.\n",
           "| format | batch | default (kernel: tok/s) | tuned (kernel: tok/s) | tuned / default |", "|---|---|---|---|---|"]  # fmt: skip
    for (f, m), c in sorted(cells.items()):
        timed = [s for s in c["all"] if s["family"] == "kurn" and "T_us" in s]
        d = min((s for s in timed if ":default-" in s["impl"]), key=lambda s: s["T_us"], default=None)
        t = min((s for s in timed if ":tuned-" in s["impl"]), key=lambda s: s["T_us"], default=None)
        ratio = f"{d['T_us'] / t['T_us']:.2f}x" if d and t else "—"
        ds = f"{d['impl'][5:]}: {_fmt_tok(d)}" if d else "unmeasured"
        ts = f"{t['impl'][5:]}: {_fmt_tok(t)}" if t else "unmeasured"
        out.append(f"| {f} | {m} | {ds} | {ts} | {ratio} |")
    return "\n".join(out) + "\n"


def unmeasured_cells(formats, batches=BATCHES):
    return {(f, m): {"fmt": f, "M": m, "kurn": None, "same": None, "overall": None, "all": [], "v_same": "unmeasured",
                     "v_overall": "unmeasured", "dispatch": "stock"} for f in formats for m in batches}  # fmt: skip


def write(results_dir, formats=None, dry=False):
    rows, meta = load(results_dir)
    agg = aggregate(rows)
    fmts = formats or sorted({k[0] for k in agg})
    cells = decide(agg, fmts) if agg else unmeasured_cells(fmts)
    md = render(cells, agg, meta, results_dir, dry=dry or not agg)
    with open(os.path.join(results_dir, "report.md"), "w") as fh:
        fh.write(md)
    with open(os.path.join(results_dir, "dispatch.json"), "w") as fh:
        json.dump(dispatch_table(cells, meta), fh, indent=1)
    return md, cells
