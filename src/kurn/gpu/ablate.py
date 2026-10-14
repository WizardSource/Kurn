"""Ablations for the fused epilogues and wide verify (`kurn gpu ablate`, run by the kit before the report).

    ffn    gate+up+SwiGLU of Llama-3-8B (W = [gate; up], 28672 x 4096): the plain kernel over 2F rows followed by a
           separate SwiGLU kernel (`--swiglu-after 1`, one CUDA graph) vs the fused epilogue (epi=swiglu), same tile
    width  wide verify, 9-16 columns: the dp4a GEMV with cols=8 (two passes over the weights for 9-16 columns) and cols=16
           (one pass), and the tensor-core engine's bn=16 tiles (staged and packed), at M = 1, 8, 9, 12, 16
    fv     the fused Freivalds check (epi=fv) per Freivalds block size vs the same tile without it: overhead, false alarms
           over every timed call, the worst passing |C - X| / tau, and whether one corrupted weight tile is caught

Results: RESULTS/ablate.jsonl (one row per harness run), rendered into report.md by kurn.gpu.report.
"""

import argparse
import json
import os
import statistics

from .spec import SpecError, resolve

FFN_SHAPE = (28672, 4096)
WIDTH_SHAPE = (4096, 14336)
FV_SHAPES = ((4096, 4096), (28672, 4096))
FFN_FORMATS = ("q4_0", "q4_K", "mxfp4", "tbm")
WIDTH_FORMATS = ("q4_0", "mxfp4", "q1_0")
FV_FORMATS = ("q4_0", "q8_0", "tbm")
WIDTH_M = (1, 8, 9, 12, 16)

# engine tiles with two row tiles per warp (epi=swiglu pairs gate and up tiles), first legal option wins
_T8 = [{"bn": 8, "bm": 64, "wm": 2, "wn": 1, "xin": "f32", "stages": 4, "bk": b, **r} for b in (256, 128)
       for r in ({}, {"wload": "reg", "pf": 2}, {"wload": "reg", "pf": 1})]  # fmt: skip
_T16 = [{"bn": 16, "bm": 64, "wm": 2, "wn": 1, "xin": x, "stages": s, "bk": b, **r} for x, s in (("f32", 4), ("f16", 3))
        for b in (256, 128) for r in ({}, {"wload": "reg", "pf": 2}, {"wload": "reg", "pf": 1})]  # fmt: skip
_FV_TILES = (("mma8", (1, 8), [{"bn": 8, "bm": 64, "wm": 4, "wn": 1, "xin": "f32", "stages": 4, **r}
                               for r in ({}, {"wload": "reg", "pf": 2})]),
             ("mma16", (16,), [{"bn": 16, "bm": 64, "wm": 4, "wn": 1, "xin": x, "stages": s, **r}
                               for x, s in (("f32", 4), ("f16", 3)) for r in ({}, {"wload": "reg", "pf": 2})]),
             ("mma256", (256,), [{"bn": 128, "bm": 128, "wm": 2, "wn": 4, "xin": "f16", "bk": b, "stages": s}
                                 for b, s in ((128, 3), (64, 3), (256, 2))]))  # fmt: skip


def _first(base, options):
    for ov in options:
        try:
            return resolve({**base, **ov})
        except SpecError:
            continue
    return None


def _gemv(fmt, arch, **ov):
    base = {"op": "gemv", "weights": fmt, "arch": arch}
    return _first(base, [ov, {**ov, "unroll": 2}, {**ov, "unroll": 1}, {**ov, "unroll": 1, "sub": 2}])


def plan(arch, quick=False):
    """[(exp, fmt, kernel name, variant, config, [(N, K, M, extra harness args)])]."""
    out = []
    ffn = FFN_FORMATS[:1] if quick else FFN_FORMATS
    n, k = FFN_SHAPE
    for f in ffn:
        g = {"op": "gemm", "weights": f, "arch": arch}
        pairs = []
        if f != "tbm":
            pairs.append(("gemv", (1,), _gemv(f, arch), _gemv(f, arch, epi="swiglu")))
        for name, ms, opts in (("mma8", (1, 8), _T8), ("mma16", (16,), _T16)):
            sw = _first({**g, "epi": "swiglu"}, opts)
            plain = resolve({**sw, "epi": "none", "fvb": 0}) if sw else None
            pairs.append((name, ms, plain, sw))
        for name, ms, plain, sw in pairs:
            if plain and sw:
                out.append(("ffn", f, name, "plain+swiglu", plain, [(n, k, m, ("--swiglu-after", "1")) for m in ms]))
                out.append(("ffn", f, name, "fused", sw, [(n, k, m, ()) for m in ms]))
    n, k = WIDTH_SHAPE
    for f in WIDTH_FORMATS[:1] if quick else WIDTH_FORMATS:
        from .matrix import ENGINE_TILES, engine_tile

        tiles = {name: engine_tile(f, arch, opts) for name, _, opts in ENGINE_TILES if name in ("mma16", "mma16-reg")}
        for name, c in (("gemv-c1", _gemv(f, arch)), ("gemv-c8", _gemv(f, arch, cols=8)), ("gemv-c16", _gemv(f, arch, cols=16)),
                        *tiles.items()):  # fmt: skip
            if c is None:
                continue
            ms = (1,) if name == "gemv-c1" else WIDTH_M
            out.append(("width", f, name, "-", c, [(n, k, m, ()) for m in ms]))
    for f in FV_FORMATS[:1] if quick else FV_FORMATS:
        for name, ms, opts in _FV_TILES:
            base = _first({"op": "gemm", "weights": f, "arch": arch}, opts)
            if base is None:
                continue
            for fvb in (0, 64, 256, 1024):
                try:
                    c = resolve({**base, "epi": "fv", "fvb": fvb}) if fvb else base
                except SpecError:
                    continue
                runs = [(sn, sk, m, ("--corrupt", "1") if fvb else ()) for sn, sk in (FV_SHAPES[:1] if quick else FV_SHAPES)
                        for m in ms]  # fmt: skip
                out.append(("fv", f, name, f"fvb={fvb}" if fvb else "none", c, runs))
    return out


def _summ(us):
    if not us:
        return None, None
    m = statistics.mean(us)
    return m, (statistics.stdev(us) / m if len(us) > 1 and m else 0.0)


def run(results, harness, arch, quick=False, dry=False, log=print):
    from .harness import HarnessError, run_json
    from .spec import label
    from .toolchain import build_many, nvcc_build

    p = plan(arch, quick)
    built = [lib for _, lib in build_many([e[4] for e in p], lambda c: nvcc_build(c)[0])]
    path = os.path.join(results, "ablate.jsonl")
    nb = sum(not isinstance(v, Exception) for v in built)
    log(f"ablate: {len(p)} kernels ({nb} built), {sum(len(e[5]) for e in p)} runs{' (dry run: build only)' if dry else ''}")
    with open(path, "w") as fh:
        for i, (exp, f, name, var, c, runs) in enumerate(p):
            lib = built[i]
            for n, k, m, extra in runs:
                row = {"exp": exp, "fmt": f, "kernel": name, "variant": var, "N": n, "K": k, "M": m, "config": label(c)}
                if isinstance(lib, Exception):
                    row["status"] = f"build failed: {str(lib).splitlines()[0]}"
                elif dry:
                    row["status"] = "unmeasured (dry run)"
                else:
                    cmd = [harness, "run", "--lib", lib, "--fmt", f, "--N", str(n), "--K", str(k), "--M", str(m), "--reps",
                           "3" if quick else "5", "--secs", "0.2" if quick else "0.3", "--quant", "1", "--cold", "1", *extra]  # fmt: skip
                    try:
                        rows = run_json(cmd, timeout=900)
                    except (HarnessError, OSError) as e:
                        rows = [{"kind": "check", "status": f"harness error: {str(e)[:200]}"}]
                    chk = next((x for x in rows if x.get("kind") == "check"), {})
                    row["status"] = chk.get("status", "no check")
                    row["relerr"] = chk.get("relerr_exact")
                    row["us"] = [x["us"] for x in rows if x.get("kind") == "sample"]
                    fv = next((x for x in rows if x.get("kind") == "fv"), None)
                    if fv:
                        keys = ("block", "lambda", "checks", "fails", "worst_ratio", "corrupt_checks", "detected")
                        row["fv"] = {key: fv[key] for key in keys}
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                t, _ = _summ(row.get("us", []))
                log(f"  {exp:5} {f:6} {name:9} {var:13} N={n} K={k} M={m}: {row['status']}" + (f"  {t:.1f} us" if t else ""))
    return path


def load(results):
    path = os.path.join(results, "ablate.jsonl")
    if not os.path.exists(path):
        return []
    with open(path) as fh:
        return [json.loads(ln) for ln in fh if ln.strip().startswith("{")]


def _t(r):
    m, cv = _summ(r.get("us") or [])
    return (m, cv, len(r.get("us") or [])) if m else None


def _us(t):
    return f"{t[0]:.1f} ± {t[0] * t[1]:.1f}" if t else "unmeasured"


def section(rows):
    """Markdown: the three ablation tables (every cell "unmeasured" without GPU results)."""
    from .report import verdict

    L = ["## Fused epilogues and wide verify (ablations)\n",
         "From `kurn gpu ablate` (ablate.jsonl): harness `run` per kernel, cold weights (copies > L2), activation quantization "
         "included, times in us per call (mean ± sd over rounds); verdicts by the 2-sigma rule.\n"]  # fmt: skip
    by = {}
    for r in rows:
        by[(r["exp"], r["fmt"], r["kernel"], r["variant"], r["N"], r["K"], r["M"])] = r
    L += ["### Fused gate+up+SwiGLU\n",
          "W = [gate; up] (2F x K = 28672 x 4096, Llama-3-8B). plain+swiglu: the same kernel over 2F rows writes Y [M][2F], then "
          "a separate SwiGLU kernel reads it back; fused: silu(gate) * up in the epilogue from one activation load / convert, "
          "Y [M][F] only. Speedup = unfused / fused.\n",
          "| format | kernel | M | config (fused) | plain + SwiGLU us | fused us | speedup | verdict |",
          "|---|---|---|---|---|---|---|---|"]  # fmt: skip
    n = 0
    for (exp, f, name, var, nn, k, m), r in sorted(by.items()):
        if exp != "ffn" or var != "fused":
            continue
        p = by.get(("ffn", f, name, "plain+swiglu", nn, k, m))
        tf, tp = _t(r), _t(p) if p else None
        sp, v = "—", r["status"] if not r["status"].startswith("ok") else "—"
        if tf and tp:
            v, ratio = verdict(tf[0], tf[1], tp[0], tp[1], tf[2], tp[2])
            sp = f"{ratio:.2f}x"
        L.append(f"| {f} | {name} | {m} | `{r['config']}` | {_us(tp)} | {_us(tf)} | {sp} | {v} |")
        n += 1
    if not n:
        L.append("| — | — | — | — | unmeasured | unmeasured | — | — |")
    L += ["", "### Wide verify (9-16 columns)\n",
          f"{WIDTH_SHAPE[0]} x {WIDTH_SHAPE[1]}. gemv-c8 makes two passes over the weights for 9-16 columns, gemv-c16 one; "
          "mma16 / mma16-reg are the tensor-core bn=16 tiles (one pass). Cells: us per call, and in brackets the time relative "
          "to the format's batch-1 GEMV (gemv-c1); a flat row means the weights are streamed once.\n"]  # fmt: skip
    L += ["| format | kernel | " + " | ".join(f"M={m}" for m in WIDTH_M) + " |", "|---|---|" + "---|" * len(WIDTH_M)]
    n = 0
    for f in sorted({k[1] for k in by if k[0] == "width"}):
        base = next((_t(r) for key, r in by.items() if key[0] == "width" and key[1] == f and key[2] == "gemv-c1"), None)
        for name in ("gemv-c1", "gemv-c8", "gemv-c16", "mma16", "mma16-reg"):
            cells = []
            for m in WIDTH_M:
                r = next((r for key, r in by.items() if key[:3] == ("width", f, name) and key[6] == m), None)
                t = _t(r) if r else None
                cells.append(f"{t[0]:.1f} ({t[0] / base[0]:.2f})" if t and base else f"{t[0]:.1f}" if t else ("—" if r is None else
                             "unmeasured"))  # fmt: skip
            if any(c != "—" for c in cells):
                L.append(f"| {f} | {name} | " + " | ".join(cells) + " |")
                n += 1
    if not n:
        L.append("| — | — | " + " | ".join("unmeasured" for _ in WIDTH_M) + " |")
    L += ["", "### Fused Freivalds check (epi=fv)\n",
          "Secret +-1 probes per block of `fvb` output rows (2 probes), columns folded into 16 slots with secret signs, "
          "statistical threshold lambda K 2^-24 sqrt(sum x^2 W^2) (lambda 8). Overhead = fv time / plain time - 1. "
          "False alarms = fails over every check of every timed call (must be 0); worst = the largest passing |C - X| / tau "
          "(the margin: lambda could shrink by about 1/worst); caught = blocks flagged after one 16 x K-tile of weights is "
          "corrupted (must be > 0).\n",
          "| format | tile | N x K | M | fvb | plain us | fv us | overhead | false alarms / checks | worst | caught |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
    n = 0
    for (exp, f, name, var, nn, k, m), r in sorted(by.items()):
        if exp != "fv" or var == "none":
            continue
        p = by.get(("fv", f, name, "none", nn, k, m))
        tf, tp = _t(r), _t(p) if p else None
        ov = f"{tf[0] / tp[0] - 1:+.1%}" if tf and tp else "—"
        fv = r.get("fv") or {}
        fa = f"{fv['fails']} / {fv['checks']}" if fv else r["status"] if not r["status"].startswith("ok") else "—"
        wst = f"{fv['worst_ratio']:.2g}" if fv else "—"
        cg = str(fv.get("detected", "—")) if fv else "—"
        L.append(f"| {f} | {name} | {nn}x{k} | {m} | {var[4:]} | {_us(tp)} | {_us(tf)} | {ov} | {fa} | {wst} | {cg} |")
        n += 1
    if not n:
        L.append("| — | — | — | — | — | unmeasured | unmeasured | — | — | — | — |")
    return "\n".join(L) + "\n"


def main(argv=None):
    from .harness import build_harness, detect_arch

    ap = argparse.ArgumentParser(prog="kurn gpu ablate", description=__doc__.split("\n")[0])
    ap.add_argument("--results", required=True)
    ap.add_argument("--harness")
    ap.add_argument("--arch")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--dry", action="store_true", help="build the kernels only (no GPU)")
    a = ap.parse_args(argv)
    arch = a.arch or detect_arch() or "sm_80"
    os.makedirs(a.results, exist_ok=True)
    h = None if a.dry else (a.harness or build_harness(arch, explicit=bool(a.arch)))
    run(a.results, h, arch, a.quick, a.dry)
    print(section(load(a.results)))
    return 0
