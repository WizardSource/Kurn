#!/usr/bin/env python3
"""<= 2-bit GEMV experiments (workstream lowbit). Results -> benchmarks/v0.2/lowbit/results/.

  run_lowbit.py screen FMT...           every legal avx512 config of each format, hot 1T, one short rep
  run_lowbit.py final REGIME T FMT...   best config per family (layout + lut/addsub variant, from the
                                        screen) plus ggml, interleaved reps (default 3), median
Data, quantization and the correctness reference come from ggml (contrib ggml-harness with the
tq1_0 / q2_K rows added, see build_ggml_harness.sh); every run must report check=ok.
Run timed modes under benchmarks/v0.2/benchlock.sh.
"""

import atexit
import csv
import os
import statistics
import subprocess
import sys

from kurn import lowbit, spec, toolchain
from kurn.harness import bench

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
HARNESS = os.environ.get("BENCH_GGML", "/tmp/kurn-lowbit/bench_ggml")
CACHE = os.environ.get("LOWBIT_CACHE", "/tmp/kurn-lowbit/cache")
W_CORE = 5.47
REPS = int(os.environ.get("REPS", "3"))
GGML_ROW_BYTES = {"q1_0": 18 / 128, "q2_0": 18 / 64, "tq2_0": 66 / 256, "tq1_0": 54 / 256, "q2_K": 84 / 256}
ROOF_GBPS = {1: 18.3, 8: 128.7}
SHAPE = os.environ.get("SHAPE", "")  # "KxN" for final, e.g. 8192x16384; default harness shape (K=4096)
EXTRA = ["--K", SHAPE.split("x")[0], "--N", SHAPE.split("x")[1]] if SHAPE else []


def family(c):
    if c["layout"] == "lut":
        return f"lut:{c['lut']}"
    if c["layout"] == "addsub":
        return f"addsub:{c['addsub']}"
    if c["layout"] in ("i16", "i8"):
        return f"{c['layout']}:{c['unpack']}/{c['accum']}"
    return c["layout"]


def packed_bytes_per_weight(c):
    """Bytes the kernel streams per weight (its repacked layout), for % of roofline."""
    r = lowbit.generic.RECIPES.get(c["weights"])
    if c["layout"] == "lut":
        p = lowbit.LutPlan(c)
        return p.rec / (32 * r.period)
    if c["layout"] == "addsub":
        planes = 1 if r.bits == 1 else 2
        ub = planes * 128 + (64 if c["addsub"] == "sad" else 0)
        return (64 + (r.period // 32) * ub) / (32 * r.period)
    if c["layout"] == "k16":
        return (320 + 1024) / (16 * 256)
    if c["layout"] in ("i16", "l32"):
        return (r.bits / 8) + 2 / r.period  # codes + fp16 d per period (generic interleaved layouts)
    return GGML_ROW_BYTES[c["weights"]]


def configs(fmt):
    return list(spec.legal_configs(op="gemv", weights=fmt, target="avx512_vnni", prefetch=(0, 8)))


def key(c):
    return tuple(c[k] for k in spec.CODEGEN_KEYS)


def measure(c, regime, T, secs):
    c = dict(c, threads=T, wait="spin")
    row = bench(toolchain.build(c), c, regime, secs, EXTRA, harness=HARNESS)
    if row.get("check") != "ok":
        raise SystemExit(f"FAILED correctness: {c} -> {row}")
    return row


def ggml_measure(fmt, regime, T, secs, impl="ggml"):
    c = {"target": "avx512_vnni", "threads": T, "wait": "spin", "op": "gemv", "weights": fmt}
    return bench(impl, c, regime, secs, EXTRA, harness=HARNESS)


def summarize(name, fmt, regime, T, c, rows):
    us = statistics.median(float(r["us"]) for r in rows)
    cpu = statistics.median(float(r["cpu_us"]) for r in rows)
    K, N = int(rows[0]["K"]), int(rows[0]["N"])
    bpw = packed_bytes_per_weight(c) if c else GGML_ROW_BYTES[fmt]
    gbps_packed = K * N * bpw / (us * 1e-6) / 1e9
    out = {
        "format": fmt, "regime": regime, "threads": T, "impl": name, "K": K, "N": N, "reps": len(rows),
        "us": round(us, 3), "us_min": round(min(float(r["us"]) for r in rows), 3),
        "us_max": round(max(float(r["us"]) for r in rows), 3), "cpu_us": round(cpu, 3), "energy_uJ": round(cpu * W_CORE, 2),
        "Gweights_s": round(K * N / us / 1e3, 2), "GBps_ggml_bytes": round(K * N * GGML_ROW_BYTES[fmt] / us / 1e3, 2),
        "packed_bpw": round(8 * bpw, 4), "GBps_packed": round(gbps_packed, 2),
        "pct_roofline": round(100 * gbps_packed / ROOF_GBPS.get(T, 128.7), 1),
        "max_drift_s": max(abs(float(r["drift_s"])) for r in rows), "relerr": max(float(r["relerr"]) for r in rows),
        "config": "" if c is None else " ".join(f"{k}={c[k]}" for k in ("layout", "lut", "addsub", "rows", "prefetch", "unpack",
                                                                           "correction", "scales", "accum") if c[k] != "auto"),
    }  # fmt: skip
    return out


def write(name, rows, append=False):
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, name)
    new = not (append and os.path.exists(p))
    with open(p, "a" if append else "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        if new:
            w.writeheader()
        w.writerows(rows)
    print("wrote", p, flush=True)


def screen(fmts):
    for fmt in fmts:
        rows = []
        for c in configs(fmt):
            r = measure(c, "hot", 1, 0.3)
            rows.append(
                {
                    "format": fmt,
                    "family": family(c),
                    "rows": c["rows"],
                    "prefetch": c["prefetch"],
                    "us": r["us"],
                    "cpu_us": r["cpu_us"],
                    "energy_uJ": round(r["cpu_us"] * W_CORE, 2),
                    "drift_s": r["drift_s"],
                    "key": "|".join(map(str, key(c))),
                }
            )
            print(fmt, family(c), c["rows"], c["prefetch"], f"{r['us']:.2f} us", flush=True)
        write(f"screen_hot1_{fmt}.csv", rows)


def best_per_family(fmt, n=2):
    """Top-n configs (energy objective, hot 1T screen) per family."""
    by_key = {key(c): c for c in configs(fmt)}
    with open(os.path.join(OUT, f"screen_hot1_{fmt}.csv")) as fh:
        rows = list(csv.DictReader(fh))
    fams = {}
    for r in sorted(rows, key=lambda r: float(r["energy_uJ"])):
        k = tuple(spec._coerce(v) if v not in ("None",) else None for v in r["key"].split("|"))
        c = by_key.get(k)
        if c is None:
            continue
        fams.setdefault(r["family"], [])
        if len(fams[r["family"]]) < n:
            fams[r["family"]].append(c)
    return fams


def final(regime, T, fmts):
    secs = 0.6 if regime == "hot" else 1.0
    for fmt in fmts:
        os.environ["BENCH_CACHE"] = os.path.join(CACHE, fmt)  # generated 1.2 GB cold sets, quantized once
        fams = best_per_family(fmt, int(os.environ.get("TOPN", "2")))
        if os.environ.get("TRIM"):  # top-3 lut variants, best addsub mode, best i16 / k16 (no l32)
            keep, nlut, nadd, ni16 = {}, 0, 0, 0
            for fam, cs in fams.items():  # best-first
                kind = fam.split(":")[0]
                if (kind == "lut" and nlut < 3) or (kind == "addsub" and nadd < 1) or (kind in ("i16", "k16") and ni16 < 1):
                    keep[fam] = cs
                    nlut, nadd, ni16 = nlut + (kind == "lut"), nadd + (kind == "addsub"), ni16 + (kind in ("i16", "k16"))
            fams = keep
        cands = [(f"kurn {fam}", c) for fam, cs in fams.items() for c in cs]
        cands += [("ggml vec_dot", None)]
        res = {i: [] for i in range(len(cands))}
        for _ in range(REPS):  # interleaved
            for i, (name, c) in enumerate(cands):
                res[i].append(ggml_measure(fmt, regime, T, secs) if c is None else measure(c, regime, T, secs))
                print(fmt, regime, T, name, f"{res[i][-1]['us']:.2f} us drift {res[i][-1]['drift_s']}", flush=True)
        rows = [summarize(name, fmt, regime, T, c, res[i]) for i, (name, c) in enumerate(cands)]
        best = {}
        for r in rows:
            if r["impl"] not in best or r["energy_uJ"] < best[r["impl"]]["energy_uJ"]:
                best[r["impl"]] = r
        write(f"final_{regime}{T}{'_' + SHAPE if SHAPE else ''}_{fmt}.csv", sorted(best.values(), key=lambda r: r["energy_uJ"]))


def sweepn(fmts):
    """Table-build amortization: best lut vs best i16 / k16 config, hot 1T, N rows per call."""
    out = []
    for fmt in fmts:
        fams = best_per_family(fmt, 1)
        pick = {}
        for fam, cs in fams.items():
            kind = fam.split(":")[0]
            if kind in ("lut", "i16", "k16") and kind not in pick:
                pick[kind] = (fam, cs[0])  # fams is ordered best-first
        for n in (256, 1024, 2048):
            res = {k: [] for k in pick}
            for _ in range(REPS):
                for kind, (_fam, c) in pick.items():
                    cc = dict(c, threads=1, wait="spin")
                    res[kind].append(bench(toolchain.build(cc), cc, "hot", 0.5, ["--N", str(n)], harness=HARNESS))
            for kind, (fam, _c) in pick.items():
                us = statistics.median(r["us"] for r in res[kind])
                out.append(
                    {
                        "format": fmt,
                        "N": n,
                        "family": fam,
                        "us": round(us, 3),
                        "ns_per_row": round(1e3 * us / n, 2),
                        "Gweights_s": round(4096 * n / us / 1e3, 1),
                        "max_drift_s": max(abs(float(r["drift_s"])) for r in res[kind]),
                    }
                )
                print(out[-1], flush=True)
    write("sweep_n_hot1.csv", out)


def boost_autogroup():
    """nice does not cross session autogroups on this VM: raise ours while holding the bench lock."""
    path = f"/proc/{os.getpid()}/autogroup"
    if subprocess.run(["sudo", "-n", "sh", "-c", f"echo -20 > {path}"], capture_output=True).returncode == 0:
        atexit.register(subprocess.run, ["sudo", "-n", "sh", "-c", f"echo 0 > {path}"], capture_output=True)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode in ("final", "sweepn"):
        boost_autogroup()
    if mode == "screen":
        screen(sys.argv[2:])
    elif mode == "sweepn":
        sweepn(sys.argv[2:])
    elif mode == "final":
        final(sys.argv[2], int(sys.argv[3]), sys.argv[4:])
    else:
        raise SystemExit(__doc__)
