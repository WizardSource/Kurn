"""Cost of split-bf16 scores (qk_hi): kurn attention with the default qk_hi against qk_hi 0, paired.

    python qk_precision.py [--reps 8] [--secs 0.5] [--threads 8] [--only SUBSTR] [--out DIR]

Every case runs both kernels once per rep, in alternating order. The change is the geometric mean of the
per-rep time ratios (default / qk_hi 0) with a 95% t-interval on the log ratios, so that clock drift
between reps cancels. Prefill is n_q 512 at depth (hot KV, as matrix.py; GQA shapes, also from
kattn_pack's buffer as llama.cpp's prefill reads it, and MLA with 16 heads). Decode is n_q 1 with cold
KV: GQA decode runs the f32 row engine, which qk_hi does not touch; MLA decode with 16 or 128 heads and
4-token verify of the 8B shape run the tile engine. Raw times go to DIR/qk_precision_raw.csv, the
summary to DIR/qk_precision.csv.
"""

import argparse
import csv
import math
import os
import statistics

from kurn import attention as A

HERE = os.path.dirname(os.path.abspath(__file__))
DEPTHS = (4096, 8192, 16384, 32768)
# two-sided 95% t quantiles by degrees of freedom
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
       11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131, 19: 2.093, 24: 2.064, 29: 2.045}  # fmt: skip


def cases():
    out = []
    for d in DEPTHS:
        for shape, h in (("1.7B", 16), ("8B", 32)):
            out.append((f"prefill-{shape}-d{d}", "amx_bf16", {"nq": 512, "nkv": d, "heads": h, "kv_heads": 8, "dk": 128}, "hot"))
            out.append((f"prefill-{shape}-d{d}-packed", "amx_bf16", {"nq": 512, "nkv": d, "heads": h, "kv_heads": 8, "dk": 128}, "hot"))
        out.append((f"prefill-8B-d{d}-avx512bf16", "avx512_bf16", {"nq": 512, "nkv": d, "heads": 32, "kv_heads": 8, "dk": 128}, "hot"))
        out.append((f"prefill-mla-h16-d{d}", "amx_bf16", {"nq": 512, "nkv": d, "heads": 16, "kv_heads": 1, "dk": 576, "mla": 1}, "hot"))
        for h in (16, 128):
            out.append((f"decode-mla-h{h}-d{d}", "amx_bf16", {"nq": 1, "nkv": d, "heads": h, "kv_heads": 1, "dk": 576, "mla": 1}, "cold"))
        out.append((f"verify4-8B-d{d}", "amx_bf16", {"nq": 4, "nkv": d, "heads": 32, "kv_heads": 8, "dk": 128}, "cold"))
        out.append((f"decode-8B-d{d}", "amx_bf16", {"nq": 1, "nkv": d, "heads": 32, "kv_heads": 8, "dk": 128}, "cold"))
    return out


def paired(t_new, t_old):
    """geometric-mean change and its 95% interval, in percent"""
    d = [math.log(a / b) for a, b in zip(t_new, t_old)]
    m = statistics.mean(d)
    h = T95.get(len(d) - 1, 1.96) * statistics.stdev(d) / math.sqrt(len(d)) if len(d) > 1 else float("nan")
    return tuple(100 * (math.exp(x) - 1) for x in (m, m - h, m + h))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--secs", type=float, default=0.5)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--only", default="")
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    raw = open(os.path.join(a.out, "qk_precision_raw.csv"), "w", newline="")
    rw = csv.writer(raw)
    rw.writerow(["case", "rep", "qk_hi", "us", "relerr"])
    rows = []
    for name, target, prob, regime in cases():
        if a.only and a.only not in name:
            continue
        spec = {"target": target, "kv": "f16", "causal": 1, **prob, "threads": a.threads}
        cs = {None: A.resolve(spec), 0: A.resolve(spec, {"qk_hi": 0})}
        so = {h: A.build(c) for h, c in cs.items()}
        t = {h: [] for h in cs}
        err = {h: 0.0 for h in cs}
        for r in range(a.reps):
            for h in ((None, 0) if r % 2 == 0 else (0, None)):
                row = A.bench(so[h], cs[h], regime, a.secs, ("--check-toks", "1", *(("--packed",) if name.endswith("-packed") else ())))
                t[h].append(row["us"])
                err[h] = max(err[h], row["relerr"])
                rw.writerow([name, r, cs[h]["qk_hi"], row["us"], row["relerr"]])
            raw.flush()
        chg, lo, hi = paired(t[None], t[0])
        rows.append({"case": name, "qk_hi": cs[None]["qk_hi"], "us_qk_hi0": statistics.median(t[0]),
                     "us_default": statistics.median(t[None]), "change_pct": round(chg, 2), "ci95_lo": round(lo, 2),
                     "ci95_hi": round(hi, 2), "reps": a.reps, "relerr_qk_hi0": err[0], "relerr_default": err[None]})  # fmt: skip
        print(f"{name:30s} {rows[-1]['us_qk_hi0']:10.0f} -> {rows[-1]['us_default']:10.0f} us  {chg:+6.1f}% [{lo:+.1f}, {hi:+.1f}]"
              f"  relerr {err[0]:.1e} -> {err[None]:.1e}", flush=True)  # fmt: skip
    raw.close()
    with open(os.path.join(a.out, "qk_precision.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
