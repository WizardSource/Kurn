"""KLD / PPL vs bpw table over results/ppl.csv with the Pareto front marked. bpw is over the 2-D
weights incl. token_embd; F16 writebacks (E8P, LoRC) take their true bpw from their stats JSON.
    pareto.py results/ppl.csv [results/e2e.csv]"""

import csv
import glob
import json
import os
import sys

here = os.path.dirname(os.path.abspath(sys.argv[1]))
true_bpw = {}
for f in glob.glob(os.path.join(here, "e8p_model_*.json")):
    true_bpw["qwen3-1.7b-e8p-" + os.path.basename(f)[len("e8p_model_") : -5]] = json.load(open(f))["tensors"]["_total"]["bpw"]
for f in glob.glob(os.path.join(here, "lorc-*.json")):
    true_bpw["qwen3-1.7b-" + os.path.basename(f)[:-5]] = json.load(open(f))["_total"]["bpw"]
e2e = {}
if len(sys.argv) > 2 and os.path.exists(sys.argv[2]):
    for r in csv.DictReader(open(sys.argv[2])):
        e2e[(r["model"], r["mode"])] = r
rows = []
for r in csv.DictReader(open(sys.argv[1])):
    m = r["model"]
    b = true_bpw.get(m, float(r["bpw"]))
    rows.append((b, float(r["kld"]), float(r["kld_err"]), float(r["ppl"]), float(r["ppl_err"]), r["top1"], m))
rows.sort()
best = float("inf")
front = set()
for _, k, *_, m in rows:
    if k < best:
        best = k
        front.add(m)
print("| model | bpw | KLD | PPL | top-1 % | Pareto | decode tok/s (repack / plain) | J/token (repack) |")
print("|---|---|---|---|---|---|---|---|")
for b, k, ke, p, pe, t, m in rows:
    d, pl = e2e.get((m, "default")), e2e.get((m, "plain"))
    tok = f"{d['decode_tok_s']} / {pl['decode_tok_s']}" if d and pl else "-"
    j = d["decode_J_per_tok"] if d else "-"
    star = "*" if m in front else ""
    print(f"| {m.replace('qwen3-1.7b-', '')} | {b:.3f} | {k:.4f} ± {ke:.4f} | {p:.3f} ± {pe:.2f} | {t} | {star} | {tok} | {j} |")
