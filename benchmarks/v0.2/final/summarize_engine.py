"""Summarize the quiet-machine engine rows: the rows appended to WS-E's committed
results CSVs after the integration commit (`git show HEAD:<csv>` gives the committed row count).

    python summarize_engine.py gen_qwen3 gen_olmoe ...
"""

import csv
import statistics
import subprocess
import sys
from pathlib import Path

RES = Path(__file__).resolve().parent.parent / "engine" / "results"
COLS = ("decode_tok_s", "med_ms", "J_per_tok", "J_per_tok_10W", "wait_share", "quiet_wait_share", "foreign_cpu", "load1")


def committed_rows(name):
    rel = f"kurn/benchmarks/v0.2/engine/results/{name}.csv"
    r = subprocess.run(["git", "show", f"HEAD:{rel}"], capture_output=True, text=True, cwd=RES)
    return max(0, len(r.stdout.splitlines()) - 1) if r.returncode == 0 else 0


def fmt(vals):
    vals = [float(v) for v in vals if v not in ("", None)]
    if not vals:
        return "-"
    return f"{statistics.median(vals):.3g} [{min(vals):.3g}-{max(vals):.3g}]"


for name in sys.argv[1:]:
    rows = list(csv.DictReader(open(RES / f"{name}.csv")))[committed_rows(name) :]
    print(f"## {name} ({len(rows)} new rows)")
    print("| config | " + " | ".join(COLS) + " |")
    print("|" + "---|" * (len(COLS) + 1))
    for cfg in dict.fromkeys(r["config"] for r in rows):
        sel = [r for r in rows if r["config"] == cfg]
        print(f"| {cfg} | " + " | ".join(fmt([r[c] for r in sel]) for c in COLS) + " |")
    print()
