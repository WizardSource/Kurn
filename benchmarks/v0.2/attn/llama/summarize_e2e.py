"""Median (min-max) tok/s per variant from run_e2e.sh bench CSVs.

python summarize_e2e.py results/e2e/bench_d4096.csv [...]
"""

import csv
import statistics
import sys
from collections import defaultdict


def main(paths):
    for path in paths:
        rows = defaultdict(list)
        pin = {}
        with open(path) as fh:
            for r in csv.reader(fh):
                if not r or r[0].startswith("#"):
                    continue
                name, n_prompt, n_gen, avg_ts = r[0], r[-8], r[-7], float(r[-2])
                test = f"pp{n_prompt}" if n_prompt != "0" else f"tg{n_gen}"
                rows[(name, test)].append(avg_ts)
                pin[name] = r[14:16]
        print(f"# {path}")
        print("variant,test,reps,tok_s_med,tok_s_min,tok_s_max,cpu_mask,cpu_strict")
        for (name, test), v in rows.items():
            print(f"{name},{test},{len(v)},{statistics.median(v):.2f},{min(v):.2f},{max(v):.2f},{','.join(pin[name])}")


if __name__ == "__main__":
    main(sys.argv[1:])
