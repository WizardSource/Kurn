"""Per-tensor empirical entropy of quantized weight codes vs their fixed width:
    entropy_report.py OUT.csv GGUF [GGUF...]
Columns: file, tensor, fmt, n, entropy (bits/code), huff12 (avg length of a 12-bit-limited Huffman
code), bits (stored width), bpw, bpw_entropy / bpw_huff12 (codes coded + scales raw)."""

import csv
import os
import sys

from kurn import entropy as en
from kurn import mixed

out, files = sys.argv[1], sys.argv[2:]
rows = []
for f in files:
    tot = {}
    for name, t in mixed.tensors(f).items():
        fmt = mixed.type_name(t)
        if not mixed.is_matrix(name, t) or fmt not in en.CODES:
            continue
        r = {"file": os.path.basename(f), "tensor": name, **en.tensor_report(t, fmt)}
        rows.append(r)
        a = tot.setdefault(fmt, [0, 0.0, 0.0])
        a[0] += r["n"]
        a[1] += r["n"] * r["entropy"]
        a[2] += r["n"] * r["huff12"]
    for fmt, (n, h, hl) in tot.items():
        side = en.BPW[fmt] - en.BITS[fmt]
        bs, bh = h / n + side, hl / n + side
        print(f"{os.path.basename(f):28s} {fmt:5s} {n / 1e6:8.1f}M codes  H={h / n:.3f}  huff12={hl / n:.3f} "
              f"of {en.BITS[fmt]} bits -> {en.BPW[fmt]:.3f} bpw becomes {bs:.3f} (Shannon) / {bh:.3f} (Huffman): "
              f"{1 - bs / en.BPW[fmt]:.1%} / {1 - bh / en.BPW[fmt]:.1%} fewer bytes", flush=True)  # fmt: skip
with open(out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
