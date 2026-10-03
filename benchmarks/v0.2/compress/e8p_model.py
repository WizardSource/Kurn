"""Quantize Qwen3-1.7B with RHT + E8P (kurn.codebook.quantize_model) and write the dequantized
weights as an F16 GGUF for llama-perplexity:
    e8p_model.py OUT.gguf [--alpha 0.5] [--stages 1] [--rvq-json recipe.json] [--raw-from MIX.gguf]
Writes OUT.json with per-tensor errors and the achieved bpw. Untimed; stays under ~1 GB RSS."""

import argparse
import json
import os

from kurn import codebook as cb

M = os.path.expanduser("~/models")
ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--src", default=f"{M}/Qwen3-1.7B-BF16.gguf")
ap.add_argument("--imatrix", default=f"{M}/compress/qwen3-1.7b.imatrix.gguf")
ap.add_argument("--alpha", type=float, default=0.5)
ap.add_argument("--stages", type=int, default=1)
ap.add_argument("--rvq-json", help="kurn mix recipe whose assign maps tensors to E8P / E8P2 (stages)")
ap.add_argument("--raw-from", help="copy token_embd as stored in this GGUF (default: the BF16 source)")
a = ap.parse_args()
stages = a.stages
if a.rvq_json:
    assign = json.load(open(a.rvq_json))["assign"]
    stages = {n: int(q[3:] or 1) for n, q in assign.items() if q.startswith("E8P")}
stats = cb.quantize_model(a.src, a.out, a.imatrix, a.alpha, stages, raw_from=a.raw_from)
meta = {"alpha": a.alpha, "stages": a.stages, "raw_from": a.raw_from, "tensors": stats}
json.dump(meta, open(os.path.splitext(a.out)[0] + ".json", "w"), indent=1)
print(f"{a.out}: {stats['_total']['bpw']:.4f} bpw overall, {stats['_total']['bpw_blocks']:.4f} bpw on the quantized tensors")
