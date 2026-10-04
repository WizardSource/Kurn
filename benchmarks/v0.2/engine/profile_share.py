#!/usr/bin/env python3
"""Classify `perf report --no-children --sort dso,sym --stdio` output into CPU-time shares:
matmul (quantized GEMV/GEMM kernels), spin-wait (barriers, OpenMP waits, the engine's epoch
waits), dispatch (llama.cpp graph build / scheduling / allocation: work the compiled engine
does once at load instead of every token), attention+other ops, kernel/syscalls, rest.

  profile_share.py REPORT.txt [--json]
"""

import json
import re
import sys

RULES = [
    ("spin", r"ggml_barrier|libgomp|sched_yield|\bawait_slow\b|\bwait_kind\b|gomp_|do_spin|do_wait"),
    (
        "matmul",
        r"kq8e_(store|axpy|swiglu_q8|store_sumsq)|kq8_gemv|vec_dot|gemv|gemm|ggml_compute_forward_mul_mat|amx|tinyblas|"
        r"repack|mul_mat|kq8",
    ),
    (
        "dispatch",
        r"libllama|ggml_backend_sched|ggml_gallocr|ggml_graph_(?!compute)|ggml_new_tensor|ggml_visit_parents|"
        r"ggml_hash|llm_build|llama_context|ggml_build_forward|ggml_view|ggml_reshape|ggml_permute|ggml_cont|"
        r"ggml_set_name|ggml_format_name|ggml_init|ggml_free|ggml_graph_compute_thread|ggml_compute_forward\b|"
        r"ggml_backend_(?!cpu_graph)|malloc|free|memset|operator new|_int_malloc|_int_free",
    ),
    (
        "attn_ops",
        r"attention|flash_attn|soft_max|rope|rms_norm|ggml_compute_forward_|ggml_vec_|swiglu|silu|quantize|"
        r"rmsnorm|norm_quant|reduce_into|step|cpy|get_rows|add|mul\b|exp",
    ),
    ("kernel", r"\[kernel|\[k\]"),
]


def classify(path):
    tot = {k: 0.0 for k, _ in RULES}
    tot["other"] = 0.0
    top = []
    for line in open(path):
        m = re.match(r"\s*([\d.]+)%\s+(\S+)\s+\[(.)\]\s+(.*)", line)
        if not m:
            continue
        pct, dso, kind, sym = float(m.group(1)), m.group(2), m.group(3), m.group(4).strip()
        text = f"{dso} {sym}" + (" [k]" if kind == "k" else "")
        cat = next((k for k, pat in RULES if re.search(pat, text)), "other")
        if kind == "k" and cat not in ("spin",):
            cat = "kernel"
        tot[cat] += pct
        top.append((pct, cat, dso, sym))
    return tot, sorted(top, reverse=True)[:15]


def main():
    tot, top = classify(sys.argv[1])
    if "--json" in sys.argv:
        print(json.dumps({k: round(v, 2) for k, v in tot.items()}))
        return
    print("  ".join(f"{k} {v:.1f}%" for k, v in tot.items()))
    for pct, cat, dso, sym in top:
        print(f"  {pct:6.2f}%  {cat:9s} {dso:28s} {sym[:90]}")


if __name__ == "__main__":
    main()
