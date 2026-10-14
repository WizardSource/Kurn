# Ternary bitmap + signs benchmarks (kurn-tbm.cpp)

Scripts behind the numbers in `integration/llama.cpp/README.md` ("Ternary weights") and the project's
`kurn-ternary-results.md`. They run from a work dir `KW` holding:

- `opbench`, built against the patched llama.cpp (`llama-kurn`; see `tools/opbench.c`);
- `opbench-tq2t`, the same built against a second checkout (`llama-tq2t`) with the `rows=4 prefetch=4 pfgran=line`
  TQ2_0 kernels (the default since patch 0068; before it, `apply.sh --config '{"tq2_0": {"rows": 4, "prefetch": 4,
  "pfgran": "line"}}'`); `opbench` then needs a checkout generated with `--config '{"tq2_0": {"rows": 2}}'` for the old schedule;
- `models/` (the GGUFs named in `run_e2e.sh`) and `results/`.

| script | what |
|---|---|
| `bench_ops.sh NAME SHAPES LAYERS M THREADS Z ROUNDS` | decode-layer matmuls through opbench, configurations alternated per round, medians |
| `run_matrix.sh` | the op-level matrix: five models' shapes, zero-fraction sweep, threads, verify widths, prefill |
| `run_e2e.sh` | llama-bench pp512 / tg128 on real models, exact mode, llama-batched-bench 1..8 sequences |
| `run_identity.sh` | greedy text and perplexity identity (TQ2_0 vs tbm), load logs, resident memory |
| `tbm_microbench.cpp` | single-core L2-resident cycles per 64 weights, with decode knock-outs (build line in the file) |
| `paired_tq2.sh` | paired A/B (alternating order) of the TQ2_0 schedule: old `rows=2` (`opbench`, `llama-kurn` before regeneration) vs `rows=4` + line prefetch (`opbench-tq2t`, `llama-tq2t`) |
| `paired_ci.py A B FILE...` | geometric-mean speedup, Student-t and bootstrap 95% CIs, pairs won |
| `dump_tq2.c` | raw TQ2_0 MUL_MAT outputs over 351 cases; build against two checkouts and `cmp` (bit-identity of a schedule change) |

`kurn ternary stats FILE...` measures zero fractions (GGUF TQ1_0 / TQ2_0, packed 2-bit safetensors).
