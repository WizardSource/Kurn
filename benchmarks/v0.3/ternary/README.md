# Ternary bitmap + signs benchmarks (kurn-tbm.cpp)

Scripts behind the numbers in `integration/llama.cpp/README.md` ("Ternary weights") and the project's
`kurn-ternary-results.md`. They run from a work dir `KW` holding:

- `opbench`, built against the patched llama.cpp (`llama-kurn`; see `tools/opbench.c`);
- `opbench-tq2t`, the same built against a second checkout (`llama-tq2t`) where `apply.sh` was run with
  `--config '{"tq2_0": {"rows": 4, "prefetch": 4, "pfgran": "line"}}'` (the tuned TQ2_0 baseline);
- `models/` (the GGUFs named in `run_e2e.sh`) and `results/`.

| script | what |
|---|---|
| `bench_ops.sh NAME SHAPES LAYERS M THREADS Z ROUNDS` | decode-layer matmuls through opbench, configurations alternated per round, medians |
| `run_matrix.sh` | the op-level matrix: five models' shapes, zero-fraction sweep, threads, verify widths, prefill |
| `run_e2e.sh` | llama-bench pp512 / tg128 on real models, exact mode, llama-batched-bench 1..8 sequences |
| `run_identity.sh` | greedy text and perplexity identity (TQ2_0 vs tbm), load logs, resident memory |
| `tbm_microbench.cpp` | single-core L2-resident cycles per 64 weights, with decode knock-outs (build line in the file) |

`kurn ternary stats FILE...` measures zero fractions (GGUF TQ1_0 / TQ2_0, packed 2-bit safetensors).
