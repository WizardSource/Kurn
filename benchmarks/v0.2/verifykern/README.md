# Narrow verify kernels: measurements (verify-kernels series)

Results write under `results/`. The scripts expect local checkouts of llama.cpp (before/after),
optional ik_llama.cpp, models, and binaries, and take the bench lock (`../benchlock.sh`) for every timed run.

| script | what |
|---|---|
| `opsweep.sh`, `ab.sh` | opbench (every MUL_MAT of 8 Qwen3-8B layers) per width; `ab.sh` alternates env configs and reports medians |
| `sweep_sched.sh`, `sweep_pf.sh` | verify schedule sweeps (row groups x prefetch distance, one-line vs whole-record prefetch) |
| `verify_cost.cpp`, `run_vcost.sh` | whole-forward verify cost (M tokens, logits for all M, KV 292) on stock / before / after / ik_llama.cpp, builds interleaved |
| `run_bb.sh`, `summ_bb.py` | `llama-batched-bench -npp 512 -ntg 64 -npl 1,2,3,4,6,8 -pps`, aggregate decode tok/s |
| `run_width_defaults.py` | `../specwidth/run_width.py` without the forced `GGML_KURN_AMX=0` (shipped defaults) |
| `server_bench.py` | llama-server: no draft, fixed k, `--spec-width` policy; tok/s from the server's timings |
| `exact_check.py` | llama-server with `GGML_KURN_FA_MODE=exact`: speculative output == no-draft output |
| `run_final.sh`, `summ_op.py` | the whole end-to-end set; final kernel table summary |

`results/`: raw outputs (opbench sweeps, A/B rounds, cost tables, batched-bench, speculative CSVs, server jsonl).
