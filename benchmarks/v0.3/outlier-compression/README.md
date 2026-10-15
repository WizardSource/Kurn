# Outlier-aware compression and R8: scripts and raw results

Write-up: the Project store's `docs/kurn-outlier-compression.md` (copied numbers below come from these files).

Machine: 8-vCPU Emerald Rapids KVM guest (AMX), Ubuntu 24.04, gcc 13.3; llama.cpp `4ebdf2c` + kurn
(rsi-build + matmul-opt + 0077-0078). Models: Qwen3-1.7B, Qwen3-8B, Llama-3.2-3B-Instruct (HF BF16 -> F16 GGUF).

## Scripts (`scripts/`)

| Script | Purpose |
|---|---|
| `qstream.py` | layer-streaming fp32 Qwen3 / Llama forward (within 5e-5 of HF logits). `--mode stats` (calibration E[x^2], means, outlier statistics), `kl` (activation-numerics KL sweep), `evalgguf` (KL of GGUF variants + OA sidecars), `kv` (K/V channel statistics and k4c variants), `check` (vs HF) |
| `pipeline.sh TAG HF_DIR` | F16 GGUF, llama-quantize baselines (+ imatrix), calibration, OA / R8 variants, F16 KL base, perplexity matrix |
| `queue2.sh` | ablations, Q6_K / exact-mode references, KV analysis, GSM8K |
| `ppl_matrix.sh`, `ppl_table.py` | llama-perplexity KL vs the F16 base for a list of models; markdown table from the logs |
| `gsm8k.py MODEL N` | first N GSM8K test problems through llama-server (4 slots, greedy, Qwen3 with `/no_think`) |
| `bench1.sh`, `speed8b.sh` | llama-bench wrapper for `../matmul-opt/paired.py` (paired-CI gate); the Qwen3-8B speed set |
| `pause.sh stop\|cont` | SIGSTOP / SIGCONT the measurement queue's workers for quiet speed windows |
| `sizes.py` | file + sidecar size, bits per weight over the linear layers |
| `np_gsm.py`, `show_stats.py` | debugging helpers (concurrent server requests; calibration statistics dump) |

Calibration text: the first 400 KB of WikiText-2 *train* (32 x 512 tokens for the statistics, 64 x 512 for
llama-imatrix). Evaluation: WikiText-2 *test*, 64 x 512 tokens (1.7B) or 32 x 512 (3B, 8B).

## Results (`results/`)

- `ppl-logs/`: every llama-perplexity run (`ppl-TAG-NAME.log`; `-extra-` = ablations / references); `ppl-*.md` tables.
- `harness/`: calibration statistics (`calib-*.json`, with per-layer residual / linear-input outlier statistics),
  the activation-numerics sweep (`kl-1.7b.json`, `stats-1.7b.json`), the GGUF-variant harness check (`w4-1.7b.json`),
  KV analysis (`kv-*.json`).
- `speed/paired.jsonl`: paired-CI runs (label, medians, cost ratio, 95% CI, gate, raw pairs).
- `gsm8k/`: per-model GSM8K results (`*.gsm8k.json`: predictions, gold answers, replies) and `gsm8k.tsv`.
- `sizes-*.txt`: sizes and bits per weight.
