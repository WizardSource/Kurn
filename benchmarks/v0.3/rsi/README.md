# RSI build measurements (llama-server, 8-vCPU Emerald Rapids VM, exact mode)

- `bench.py CONFIG WORKLOAD` starts llama-server per run (`LLAMA_SERVER`, `MODELS`) and appends to `results.jsonl`:
  - configs: `none`, `draft_selector`, `ngram`, `ngram_selector`, `union_naive`, `arms`, `arms_eager`;
  - workloads: `refine` (re-emit a corrected 70-line file, `buggy.py`), `think` (a counting problem with thinking),
    `eight` (Qwen3-8B, 8 prompts).
- `run_matrix.sh`, `run_matrix2.sh` and `run_v2.sh` are the rounds that were run, in that order.
  - `results/results_r1r2.jsonl`: rounds 1-2. Round 1 of `refine`/`think` overlapped unrelated test runs and is dropped.
  - `results/results_r3r4.jsonl`: rounds 3-4. `arms` there is the first selector, without the catch-up charge
    (`arms_v1` in the summary).
  - `results/results_v2.jsonl`: the final selector.
  - `results/combined.jsonl` and `results/summary.txt` (`summarize.py results/combined.jsonl`) merge the quiet rounds.
- `aa_tune.py N` is the A/A test of the tuner's confirmation on the real harness (`results/aa_tune.*`).
- `results/rsi_check.json` is `integration/llama.cpp/rsi/rsi_check.py` on Qwen3-1.7B + 0.6B.
- `results/q17.cost` and `results/q8b.cost` are the `kurn-spec-calib` tables, measured in exact mode (`GGML_KURN_FA_MODE=exact`, the mode of these runs). They were recalibrated after kv-backlog made exact-mode decode faster: median of 3 interleaved runs, raw runs and the pre-kv-backlog comparison in `results/exact-recal/`. At this context (76 tokens) the change is small: Qwen3-1.7B verify 6-12% cheaper at M <= 8 (M=1 21.1 -> 19.8 ms, M=8 29.1 -> 27.0), Qwen3-8B within noise (M=8 109.6 vs 111.7). The measurements above were taken with the earlier tables.
- `overhead.py` measures the cost of `kurn_replay` / `kurn_rng_position` on the `think` prompt (`results/overhead.json`).
- `results/rsi_check_before.json` is the `probs` check on a llama-server built without `rsi/apply.sh`: accepted
  speculative tokens without probabilities. `results/rsi_check_fresh.json` is the full check on a fresh
  `4ebdf2c` checkout with all four apply scripts.
