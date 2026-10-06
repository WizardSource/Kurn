#!/usr/bin/env bash
# Final serial pass, stage 5: J/token and RSS for the buffer-type matrix (speed runs only, 1 rep; the stage-3 harness
# was stopped during the hour-long stock batched perplexity of Ternary-Bonsai-8B, so its CSV was never written).
set -u
K=$(cd "$(dirname "$0")/../../.." && pwd)
export PYTHONPATH=$K/src KURN_CACHE_DIR=/tmp/kurn-cache-final
cd "$K/benchmarks/v0.2"
M=$HOME/models
OUT=final/results
echo "=== $(date -u +%T) energy matrix"
/opt/kenv/bin/python e2e/run_e2e.py --models $M/Qwen3-1.7B-Q8_0.gguf $M/Qwen3-1.7B-Q4_0.gguf $M/Qwen3-1.7B-Q4_K_M.gguf \
  $M/Qwen3-1.7B-IQ4_NL.gguf $M/OLMoE-1B-7B-0125-Instruct-Q8_0.gguf $M/Bonsai-1.7B-Q1_0.gguf $M/bitnet-2b4t-tq2_0.gguf \
  --modes default,plain,kurn --reps 1 --skip ppl_ub1,ppl_batched --out "$OUT/e2e_energy.csv" > "$OUT/e2e_energy.log" 2>&1
/opt/kenv/bin/python e2e/run_e2e.py --summary "$OUT/e2e_energy.csv" > "$OUT/e2e_energy_summary.md" 2>&1
cat "$OUT/e2e_energy_summary.md"
echo "=== $(date -u +%T) STAGE5_DONE"
