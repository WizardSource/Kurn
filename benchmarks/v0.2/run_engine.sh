#!/usr/bin/env bash
# Whole-decode-step engine (kurn model) vs llama.cpp: decode tok/s, J/token proxy, perplexity on
# identical tokens, greedy identity and the perf CPU-time shares (spin / matmul / dispatch).
# Runs benchmarks/v0.2/engine/measure.sh batches, each under the bench lock; results go to
# benchmarks/v0.2/engine/results/ (summarize.py / greedy_pairs.py print the tables).
#   run_engine.sh [MODEL...]     MODEL = qwen3 | olmoe | qwen3q4 | olmoeq4 (default: qwen3 olmoe)
#   BATCHES="gen ppl" run_engine.sh qwen3
set -u
H=$(cd "$(dirname "$0")" && pwd)
export PYTHONPATH=${PYTHONPATH:-$H/../../src} KURN_CACHE_DIR=${KURN_CACHE_DIR:-/tmp/kurn-cache-engine}
export DONE_DIR=$(mktemp -d)
for m in ${@:-qwen3 olmoe}; do
  for b in ${BATCHES:-gen ppl greedy perf}; do
    "$H/benchlock.sh" "$H/engine/measure.sh" "$b-$m"
  done
done
rm -rf "$DONE_DIR"
