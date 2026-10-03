#!/usr/bin/env bash
# Final serial pass, stage 1: whole-step engine vs llama.cpp on a quiet machine (coordinator).
# Runs WS-E's measure.sh batches from this checkout; results land in benchmarks/v0.2/engine/results/.
set -u
cd "$(dirname "$0")/../../.."
export PYTHONPATH=$PWD/src KURN_CACHE_DIR=/tmp/kurn-cache-final DONE_DIR=/tmp/kurn-final-done
L=benchmarks/v0.2/benchlock.sh
for b in gen-qwen3 gen-olmoe gen-qwen3q4 gen-olmoeq4 wait-olmoe wait-qwen3 perf-olmoe perf-qwen3 greedy-qwen3; do
  echo "=== $(date -u +%T) $b"
  $L benchmarks/v0.2/engine/measure.sh "$b"
done
echo "=== $(date -u +%T) STAGE1_DONE"
