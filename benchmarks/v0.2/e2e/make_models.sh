#!/usr/bin/env bash
# Quantize Qwen3-1.7B BF16 into the formats the e2e harness covers that have no ready-made GGUF.
# Real low-bit models (Bonsai Q1_0, BitNet TQ2_0, Ternary-Bonsai Q2_0) are downloaded separately.
#   make_models.sh [QUANTIZE_BIN] [MODELS_DIR]
set -euo pipefail
Q=${1:-$HOME/src/llama.cpp/build/bin/llama-quantize}
D=${2:-$HOME/models}
SRC=$D/Qwen3-1.7B-BF16.gguf
for t in Q4_0 IQ4_NL; do
  out=$D/Qwen3-1.7B-$t.gguf
  [ -f "$out" ] || nice -n 19 "$Q" "$SRC" "$out" "$t" 8
done
