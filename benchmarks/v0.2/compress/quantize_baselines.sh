#!/usr/bin/env bash
# Uniform / llama.cpp-mix baselines for the compress workstream, all from the BF16 model with
# the same importance matrix (wiki train, 128 x 512 tokens):
#   quantize_baselines.sh [TYPE...]
# Writes ~/models/compress/qwen3-1.7b-<type>.gguf (llama.cpp's own per-tensor mixes) and
# qwen3-1.7b-pure-<type>.gguf (--pure: every tensor the same type; the per-tensor error tables
# of `kurn mix` read these). Missing outputs are made in batches of $BATCH per lock acquisition
# (one quantize is 10-50 s; the shared lock is contended).
set -euo pipefail
BIN=${LLAMA_BIN:-$HOME/src/llama.cpp/build/bin}
SRC=${SRC:-$HOME/models/Qwen3-1.7B-BF16.gguf}
OUT=${OUT:-$HOME/models/compress}
IMX=${IMX:-$OUT/qwen3-1.7b.imatrix.gguf}
LOCK=$(dirname "$0")/../benchlock.sh
BATCH=${BATCH:-8}
MIXES=${MIXES:-"Q8_0 Q6_K Q5_K_M Q4_K_M Q4_K_S IQ4_NL IQ4_XS Q3_K_L Q3_K_M Q3_K_S IQ3_XXS Q2_K IQ2_M IQ2_XS IQ2_XXS"}
PURE=${PURE:-"Q8_0 Q6_K Q5_K Q4_K Q3_K Q2_K IQ4_NL IQ4_XS IQ3_XXS IQ2_XS"}
mkdir -p "$OUT"
jobs=()
for t in ${@:-$MIXES}; do
  f=$OUT/qwen3-1.7b-$(echo "$t" | tr A-Z a-z).gguf
  [ -s "$f" ] || jobs+=("$BIN/llama-quantize --imatrix $IMX $SRC $f.part $t 8 >/dev/null 2>&1 && mv $f.part $f")
done
if [ $# -eq 0 ]; then
  for t in $PURE; do
    f=$OUT/qwen3-1.7b-pure-$(echo "$t" | tr A-Z a-z).gguf
    [ -s "$f" ] || jobs+=("$BIN/llama-quantize --pure --imatrix $IMX $SRC $f.part $t 8 >/dev/null 2>&1 && mv $f.part $f")
  done
fi
for ((i = 0; i < ${#jobs[@]}; i += BATCH)); do
  script=$(printf '%s\n' "${jobs[@]:i:BATCH}")
  "$LOCK" bash -ec "$script"
done
ls -l "$OUT"/qwen3-1.7b-*.gguf | awk '{print $NF, $5}'
