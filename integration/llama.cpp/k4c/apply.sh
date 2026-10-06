#!/usr/bin/env bash
# Add kurn's k4c key format to a llama.cpp checkout as KV cache type `k4c` (tested on 4ebdf2c, after ../apply.sh).
#   k4c/apply.sh LLAMA_CPP_DIR
# - copies ggml-k4c.c (GGML_TYPE_K4C: per-channel 4-bit keys in 32-cell groups) into ggml/src/
# - applies llama-k4c.patch: the type and its codec API in ggml, SET_ROWS into K4C caches and supports_op in
#   ggml-cpu (K4C keys are read only by kurn's FLASH_ATTN_EXT), the KV cache (no Hadamard rotation, no K-shift,
#   session state as f16 rows re-quantized on load), the context checks and -ctk k4c / k4c_q4 / k4c_q8
# Use: -ctk k4c -ctv q4_0 (kurn's k4c_q4) or -ctv q8_0 (k4c_q8); -ctk k4c_q4 / k4c_q8 set both.
# Idempotent. Rebuild with: cmake --build build -j
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
L=$(cd "$1" && pwd)
if [ ! -f "$L/ggml/src/ggml-cpu/kurn/kurn-attn.h" ]; then
  echo "run integration/llama.cpp/apply.sh first (k4c keys are read by kurn's attention)" >&2
  exit 1
fi
cp "$HERE/ggml-k4c.c" "$L/ggml/src/"
if grep -q "GGML_TYPE_K4C" "$L/ggml/include/ggml.h"; then
  echo "llama-k4c.patch already applied"
else
  git -C "$L" apply "$HERE/llama-k4c.patch"
fi
echo "k4c KV cache type applied to $L; rebuild with: cmake --build build -j"
