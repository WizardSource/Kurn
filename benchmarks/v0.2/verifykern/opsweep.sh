#!/usr/bin/env bash
# opsweep.sh BUFT TYPE "M list" [REPS] [LAYERS] -> one line per M (opbench, Qwen3-8B shapes, 8 threads)
# env: OPBENCH (binary), extra env is passed through (GGML_KURN_*)
B=${OPBENCH:-$HOME/work/bin/opbench}
SH="4096:4096,4096:1024,4096:1024,4096:4096,4096:12288,4096:12288,12288:4096"
for M in $3; do
  "$B" "$1" "$2" "$M" 8 "${4:-15}" "${5:-8}" "$SH"
done
