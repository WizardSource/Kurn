#!/usr/bin/env bash
# End-to-end decode tok/s, J/token proxy and ubatch-1 perplexity per model, in the three
# modes of benchmarks/v0.2/run_e2e.sh (ggml default = AMX/CPU_REPACK buffers, --repack 0,
# --repack 0 + GGML_KURN=1), with a llama.cpp built from llama/ggml-kurn-fourbit.patch.
# One benchlock session per model (each < 10 min).
#   e2e.sh OUT.csv MODEL...
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
BIN=${LLAMA_FOURBIT_BIN:-$HOME/src/llama-fourbit/build/bin}
OUT=$1; shift
for m in "$@"; do
    "$HERE/../benchlock.sh" bash "$HERE/../run_e2e.sh" "$BIN" "$OUT" "$m"
done
