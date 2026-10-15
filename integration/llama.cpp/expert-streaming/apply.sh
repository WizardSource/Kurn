#!/usr/bin/env bash
# Add kurn expert streaming (KXS) to a llama.cpp checkout already patched by ../apply.sh (which copies
# ggml-kurn/kurn-xs.{cpp,h} and the hooked kurn-buft.cpp), ../spec-width/apply.sh and ../k4c/apply.sh.
#   expert-streaming/apply.sh LLAMA_CPP_DIR
# Applies llama-expert-streaming.patch: ggml-cpu.cpp entry points, the loader / graph hooks in src/,
# --expert-stream / --expert-cache-mb, and tools/kurn-xs (llama-kurn-xs-convert, llama-kurn-xs-bench).
# Idempotent: a checkout that already has the patch is left as is.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
L=$(cd "$1" && pwd)
if [ ! -f "$L/ggml/src/ggml-cpu/kurn/kurn-xs.cpp" ]; then
  echo "run $HERE/../apply.sh $L first (it copies kurn-xs.cpp)" >&2
  exit 1
fi
if grep -q ggml_kurn_xs_open "$L/ggml/src/ggml-cpu/ggml-cpu.cpp"; then
  echo "expert streaming already applied to $L"
  exit 0
fi
git -C "$L" apply --whitespace=nowarn "$HERE/llama-expert-streaming.patch"
echo "expert streaming applied to $L; rebuild with: cmake -B build && cmake --build build -j"
