#!/usr/bin/env bash
# End-to-end per format in llama.cpp (decode / prefill tok/s, J/token proxy, ubatch-1 and batched
# perplexity, peak RSS) for ggml's default path, --repack 0 and the kurn buffer type.
#   run_e2e.sh OUT.csv MODEL.gguf...            (extra options: E2E_ARGS="--reps 5 --skip ppl_ub1")
# STOCK_BIN / KURN_BIN select the llama.cpp builds (default ~/src/llama.cpp/build/bin and
# ~/src/llama-kurn/build/bin). See e2e/run_e2e.py --help and e2e/README.md.
set -euo pipefail
OUT=$1; shift
PY=${KURN_PYTHON:-$( [ -x /opt/kenv/bin/python ] && echo /opt/kenv/bin/python || echo python3 )}
exec "$PY" "$(dirname "$0")/e2e/run_e2e.py" --out "$OUT" \
  ${STOCK_BIN:+--stock-bin "$STOCK_BIN"} ${KURN_BIN:+--kurn-bin "$KURN_BIN"} ${E2E_ARGS:-} --models "$@"
