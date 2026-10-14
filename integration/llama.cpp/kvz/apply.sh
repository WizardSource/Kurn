#!/usr/bin/env bash
# Add kvz, lossless entropy coding of the KV cache in session state, to a llama.cpp checkout (tested on 4ebdf2c, after
# ../apply.sh, ../spec-width/apply.sh and ../k4c/apply.sh).
#   kvz/apply.sh LLAMA_CPP_DIR
# - copies llama-kvz.{h,cpp} (the codec: per-channel static rANS over f16 byte planes, Q8_0 / Q4_0 scales and codes)
#   into src/
# - applies llama-kvz.patch:
#   - llama.h: LLAMA_STATE_SEQ_FLAGS_KVZ and llama_state_seq_save_file_ext;
#   - the KV cache's session state codes each layer's K and V rows as one kvz block under that flag (readers detect
#     it, so loading needs no flag; the bytes restored are exactly the plain state's);
#   - llama-server --cache-kvz: the RAM prompt cache (--cache-ram, idle slots) and --slot-save-path files.
# Idempotent. Rebuild with: cmake --build build -j
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
L=$(cd "$1" && pwd)
if ! grep -q "GGML_TYPE_K4C" "$L/ggml/include/ggml.h"; then
  echo "run integration/llama.cpp/k4c/apply.sh first (kvz codes k4c session state too)" >&2
  exit 1
fi
cp "$HERE/llama-kvz.h" "$HERE/llama-kvz.cpp" "$L/src/"
if grep -q "LLAMA_STATE_SEQ_FLAGS_KVZ" "$L/include/llama.h"; then
  echo "llama-kvz.patch already applied"
else
  git -C "$L" apply "$HERE/llama-kvz.patch"
fi
echo "kvz session state applied to $L; rebuild with: cmake --build build -j"
