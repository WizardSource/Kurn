#!/usr/bin/env bash
# Add the cost-aware verify width to a llama.cpp checkout (tested on 4ebdf2c, on top of ../apply.sh).
#   spec-width/apply.sh LLAMA_CPP_DIR
# - copies kurn-spec-width.h (the policy) and kurn-spec-calib.cpp (cost table + trace tool) into
#   examples/speculative-simple/
# - applies llama-spec-width.patch: draft-confidence output and a keep-drafting callback in draft-simple
#   (common/speculative.*), the policy in llama-speculative-simple (KURN_SPEC_WIDTH=table), the
#   kurn-spec-calib target
# - copies kurn-spec-width.h into tools/server/ and applies llama-server-spec-width.patch: the policy per
#   llama-server slot (--spec-width TABLE [--spec-width-mode cap], or KURN_SPEC_WIDTH=TABLE)
# Idempotent. Rebuild with: cmake --build build -j --target llama-server llama-speculative-simple kurn-spec-calib
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
L=$(cd "$1" && pwd)
cp "$HERE/kurn-spec-width.h" "$HERE/kurn-spec-calib.cpp" "$L/examples/speculative-simple/"
cp "$HERE/kurn-spec-width.h" "$L/tools/server/"
if grep -q "keep_drafting" "$L/common/speculative.h"; then
  echo "llama-spec-width.patch already applied"
else
  git -C "$L" apply "$HERE/llama-spec-width.patch"
fi
if grep -q "kurn_width" "$L/tools/server/server-context.cpp"; then
  echo "llama-server-spec-width.patch already applied"
else
  git -C "$L" apply "$HERE/llama-server-spec-width.patch"
fi
echo "kurn spec width applied to $L; rebuild with: cmake --build build -j --target llama-server llama-speculative-simple kurn-spec-calib"
