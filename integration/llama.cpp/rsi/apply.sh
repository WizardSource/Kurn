#!/usr/bin/env bash
# kurn RSI build for llama-server (tested on 4ebdf2c, on top of ../apply.sh, ../spec-width/apply.sh and ../k4c/apply.sh).
#   rsi/apply.sh LLAMA_CPP_DIR
# - copies kurn-replay.h into tools/server/ and writes tools/server/kurn-build-info.h (sha256 of the kurn
#   integration sources, kurn version) for the run manifest
# - applies llama-rsi.patch:
#   * common/speculative.*: per-sequence arm hooks (allow_impl / on_draft) in the drafting chain, the implementation
#     that produced a draft, and lazy draft-model sync (draft-simple catches up from the prompt only when it drafts)
#   * common/sampling.*: position-keyed sampling (`kurn_rng_position`): each token's uniform from (seed, position)
#   * llama-server: retrieval drafts (n-gram types) and the draft model as two arms of the cost-aware selector
#     (--spec-width TABLE with an n-gram --spec-type), correct probabilities for accepted speculative tokens,
#     and `kurn_replay` (run manifest + probability hash chain) per request
# Needs spec-width/apply.sh's kurn-spec-width.h (with kurn::arm_selector). Idempotent.
# Rebuild with: cmake --build build -j --target llama-server llama-speculative-simple
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
INTEG=$(cd "$HERE/.." && pwd)
L=$(cd "$1" && pwd)
if ! grep -q "arm_selector" "$L/tools/server/kurn-spec-width.h" 2>/dev/null; then
  echo "run spec-width/apply.sh first (tools/server/kurn-spec-width.h without kurn::arm_selector)" >&2
  exit 1
fi
cp "$HERE/kurn-replay.h" "$L/tools/server/"
SHA=$(cd "$INTEG" && find . -type f \( -name '*.patch' -o -name '*.h' -o -name '*.hpp' -o -name '*.c' -o -name '*.cpp' -o -name '*.py' -o -name '*.sh' \) \
      -not -path './*/__pycache__/*' | LC_ALL=C sort | xargs sha256sum | sha256sum | cut -d' ' -f1)
VER=$(grep -m1 -E '^version *=' "$INTEG/../../pyproject.toml" 2>/dev/null | sed -E 's/.*"(.*)".*/\1/' || true)
cat > "$L/tools/server/kurn-build-info.h" <<H
// written by kurn integration/llama.cpp/rsi/apply.sh
#define KURN_INTEGRATION_SHA256 "$SHA"
#define KURN_VERSION "${VER:-unknown}"
H
if grep -q "kurn_arm_mode" "$L/tools/server/server-context.cpp"; then
  echo "llama-rsi.patch already applied"
else
  git -C "$L" apply "$HERE/llama-rsi.patch"
fi
echo "kurn RSI build applied to $L; rebuild with: cmake --build build -j --target llama-server llama-speculative-simple"
