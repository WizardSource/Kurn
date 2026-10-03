#!/usr/bin/env bash
# Repeat-run determinism of the AMX paths (kurn's Q8_0 AMX prefill kernel, and ggml's own AMX
# buffer for comparison), 8 threads pinned 1:1 to vCPUs. Run under benchlock.sh: on a VM that
# does not preserve AMX tile state, a second AMX user on the same vCPU corrupts results.
#   benchlock.sh tools/amx_check.sh [LLAMA_DIR] [REPS]
set -uo pipefail
L=${1:-$HOME/src/llama-kurn}
REPS=${2:-20}
HERE=$(cd "$(dirname "$0")/.." && pwd)
T=$(mktemp -d)
gcc -O2 -I"$L/ggml/include" "$HERE/test_kurn_buft.c" -L"$L/build/bin" -lggml -lggml-base -lggml-cpu -lm \
    -Wl,-rpath,"$L/build/bin" -o "$T/t" || exit 1
fail=0
for shape in "2048 2048 64" "2048 6144 128" "6144 2048 512" "4096 4096 37"; do
  set -- $shape
  VERBOSE=1 "$T/t" determinism q8_0 "$1" "$2" "$3" 8 "$REPS" | sed 's/^/kurn-amx /'
  [ "${PIPESTATUS[0]}" = 0 ] || fail=$((fail + 1))
  # ggml's AMX buffer: only the run-to-run count matters (its column 0 is not batch invariant)
  VERBOSE=1 TEST_BUFT=AMX "$T/t" determinism q8_0 "$1" "$2" "$3" 8 "$REPS" | sed 's/^/ggml-amx /'
done
rm -rf "$T"
exit $fail
