#!/usr/bin/env bash
# Build contrib/ggml-harness/bench_ggml.c with two extra GEMV rows (tq10gemv = TQ1_0,
# q2kgemv = Q2_K) against the mainline llama.cpp shared libraries. Output: $OUT (default
# /tmp/kurn-lowbit/bench_ggml). The contrib source is not modified.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
KURN=$(cd "$HERE/../../.." && pwd)
L=${LLAMA:-$HOME/src/llama.cpp}
OUT=${OUT:-/tmp/kurn-lowbit/bench_ggml}
mkdir -p "$(dirname "$OUT")"
SRC="$(dirname "$OUT")/bench_ggml_lowbit.c"
sed 's|    {"q10gemv", "q1_0_gemv", "kq10_gemv", GGML_TYPE_Q1_0},|&\n    {"tq10gemv", "tq1_0_gemv", "ktq10_gemv", GGML_TYPE_TQ1_0}, {"q2kgemv", "q2_K_gemv", "kq2k_gemv", GGML_TYPE_Q2_K},|' \
    "$KURN/contrib/ggml-harness/bench_ggml.c" > "$SRC"
grep -q tq10gemv "$SRC"
gcc -O3 -march=native -I "$KURN/src/kurn/data" -I "$L/ggml/include" "$SRC" -o "$OUT" \
    -L"$L/build/bin" -lggml -lggml-base -lggml-cpu -Wl,-rpath,"$L/build/bin" -lpthread -ldl -lm
echo "$OUT"
