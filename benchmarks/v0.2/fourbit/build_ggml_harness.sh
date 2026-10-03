#!/usr/bin/env bash
# Build the ggml-linked harness (contrib/ggml-harness) with two extra decode GEMV rows,
# mxfp4gemv / nvfp4gemv, without editing the shared source: a patched copy goes to $OUT.
# Also keys the BENCH_CACHE file on the GEMV table row: upstream keys it on the kernel kind
# and block count only, so q4_0 / iq4_nl / nvfp4 cold data (128 blocks each) collide.
#   build_ggml_harness.sh [OUT=/tmp/fb_ggml/bench_ggml]
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
KURN=$(cd "$HERE/../../.." && pwd)
L=${LLAMA:-$HOME/src/llama.cpp}
OUT=${1:-/tmp/fb_ggml/bench_ggml}
mkdir -p "$(dirname "$OUT")"
SRC="$(dirname "$OUT")/bench_ggml_fourbit.c"
sed -e 's|    {"q10gemv", "q1_0_gemv", "kq10_gemv", GGML_TYPE_Q1_0},|&\n    {"mxfp4gemv", "mxfp4_gemv", "kmxfp4_gemv", GGML_TYPE_MXFP4}, {"nvfp4gemv", "nvfp4_gemv", "knvfp4_gemv", GGML_TYPE_NVFP4},|' \
    -e 's|"%s/k%d_K%ld_N%ld_M%d_c%ld.bin", dir, G.kernel,|"%s/k%d_g%d_K%ld_N%ld_M%d_c%ld.bin", dir, G.kernel, gemv_idx,|' \
    "$KURN/contrib/ggml-harness/bench_ggml.c" > "$SRC"
grep -q mxfp4gemv "$SRC" && grep -q 'k%d_g%d_K' "$SRC" || { echo "patch did not apply" >&2; exit 1; }
gcc -O3 -march=native -I "$KURN/src/kurn/data" -I "$L/ggml/include" "$SRC" -o "$OUT.tmp" \
    -L"$L/build/bin" -lggml -lggml-base -lggml-cpu -Wl,-rpath,"$L/build/bin" -lpthread -ldl -lm
mv "$OUT.tmp" "$OUT"
echo "$OUT"
