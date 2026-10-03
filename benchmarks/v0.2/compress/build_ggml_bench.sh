#!/usr/bin/env bash
# Build the ggml-linked harness (contrib/ggml-harness/bench_ggml.c) with extra ggml weight types
# for the compress baselines (Q2_K, Q3_K, Q5_K, Q6_K, IQ2_XXS, IQ2_XS, IQ3_XXS, IQ4_XS): `--impl ggml
# --kernel <name>` then times ggml's own vec_dot (the --repack 0 path), and `ggml-graph-cpu` /
# `ggml-graph-repack` a full MUL_MAT graph. IQ types need an importance matrix to quantize; an
# all-ones one is passed (the data are synthetic anyway). The cold working set is one quantized
# matrix replicated (quantizing 1.2 GB of floats to IQ types takes minutes per run).
#   build_ggml_bench.sh [OUT]   (default /tmp/kurn-cache-compress/bench_ggml_types)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../../.." && pwd)
L=${LLAMA:-$HOME/src/llama.cpp}
OUT=${1:-/tmp/kurn-cache-compress/bench_ggml_types}
mkdir -p "$(dirname "$OUT")"
SRC=$(mktemp --suffix=.c)
sed -e 's|    {"q10gemv", "q1_0_gemv", "kq10_gemv", GGML_TYPE_Q1_0},|&\n    {"q2kgemv", "q2_K_gemv", "kq2k_gemv", GGML_TYPE_Q2_K}, {"q3kgemv", "q3_K_gemv", "kq3k_gemv", GGML_TYPE_Q3_K},\n    {"q5kgemv", "q5_K_gemv", "kq5k_gemv", GGML_TYPE_Q5_K}, {"q6kgemv", "q6_K_gemv", "kq6k_gemv", GGML_TYPE_Q6_K},\n    {"iq2xxsgemv", "iq2_xxs_gemv", "kiq2xxs_gemv", GGML_TYPE_IQ2_XXS}, {"iq2xsgemv", "iq2_xs_gemv", "kiq2xs_gemv", GGML_TYPE_IQ2_XS},\n    {"iq3xxsgemv", "iq3_xxs_gemv", "kiq3xxs_gemv", GGML_TYPE_IQ3_XXS}, {"iq4xsgemv", "iq4_xs_gemv", "kiq4xs_gemv", GGML_TYPE_IQ4_XS},|' \
    -e 's|ggml_quantize_chunk(wt, tmp, G.W + m \* G.mat_bytes + r0 \* G.row_bytes, 0, nr, G.K, NULL);|{ static float *ones = NULL; if (!ones) { ones = malloc(sizeof(float) * G.K); for (int64_t j = 0; j < G.K; j++) ones[j] = 1.0f; ggml_quantize_init(wt); }\n            ggml_quantize_chunk(wt, tmp, G.W + m * G.mat_bytes + r0 * G.row_bytes, 0, nr, G.K, ggml_quantize_requires_imatrix(wt) ? ones : NULL); }|' \
    -e '/Weights: roughly Gaussian/,/free(tmp);/ { s/for (int64_t m = 0; m < G.count; m++) {/for (int64_t m = 0; m < 1; m++) {/; s/^    free(tmp);/    free(tmp);\n    for (int64_t m = 1; m < G.count; m++) memcpy(G.W + m * G.mat_bytes, G.W, G.mat_bytes);/ }' \
    "$ROOT/contrib/ggml-harness/bench_ggml.c" > "$SRC"
grep -q iq2xxsgemv "$SRC" && grep -q ggml_quantize_requires_imatrix "$SRC" && grep -q "memcpy(G.W + m" "$SRC" || { echo "patch did not apply" >&2; exit 1; }
gcc -O3 -march=native -I "$ROOT/src/kurn/data" -I "$L/ggml/include" "$SRC" -o "$OUT" \
    -L"$L/build/bin" -lggml -lggml-base -lggml-cpu -Wl,-rpath,"$L/build/bin" -lpthread -ldl -lm
rm -f "$SRC"
echo "$OUT"
