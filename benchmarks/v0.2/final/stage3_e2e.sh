#!/usr/bin/env bash
# Final serial pass, stage 3: end to end in llama.cpp on a quiet machine (coordinator).
#   1. rebuild ~/src/llama-kurn from this checkout's integration/llama.cpp (KURN buffer type, AMX opt-in)
#   2. format matrix: ggml default (AMX/CPU_REPACK) vs --repack 0 vs kurn; decode/prefill tok/s, J/token,
#      ub-1 and batched perplexity, RSS (WS-H harness)
#   3. draft-model speculative decoding, Qwen3-8B Q8_0 target + Qwen3-0.6B Q8_0 draft (WS-H harness)
#   4. kurn attention hook at 4K / 16K depth (WS-G harness)
set -u
K=$(cd "$(dirname "$0")/../../.." && pwd)
cd "$K"
export PYTHONPATH=$K/src KURN_CACHE_DIR=/tmp/kurn-cache-final
L=$K/benchmarks/v0.2/benchlock.sh
PY=/opt/kenv/bin/python
OUT=$K/benchmarks/v0.2/final/results
mkdir -p "$OUT"
say() { echo "=== $(date -u +%T) load=$(cut -d' ' -f1 /proc/loadavg) $*"; }
M=$HOME/models

say "rebuild llama-kurn"
LK=$HOME/src/llama-kurn
git -C "$LK" log --oneline -1
integration/llama.cpp/apply.sh "$LK" > "$OUT/apply.txt" 2>&1 || { echo "apply.sh failed"; tail "$OUT/apply.txt"; }
( cd "$LK" && cmake --build build -j 8 --target llama-bench llama-perplexity llama-cli llama-server llama-speculative-simple llama-completion ) \
  > "$OUT/llama_kurn_build.txt" 2>&1 && echo built || { echo BUILD_FAILED; tail -20 "$OUT/llama_kurn_build.txt"; exit 1; }
gcc -O2 -I$LK/ggml/include integration/llama.cpp/test_kurn_buft.c -L$LK/build/bin -lggml -lggml-base -lggml-cpu -lm \
  -Wl,-rpath,$LK/build/bin -o /tmp/final_buft && /tmp/final_buft quick > "$OUT/buft_check.txt" 2>&1; echo "buft checker exit=$?"; tail -2 "$OUT/buft_check.txt"

cd benchmarks/v0.2
say "format matrix"
$PY e2e/run_e2e.py --models $M/Qwen3-1.7B-Q8_0.gguf $M/Qwen3-1.7B-Q4_0.gguf $M/Qwen3-1.7B-Q4_K_M.gguf $M/Qwen3-1.7B-IQ4_NL.gguf \
    $M/OLMoE-1B-7B-0125-Instruct-Q8_0.gguf $M/Bonsai-1.7B-Q1_0.gguf $M/bitnet-2b4t-tq2_0.gguf $M/Ternary-Bonsai-8B-Q2_0_g64.gguf \
    --modes default,plain,kurn --reps 3 --out "$OUT/e2e.csv" > "$OUT/e2e.log" 2>&1
$PY e2e/run_e2e.py --summary "$OUT/e2e.csv" > "$OUT/e2e_summary.md" 2>&1; cat "$OUT/e2e_summary.md"

say "speculative 8B"
$PY e2e/run_spec.py --verify-m --target $M/Qwen3-8B-Q8_0.gguf --out "$OUT/verify_m_8b.csv" > "$OUT/spec.log" 2>&1
$PY e2e/run_spec.py --target $M/Qwen3-8B-Q8_0.gguf --draft $M/Qwen3-0.6B-Q8_0.gguf --out "$OUT/spec_8b.csv" >> "$OUT/spec.log" 2>&1
cat "$OUT/spec_8b.csv"

say "attention e2e"
bash attn/llama/build.sh > "$OUT/attn_build.txt" 2>&1 || { echo ATTN_BUILD_FAILED; tail "$OUT/attn_build.txt"; }
VARS=kurn-f16,kurn-q8_0,ggml-fa-f16,ggml-nofa-f16 bash attn/llama/run_e2e.sh bench 16384 0 > "$OUT/attn_e2e_16k.txt" 2>&1
bash attn/llama/run_e2e.sh bench 4096 0,1,2 > "$OUT/attn_e2e_4k.txt" 2>&1
tail -12 "$OUT/attn_e2e_16k.txt" "$OUT/attn_e2e_4k.txt"
say STAGE3_DONE
