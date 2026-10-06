#!/usr/bin/env bash
# Final serial pass, stage 2: kernel-level headline numbers on a quiet machine (coordinator).
# Re-runs each workstream's own comparison scripts (vs ggml and kurn v0.1) from this checkout.
set -u
K=$(cd "$(dirname "$0")/../../.." && pwd)
cd "$K"
export PYTHONPATH=$K/src
L=$K/benchmarks/v0.2/benchlock.sh
PY=/opt/kenv/bin/python
OUT=$K/benchmarks/v0.2/final/results
mkdir -p "$OUT"
say() { echo "=== $(date -u +%T) load=$(cut -d' ' -f1 /proc/loadavg) $*"; }

say roofline
for t in 1 8; do $L $PY -m kurn roofline --threads $t --streams 4; done | tee "$OUT/roofline.txt"

say "layout (WS-A): Q4_K compare hot 1T + DRAM 8T"
export KURN_CACHE_DIR=/tmp/kurn-cache-layout
[ -x /tmp/bench_ggml ] || gcc -O3 -march=native -I src/kurn/data -I ~/src/llama.cpp/ggml/include contrib/ggml-harness/bench_ggml.c \
  -o /tmp/bench_ggml -L ~/src/llama.cpp/build/bin -lggml -lggml-base -lggml-cpu -Wl,-rpath,$HOME/src/llama.cpp/build/bin -lpthread -ldl -lm
LY=benchmarks/v0.2/layout
$L $PY $LY/bench_configs.py $LY/q4k_compare_hot1.json --harness /tmp/bench_ggml --threads 1 --reps 5 --secs 0.5 > "$OUT/layout_q4k_hot1.txt" 2>&1
$L $PY $LY/bench_configs.py $LY/q4k_compare_cold8.json --harness /tmp/bench_ggml --regime cold --threads 8 --reps 3 --secs 1 > "$OUT/layout_q4k_cold8.txt" 2>&1
tail -15 "$OUT/layout_q4k_cold8.txt"

say "fourbit (WS-B): headline + compare (hot 1T, DRAM 8T, incl. ggml paths)"
export KURN_CACHE_DIR=/tmp/kurn-cache-fourbit
bash benchmarks/v0.2/fourbit/queue.sh headline compare > "$OUT/fourbit_queue.txt" 2>&1
tail -5 "$OUT/fourbit_queue.txt"

say "lowbit (WS-C): finals hot 1T and DRAM 8T vs ggml vec_dot"
export KURN_CACHE_DIR=/tmp/kurn-cache-lowbit
( cd benchmarks/v0.2/lowbit
  $L env TOPN=1 TRIM=1 $PY run_lowbit.py final hot 1 q1_0 q2_0 tq2_0 tq1_0 q2_K
  $L env TOPN=1 TRIM=1 $PY run_lowbit.py final cold 8 q1_0 q2_0 tq2_0
  $L env TOPN=1 TRIM=1 $PY run_lowbit.py final cold 8 tq1_0 q2_K ) > "$OUT/lowbit_finals.txt" 2>&1
tail -12 "$OUT/lowbit_finals.txt"

say "compress (WS-D): E8P / entropy kernels vs ggml IQ2_XXS"
export KURN_CACHE_DIR=/tmp/kurn-cache-compress
$L bash benchmarks/v0.2/compress/bench_kernels.sh "$OUT/compress" > "$OUT/compress_kernels.txt" 2>&1
tail -8 "$OUT/compress_kernels.txt"

say STAGE2_DONE
