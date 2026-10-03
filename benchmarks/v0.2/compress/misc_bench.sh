#!/usr/bin/env bash
# Locked microbenchmarks (~6 min): fused Huffman-decode Q8_0 GEMV (cold 1/2/4/8T on the model's
# Q8_0 matrices, hot 1T), vq2x8 decode vs LUT GEMV, fused low-rank GEMV.
#   benchlock.sh paused.sh misc_bench.sh OUTDIR
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${1:-$HERE/results}
C=${KURN_CACHE_DIR:-/tmp/kurn-cache-compress}
PY=/opt/kenv/bin/python
gcc -O3 -march=native -pthread -o "$C/lut_bench" "$HERE/lut_bench.c" -lm
gcc -O3 -march=native -pthread -o "$C/lowrank_bench" "$HERE/lowrank_bench.c" -lm
$PY "$HERE/entropy_bench.py" run "$C/ent_full.bin" --threads 1,2,4,8 --secs 1 > "$OUT/entropy_cold.log" 2>&1
$PY "$HERE/entropy_bench.py" run "$C/ent_small.bin" --threads 1 --hot --secs 0.5 > "$OUT/entropy_hot.log" 2>&1
{ "$C/lut_bench" 2048 2048 1 0.5 hot; "$C/lut_bench" 256 4096 1 0.5 hot; "$C/lut_bench" 4096 4096 1 0.5 cold; "$C/lut_bench" 4096 4096 8 0.5 cold; } > "$OUT/lut.log" 2>&1
{ "$C/lowrank_bench" 2048 6144 0,8,32,128 1 0.5 hot; "$C/lowrank_bench" 2048 6144 0,8,32,128 1 0.5 cold; "$C/lowrank_bench" 2048 6144 0,8,32,128 8 0.5 cold; } > "$OUT/lowrank.log" 2>&1
echo "load at end: $(cut -d' ' -f1-3 /proc/loadavg)"
