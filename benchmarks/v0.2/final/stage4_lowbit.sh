#!/usr/bin/env bash
# Final serial pass, stage 4: WS-C low-bit finals on the integrated branch. The screen is re-run first,
# because config keys gained WS-A's layout keys at integration and the branch's screen files no longer match.
set -u
K=$(cd "$(dirname "$0")/../../.." && pwd)
export PYTHONPATH=$K/src KURN_CACHE_DIR=/tmp/kurn-cache-lowbit-final
L=$K/benchmarks/v0.2/benchlock.sh
PY=/opt/kenv/bin/python
OUT=$K/benchmarks/v0.2/final/results
cd "$K/benchmarks/v0.2/lowbit"
echo "=== $(date -u +%T) screen"
$L $PY run_lowbit.py screen q1_0 q2_0 tq2_0 tq1_0 q2_K > "$OUT/lowbit_screen.txt" 2>&1
echo "=== $(date -u +%T) finals"
{ $L env TOPN=1 TRIM=1 $PY run_lowbit.py final hot 1 q1_0 q2_0 tq2_0 tq1_0 q2_K
  $L env TOPN=1 TRIM=1 $PY run_lowbit.py final cold 8 q1_0 q2_0 tq2_0
  $L env TOPN=1 TRIM=1 $PY run_lowbit.py final cold 8 tq1_0 q2_K; } > "$OUT/lowbit_finals.txt" 2>&1
echo "=== $(date -u +%T) STAGE4_DONE"
