#!/bin/bash
cd "$(dirname "$0")"
for round in 3 4; do
  for wl in refine think; do
    for cfg in none draft_selector ngram ngram_selector union_naive arms arms_eager; do
      python3 bench.py $cfg $wl 1 || echo "FAIL $cfg $wl"
    done
  done
done
for cfg in none draft_selector arms; do
  python3 bench.py $cfg eight 1 || echo "FAIL $cfg eight"
done
echo MATRIX_DONE
