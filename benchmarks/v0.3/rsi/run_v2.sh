#!/bin/bash
cd "$(dirname "$0")"
for round in 1 2 3; do
  for wl in refine think; do
    for cfg in none ngram_selector draft_selector arms; do
      python3 bench.py $cfg $wl 1 || echo "FAIL $cfg $wl"
    done
  done
done
python3 bench.py arms eight 1
echo MATRIX_DONE
