#!/usr/bin/env bash
# sweep verify schedules with pfgran=line: "G/Pq8/Pq4" triples
cd $HOME/work
OUT=${OUT:-results/pf-sweep.txt}
SET=${SET:-"1/4/2 1/8/4 1/16/8 2/4/2 2/8/4 4/4/2 4/8/4"}
for S in $SET; do
  IFS=/ read G P8 P4 <<< "$S"
  cfg=$(python3 -c "
import json
G,P8,P4=$G,$P8,$P4
def rows(m):
    r=G
    while r*m>8: r//=2
    return max(r,1)
q8={str(m):{'rows':rows(m),'prefetch':P8,'pfgran':'line'} for m in range(2,9)}
q4={str(m):{'rows':min(rows(m),4),'prefetch':P4,'pfgran':'line'} for m in range(2,9)}
print(json.dumps({'q8_0':{'vfy':q8},'q4_K':{'vfy':q4}}))")
  ./regen.sh "$cfg" > /dev/null || exit 1
  for t in q8_0 q4_K; do
    Kurn-gpu/benchmarks/v0.2/benchlock.sh ./opsweep.sh KURN $t "2 3 4 5 6 7 8" 11 8 2>/dev/null | sed "s|^|S=$S |"
  done | tee -a $OUT
done
echo SWEEP_DONE | tee -a $OUT
