#!/usr/bin/env bash
# sweep verify schedules: for G in rows, P in prefetch -> regen, opbench q8_0 and q4_K at M=2..8
cd $HOME/work
OUT=${OUT:-results/sched-sweep.txt}
GPS=${GPS:-"1/0 1/4 1/8 1/16 2/0 2/8 2/16 4/0 4/8 4/16"}
for GP in $GPS; do
  G=${GP%/*}; P=${GP#*/}
  cfg=$(python3 -c "
import json
G,P=$G,$P
def rows(m):
    r=G
    while r*m>8: r//=2
    return max(r,1)
v={str(m):{'rows':rows(m),'prefetch':P} for m in range(2,9)}
q4=dict(v)
print(json.dumps({'q8_0':{'vfy':v},'q4_K':{'vfy':{k:{'rows':min(x['rows'],4),'prefetch':P} for k,x in v.items()}}}))")
  ./regen.sh "$cfg" > /dev/null || exit 1
  for t in q8_0 q4_K; do
    Kurn-gpu/benchmarks/v0.2/benchlock.sh ./opsweep.sh KURN $t "2 3 4 5 6 7 8" 9 8 2>/dev/null | sed "s|^|G=$G P=$P |"
  done | tee -a $OUT
done
echo SWEEP_DONE | tee -a $OUT
