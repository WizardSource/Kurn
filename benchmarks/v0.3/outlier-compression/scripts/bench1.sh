#!/bin/bash
# bench1.sh MODEL pp|tg N [extra llama-bench args]: one llama-bench run (8 threads, FA on), prints tokens/s last
M=$1; K=$2; N=$3; shift 3
B=${B:-/tmp/llama-kurn/build/bin}
if [ $K = pp ]; then a="-p $N -n 0"; else a="-p 0 -n $N"; fi
$B/llama-bench -m $M $a -t 8 -fa 1 -r 1 -o csv "$@" 2>/dev/null | tail -1 | awk -F, '{gsub(/"/,"",$0); print $(NF-1)}'
