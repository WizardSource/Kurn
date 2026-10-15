#!/bin/bash
# ppl_matrix.sh BASE.kld OUT.tsv NAME=FILE ...   llama-perplexity KL vs the F16 base logits, one row per model
# (a FILE.oa.gguf sidecar next to FILE is picked up by llama.cpp automatically)
B=${B:-/tmp/llama-kurn/build/bin}
KLD=$1; OUT=$2; shift 2
CH=${CHUNKS:-64}
for nv in "$@"; do
  n=${nv%%=*}; f=${nv#*=}
  log=/tmp/ppl-logs/$(basename $OUT .tsv)-$n.log; mkdir -p /tmp/ppl-logs
  $B/llama-perplexity -m $f -f /tmp/wiki.txt -c 512 --chunks $CH -t 8 --kl-divergence-base $KLD --kl-divergence > $log 2>&1
  sz=$(( $(stat -c %s $f) + $( [ -f $f.oa.gguf ] && stat -c %s $f.oa.gguf || echo 0) ))
  ppl=$(grep "Mean PPL(Q)  " $log | awk '{print $4, $6}')
  kld=$(grep "Mean    KLD" $log | awk '{print $3, $5}')
  top=$(grep "Same top p" $log | awk '{print $4}')
  printf "%s\t%.1f\t%s\t%s\t%s\n" "$n" "$(echo "$sz/1048576" | bc -l)" "$ppl" "$kld" "$top" | tee -a $OUT
done
