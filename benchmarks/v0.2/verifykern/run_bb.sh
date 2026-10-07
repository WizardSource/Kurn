#!/usr/bin/env bash
# llama-batched-bench: 512-token shared prefix (-pps), aggregate decode tok/s at 1/2/3/4/6/8 streams, builds interleaved.
# run_bb.sh MODEL ROUNDS [builds...]
cd $HOME/work
MODEL=${1:-Qwen3-8B-Q8_0}; R=${2:-3}; shift 2
BUILDS=${*:-after before stock}
OUT=results/final/bb-$MODEL.txt
L=Kurn-gpu/benchmarks/v0.2/benchlock.sh
for r in $(seq $R); do
  for b in $BUILDS; do
    case $b in
      after)  bin=llama-vfy/build/bin;  envv="X=1";;
      before) bin=llama-amx/build/bin;  envv="X=1";;
      stock)  bin=llama-vfy/build/bin;  envv="GGML_KURN=0";;
      ik)     bin=ik-src/build/bin;     envv="X=1";;
    esac
    extra="-lm none"; [ $b = ik ] && extra="--no-mmap -rtr"
    env $envv $L $bin/llama-batched-bench -m models/$MODEL.gguf -c 8192 -b 2048 -ub 512 -npp 512 -ntg 64 \
      -npl 1,2,3,4,6,8 -pps -t 8 -fa on $extra 2>/dev/null | grep -E '^\|\s+[0-9]' | sed "s/^/$r $b /" | tee -a $OUT
  done
done
echo BB_DONE >> $OUT
