#!/usr/bin/env bash
# Whole-forward verify cost (M tokens, logits for all M, KV depth 292), builds interleaved per round.
cd $HOME/work
OUT=results/final/vcost.txt
MS="1 2 3 4 5 6 7 8 12 16"
L=Kurn-gpu/benchmarks/v0.2/benchlock.sh
for r in 1 2; do
  for model in Qwen3-8B-Q8_0 Qwen3-8B-Q4_K_M; do
    for b in after before stock ik; do
      case $b in
        after)  cmd="bin/verify_cost_vfy";   envv="X=1";;
        before) cmd="bin/verify_cost_amx";   envv="X=1";;
        stock)  cmd="bin/verify_cost_vfy";   envv="GGML_KURN=0";;
        ik)     cmd="bin/verify_cost_ik";    envv="IK_RTR=1";;
      esac
      env $envv $L $cmd models/$model.gguf 8 "$MS" 7 292 2>/dev/null | sed "s/^/$r $model $b /" | tee -a $OUT
    done
  done
done
echo VCOST_DONE >> $OUT
