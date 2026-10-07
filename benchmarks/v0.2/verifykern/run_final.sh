#!/usr/bin/env bash
# final end-to-end set on the rebased trees: before = llama-amx (integrated + amx-prefill), after = llama-vfy, stock = GGML_KURN=0
cd $HOME/work
L=Kurn-gpu/benchmarks/v0.2/benchlock.sh
PY=Kurn-gpu/.venv/bin/python
T=models/Qwen3-8B-Q8_0.gguf; D=models/Qwen3-0.6B-Q8_0.gguf
P="Write a Python function that checks whether a string is a palindrome."
R=results/final; mkdir -p $R/calib $R/spec
B=$HOME/work
bins() { case $1 in after) echo llama-vfy/build/bin;; before) echo llama-amx/build/bin;; stock) echo llama-vfy/build/bin;; esac; }
envs() { case $1 in stock) echo GGML_KURN=0;; *) echo X=1;; esac; }
./run_vcost.sh > logs/final-vcost.log 2>&1; echo "vcost done"
for b in after before stock; do
  env $(envs $b) KURN_CALIB_OUT=$R/calib/qwen3-8b-$b KURN_CALIB_MMAX=16 KURN_CALIB_REPS=5 KURN_CALIB_TABLE=1 \
    $L $(bins $b)/kurn-spec-calib -m $T -md $D --spec-type draft-simple -p "$P" -n 128 -t 8 -td 8 -c 4096 > logs/final-calib-$b.log 2>&1
  echo "calib $b rc=$?"
done
./run_bb.sh Qwen3-8B-Q8_0 3 after before stock; echo "bb q8 done"
./run_bb.sh Qwen3-8B-Q4_K_M 2 after before stock; echo "bb q4 done"
for set in main heldout; do
  for b in after before stock; do
    cf=k3,k7,policy; [ $b = stock ] && cf=k3,k7
    env $(envs $b) $L $PY benchmarks/v0.2/verifykern/run_width_defaults.py run --target $T --draft $D --out $R/spec/$b --bin $(bins $b) --prompt-set $set --reps 1 \
      --configs $cf --table $R/calib/qwen3-8b-$b.cost > logs/final-spec-$b-$set.log 2>&1
    echo "spec $b $set rc=$?"
  done
done
$L $PY benchmarks/v0.2/verifykern/server_bench.py $R/server.jsonl \
  after-nodraft:$B/llama-vfy/build/bin:nodraft \
  after-policy:$B/llama-vfy/build/bin:policy=$B/$R/calib/qwen3-8b-after.cost \
  after-k3:$B/llama-vfy/build/bin:k3 after-k7:$B/llama-vfy/build/bin:k7 \
  before-nodraft:$B/llama-amx/build/bin:nodraft \
  before-policy:$B/llama-amx/build/bin:policy=$B/$R/calib/qwen3-8b-before.cost \
  before-k3:$B/llama-amx/build/bin:k3 before-k7:$B/llama-amx/build/bin:k7 \
  stock-nodraft:$B/llama-vfy/build/bin:nodraft:GGML_KURN=0 \
  stock-k3:$B/llama-vfy/build/bin:k3:GGML_KURN=0 stock-k7:$B/llama-vfy/build/bin:k7:GGML_KURN=0 > logs/final-server.log 2>&1
echo "server rc=$?"
$L $PY benchmarks/v0.2/verifykern/exact_check.py $B/llama-vfy/build/bin $B/$R/calib/qwen3-8b-after.cost $R/exact-after.json > logs/final-exact.log 2>&1
echo "exact rc=$?"
echo FINAL_DONE
