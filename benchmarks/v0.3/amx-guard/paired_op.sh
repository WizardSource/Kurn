#!/usr/bin/env bash
# Op-level paired A/B of the matmul guard on one binary: GGML_KURN_AMX_GUARD=8192 (A) vs 4096 (B).
# opbench (integration/llama.cpp/tools/opbench.c): every MUL_MAT of 4 Qwen3-8B layers at M = 512, KURN buffer, 8 threads.
#   OPBENCH=./opbench paired_op.sh > results/opbench.jsonl
HERE=$(cd "$(dirname "$0")" && pwd)
P=$HERE/../matmul-opt/paired.py
PY=${KURN_PYTHON:-python3}
OB=${OPBENCH:-./opbench}
S="4096:4096,4096:1024,4096:1024,4096:4096,4096:12288,4096:12288,12288:4096"
for t in q8_0 q4_K; do
  for ex in 0 1; do
    [ $t = q4_K ] && [ $ex = 1 ] && continue
    c="GGML_KURN_EXACT=$ex $OB KURN $t 512 8 15 4 $S"
    $PY "$P" --pairs 16 --metric ms --regex 'ops=[0-9]+: ([0-9.]+) ms' --label "op-$t-M512-exact$ex" -- "GGML_KURN_AMX_GUARD=8192 $c" -- "GGML_KURN_AMX_GUARD=4096 $c"
  done
done
