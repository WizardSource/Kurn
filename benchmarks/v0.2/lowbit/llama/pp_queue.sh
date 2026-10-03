#!/usr/bin/env bash
E=/workspace/wt/lowbit/kurn/benchmarks/v0.2/lowbit/llama/run_e2e_lowbit.sh
export NGEN=0 NPP=128 ROUNDS=3
$E bench ~/models/Bonsai-1.7B-Q1_0.gguf bonsai-1.7b-q1_0 "GGML_KURN_Q1_0=lut"
$E bench ~/models/bitnet-2b4t-tq2_0.gguf bitnet-2b4t-tq2_0 "GGML_KURN_TQ2_0=lut"
