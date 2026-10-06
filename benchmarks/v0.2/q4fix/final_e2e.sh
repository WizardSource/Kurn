#!/usr/bin/env bash
# Final 0.2.1 e2e matrix (results/e2e-final.csv, run.log, summary.txt): 4 models x 5 modes x 3 interleaved rounds + PPL.
# Builds: ~/src/llama.cpp/{build,build-noamx} (stock; noamx = -mno-amx-tile -mno-amx-int8 -mno-amx-bf16),
# ~/src/llama-kurn/{build,build-noamx} (apply.sh on 4ebdf2c74 + fix-amx-broadcast.diff, -mno-avx512fp16),
# /tmp/var/base = libggml-cpu etc. of build-noamx generated from the 0.2.0 sources (d31cc24).
cd "$(dirname "$0")/../e2e"
/opt/kenv/bin/python run_e2e.py --models ~/models/Qwen3-1.7B-Q4_0.gguf ~/models/Qwen3-1.7B-Q4_K_M.gguf ~/models/Qwen3-1.7B-Q8_0.gguf ~/models/OLMoE-1B-7B-0125-Instruct-Q4_0.gguf \
  --modes "" --reps 3 --out /tmp/final/e2e-final.csv \
  --mode stock-amx=$HOME/src/llama.cpp/build/bin \
  --mode stock-noamx=$HOME/src/llama.cpp/build-noamx/bin \
  --mode kurn-0.2.0=$HOME/src/llama-kurn/build-noamx/bin:LD_LIBRARY_PATH=/tmp/var/base,GGML_KURN_AMX=0 \
  --mode kurn-0.2.1=$HOME/src/llama-kurn/build-noamx/bin:GGML_KURN_AMX=0 \
  --mode kurn-0.2.1-amxbuild=$HOME/src/llama-kurn/build/bin:GGML_KURN_AMX=0
echo FINALDONE
