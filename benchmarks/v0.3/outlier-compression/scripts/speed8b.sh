#!/bin/bash
# paired-CI whole-model speed, Qwen3-8B (run with the measurement queues paused)
. /tmp/venv/bin/activate
cd /tmp/gguf
PB=/tmp/kurn/Kurn-gpu/benchmarks/v0.3/matmul-opt/paired.py
S=/tmp/proto/bench1.sh
OUT=/tmp/proto/speed/paired.jsonl
run() {  # label A B kind n [pairs] [env]
  python $PB --pairs ${6:-8} --metric tps --label "$1" -- "$7 $S $2 $4 $5" -- "$7 $S $3 $4 $5" 2>/dev/null | tee -a $OUT |
    python -c "import json,sys; d=json.loads(sys.stdin.read()); print(d['label'], round(d['A_med'],2), round(d['B_med'],2), 'ratio', round(d['ratio_med'],3), [round(x,3) for x in d['ci']], d['gate'], flush=True)"
}
run "8B R8 vs Q8_0 pp512" qwen3-8b-Q8_0.gguf qwen3-8b-r8.gguf pp 512
run "8B R8 vs Q8_0 tg64" qwen3-8b-Q8_0.gguf qwen3-8b-r8.gguf tg 64
run "8B R8 vs Q8_0 pp8" qwen3-8b-Q8_0.gguf qwen3-8b-r8.gguf pp 8
run "8B R8 vs Q8_0 pp512 exact" qwen3-8b-Q8_0.gguf qwen3-8b-r8.gguf pp 512 6 GGML_KURN_FA_MODE=exact
run "8B OA16 vs base Q4_K_M-im tg64" qwen3-8b-Q4_K_M-im.gguf qwen3-8b-oa16nb-q4_k_m.gguf tg 64
run "8B OA16 vs base Q4_K_M-im pp512" qwen3-8b-Q4_K_M-im.gguf qwen3-8b-oa16nb-q4_k_m.gguf pp 512
run "8B OA16 vs base Q4_K_M-im pp8" qwen3-8b-Q4_K_M-im.gguf qwen3-8b-oa16nb-q4_k_m.gguf pp 8
run "8B OA16 vs base IQ2_XS-im tg64" qwen3-8b-IQ2_XS-im.gguf qwen3-8b-oa16nb-iq2_xs.gguf tg 64
run "8B OA16 vs base IQ2_XS-im pp512" qwen3-8b-IQ2_XS-im.gguf qwen3-8b-oa16nb-iq2_xs.gguf pp 512
