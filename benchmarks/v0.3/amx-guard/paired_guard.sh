#!/usr/bin/env bash
# Paired A/B prefill: A = guard base 8192 (build dir $OLD), B = 4096 ($NEW). Qwen3-8B Q8_0, 8 threads.
#   MODEL=Qwen3-8B-Q8_0.gguf OLD=llama-kurn NEW=llama-g4096 paired_guard.sh > results/whole-model.jsonl
HERE=$(cd "$(dirname "$0")" && pwd)
P=$HERE/../matmul-opt/paired.py
PY=${KURN_PYTHON:-python3}
M=${MODEL:?set MODEL to a Qwen3-8B Q8_0 gguf}
OLD=${OLD:-llama-kurn}
NEW=${NEW:-llama-g4096}
RX='"([0-9.]+)","[0-9.]+"\s*$'
lb() { echo "$1/build/bin/llama-bench -m $M -n 0 -t 8 -fa 1 -o csv $2 2>/dev/null | tail -1"; }
$PY "$P" --pairs 12 --metric tps --regex "$RX" --label pp512 -- "$(lb "$OLD" '-p 512 -r 3')" -- "$(lb "$NEW" '-p 512 -r 3')"
$PY "$P" --pairs 10 --metric tps --regex "$RX" --label pp2048 -- "$(lb "$OLD" '-p 2048 -r 2')" -- "$(lb "$NEW" '-p 2048 -r 2')"
$PY "$P" --pairs 10 --metric tps --regex "$RX" --label pp512-exact -- "GGML_KURN_EXACT=1 $(lb "$OLD" '-p 512 -r 3')" -- "GGML_KURN_EXACT=1 $(lb "$NEW" '-p 512 -r 3')"
