#!/usr/bin/env bash
# Plan + quantize the `kurn mix` sweep (untimed, in-process libggml quantizers; no lock needed):
#   mix_sweep.sh PROFILE.json [NAME:TYPES:BPW[:EMBD] ...]
# TYPES "all" = every profiled type; EMBD pins token_embd (tied lm_head) to a type, as llama.cpp
# does (Q6_K; Q5_K for its IQ2/IQ3 mixes). EXTRA: more `kurn mix plan` flags (e.g. --weights).
# Recipes go to results/mix/, models to ~/models/compress/.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
export PYTHONPATH=${PYTHONPATH:-$here/../../../src}
PY=${PY:-/opt/kenv/bin/python}
B=${B:-$HOME/models/compress}
REF=${REF:-$HOME/models/Qwen3-1.7B-BF16.gguf}
IMX=${IMX:-$B/qwen3-1.7b.imatrix.gguf}
PROF=$1; shift
K=Q8_0,Q6_K,Q5_K,Q4_K,Q3_K,Q2_K
specs=("$@")
[ ${#specs[@]} -gt 0 ] || specs=(all:all:2.9:Q5_K all:all:3.2:Q5_K all:all:3.5:Q6_K all:all:4.0:Q6_K
  all:all:4.65:Q6_K all:all:5.1:Q6_K k:$K:3.6:Q6_K k:$K:4.0:Q6_K k:$K:4.35:Q6_K k:$K:4.9:Q6_K
  taalas:Q3_K,Q6_K:4.0:Q6_K taalas:Q3_K,Q6_K:4.65:Q6_K)
mkdir -p "$here/results/mix"
for s in "${specs[@]}"; do
  IFS=: read -r name types bpw embd <<< "$s"
  tag=mix-$name-$bpw
  t=(); [ "$types" = all ] || t=(--types "$types")
  [ -z "$embd" ] || t+=(--fix "token_embd.weight=$embd")
  # shellcheck disable=SC2086
  nice -n 19 $PY -m kurn mix plan "$PROF" --bpw "$bpw" "${t[@]}" ${EXTRA:-} -o "$here/results/mix/$tag.txt"
  [ -s "$B/qwen3-1.7b-$tag.gguf" ] && continue
  nice -n 19 $PY -m kurn mix quantize "$here/results/mix/$tag.json" "$REF" "$B/qwen3-1.7b-$tag.gguf.part" \
    --imatrix "$IMX" --inproc --threads 4
  mv "$B/qwen3-1.7b-$tag.gguf.part" "$B/qwen3-1.7b-$tag.gguf"
done
