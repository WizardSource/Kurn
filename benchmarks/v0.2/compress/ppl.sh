#!/usr/bin/env bash
# PPL, mean KLD and top-1 agreement vs BF16 on wikitext-2 test (CHUNKS x 512 tokens), llama-perplexity.
#   ppl.sh base                 # once: BF16 logits -> $KLD_BASE
#   ppl.sh OUT.csv MODEL...     # appends model,bpw,ppl,ppl_err,kld,kld_err,top1
# Runs hold the bench lock with threads pinned 1:1 (paused.sh; AMX).
set -uo pipefail
BIN=${LLAMA_BIN:-$HOME/src/llama.cpp/build/bin}
TXT=${TXT:-$HOME/data/wiki.test.raw}
CHUNKS=${CHUNKS:-32}
KLD_BASE=${KLD_BASE:-$HOME/models/compress/kld-base-bf16-c${CHUNKS}.bin}
BF16=${BF16:-$HOME/models/Qwen3-1.7B-BF16.gguf}
H=$(dirname "$0")
LOCK="$H/../benchlock.sh $H/paused.sh"
PY=${PY:-$HOME/venv-compress/bin/python}
export PYTHONPATH=${PYTHONPATH:-$(cd "$H/../../.." && pwd)/src}
if [ "$1" = base ]; then
  $LOCK "$BIN/llama-perplexity" -m "$BF16" -f "$TXT" -c 512 --chunks "$CHUNKS" -t 8 -b 512 \
    --kl-divergence-base "$KLD_BASE" 2>&1 | grep -E "Final estimate"
  exit
fi
OUT=$1; shift
BATCH=${BATCH:-5}  # models per lock acquisition (~1-2 min each)
[ -s "$OUT" ] || echo "model,bpw,ppl,ppl_err,kld,kld_err,top1" > "$OUT"
m=(); for M in "$@"; do grep -q "^$(basename "$M" .gguf)," "$OUT" || m+=("$M"); done
for ((i = 0; i < ${#m[@]}; i += BATCH)); do
  s=""
  for M in "${m[@]:i:BATCH}"; do
    s+="'$BIN/llama-perplexity' -m '$M' -f '$TXT' -c 512 --chunks $CHUNKS -t 8 -b 512 --kl-divergence-base '$KLD_BASE' --kl-divergence > /tmp/compress-ppl-$(basename "$M" .gguf).log 2>&1"$'\n'
  done
  $LOCK bash -c "$s"
  for M in "${m[@]:i:BATCH}"; do
    log=/tmp/compress-ppl-$(basename "$M" .gguf).log
    ppl=$(grep -E "^Mean PPL\(Q\)" "$log" | head -1 | awk '{print $(NF-2)","$NF}')
    kld=$(grep -E "^Mean +KLD:" "$log" | head -1 | awk '{print $(NF-2)","$NF}')
    top=$(grep -E "^Same top p:" "$log" | head -1 | awk '{print $(NF-3)}')
    bpw=$($PY -c "from kurn.mixed import gguf_bpw; print(round(gguf_bpw('$M'),4))")
    echo "$(basename "$M" .gguf),$bpw,$ppl,$kld,$top" | tee -a "$OUT"
  done
done
