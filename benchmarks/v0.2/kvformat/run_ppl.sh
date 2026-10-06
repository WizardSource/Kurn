#!/usr/bin/env bash
# KV-format accuracy on one model: perplexity, KL vs the f16-KV run, and argmax agreement.
#   run_ppl.sh ENGINE MODEL.gguf TOKENS CTX CHUNKS OUTDIR [formats...]
# ENGINE is a `kurn model` engine (its context capacity follows CTX). Threads = 8 (one per kv head on
# Qwen3-0.6B/1.7B/8B). Each format scores the 2nd half of CHUNKS chunks of CTX tokens.
set -euo pipefail
ENGINE=$1 MODEL=$2 TOKENS=$3 CTX=$4 CHUNKS=$5 OUT=$6
shift 6
FORMATS=${*:-q8_0 q4_0 k4c_q4 k4c_q8}
mkdir -p "$OUT"
REF="$OUT/ref_ctx${CTX}.kld"
echo "# $(date -u +%FT%TZ) ctx=$CTX chunks=$CHUNKS model=$(basename "$MODEL")" | tee -a "$OUT/ppl_ctx${CTX}.log"
PPL_CHUNKS=$CHUNKS KURN_KV=f16 KURN_KLD="ref:$REF" "$ENGINE" "$MODEL" ppl 8 "$CTX" "$TOKENS" 2>/dev/null | sed "s/^/f16 /" | tee -a "$OUT/ppl_ctx${CTX}.log"
for f in $FORMATS; do
    PPL_CHUNKS=$CHUNKS KURN_KV=$f KURN_KLD="cmp:$REF" "$ENGINE" "$MODEL" ppl 8 "$CTX" "$TOKENS" 2>/dev/null | sed "s/^/$f /" | tee -a "$OUT/ppl_ctx${CTX}.log"
done
