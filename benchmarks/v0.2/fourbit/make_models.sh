#!/usr/bin/env bash
# Qwen3-1.7B FP4 variants for the quality / end-to-end comparison.
#   Q4_0 / IQ4_NL / Q4_K_M: the standard llama-quantize mixes in ~/models (no imatrix; Q4_0 is
#   Q4_0 in every layer tensor with a Q6_K token embedding).
#   swap-<fmt>-<method>: Qwen3-1.7B-Q4_0.gguf with each Q4_0 tensor re-quantized from the BF16
#   source by kurn.mx (quality.py swap), so the FP4 models differ from Q4_0 only in the format.
# The swap streams one tensor at a time (peak anonymous memory < 1 GB, logged as peak_anon_mb),
# so it runs untimed at nice 19, outside benchlock.sh.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
SRC=${SRC:-$HOME/models/Qwen3-1.7B-BF16.gguf}
TPL=${TPL:-$HOME/models/Qwen3-1.7B-Q4_0.gguf}
D=${OUT:-$HOME/models/fourbit}
PY=${PY:-/opt/kenv/bin/python}
mkdir -p "$D"
for fm in mxfp4:ggml mxfp4:mse nvfp4:ggml nvfp4:mse; do
    f=${fm%:*} m=${fm#*:}
    out="$D/qwen3-1.7b-swap-$f-$m.gguf"
    [ -s "$out" ] && continue
    nice -n 19 "$PY" "$HERE/quality.py" swap --src "$SRC" --template "$TPL" --format "$f" --method "$m" \
        --out "$out.tmp" > "$out.log" 2>&1 &
    pid=$! peak=0
    while kill -0 $pid 2>/dev/null; do
        a=$(awk '/RssAnon/ {print int($2 / 1024)}' /proc/$pid/status 2>/dev/null || echo 0)
        [ "${a:-0}" -gt "$peak" ] && peak=$a
        [ "$peak" -gt 2048 ] && { kill $pid; echo "$out: anonymous memory > 2 GB, stopped" >&2; exit 1; }
        sleep 1
    done
    wait $pid
    mv "$out.tmp" "$out"
    echo "peak_anon_mb=$peak" >> "$out.log"
    tail -1 "$out.log"
done
ls -la "$D"/*.gguf
