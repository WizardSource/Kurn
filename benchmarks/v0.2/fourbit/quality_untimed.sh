#!/usr/bin/env bash
# Perplexity / KLD of the Qwen3-1.7B 4-bit models (quality.py ppl), untimed: nice 19, 3 threads,
# outside benchlock.sh. quality.py passes --no-repack, so no AMX runs outside the lock (AMX tile data is
# not context-switched on this VM). Resident set with repack was ~2.4 GB, of which ~1.1 GB is the file-backed (reclaimable)
# model mmap; anonymous memory is ~1.3 GB. Each run is watched and stopped if its anonymous
# memory exceeds 2 GB (peak RSS and anonymous memory logged per model).
#   quality_untimed.sh
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${PY:-/opt/kenv/bin/python}
export PYTHONPATH=$(cd "$HERE/../../../src" && pwd)
M=$HOME/models
run() {  # run MODEL LABEL
    nice -n 19 "$PY" "$HERE/quality.py" ppl "$1" --label "$2" --threads 3 &
    local py=$! peak=0 peak_anon=0
    while kill -0 $py 2>/dev/null; do
        local pid rss anon
        pid=$(pgrep -P $py | head -1)
        rss=$( [ -n "$pid" ] && awk '/VmRSS/ {print int($2 / 1024)}' /proc/$pid/status 2>/dev/null || echo 0)
        anon=$( [ -n "$pid" ] && awk '/RssAnon/ {print int($2 / 1024)}' /proc/$pid/status 2>/dev/null || echo 0)
        [ "${rss:-0}" -gt "$peak" ] && peak=$rss
        [ "${anon:-0}" -gt "$peak_anon" ] && peak_anon=$anon
        if [ "$peak_anon" -gt 2048 ]; then kill "$pid" $py; echo "$2: anonymous memory > 2 GB, stopped" >&2; return 1; fi
        sleep 2
    done
    wait $py
    echo "$2 peak_rss_mb=$peak peak_anon_mb=$peak_anon"
}
run "$M/Qwen3-1.7B-Q4_0.gguf" Q4_0
for f in mxfp4-ggml nvfp4-mse; do run "$M/fourbit/qwen3-1.7b-swap-$f.gguf" "swap-$f"; done
for f in IQ4_NL Q4_K_M; do run "$M/Qwen3-1.7B-$f.gguf" "$f"; done
for f in mxfp4-mse nvfp4-ggml; do run "$M/fourbit/qwen3-1.7b-swap-$f.gguf" "swap-$f"; done
