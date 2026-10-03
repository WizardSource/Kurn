#!/usr/bin/env bash
# The WS-B locked measurement chain. Each argument is one benchlock session (< 10 min); join
# steps with + to share a session (the lock is contended, so fewer acquisitions finish sooner).
#   queue.sh GROUP...   e.g. queue.sh headline+amx compare ppl e2e tune_q4_K
#   steps: headline compare ggml fp4 nibble amx ppl e2e tune_<spec>
# Prepare everything first, outside the lock: make_models.sh (FP4 swap models, untimed),
#   nice -n 19 python bench.py SET --build-only; nice -n 19 python prebuild_tune.py specs/*.kurn
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
LOCK=$HERE/../benchlock.sh
PY=/opt/kenv/bin/python
R=$HERE/results
M=$HOME/models/fourbit
L=${LLAMA_FOURBIT_BIN:-$HOME/src/llama-fourbit/build/bin}
export PYTHONPATH=$(cd "$HERE/../../../src" && pwd) KURN_CACHE_DIR=${KURN_CACHE_DIR:-/tmp/kurn-cache-fourbit}
# the lock holder is reniced above everyone's nice-19 jobs, so the VM is never idle: bound the wait
export FB_IDLE_PATIENCE=${FB_IDLE_PATIENCE:-2}

step() {
    echo "=== $1 $(date -u +%T) load=$(cut -d' ' -f1 /proc/loadavg)"
    case $1 in
        headline|compare|ggml|fp4|nibble)
            $PY "$HERE/bench.py" "$1" --reps 3 --secs "${FB_SECS:-0.5}" --regimes hot:1,cold:8 --out "$R/$1.csv" ;;
        *_cold)  # DRAM-only re-run of a set (with the in-session read roofline per rep)
            s=${1%_cold}
            $PY "$HERE/bench.py" "$s" --reps 3 --secs "${FB_SECS:-0.5}" --regimes cold:8 --out "$R/${s}_cold.csv" ;;
        amx)  # amx/amx4.c built as /tmp/fb_amx4; 3 interleaved reps per M
            for rep in 1 2 3; do
                for m in 1 2 4 8 16; do /tmp/fb_amx4 "$m" 1.0 | sed "s/^/rep$rep /"; done
            done | tee "$R/amx.txt" ;;
        ppl)
            for f in Q4_0 IQ4_NL Q4_K_M; do
                $PY "$HERE/quality.py" ppl "$HOME/models/Qwen3-1.7B-$f.gguf" --label "$f"
            done
            for f in mxfp4-ggml mxfp4-mse nvfp4-ggml nvfp4-mse; do
                $PY "$HERE/quality.py" ppl "$M/qwen3-1.7b-swap-$f.gguf" --label "swap-$f"
            done ;;
        e2e)
            bash "$HERE/../run_e2e.sh" "$L" "$R/e2e.csv" "$HOME/models/Qwen3-1.7B-Q8_0.gguf" \
                "$HOME/models/Qwen3-1.7B-Q4_0.gguf" "$HOME/models/Qwen3-1.7B-Q4_K_M.gguf" \
                "$HOME/models/Qwen3-1.7B-IQ4_NL.gguf" "$M/qwen3-1.7b-swap-mxfp4-ggml.gguf" \
                "$M/qwen3-1.7b-swap-nvfp4-mse.gguf" ;;
        tune_*)
            f=${1#tune_}
            $PY -m kurn tune "$HERE/specs/$f.kurn" --regime cold --objective energy --secs 0.5 -o "$R/tune_$f.csv" ;;
        *) echo "unknown step $1" ;;
    esac
}

if [ "${1:-}" = --inner ]; then
    shift
    for s in "$@"; do step "$s"; done
    exit 0
fi
for group in "$@"; do
    case $group in *ppl*|*e2e*)  # wait for make_models.sh outside the lock
        until [ -s "$M/qwen3-1.7b-swap-nvfp4-mse.gguf" ]; do sleep 30; done ;;
    esac
    case $group in *compare*|*ggml*)  # ggml's cold data (6 formats) quantized into BENCH_CACHE beforehand
        until [ "$(ls "${BENCH_CACHE:-/tmp/fb_ggml/cache}" | wc -l)" -ge 6 ] && ! pgrep -f "^/tmp/fb_ggml/bench_ggml" > /dev/null; do
            sleep 30
        done ;;
    esac
    echo "### $group queued $(date -u +%T)"
    $LOCK bash "$0" --inner ${group//+/ }
done
echo "=== done $(date -u +%T)"
