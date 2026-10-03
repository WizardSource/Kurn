#!/usr/bin/env bash
# SwiGLU activation sparsity on Qwen3-1.7B (Q8_0): calibrate on CALIB, test on /tmp/ppl.txt.
#   run_qwen.sh [phase...]   phases: calib fit test (default: all). Results in $OUT.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
LOCK=$HERE/../../benchlock.sh
L=${LLAMA_SRC:-$HOME/src/llama.cpp}
OUT=${OUT:-/tmp/attn-res/sparse}
MODEL=${MODEL:-$HOME/models/Qwen3-1.7B-Q8_0.gguf}
CALIB=${CALIB:-/tmp/attn-res/calib.txt}
TEXT=${TEXT:-/tmp/ppl.txt}
TOOL=$OUT/act_sparsity
PY=/opt/kenv/bin/python
mkdir -p "$OUT"
if [ ! -x "$TOOL" ] || [ "$HERE/act_sparsity.cpp" -nt "$TOOL" ]; then
    g++ -O2 -fopenmp -std=c++17 -I "$L/include" -I "$L/ggml/include" "$HERE/act_sparsity.cpp" -o "$TOOL" \
        -L"$L/build/bin" -lllama -lggml -lggml-base -lggml-cpu -Wl,-rpath,"$L/build/bin" || exit 1
fi
# one lock per group of runs (each group stays under ~10 min)
locked() { $LOCK bash -c "$1"; }
run() { echo "$TOOL $MODEL $TEXT --threads 8 ${*:2} > $OUT/test_$1.log 2> $OUT/test_$1.err;"; }
phases=${*:-calib fit test}
for ph in $phases; do
case $ph in
calib)
    $LOCK "$TOOL" "$MODEL" "$CALIB" --threads 8 --stats "$OUT/calib_stats.csv" --dump "$OUT/X.bin" --dump-tokens 2048 \
        > "$OUT/calib.log" 2> "$OUT/calib.err"
    ;;
fit)
    for s in 0.3 0.5 0.7; do
        $PY "$HERE/fit.py" thr "$OUT/calib_stats.csv" h $s "$OUT/thr_h_$s.txt" > /dev/null
        $PY "$HERE/fit.py" thr "$OUT/calib_stats.csv" silu_gate $s "$OUT/thr_gate_$s.txt" > /dev/null
    done
    cmd=""
    for r in 128 256; do
        for s in 0.3 0.5; do
            cmd+="$PY $HERE/fit.py pred $MODEL $OUT/X.bin $r $s $OUT/pred_r$r.bin $OUT/thr_pred_r${r}_$s.txt > $OUT/fit_r${r}_$s.log 2>&1;"
        done
    done
    locked "$cmd"
    ;;
test)
    locked "$(run base --stats "$OUT/test_stats.csv") $(for s in 0.3 0.5 0.7; do run h_$s --kind h --thr "$OUT/thr_h_$s.txt"; done)"
    locked "$(for s in 0.3 0.5 0.7; do run gate_$s --kind gate --thr "$OUT/thr_gate_$s.txt"; done)"
    locked "$(for r in 128 256; do for s in 0.3 0.5; do
        run pred_r${r}_$s --kind pred --pred "$OUT/pred_r$r.bin" --thr "$OUT/thr_pred_r${r}_$s.txt"; done; done)"
    ;;
esac
done
grep -H -E "^PPL|^ALL" "$OUT"/test_*.log 2>/dev/null
