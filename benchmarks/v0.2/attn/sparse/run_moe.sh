#!/usr/bin/env bash
# OLMoE-1B-7B: expert-FFN activation sparsity (oracle thresholds) and routed-expert dropping;
# PLM-1.8B: natural ReLU^2 FFN sparsity for contrast. Thresholds from CALIB, PPL on TEXT.
#   run_moe.sh [phase...]   phases: calib test drop (default all). Results in $OUT.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
LOCK=$HERE/../../benchlock.sh
L=${LLAMA_SRC:-$HOME/src/llama.cpp}
OUT=${OUT:-/tmp/attn-res/sparse}
CALIB=${CALIB:-/tmp/attn-res/calib.txt}
TEXT=${TEXT:-/tmp/ppl.txt}
TOOL=$OUT/act_sparsity_moe
PY=/opt/kenv/bin/python
mkdir -p "$OUT"
if [ ! -x "$TOOL" ] || [ "$HERE/act_sparsity.cpp" -nt "$TOOL" ]; then
    g++ -O2 -fopenmp -std=c++17 -I "$L/include" -I "$L/ggml/include" "$HERE/act_sparsity.cpp" -o "$TOOL" \
        -L"$L/build/bin" -lllama -lggml -lggml-base -lggml-cpu -Wl,-rpath,"$L/build/bin" || exit 1
fi
M=$HOME/models/OLMoE-1B-7B-0125-Instruct-Q8_0.gguf
P=$HOME/models/PLM-1.8B-Instruct-Q8_0.gguf
for ph in ${*:-calib test drop}; do
case $ph in
calib)
    $LOCK bash -c "
        $TOOL $M $CALIB --threads 8 --chunks 8 --stats $OUT/olmoe_calib_stats.csv > $OUT/olmoe_calib.log 2>/dev/null"
    for s in 0.3 0.5 0.7; do $PY "$HERE/fit.py" thr "$OUT/olmoe_calib_stats.csv" h $s "$OUT/olmoe_thr_h_$s.txt" > /dev/null; done
    ;;
test)
    $LOCK bash -c "
        $TOOL $M $TEXT --threads 8 --stats $OUT/olmoe_test_stats.csv > $OUT/olmoe_test_base.log 2>/dev/null
        for s in 0.3 0.5 0.7; do $TOOL $M $TEXT --threads 8 --kind h --thr $OUT/olmoe_thr_h_\$s.txt > $OUT/olmoe_test_h_\$s.log 2>/dev/null; done"
    ;;
drop)
    $LOCK bash -c "
        for tau in 0.02 0.05 0.1; do $TOOL $M $TEXT --threads 8 --moe-tau \$tau > $OUT/olmoe_test_tau_\$tau.log 2>/dev/null; done
        $TOOL $P $TEXT --threads 8 --act 'ffn_sqr(relu)' --stats $OUT/plm_test_stats.csv > $OUT/plm_test_base.log 2>/dev/null"
    ;;
esac
done
grep -H -E "^PPL|^ALL|^MOE" "$OUT"/olmoe_test_*.log "$OUT"/plm_test_*.log 2>/dev/null
