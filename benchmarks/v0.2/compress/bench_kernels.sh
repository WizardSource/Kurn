#!/usr/bin/env bash
# Kernel-level comparison for the compress workstream (run under benchlock.sh; ~8 min):
#   benchlock.sh benchmarks/v0.2/compress/bench_kernels.sh [OUTDIR]
# 1) kurn tune of the e8p GEMV (energy objective; hot 1T and cold 8T), 2) 3 interleaved reps of
# hot 1T / cold 1T / cold 8T for: kurn v0.2 e8p (2 configs), Q8_0 vnni16, Q4_K; kurn v0.1 Q8_0
# vnni16, Q4_K; ggml Q8_0, Q4_K, Q2_K, IQ2_XXS, IQ2_XS, IQ3_XXS, IQ4_XS (same harness CLI).
set -uo pipefail
cd "$(dirname "$0")/../../.."
OUT=${1:-benchmarks/v0.2/compress/results}
SECS=${SECS:-0.4}
mkdir -p "$OUT"
export PYTHONPATH=$PWD/src KURN_CACHE_DIR=${KURN_CACHE_DIR:-/tmp/kurn-cache-compress}
PY=/opt/kenv/bin/python
SPEC=benchmarks/v0.2/compress/specs/e8p_gemv.kurn
echo "load at start: $(cat /proc/loadavg)"
H2=$($PY -c "from kurn.toolchain import build_harness; print(build_harness())")
E8P_R4=$($PY -m kurn build $SPEC rows=4 prefetch=4 | tail -1 | cut -d' ' -f1)
E8P_R1=$($PY -m kurn build $SPEC rows=1 prefetch=0 | tail -1 | cut -d' ' -f1)
Q8=$($PY -m kurn build examples/q8_0_gemv_vnni16.kurn | tail -1 | cut -d' ' -f1)
Q4K=$($PY -m kurn build examples/q4_K_gemv.kurn | tail -1 | cut -d' ' -f1)
V01="env -u PYTHONPATH KURN_CACHE_DIR=/tmp/kurn-cache-compress-v01"
H1=$(cd /tmp && $V01 /opt/kenv01/bin/python -c "from kurn.toolchain import build_harness; print(build_harness())")
Q8_01=$(cd /tmp && $V01 /opt/kenv01/bin/kurn build "$OLDPWD/examples/q8_0_gemv_vnni16.kurn" | tail -1 | cut -d' ' -f1)
Q4K_01=$(cd /tmp && $V01 /opt/kenv01/bin/kurn build "$OLDPWD/examples/q4_K_gemv.kurn" | tail -1 | cut -d' ' -f1)
GG=$KURN_CACHE_DIR/bench_ggml_types

if [ "${TUNE:-1}" = 1 ]; then
  $PY -m kurn tune $SPEC --regime hot threads=1 --secs $SECS -o "$OUT/e8p_tune_hot1.csv" | tail -8
  $PY -m kurn tune $SPEC --regime cold --secs $SECS -o "$OUT/e8p_tune_cold8.csv" | tail -8
fi

CSV=$OUT/kernels.csv
rm -f "$CSV"
run() {  # label harness impl kernel regime threads
  "$2" --impl "$3" --kernel "$4" --regime "$5" --threads "$6" --secs "$SECS" --csv "$CSV" --label "$1" | sed "s/^/$1 /"
}
GGK_ALL="q8gemv q4kgemv q2kgemv iq2xxsgemv iq2xsgemv iq3xxsgemv iq4xsgemv"
for rep in 1 2 3; do
  for rt in "hot 1" "cold 1" "cold 8"; do
    set -- $rt; R=$1; T=$2
    run v02_e8p_r4pf4 "$H2" "$E8P_R4" e8pgemv $R $T
    [ $R = hot ] && run v02_e8p_r1pf0 "$H2" "$E8P_R1" e8pgemv $R $T
    run v02_q8_0_vnni16 "$H2" "$Q8" q8gemv $R $T
    run v02_q4_K "$H2" "$Q4K" q4kgemv $R $T
    run v01_q8_0_vnni16 "$H1" "$Q8_01" q8gemv $R $T
    run v01_q4_K "$H1" "$Q4K_01" q4kgemv $R $T
    GGK=$GGK_ALL; [ "$R $T" = "cold 1" ] && GGK="q2kgemv iq2xxsgemv"
    for k in $GGK; do
      run ggml_$k "$GG" ggml $k $R $T
    done
  done
  echo "rep $rep done, load $(cat /proc/loadavg)"
done
echo "load at end: $(cat /proc/loadavg)"
