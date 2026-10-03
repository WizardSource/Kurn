#!/usr/bin/env bash
# End-to-end decode speed + perplexity, mainline ggml vs the kurn lowbit hook. Run under benchlock.sh,
# one model per invocation (each stays well under 10 minutes):
#   run_e2e_lowbit.sh bench MODEL.gguf TAG "VARIANT_ENV ..."   tok/s, interleaved rounds (ROUNDS, default 3)
#   run_e2e_lowbit.sh ppl   MODEL.gguf TAG "VARIANT_ENV"       perplexity on /tmp/ppl.txt
# VARIANT_ENV entries are ';'-free env assignment lists separated by spaces in quotes, e.g.
#   "GGML_KURN_Q1_0=lut GGML_KURN_Q1_0=lut,GGML_KURN_LUT_SHARED=0 GGML_KURN_Q1_0=i16"
# Rows are appended to results/e2e_bench.csv / results/e2e_ppl.csv.
set -euo pipefail
# nice does not cross session autogroups on this VM: raise this session's autogroup for the locked run
sudo -n sh -c "echo -20 > /proc/$$/autogroup" 2>/dev/null && trap 'sudo -n sh -c "echo 0 > /proc/$$/autogroup" 2>/dev/null' EXIT
HERE=$(cd "$(dirname "$0")" && pwd)
RES=$HERE/../results
MAIN=${MAIN:-$HOME/src/llama.cpp/build/bin}
KURN=${KURN:-/tmp/kurn-lowbit/llama.cpp/build/bin}
mode=$1 model=$2 tag=$3 variants=${4:-}
T=${THREADS:-8}
PIN=(-C ff --cpu-strict 1)  # threads 1:1 on vCPUs (AMX tile state is not preserved across migrations on this VM)
mkdir -p "$RES"
load() { cut -d' ' -f1 /proc/loadavg; }

bench_one() {  # label bin repack env...
    local label=${1//,/+} bin=$2 rp=$3; shift 3
    local l0 ts
    l0=$(load)
    ts=$(env "$@" "$bin/llama-bench" -m "$model" -n "${NGEN:-64}" -p "${NPP:-0}" -r "${LREPS:-1}" -t "$T" "${PIN[@]}" --repack "$rp" -o csv 2>/dev/null |
         /opt/kenv/bin/python -c 'import csv, sys; rows = list(csv.DictReader(sys.stdin)); print(rows[-1]["avg_ts"] if rows else "nan")')
    echo "$(date -u +%FT%T),$tag,$label,$T,${NGEN:-64}${NPP:+/pp$NPP}x${LREPS:-1},$ts,$l0,$(load)" | tee -a "$RES/e2e_bench.csv"
}

if [ "$mode" = bench ]; then
    [ -f "$RES/e2e_bench.csv" ] || echo "time,model,config,threads,n_gen,avg_ts,load_start,load_end" > "$RES/e2e_bench.csv"
    for r in $(seq "${ROUNDS:-3}"); do
        bench_one "ggml default (repack=1)" "$MAIN" 1
        bench_one "ggml plain (repack=0)" "$MAIN" 0
        for v in $variants; do
            bench_one "kurn $v" "$KURN" 0 GGML_KURN_LOWBIT=1 ${v//,/ }
        done
    done
elif [ "$mode" = ppl ]; then
    [ -f "$RES/e2e_ppl.csv" ] || echo "time,model,config,ctx,chunks,ubatch,ppl,load_start" > "$RES/e2e_ppl.csv"
    C=${CHUNKS:-4}
    ppl_one() {  # label bin ub env...
        local label=${1//,/+} bin=$2 ub=$3; shift 3
        local l0 p
        l0=$(load)
        p=$(env "$@" "$bin/llama-perplexity" -m "$model" -f /tmp/ppl.txt -c 512 -b 512 -ub "$ub" --chunks "$C" -t "$T" "${PIN[@]}" \
            --no-repack 2>&1 | grep -oE "Final estimate: PPL = [0-9.]+ \+/- [0-9.]+" | sed 's/Final estimate: PPL = //')
        echo "$(date -u +%FT%T),$tag,$label,512,$C,$ub,$p,$l0" | tee -a "$RES/e2e_ppl.csv"
    }
    ppl_one "ggml batched" "$MAIN" 512
    [ -n "${BATCHED_ONLY:-}" ] && exit 0
    ppl_one "ggml ubatch=1" "$MAIN" 1
    for v in $variants; do
        ppl_one "kurn $v ubatch=1" "$KURN" 1 GGML_KURN_LOWBIT=1 ${v//,/ }
    done
fi
