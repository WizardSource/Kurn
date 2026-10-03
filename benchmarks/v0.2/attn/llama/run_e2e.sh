#!/usr/bin/env bash
# End-to-end llama.cpp: kurn attn (through the kattn_hook build) vs stock ggml FA / non-FA.
#   run_e2e.sh bench DEPTH REPS    llama-bench pp512 + tg32 at depth DEPTH; REPS=0,1,2 runs every
#                                  variant once per rep, interleaved, inside one lock
#   run_e2e.sh ppl                 llama-perplexity (ctx 2048) of every variant on $TEXT
# Each call takes the bench lock once. Needs build.sh first (OUT=/tmp/llama-kattn).
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
LOCK=$HERE/../../benchlock.sh
L=${LLAMA_SRC:-$HOME/src/llama.cpp}/build/bin
K=${OUT:-/tmp/llama-kattn}/build/bin
MODEL=${MODEL:-$HOME/models/Qwen3-1.7B-Q8_0.gguf}
TEXT=${TEXT:-/tmp/ppl.txt}
RES=${RES:-$HERE/../results/e2e}
T=${THREADS:-8}
# threads pinned 1:1 to vCPUs (AMX tile state is not preserved across migrations on this VM)
PIN="-C 0xff --cpu-strict 1"
mkdir -p "$RES"
lib() { /opt/kenv/bin/python -c "from kurn import attention as A; print(A.build(A.resolve({'target': '$1', 'kv': '$2', 'dk': 128})))"; }
KF16=$(lib amx_bf16 f16)
KQ8=$(lib amx_bf16 q8_0)
# name | env | flags
VARIANTS=(
    "ggml-fa-f16||-fa 1"
    "ggml-nofa-f16||-fa 0"
    "ggml-fa-q8_0||-fa 1 -ctk q8_0 -ctv q8_0"
    "kurn-f16|LD_LIBRARY_PATH=$K KURN_ATTN_LIB=$KF16|-fa 1"
    "kurn-q8_0|LD_LIBRARY_PATH=$K KURN_ATTN_LIB=$KQ8|-fa 1 -ctk q8_0 -ctv q8_0"
)
case ${1:-} in
bench)
    D=$2
    REPS=${3:-0}
    cmd="set -e;"
    for REP in ${REPS//,/ }; do
        for v in "${VARIANTS[@]}"; do
            IFS='|' read -r name env flags <<< "$v"
            [[ -n ${VARS:-} && ",$VARS," != *",$name,"* ]] && continue
            cmd+=" echo '# $name'; env $env $L/llama-bench -m $MODEL -t $T $PIN -p 512 -n 32 -d $D -r 1 -o csv $flags 2>/dev/null | tail -n +2 | sed 's/^/$name,$REP,/' || true;"
        done
    done
    $LOCK bash -c "$cmd" >> "$RES/bench_d$D.csv"
    ;;
ppl)
    cmd=""
    for v in "${VARIANTS[@]}"; do
        IFS='|' read -r name env flags <<< "$v"
        flags=${flags/-fa 1/-fa on}
        flags=${flags/-fa 0/-fa off}
        cmd+=" echo \"$name \$(env $env $L/llama-perplexity -m $MODEL -f $TEXT -c 2048 -b 2048 -t $T $PIN $flags 2>&1 | grep 'Final estimate')\";"
    done
    $LOCK bash -c "$cmd" | tee "$RES/ppl.txt"
    ;;
*)
    sed -n 2,5p "$0"
    exit 1
    ;;
esac
