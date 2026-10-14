#!/usr/bin/env bash
# end-to-end llama.cpp runs on real ternary models -> results/e2e-*.txt
cd "${KW:?set KW to the work dir: opbench, opbench-tq2t, llama-kurn, llama-tq2t, models/, results/}"
B=llama-kurn/build/bin; BT=llama-tq2t/build/bin
MODELS="models/bitnet-2b4t-tq2_0.gguf models/TriLM-3.9B-TQ2_0.gguf models/Ternary-Bonsai-8B-TQ2_0.gguf models/llama3-8b-tq1/Llama3-8B-1.58-100B-tokens-TQ1_0-00001-of-00005.gguf"
R=${R:-3}
bench() {  # name env... -- binary-dir model
    local name=$1; shift
    local envs=()
    while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
    local bin=$1 m=$2
    env "${envs[@]}" $bin/llama-bench -m $m -t 8 -p 512 -n 128 -r $R -o csv 2>/dev/null | tail -n +2 |
        awk -F, -v n=$name -v m=$(basename $m) '{gsub(/"/,""); print m, n, $0}'
}
{
for round in 1 2; do
for m in $MODELS; do
    bench stock GGML_KURN=0 -- $B $m
    bench kurn-tq2 GGML_KURN_TBM=0 -- $B $m
    bench kurn-tq2t GGML_KURN_TBM=0 -- $BT $m
    bench tbm GGML_KURN_TBM=1 -- $B $m
    bench tbm-auto X=1 -- $B $m
done
done
} > results/e2e-bench.csv
# exact mode (batch-invariant matmuls + attention)
{
for m in models/bitnet-2b4t-tq2_0.gguf models/Ternary-Bonsai-8B-TQ2_0.gguf; do
    bench exact-kurn-tq2t GGML_KURN_FA_MODE=exact GGML_KURN_TBM=0 -- $BT $m
    bench exact-tbm GGML_KURN_FA_MODE=exact GGML_KURN_TBM=1 -- $B $m
done
} > results/e2e-exact.csv
# batched decode (2..8 sequences: the verify-width regime)
for m in models/bitnet-2b4t-tq2_0.gguf models/Ternary-Bonsai-8B-TQ2_0.gguf; do
    for cfg in "kurn-tq2t $BT 0" "tbm $B 1"; do
        set -- $cfg
        echo "== $(basename $m) $1"
        GGML_KURN_TBM=$3 $2/llama-batched-bench -m $m -t 8 -c 4096 -b 2048 -ub 512 -npp 128 -ntg 64 -npl 1,2,4,8 2>/dev/null | grep -E "^\|"
    done
done > results/e2e-batched.txt
echo E2E DONE
