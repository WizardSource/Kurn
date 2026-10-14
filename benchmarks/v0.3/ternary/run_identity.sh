#!/usr/bin/env bash
# bit-identity (greedy text, perplexity), load logs (per-tensor layout, bytes) and resident memory -> results/id-*
cd "${KW:?set KW to the work dir: opbench, opbench-tq2t, llama-kurn, llama-tq2t, models/, results/}"
B=llama-kurn/build/bin
MS="models/bitnet-2b4t-tq2_0.gguf models/Ternary-Bonsai-8B-TQ2_0.gguf models/llama3-8b-tq1/Llama3-8B-1.58-100B-tokens-TQ1_0-00001-of-00005.gguf"
P="The three most important inventions of the twentieth century were"
for m in $MS; do
    n=$(basename $m .gguf)
    for cfg in "stock GGML_KURN=0" "tq2 GGML_KURN_TBM=0" "tbm GGML_KURN_TBM=1" "tbm-exact GGML_KURN_TBM=1 GGML_KURN_FA_MODE=exact" "tq2-exact GGML_KURN_TBM=0 GGML_KURN_FA_MODE=exact"; do
        set -- $cfg; c=$1; shift
        env "$@" $B/llama-completion -m $m -t 8 -p "$P" -n 64 --temp 0 --seed 1 -no-cnv --no-display-prompt 2>/dev/null > results/id-$n-$c.txt < /dev/null
    done
    for cfg in "stock GGML_KURN=0" "tq2 GGML_KURN_TBM=0" "tbm GGML_KURN_TBM=1"; do
        set -- $cfg; c=$1; shift
        env "$@" $B/llama-perplexity -m $m -t 8 -f /usr/share/common-licenses/GPL-3 -c 512 --chunks 12 2>&1 | grep -E "^\[|Final estimate" > results/id-$n-ppl-$c.txt
    done
    GGML_KURN_VERBOSE=1 $B/llama-completion -m $m -t 8 -p "hi" -n 1 --temp 0 -no-cnv 2>&1 < /dev/null | grep -E "kurn: repack|KURN model buffer|CPU_Mapped model buffer|CPU model buffer" > results/id-$n-load.txt
    for tb in 0 1; do
        GGML_KURN_TBM=$tb $B/llama-server -m $m -t 8 -c 2048 --port 18088 > /tmp/srv.log 2>&1 &
        pid=$!
        for i in $(seq 1 240); do grep -q "server is listening" /tmp/srv.log && break; sleep 1; done
        echo "TBM=$tb $(grep -E 'RssAnon|RssFile|VmRSS' /proc/$pid/status | tr -s ' \t' ' ' | tr '\n' ' ')" >> results/id-$n-mem.txt
        kill $pid; wait $pid 2>/dev/null
    done
done
echo IDENTITY DONE
