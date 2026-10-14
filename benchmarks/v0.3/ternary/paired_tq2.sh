#!/usr/bin/env bash
# Paired A/B of the TQ2_0 kernel schedule: A = rows=2 prefetch=0 (old default, opbench / llama-kurn-old),
# B = rows=4 prefetch=4 pfgran=line (new default, opbench-tq2t / llama-tq2t). Order alternates per pair (ABBA).
# Writes results/paired-tq2-{ops,e2e}.tsv; summarize with paired_ci.py A B FILE.
HERE=$(cd "$(dirname "$0")" && pwd)
cd "${KW:?set KW to the work dir (see README.md)}"
P=${PAIRS:-10}; PE=${PAIRS_E2E:-6}
BN="2560:2560,2560:640,2560:640,2560:2560,2560:6912,2560:6912,6912:2560"
TR="3072:3072,3072:3072,3072:3072,3072:3072,3072:9216,3072:9216,9216:3072"
BO="4096:4096,4096:1024,4096:1024,4096:4096,4096:12288,4096:12288,12288:4096"
L3="4096:4096,4096:1024,4096:1024,4096:4096,4096:14336,4096:14336,14336:4096"
FA="3072:3072,3072:1024,3072:1024,3072:3072,3072:23040,3072:23040,23040:3072"
ops=results/paired-tq2-ops.tsv; e2e=results/paired-tq2-e2e.tsv
: > $ops; : > $e2e
one_op() {  # name exe M layers shapes z
    GGML_KURN_TBM=0 OPBENCH_ZERO=$6 $2 KURN tq2_0 $3 8 15 $4 $5 | sed -n 's/.*: \([0-9.]*\) ms.*/\1/p'
}
pair_op() {  # name M layers shapes z pairs
    for i in $(seq 1 $6); do
        if [ $((i % 2)) = 1 ]; then o="A B"; else o="B A"; fi
        for c in $o; do
            exe=./opbench; [ $c = B ] && exe=./opbench-tq2t
            echo "$1 $i $c $(one_op $1 $exe $2 $3 $4 $5) ms" >> $ops
        done
    done
}
pair_op dec-bitnet 1 30 $BN 0.422 $P
pair_op dec-trilm 1 30 $TR 0.387 $P
pair_op dec-bonsai8b 1 36 $BO 0.383 $P
pair_op dec-llama3-8b 1 32 $L3 0.357 $P
pair_op dec-falcon3-7b 1 28 $FA 0.313 $P
for m in 2 4 8; do pair_op vfy$m-bitnet $m 30 $BN 0.422 $((P * 4 / 5)); done
pair_op pp128-bitnet 128 6 $BN 0.422 $((P * 4 / 5))
for m in models/bitnet-2b4t-tq2_0.gguf models/TriLM-3.9B-TQ2_0.gguf models/Ternary-Bonsai-8B-TQ2_0.gguf \
         models/llama3-8b-tq1/Llama3-8B-1.58-100B-tokens-TQ1_0-00001-of-00005.gguf; do
    n=$(basename $m .gguf | sed 's/-00001-of-00005//')
    for i in $(seq 1 $PE); do
        if [ $((i % 2)) = 1 ]; then o="A B"; else o="B A"; fi
        for c in $o; do
            bin=llama-kurn/build/bin; [ $c = B ] && bin=llama-tq2t/build/bin
            GGML_KURN_TBM=0 $bin/llama-bench -m $m -t 8 -p 512 -n 128 -r 2 -o csv 2>/dev/null | tail -n +2 |
                python3 -c "
import csv, sys
for r in csv.reader(sys.stdin):
    print(('tg128-' if int(r[-8]) == 0 else 'pp512-') + '$n', $i, '$c', r[-2], 'tps')" >> $e2e
        done
    done
done
python3 "$HERE/paired_ci.py" A B $ops $e2e
echo PAIRED DONE
