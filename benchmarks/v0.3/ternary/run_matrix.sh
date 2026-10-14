#!/usr/bin/env bash
# op-level benchmark matrix (opbench through the KURN / CPU buffer types); results/ops-*.tsv + matrix.log
HERE=$(cd "$(dirname "$0")" && pwd)
cd "${KW:?set KW to the work dir: opbench, opbench-tq2t, llama-kurn, llama-tq2t, models/, results/}"
BN="2560:2560,2560:640,2560:640,2560:2560,2560:6912,2560:6912,6912:2560"
TR="3072:3072,3072:3072,3072:3072,3072:3072,3072:9216,3072:9216,9216:3072"
BO="4096:4096,4096:1024,4096:1024,4096:4096,4096:12288,4096:12288,12288:4096"
L3="4096:4096,4096:1024,4096:1024,4096:4096,4096:14336,4096:14336,14336:4096"
FA="3072:3072,3072:1024,3072:1024,3072:3072,3072:23040,3072:23040,23040:3072"
run() { echo "=== $*"; "$HERE/bench_ops.sh" "$@"; }
{
CONFIGS="stock-tq1 stock-tq2 kurn-tq2 kurn-tq2t tbm tbm-tq1" run dec-bitnet $BN 30 1 8 0.422 5
CONFIGS="stock-tq1 stock-tq2 kurn-tq2 kurn-tq2t tbm" run dec-trilm $TR 30 1 8 0.387 5
CONFIGS="stock-tq1 stock-tq2 kurn-tq2 kurn-tq2t tbm" run dec-bonsai8b $BO 36 1 8 0.383 5
CONFIGS="stock-tq1 stock-tq2 kurn-tq2 kurn-tq2t tbm" run dec-llama3-8b $L3 32 1 8 0.357 5
CONFIGS="stock-tq1 stock-tq2 kurn-tq2 kurn-tq2t tbm" run dec-falcon3-7b $FA 28 1 8 0.313 5
for z in 0.25 0.31 0.38 0.45 0.52; do CONFIGS="kurn-tq2t tbm" run zsweep-$z $BN 30 1 8 $z 3; done
CONFIGS="kurn-tq2 kurn-tq2t tbm" run dec-bitnet-t4 $BN 30 1 4 0.422 3
CONFIGS="kurn-tq2t tbm" run l3-bitnet $BN 4 1 8 0.422 5
for m in 2 3 4 8; do CONFIGS="stock-tq2 kurn-tq2 kurn-tq2t tbm" run vfy$m-bitnet $BN 30 $m 8 0.422 3; done
for m in 2 4 8; do CONFIGS="kurn-tq2 kurn-tq2t tbm" run vfy$m-bonsai8b $BO 36 $m 8 0.383 3; done
for m in 128 512; do CONFIGS="stock-tq1 stock-tq2 kurn-tq2 kurn-tq2t tbm tbm-noamx" REPS=5 run pp$m-bitnet $BN 6 $m 8 0.422 3; done
echo MATRIX DONE
} 2>&1 | tee results/matrix.log
