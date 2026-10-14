#!/usr/bin/env bash
# bench_ops.sh NAME SHAPES LAYERS M THREADS Z ROUNDS -> results/ops-NAME.tsv (config, round, ms)
# configs: stock-tq1 (ggml CPU TQ1_0), stock-tq2 (ggml CPU TQ2_0), kurn-tq2 (shipped), kurn-tq2t (tuned
# rows=4 prefetch=4 line), tbm (bitmap + signs; TQ2_0 source), tbm-tq1 (bitmap + signs; TQ1_0 source)
set -u
cd "${KW:?set KW to the work dir: opbench, opbench-tq2t, llama-kurn, llama-tq2t, models/, results/}"
NAME=$1; SH=$2; LAY=$3; M=$4; T=$5; Z=$6; R=${7:-5}; CONFIGS=${CONFIGS:-"stock-tq1 stock-tq2 kurn-tq2 kurn-tq2t tbm"}
REPS=${REPS:-15}
out=results/ops-$NAME.tsv
: > $out
for r in $(seq 1 $R); do
  for c in $CONFIGS; do
    case $c in
      stock-tq1) cmd="./opbench CPU tq1_0";;
      stock-tq2) cmd="./opbench CPU tq2_0";;
      kurn-tq2) cmd="env GGML_KURN_TBM=0 ./opbench KURN tq2_0";;
      kurn-tq2t) cmd="env GGML_KURN_TBM=0 ./opbench-tq2t KURN tq2_0";;
      tbm) cmd="env GGML_KURN_TBM=1 ./opbench KURN tq2_0";;
      tbm-tq1) cmd="env GGML_KURN_TBM=1 ./opbench KURN tq1_0";;
      tbm-noamx) cmd="env GGML_KURN_TBM=1 GGML_KURN_AMX=0 ./opbench KURN tq2_0";;
      kurn-tq2-noamx) cmd="env GGML_KURN_TBM=0 GGML_KURN_AMX=0 ./opbench KURN tq2_0";;
    esac
    ms=$(OPBENCH_ZERO=$Z $cmd $M $T $REPS $LAY $SH | sed -n 's/.*: \([0-9.]*\) ms.*/\1/p')
    printf "%s\t%s\t%s\n" "$c" "$r" "$ms" >> $out
  done
done
python3 - "$out" <<'EOF'
import sys, statistics, collections
d = collections.defaultdict(list)
for l in open(sys.argv[1]):
    c, r, ms = l.split()
    d[c].append(float(ms))
base = statistics.median(d["kurn-tq2t"]) if "kurn-tq2t" in d else None
for c, v in d.items():
    m = statistics.median(v)
    print(f"{c:16} median {m:8.3f} ms  min {min(v):8.3f}  max {max(v):8.3f}" + (f"  vs kurn-tq2t {base / m:5.2f}x" if base else ""))
EOF
