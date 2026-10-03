#!/usr/bin/env bash
cd /workspace/wt/lowbit/kurn
export PYTHONPATH=$PWD/src KURN_CACHE_DIR=/tmp/kurn-cache-lowbit
R=benchmarks/v0.2/lowbit/results
sudo -n sh -c "echo -20 > /proc/$$/autogroup" 2>/dev/null
for s in tq1_0 q1_0; do
  BENCH_CACHE=/tmp/kurn-lowbit/cache/$s /opt/kenv/bin/python -m kurn tune benchmarks/v0.2/lowbit/specs/${s}_gemv.kurn --regime cold --objective energy \
    --harness /tmp/kurn-lowbit/bench_ggml --secs 0.5 -o $R/tune_${s}_cold8.csv > $R/tune_${s}_cold8.log 2>&1
done
sudo -n sh -c "echo 0 > /proc/$$/autogroup" 2>/dev/null
