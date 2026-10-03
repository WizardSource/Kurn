#!/usr/bin/env bash
# End-to-end decode speed and energy proxy in stock llama.cpp, same method as ../run_e2e.sh:
# decode tok/s from llama-bench (-p 0 -n 128, 8 threads) and J/token =
# (CPU_S(n=128) - CPU_S(n=8)) * 5.47 W / 120 tokens, for ggml's default path (repack on) and
# --repack 0. REPS interleaved reps (models x modes inner), median + min/max reported.
# BATCH models per lock acquisition; threads pinned 1:1 (paused.sh; AMX).
#   e2e.sh OUT.csv MODEL...
set -uo pipefail
BIN=${LLAMA_BIN:-$HOME/src/llama.cpp/build/bin}
HERE=$(cd "$(dirname "$0")" && pwd)
LOCK="$HERE/../benchlock.sh $HERE/paused.sh"
W=$HERE/../cputime.py
BATCH=${BATCH:-5}
REPS=${REPS:-3}
OUT=$1; shift
[ -s "$OUT" ] || echo "model,mode,decode_tok_s,tok_s_min,tok_s_max,decode_J_per_tok,load" > "$OUT"
models=("$@")
for ((i = 0; i < ${#models[@]}; i += BATCH)); do
  script='set -u'$'\n'
  for ((r = 0; r < REPS; r++)); do
    for M in "${models[@]:i:BATCH}"; do
      n=$(basename "$M" .gguf)
      for mode in default plain; do
        f="--repack 1"; [ $mode = plain ] && f="--repack 0"
        p=/tmp/compress-e2e-$n-$mode-$r
        script+="python3 $W $BIN/llama-bench -m $M -t 8 $f -r 1 -p 0 -n 128 -o csv 2>$p-a.txt | tail -1 > $p.csv"$'\n'
        script+="python3 $W $BIN/llama-bench -m $M -t 8 $f -r 1 -p 0 -n 8 -o csv >/dev/null 2>$p-b.txt"$'\n'
      done
    done
  done
  script+="cut -d' ' -f1-3 /proc/loadavg | tr ' ' / > /tmp/compress-e2e-load.txt"$'\n'
  $LOCK bash -c "$script"
  for M in "${models[@]:i:BATCH}"; do
    n=$(basename "$M" .gguf)
    for mode in default plain; do
      tg=() ; js=()
      for ((r = 0; r < REPS; r++)); do
        p=/tmp/compress-e2e-$n-$mode-$r
        tg+=("$(awk -F, '{print $(NF-1)}' $p.csv | tr -d '"')")
        read -r _ ca _ < $p-a.txt
        read -r _ cb _ < $p-b.txt
        js+=("$ca-$cb")
      done
      python3 - "$n" "$mode" "$(cat /tmp/compress-e2e-load.txt)" "${tg[*]}" "${js[*]}" <<'EOF' | tee -a "$OUT"
import statistics as st, sys
n, mode, load, tg, js = sys.argv[1:]
tg = [float(x) for x in tg.split()]
j = [eval(x) * 5.47 / 120 for x in js.split()]
print(f"{n},{mode},{st.median(tg):.2f},{min(tg):.2f},{max(tg):.2f},{st.median(j):.4f},{load}")
EOF
    done
  done
done
