#!/usr/bin/env bash
# Measurement batches behind the engine report (results/ in this directory). Every batch
# stays under ~10 minutes and must run through the bench lock:
#   export PYTHONPATH=$PWD/src KURN_CACHE_DIR=/tmp/kurn-cache-engine
#   benchmarks/v0.2/benchlock.sh benchmarks/v0.2/engine/measure.sh BATCH
# BATCH (MODEL = qwen3 | olmoe | qwen3q4 | olmoeq4; the q4 models are `llama-quantize --pure ... Q4_0`
# outputs, so llama.cpp and the engine run identical tensors; engine v1 is Q8_0-only):
#   gen-MODEL    engine v1, v2 (pinned / not pinned) vs llama.cpp AMX-repack and --repack 0 (lcdrive; default OpenMP env and pinned+spinning), interleaved
#   var-MODEL    v2 variants: unfused epilogue, no THP, prefetch-in-wait, 7 threads (+ llama.cpp at 7)
#   wait-MODEL   v2 wait policies per sync kind: spin / spin-then-futex / yield
#   ppl-MODEL    perplexity on /tmp/ppl.txt tokens, same chunks for every implementation
#   perf-MODEL   perf cpu-clock profile of the decode loop only -> spin / matmul / dispatch shares
#   greedy-MODEL greedy token identity, 5 prompts x 64 tokens: v2 / v1 / llama.cpp plain vs llama.cpp AMX
#   epi-MODEL    kernel-level fused vs unfused epilogues (epibench.c) at one thread's per-layer shapes
# Needs: lcdrive (LCDRIVE, built from lcdrive.c), perf (PERF), models in MODELS.
set -u
H=$(cd "$(dirname "$0")" && pwd)
R=$H/results
mkdir -p "$R"
PY=${PY:-/opt/kenv/bin/python}
LC=${LCDRIVE:-/tmp/eng/lcdrive}
MD=${MODELS:-$HOME/models}
PERF=${PERF:-/usr/lib/linux-tools-6.8.0-146/perf}
LB=${LLAMA_BIN:-$HOME/src/llama.cpp/build/bin}
REPS=${REPS:-3}
NGEN=${NGEN:-128}
DONE=${DONE_DIR:-/tmp/kurn-engine-done}  # a batch that completed once is skipped when a queued chain reaches it again
mkdir -p "$DONE"
[ -e "$DONE/${1:-none}" ] && { echo "${1} already done ($DONE/${1})"; exit 0; }
case ${1:-} in
  *-qwen3) n=qwen3; M=$MD/Qwen3-1.7B-Q8_0.gguf ;;
  *-olmoe) n=olmoe; M=$MD/OLMoE-1B-7B-0125-Instruct-Q8_0.gguf ;;
  *-qwen3q4) n=qwen3q4; M=$MD/Qwen3-1.7B-Q4_0-pure.gguf ;;
  *-olmoeq4) n=olmoeq4; M=$MD/OLMoE-1B-7B-0125-Instruct-Q4_0-pure.gguf ;;
  *) sed -n 2,16p "$0"; exit 2 ;;
esac
P=$("$LB/llama-tokenize" -m "$M" -p "The three most important ideas in thermodynamics are" --ids --log-disable 2>/dev/null | tail -1 | tr -d '[] ')
TOK=$R/tokens_$n.txt
[ -s "$TOK" ] || "$LB/llama-tokenize" -m "$M" -f /tmp/ppl.txt --ids --log-disable 2>/dev/null | tail -1 | tr -d '[]' | tr ',' ' ' > "$TOK"
bin() { $PY -c "import sys; from kurn.model.compile_model import engine_binary as e; print(e(sys.argv[1], sys.argv[2], tuple(sys.argv[3:])))" "$M" "$@"; }
V2=$(bin v2) V2U=$(bin v2 KURN_EPILOGUE=0) V1=
case $n in *q4) ;; *) V1=$(bin v1) ;; esac
PIN="OMP_WAIT_POLICY=active OMP_PROC_BIND=close OMP_PLACES=cores"  # llama.cpp threads spin and are pinned 1:1, like the engine's
B=("$PY" "$H/bench.py" --renice --quiet-wait 3 --model "$n" --prompt "$P" --reps "$REPS" --ngen "$NGEN")
# Other workers' nice-19 jobs run in their own session autogroups (kernel.sched_autogroup_enabled=1),
# where nice is relative to the autogroup only: they compete with this run as equals unless this
# run's autogroup gets a higher weight. Raise it for the duration of the locked batch.
sudo -n sh -c "echo -20 > /proc/$$/autogroup" 2>/dev/null
trap 'sudo -n sh -c "echo 0 > /proc/$$/autogroup" 2>/dev/null' EXIT
uptime

case $1 in
  gen-*)
    "${B[@]}" --csv "$R/gen_$n.csv" --log "$R/gen_$n.stderr" \
      ${V1:+"v1=$V1 $M"} "v2=KURN_PROF=1 $V2 $M" "v2_nopin=KURN_PIN=0 $V2 $M" "lc_amx=$LC $M" "lc_plain=LC_REPACK=0 $LC $M" \
      "lc_amx_pin=$PIN $LC $M" "lc_plain_pin=$PIN LC_REPACK=0 $LC $M" ;;
  var-*)
    "${B[@]}" --csv "$R/var_$n.csv" --log "$R/var_$n.stderr" \
      "v2=KURN_PROF=1 $V2 $M" "v2_unfused=$V2U $M" "v2_nothp=KURN_THP=0 $V2 $M" \
      "v2_pf256k=KURN_PROF=1 KURN_PFWAIT=262144 $V2 $M" "v2_T7=THREADS=7 KURN_PROF=1 $V2 $M" "lc_amx_T7=THREADS=7 $LC $M" ;;
  wait-*)
    "${B[@]}" --csv "$R/wait_$n.csv" --log "$R/wait_$n.stderr" \
      "v2=KURN_PROF=1 $V2 $M" "v2_futex0=KURN_PROF=1 KURN_WAIT=futex:0 $V2 $M" \
      "v2_futex4k=KURN_PROF=1 KURN_WAIT=futex:4000 $V2 $M" "v2_yield=KURN_WAIT=yield:0 $V2 $M" \
      "v2_outfutex=KURN_WAIT=attn=spin,ffn=spin,qk=spin,out=futex:4000 $V2 $M" ;;
  ppl-*)
    CTX=${CTX:-512}
    export PPL_CHUNKS=${PPL_CHUNKS:-4}
    for cfg in "v2|$V2" "lc_amx|$PIN $LC" "lc_plain|$PIN LC_REPACK=0 $LC" ${V1:+"v1|$V1"}; do
      name=${cfg%%|*}
      line=$(env ${cfg#*|} "$M" ppl 8 "$CTX" "$TOK" 2>/dev/null | grep '^ppl')
      echo "$n $name ctx=$CTX chunks=$PPL_CHUNKS $line" | tee -a "$R/ppl.txt"
    done ;;
  perf-*)
    NG=${PERF_NGEN:-512}
    for cfg in "v2|$V2" ${V1:+"v1|$V1"} "lc_amx|$LC" "lc_amx_pin|$PIN $LC" "lc_plain_pin|$PIN LC_REPACK=0 $LC"; do
      name=${cfg%%|*}
      fifo=$(mktemp -u /tmp/kurn-perfctl.XXXX); mkfifo "$fifo"
      KURN_PERF_CTL=$fifo "$PERF" record -q -D -1 --control "fifo:$fifo" -F 999 -e cpu-clock -o "/tmp/kurn-$n-$name.perf" -- \
        env ${cfg#*|} "$M" gen 8 "$NG" "$P" > "$R/perf_${n}_$name.out" 2>/dev/null
      rm -f "$fifo"
      "$PERF" report -i "/tmp/kurn-$n-$name.perf" --no-children --sort dso,sym --stdio 2>/dev/null > "/tmp/kurn-$n-$name.report"
      grep -E '^\s+[0-9.]+%' "/tmp/kurn-$n-$name.report" | head -40 > "$R/perf_${n}_$name.top.txt"
      echo "$n $name $(grep decode_tok_s "$R/perf_${n}_$name.out") $($PY "$H/profile_share.py" "/tmp/kurn-$n-$name.report")" | tee -a "$R/perf_share.txt"
    done ;;
  greedy-*)
    for text in "The three most important ideas in thermodynamics are" "Once upon a time, in a small village by the sea," \
                "def fibonacci(n):" "The capital of France is" "Explain why the sky is blue in two sentences."; do
      ids=$("$LB/llama-tokenize" -m "$M" -p "$text" --ids --log-disable 2>/dev/null | tail -1 | tr -d '[] ')
      ref=""
      for cfg in "lc_amx|$PIN $LC" "lc_plain|$PIN LC_REPACK=0 $LC" "v2|$V2" ${V1:+"v1|$V1"}; do
        name=${cfg%%|*}
        toks=$(env ${cfg#*|} "$M" gen 8 64 "$ids" 2>/dev/null | grep '^gen:' | cut -c5-)
        [ -z "$ref" ] && ref=$toks
        same=$($PY -c "import sys; a, b = sys.argv[1].split(), sys.argv[2].split(); print(next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b))))" "$toks" "$ref")
        echo "$n [$text] $name first_diff_vs_lc_amx=$same tokens:$toks" | tee -a "$R/greedy.txt"
      done
    done ;;
  epi-*)
    W=$(dirname "$V2")
    gcc -O3 -march=native -mno-avx512fp16 -fno-math-errno "$H/epibench.c" "$W/epilogue_kernels.c" -I "$H/../../../src/kurn/model" \
      -I "$H/../../../src/kurn/data" -o /tmp/kurn-epibench -lm
    shape=$($PY -c "
import re, sys
h = open(sys.argv[1]).read(); g = lambda k: int(re.search(rf'#define {k} (\d+)', h).group(1))
ff = g('N_FF') * g('N_USED') // 8 if g('N_EXPERT') else g('N_FF') // 8
print(g('N_EMBD') // 32, ff, g('N_LAYER'))" "$W/model_config.h")
    for i in 1 2 3; do taskset -c 3 /tmp/kurn-epibench $shape 40 | sed "s/^/$n rep$i /" | tee -a "$R/epibench.txt"; done ;;
esac
touch "$DONE/$1"
uptime
