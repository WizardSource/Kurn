#!/usr/bin/env bash
# The e2e measurement set, one benchlock acquisition per (model, mode) so each locked run stays short.
#   e2e_all.sh [ITEM ...]     items: see the case below (default: all, in order)
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
LOCK=${LOCK:-$HERE/../../benchlock.sh}
E2E=$HERE/run_e2e_lowbit.sh
M=$HOME/models
run() { echo "== $(date -u +%T) $*"; "$LOCK" env "${@:1:$#-4}" "$E2E" "${@: -4}"; }

item() {
    case $1 in
    bonsai17-bench) run ROUNDS=5 NGEN=256 LREPS=3 bench $M/Bonsai-1.7B-Q1_0.gguf bonsai-1.7b-q1_0 \
        "GGML_KURN_Q1_0=lut GGML_KURN_Q1_0=lut4 GGML_KURN_Q1_0=lut,GGML_KURN_LUT_SHARED=0 GGML_KURN_Q1_0=i16" ;;
    bonsai17-ppl) run CHUNKS=4 ppl $M/Bonsai-1.7B-Q1_0.gguf bonsai-1.7b-q1_0 "GGML_KURN_Q1_0=lut GGML_KURN_Q1_0=i16" ;;
    bonsai8-bench) run ROUNDS=3 NGEN=128 LREPS=2 bench $M/Bonsai-8B-Q1_0.gguf bonsai-8b-q1_0 \
        "GGML_KURN_Q1_0=lut GGML_KURN_Q1_0=lut4 GGML_KURN_Q1_0=i16" ;;
    bonsai8-ppl) run CHUNKS=2 ppl $M/Bonsai-8B-Q1_0.gguf bonsai-8b-q1_0 "GGML_KURN_Q1_0=lut" ;;
    bitnet2-bench) run ROUNDS=3 NGEN=256 LREPS=3 bench $M/bitnet-2b4t-tq2_0.gguf bitnet-2b4t-tq2_0 \
        "GGML_KURN_TQ2_0=i16 GGML_KURN_TQ2_0=lut GGML_KURN_TQ2_0=addsub" ;;
    bitnet2-ppl) run CHUNKS=4 ppl $M/bitnet-2b4t-tq2_0.gguf bitnet-2b4t-tq2_0 "GGML_KURN_TQ2_0=i16 GGML_KURN_TQ2_0=lut" ;;
    bitnet1-bench) run ROUNDS=3 NGEN=256 LREPS=3 bench $M/lowbit/bitnet-2b4t-tq1_0-f16emb.gguf bitnet-2b4t-tq1_0 \
        "GGML_KURN_TQ1_0=i16 GGML_KURN_TQ1_0=tern GGML_KURN_TQ1_0=lut" ;;
    bitnet1-ppl) run CHUNKS=4 ppl $M/lowbit/bitnet-2b4t-tq1_0-f16emb.gguf bitnet-2b4t-tq1_0 \
        "GGML_KURN_TQ1_0=i16 GGML_KURN_TQ1_0=tern" ;;
    tbonsai8-bench) run ROUNDS=3 NGEN=128 LREPS=2 bench $M/Ternary-Bonsai-8B-Q2_0_g64.gguf ternary-bonsai-8b-q2_0 \
        "GGML_KURN_Q2_0=i16 GGML_KURN_Q2_0=lut" ;;
    tbonsai8-ppl) run CHUNKS=2 ppl $M/Ternary-Bonsai-8B-Q2_0_g64.gguf ternary-bonsai-8b-q2_0 "GGML_KURN_Q2_0=i16" ;;
    qwen-q2k-bench) run ROUNDS=3 NGEN=256 LREPS=3 bench $M/lowbit/qwen3-1.7b-q2_k.gguf qwen3-1.7b-q2_k "GGML_KURN_Q2_K=k16" ;;
    qwen-q2k-ppl) run CHUNKS=4 ppl $M/lowbit/qwen3-1.7b-q2_k.gguf qwen3-1.7b-q2_k "GGML_KURN_Q2_K=k16" ;;
    qwen-ref-ppl) for q in BF16 Q4_K_M; do
        run CHUNKS=4 BATCHED_ONLY=1 ppl $M/Qwen3-1.7B-$q.gguf qwen3-1.7b-${q,,} ""; done ;;
    *) echo "unknown item $1" >&2; return 1 ;;
    esac
}

ITEMS=("$@")
[ ${#ITEMS[@]} -gt 0 ] || ITEMS=(bonsai17-bench bonsai17-ppl qwen-q2k-bench qwen-q2k-ppl qwen-ref-ppl bitnet2-bench
    bitnet2-ppl bitnet1-bench bitnet1-ppl bonsai8-bench bonsai8-ppl tbonsai8-bench tbonsai8-ppl)
for it in "${ITEMS[@]}"; do item "$it"; done
echo E2E_DONE
