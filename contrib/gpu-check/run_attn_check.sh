#!/usr/bin/env bash
# KURN GPU attention check: one command on a Linux CUDA machine. Tiers: A100 (sm_80, the measured target), B200/GB200
# (sm_100) and RTX 50 (sm_120); every build carries SASS for all three (CUDA >= 12.8) plus PTX, and the run reports
# which arch it ran on and whether that was native SASS or JIT from PTX.
#
#   ./run_attn_check.sh               ~20-30 min on an A100: build, GPU correctness of every covering config on the awkward
#                                     shapes vs a float64 reference, then the decode matrix (correctness + timings)
#   QUICK=1 ./run_attn_check.sh       ~5-10 min: 5 shapes per config, matrix at 1K and 16K context
#   DRYRUN=1 ./run_attn_check.sh      no GPU needed: compile (sm_80 + sm_120 fatbins) and CPU-emulator numerics only
#
# Writes kurn-attn-results-<host>-<date>.tar.gz next to this script. Exit status 0 only if every check passed.
# Needs: Linux, NVIDIA driver + CUDA toolkit (nvcc) 11.x/12.x, python3 >= 3.9, g++. Installs nothing; no root.
# CUDA runtime outside the toolkit dir (split install)? KURN_CUDA_INCLUDE=<dir>/include KURN_CUDA_LIB=<dir>/lib ./...
# Pick the GPU with CUDA_VISIBLE_DEVICES=N.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
NAME="kurn-attn-results-$(hostname -s)-$STAMP"
OUT="$HERE/$NAME"
mkdir -p "$OUT"
log() { echo "[$(date -u +%T)] $*" | tee -a "$OUT/run.log"; }
QUICK=${QUICK:-0}
DRYRUN=${DRYRUN:-0}
START=$(date +%s)
FAILS=0

for d in "${CUDA_HOME:-}" /usr/local/cuda /usr/local/cuda-*; do
  if [ -n "$d" ] && [ -x "$d/bin/nvcc" ] && ! command -v nvcc >/dev/null; then export PATH="$d/bin:$PATH"; fi
done
[ -n "${KURN_NVCC:-}" ] || command -v nvcc >/dev/null || { log "nvcc not found: install the CUDA toolkit (or put it on PATH / set CUDA_HOME or KURN_NVCC)"; exit 1; }
export KURN_NVCC=${KURN_NVCC:-$(command -v nvcc)}
PY=${PYTHON:-python3}
"$PY" -c 'import sys; assert sys.version_info >= (3, 9)' || { log "python3 >= 3.9 needed"; exit 1; }
export PYTHONPATH="$HERE/kurn/src${PYTHONPATH:+:$PYTHONPATH}"
export KURN_CACHE_DIR="$HERE/kurn-cache"
K() { "$PY" -m kurn "$@"; }

GPU=0
if command -v nvidia-smi >/dev/null && nvidia-smi -L 2>/dev/null | grep -q GPU; then GPU=1; fi
if [ "$GPU" = "0" ] && [ "$DRYRUN" = "0" ]; then log "no NVIDIA GPU found: running the dry run (compile + emulator only)"; DRYRUN=1; fi

log "== system"
{
  uname -a; head -3 /etc/os-release 2>/dev/null; nvcc --version | tail -2; g++ --version | head -1; "$PY" --version
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"; nvidia-smi 2>/dev/null || echo "nvidia-smi: none"
} > "$OUT/system.txt" 2>&1
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv,noheader 2>/dev/null | tee -a "$OUT/run.log"
K --version | tee -a "$OUT/run.log"

# CUDA toolchain preflight: nvcc plus the runtime headers / libcudart it needs (they can live outside the toolkit, e.g. a
# monorepo third_party dir with include_no_implicit/ and lib/), checked once with a tiny .cu and the exact build flags,
# before hundreds of builds can fail the same way. Fix: KURN_CUDA_INCLUDE / KURN_CUDA_LIB (or CUDA_HOME, or
# NVCC_APPEND_FLAGS="-I... -L...", passed through unchanged).
if [ -n "${KURN_NVCC:-}" ]; then
  log "== CUDA toolchain preflight (nvcc + runtime headers/libs, compile + link, run on the GPU if present)"
  PFNR=(); [ "$DRYRUN" = "1" ] && PFNR=(--no-run)
  if K gpu doctor "${PFNR[@]}" --json "$OUT/toolchain.json" > "$OUT/toolchain.txt" 2>&1; then
    sed 's/^/  /' "$OUT/toolchain.txt" | tee -a "$OUT/run.log"
  else
    sed 's/^/  /' "$OUT/toolchain.txt" | tee -a "$OUT/run.log"
    log "stopping before any build: fix the CUDA toolchain as shown above, then re-run"
    tar czf "$HERE/$NAME.tar.gz" -C "$HERE" "$NAME"
    log "results: $HERE/$NAME.tar.gz"
    exit 1
  fi
fi
ARCH=sm_80
[ "$DRYRUN" = "0" ] && ARCH=$("$PY" -c 'from kurn.gpu.harness import detect_arch; print(detect_arch() or "sm_80")')
# Which archs a build here embeds is decided by this box's nvcc (or KURN_GPU_ARCHS=sm_80,sm_120 to choose); archs it can't
# build are skipped with a message and those GPUs JIT the embedded PTX. archs.json records the decision.
NOGPU=(); [ "$DRYRUN" = "1" ] && NOGPU=(--no-gpu)
K gpu attn archs "${NOGPU[@]}" > "$OUT/archs.json" 2> "$OUT/archs_warnings.txt" || { log "arch selection failed:"; cat "$OUT/archs.json" "$OUT/archs_warnings.txt" | tee -a "$OUT/run.log"; exit 1; }
[ -s "$OUT/archs_warnings.txt" ] && sed 's/^/note: /' "$OUT/archs_warnings.txt" | tee -a "$OUT/run.log"
IFS='|' read -r TIER HOW FATB <<< "$("$PY" -c "
from kurn.gpu import attn as A
a = '$ARCH'
print(A.tier_for(a), A.run_mode(a), ' '.join(A.target_archs()), sep='|')" 2>/dev/null)"
if [ "$DRYRUN" = "1" ]; then
  log "arch: none (dry run); fatbin SASS: $FATB"
else
  log "ran on: $ARCH -> tier $TIER ($HOW; fatbin SASS: $FATB)"
fi
RAN=$([ "$DRYRUN" = "1" ] && echo none || echo "$ARCH")
echo "{\"ran_on\": \"$RAN\", \"tier\": \"$TIER\", \"how\": \"$HOW\", \"fatbin\": \"$FATB\", \"dryrun\": $DRYRUN}" > "$OUT/arch.json"

log "== build: the covering sets of all three tiers as fatbins ($FATB + PTX); registers, spills"
K gpu attn ptxas --all > "$OUT/attn_ptxas.txt" 2>&1 || FAILS=$((FAILS + 1))
tail -1 "$OUT/attn_ptxas.txt" | tee -a "$OUT/run.log"

if [ "$DRYRUN" = "1" ]; then
  log "== GPU harness for each fatbin arch (compile only)"
  for a in $FATB; do K gpu attn harness --arch "$a" > /dev/null 2>> "$OUT/harness_build.txt" && log "harness $a: ok" || { log "harness $a: FAILED"; FAILS=$((FAILS + 1)); }; done
  log "== CPU-emulator numerics: the sm_80 covering set and every tier's default kernels (no GPU)"
  K gpu attn verify --all --tier sm_80 --quick > "$OUT/attn_verify_emu.txt" 2>&1 || FAILS=$((FAILS + 1))
  tail -1 "$OUT/attn_verify_emu.txt" | tee -a "$OUT/run.log"
  K gpu attn verify --defaults --quick > "$OUT/attn_verify_emu_defaults.txt" 2>&1 || FAILS=$((FAILS + 1))
  tail -1 "$OUT/attn_verify_emu_defaults.txt" | tee -a "$OUT/run.log"
else
  log "== GPU harness"
  HARNESS=$(K gpu attn harness --arch "$ARCH" 2> "$OUT/harness_build.txt" | tail -1)
  [ -n "$HARNESS" ] || { log "harness build failed (see harness_build.txt)"; FAILS=$((FAILS + 1)); }
  if [ -n "$HARNESS" ]; then
    log "== correctness on the GPU: every covering config of tier $TIER on the awkward shapes vs the float64 reference"
    VQ=(); [ "$QUICK" = "1" ] && VQ=(--quick)
    K gpu attn verify --all --gpu --arch "$ARCH" "${VQ[@]}" > "$OUT/attn_verify_gpu.txt" 2>&1 || FAILS=$((FAILS + 1))
    tail -1 "$OUT/attn_verify_gpu.txt" | tee -a "$OUT/run.log"
    grep FAIL "$OUT/attn_verify_gpu.txt" | head -5 | tee -a "$OUT/run.log"

    log "== decode matrix: Llama-3-8B / Qwen3-1.7B / MLA shapes x context x F16/BF16/Q8_0 KV (cold KV > L2; correctness + time)"
    MQ=(); [ "$QUICK" = "1" ] && MQ=(--quick)
    K gpu attn matrix --results "$OUT" --harness "$HARNESS" "${MQ[@]}" > "$OUT/attn_matrix_stdout.txt" 2>&1 || FAILS=$((FAILS + 1))
    tail -1 "$OUT/attn_matrix_stdout.txt" | tee -a "$OUT/run.log"
    K gpu attn report "$OUT" > "$OUT/attn_report.md" 2>&1
    cat "$OUT/attn_report.md" | tee -a "$OUT/run.log"
  fi
fi

log "== done in $(( ($(date +%s) - START) / 60 )) min: $([ "$FAILS" = 0 ] && echo "all checks passed" || echo "$FAILS step(s) FAILED")"
mkdir -p "$OUT/kernels"
cp "$KURN_CACHE_DIR"/gpu/cuda/attn_*.cu "$OUT/kernels/" 2>/dev/null
tar czf "$HERE/$NAME.tar.gz" -C "$HERE" "$NAME"
log "results: $HERE/$NAME.tar.gz ($(du -h "$HERE/$NAME.tar.gz" | cut -f1))"
echo "Copy $NAME.tar.gz into your local artifacts/ folder (or wherever you keep run outputs)."
[ "$FAILS" = 0 ]
