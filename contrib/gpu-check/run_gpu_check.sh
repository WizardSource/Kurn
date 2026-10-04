#!/usr/bin/env bash
# KURN GPU check: build, verify, tune briefly and benchmark KURN's CUDA kernels against every installed
# competitor on a Linux CUDA machine. Writes one archive, kurn-gpu-results-<host>-<date>.tar.gz.
#
#   ./run_gpu_check.sh                 full run (~1.5 h on an A100): correctness, roofline, tune sweep per format, full matrix
#   QUICK=1 ./run_gpu_check.sh         ~25 min: smaller tune, matrix at batch 1 and 16 with 3 rounds
#   DRYRUN=1 ./run_gpu_check.sh        no GPU needed: compile for sm_80/90/100, CPU-emulator numerics, packaging
#
# Options (environment):
#   LLAMA_CPP_DIR=/path   use this llama.cpp checkout for the ggml-cuda competitor (built here if needed)
#   NO_LLAMA=1            skip the ggml-cuda competitor (otherwise llama.cpp is cloned if git and network work)
#   NO_MARLIN=1           skip the Marlin competitor (runs only if PyTorch + vLLM or `marlin` are importable)
#   MARLIN_PYTHON=python  interpreter that has torch (default: python3)
#   FORMATS=q4_0,tq2_0    restrict formats (default: all eight)
#   CUDA_VISIBLE_DEVICES  pick the GPU (default: GPU 0)
#
# Needs: Linux, NVIDIA driver + CUDA toolkit (nvcc) 12.x, python3 >= 3.9, g++. cmake + git for llama.cpp.
# Installs nothing outside this folder; no root, no clock or power-limit changes. Close other GPU jobs while it runs.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
NAME="kurn-gpu-results-$(hostname -s)-$STAMP"
OUT="$HERE/$NAME"
mkdir -p "$OUT"
log() { echo "[$(date -u +%T)] $*" | tee -a "$OUT/run.log"; }
QUICK=${QUICK:-0}
DRYRUN=${DRYRUN:-0}
START=$(date +%s)

if [ "$(uname -s)" != "Linux" ]; then
  echo "This kit needs Linux with an NVIDIA GPU (or DRYRUN=1 on any Linux box)." >&2
  exit 1
fi

# ---------------------------------------------------------------- toolchain
for d in "${CUDA_HOME:-}" /usr/local/cuda /usr/local/cuda-*; do
  if [ -n "$d" ] && [ -x "$d/bin/nvcc" ] && ! command -v nvcc >/dev/null; then export PATH="$d/bin:$PATH"; fi
done
if command -v nvcc >/dev/null; then export KURN_NVCC=$(command -v nvcc); fi
GPU=0
if command -v nvidia-smi >/dev/null && nvidia-smi -L 2>/dev/null | grep -q GPU; then GPU=1; fi
if [ "$GPU" = "0" ] && [ "$DRYRUN" = "0" ]; then
  log "no NVIDIA GPU found: running the dry run (compile + emulator checks only)"
  DRYRUN=1
fi

log "== system"
{
  uname -a
  head -3 /etc/os-release 2>/dev/null
  echo "virt: $(systemd-detect-virt 2>/dev/null || echo unknown)"
  lscpu 2>/dev/null | head -20
  free -g
  echo; nvcc --version 2>/dev/null | tail -2 || echo "nvcc: none"
  g++ --version 2>/dev/null | head -1
  python3 --version
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
  nvidia-smi 2>/dev/null || echo "nvidia-smi: none"
  nvidia-smi -q -d CLOCK,POWER,PERFORMANCE,ECC 2>/dev/null | head -120
} > "$OUT/system.txt" 2>&1
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version,power.limit --format=csv 2>/dev/null | tee -a "$OUT/run.log"

log "== kurn from ./kurn/src (pure Python, no dependencies, nothing installed)"
PY=${PYTHON:-python3}
"$PY" -c 'import sys; assert sys.version_info >= (3, 9)' || { log "python3 >= 3.9 needed"; exit 1; }
export PYTHONPATH="$HERE/kurn/src${PYTHONPATH:+:$PYTHONPATH}"
export KURN_CACHE_DIR="$HERE/kurn-cache"
K() { "$PY" -m kurn "$@"; }
K --version | tee -a "$OUT/run.log"
K gpu targets > "$OUT/targets.txt" 2>&1; head -2 "$OUT/targets.txt" | tee -a "$OUT/run.log"

FORMATS=${FORMATS:-q8_0,q4_0,iq4_nl,q4_K,q2_0,tq2_0,q1_0,e8p}
if [ "$DRYRUN" = "1" ]; then
  ARCH=sm_80
  ARCHS=sm_80,sm_90,sm_100
else
  ARCH=$("$PY" -c 'from kurn.gpu.harness import detect_arch; print(detect_arch() or "sm_80")')
  ARCHS=$ARCH
fi
log "arch: $ARCH (compile checks: $ARCHS)"

# ---------------------------------------------------------------- static checks (no GPU needed)
if [ -n "${KURN_NVCC:-}" ]; then
  log "== nvcc/ptxas: every kernel in the covering set for $ARCHS (registers, shared memory, spills, occupancy)"
  K gpu ptxas --all --extra 0 --archs "$ARCHS" --strict > "$OUT/ptxas.txt" 2>&1
  tail -1 "$OUT/ptxas.txt" | tee -a "$OUT/run.log"
  log "== SASS of the default kernels (tensor-core MMA / ldmatrix / cp.async counts, local memory, hot-loop mix)"
  K gpu sass --defaults --arch "${ARCH}" > "$OUT/sass.txt" 2>&1
  grep -c "^ok" "$OUT/sass.txt" | sed 's/^/default kernels without local memory: /' | tee -a "$OUT/run.log"
else
  log "nvcc not found: skipping compile checks (install the CUDA toolkit; on a GPU host it is required)"
fi
log "== CPU-emulator numerics (generated kernels vs the exact reference, no GPU)"
EXTRA=4; [ "$QUICK" = "1" ] && EXTRA=0
K gpu verify --all --extra "$EXTRA" > "$OUT/verify_emu.txt" 2>&1
tail -1 "$OUT/verify_emu.txt" | tee -a "$OUT/run.log"

# ---------------------------------------------------------------- llama.cpp (ggml-cuda competitor)
LLAMA=""
if [ "${NO_LLAMA:-0}" != "1" ] && [ -n "${KURN_NVCC:-}" ]; then
  if [ -n "${LLAMA_CPP_DIR:-}" ]; then
    LLAMA=$LLAMA_CPP_DIR
  elif [ "$DRYRUN" = "0" ] && command -v git >/dev/null && command -v cmake >/dev/null; then
    log "== cloning llama.cpp (shallow) for the ggml-cuda competitor"
    git clone -q --depth 1 https://github.com/ggml-org/llama.cpp "$HERE/llama.cpp" > "$OUT/llama_clone.txt" 2>&1 && LLAMA="$HERE/llama.cpp" ||
      log "clone failed (no network?): ggml-cuda will be reported as not installed. Set LLAMA_CPP_DIR to an existing checkout."
  fi
  if [ -n "$LLAMA" ] && ! ls "$LLAMA"/build*/bin/libggml-cuda.so >/dev/null 2>&1; then
    log "== building llama.cpp's ggml with CUDA for $ARCH (10-30 min)"
    (cd "$LLAMA" && CC=gcc CXX=g++ cmake -B build-kurn -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="${ARCH#sm_}" -DGGML_NATIVE=OFF \
       -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_TOOLS=OFF &&
     cmake --build build-kurn -j"$(nproc)" --target ggml) > "$OUT/llama_build.txt" 2>&1 || { log "llama.cpp build failed (see llama_build.txt)"; LLAMA=""; }
  fi
  [ -n "$LLAMA" ] && (cd "$LLAMA" && git log --oneline -1 2>/dev/null) > "$OUT/llama_commit.txt"
fi

log "== GPU harness"
HARNESS=""
if [ -n "${KURN_NVCC:-}" ]; then
  HARGS=(--arch "$ARCH"); [ -n "$LLAMA" ] && HARGS+=(--llama "$LLAMA")
  HARNESS=$(K gpu harness "${HARGS[@]}" 2> "$OUT/harness_build.txt" | tail -1)
  if [ -z "$HARNESS" ] && [ -n "$LLAMA" ]; then
    log "harness with ggml failed to build (see harness_build.txt); building without the ggml competitor"
    HARNESS=$(K gpu harness --arch "$ARCH" 2>> "$OUT/harness_build.txt" | tail -1)
    LLAMA=""
  fi
  [ -n "$HARNESS" ] && log "harness: $(basename "$HARNESS")${LLAMA:+ (with ggml-cuda from $(cat "$OUT/llama_commit.txt" 2>/dev/null))}" || log "harness build failed (see harness_build.txt)"
fi

if [ "$DRYRUN" = "1" ]; then
  if [ -n "$LLAMA" ]; then
    log "== ggml comparison path on ggml's CPU backend (no GPU)"
    LIBD=$(dirname "$(ls "$LLAMA"/build*/bin/libggml-base.so | head -1)")
    D="$("$PY" -c 'import kurn.gpu.toolchain as t; import os; print(os.path.dirname(t.data_path("kurn_gpu.h")))')"
    if g++ -O2 -std=c++17 -I "$D" -I "$LLAMA/ggml/include" "$D/ggml_mm_check.cpp" -L "$LIBD" -lggml-base -lggml-cpu \
         -Wl,-rpath="$LIBD" -o "$OUT/ggml_mm_check" 2> "$OUT/ggml_mm_check_build.txt"; then
      for f in ${FORMATS//,/ }; do "$OUT/ggml_mm_check" "$f" 64 1024 3; done > "$OUT/ggml_mm_check.jsonl" 2>&1
      cat "$OUT/ggml_mm_check.jsonl" | tee -a "$OUT/run.log"
    fi
  fi
  log "== dry run: no GPU measurements; the report lists every cell as unmeasured"
  echo "{\"mode\": \"dryrun\", \"formats\": \"$FORMATS\"}" > "$OUT/kit.json"
  K gpu report "$OUT" --formats "$FORMATS" --dry > /dev/null 2>&1
else
  log "== device info and roofline"
  K gpu info --harness "$HARNESS" > "$OUT/info.json" 2> "$OUT/info_err.txt"
  K gpu roofline --harness "$HARNESS" > "$OUT/roofline.json" 2>> "$OUT/info_err.txt"
  cat "$OUT/roofline.json" | tr -d '\n' | head -c 600 | tee -a "$OUT/run.log"; echo

  log "== correctness on the GPU (covering set vs the exact reference)"
  K gpu verify --all --extra $([ "$QUICK" = "1" ] && echo 0 || echo 2) --gpu --arch "$ARCH" > "$OUT/verify_gpu.txt" 2>&1
  tail -1 "$OUT/verify_gpu.txt" | tee -a "$OUT/run.log"
  grep FAIL "$OUT/verify_gpu.txt" | head -5 | tee -a "$OUT/run.log"

  log "== tuning sweep for every format before the matrix (dp4a GEMV + tensor-core engine per batch range; speed and NVML energy)"
  TQ=(); [ "$QUICK" = "1" ] && TQ=(--quick)
  K gpu kit-tune --harness "$HARNESS" --arch "$ARCH" --formats "$FORMATS" --out "$OUT/tuned.json" "${TQ[@]}" > "$OUT/tune.txt" 2>&1
  grep -c " us " "$OUT/tune.txt" | sed 's/^/configurations timed: /' | tee -a "$OUT/run.log"

  log "== benchmark matrix (KURN default and tuned kernels vs ggml-cuda, cuBLAS FP16/INT8; interleaved rounds)"
  MQ=(); [ "$QUICK" = "1" ] && MQ=(--quick)
  K gpu matrix --results "$OUT" --harness "$HARNESS" --arch "$ARCH" --formats "$FORMATS" --kernels "$OUT/tuned.json" \
    --secs 0.4 --reps 5 "${MQ[@]}" > "$OUT/matrix_report_stdout.txt" 2>&1
  tail -3 "$OUT/matrix.log" 2>/dev/null | tee -a "$OUT/run.log"

  if [ "${NO_MARLIN:-0}" != "1" ]; then
    log "== Marlin (optional cross-format competitor)"
    MP=${MARLIN_PYTHON:-python3}
    $MP "$HERE/bench_marlin.py" --out "$OUT/marlin.jsonl" $([ "$QUICK" = "1" ] && echo --quick) 2>&1 | tail -2 | tee -a "$OUT/run.log"
  fi
  echo "{\"mode\": \"$([ "$QUICK" = "1" ] && echo quick || echo full)\", \"formats\": \"$FORMATS\", \"llama\": \"$(cat "$OUT/llama_commit.txt" 2>/dev/null)\"}" > "$OUT/kit.json"
  K gpu report "$OUT" --formats "$FORMATS" > /dev/null 2>&1
fi

log "== summary"
grep -A40 "^## Wins" "$OUT/report.md" 2>/dev/null | head -45 | tee -a "$OUT/run.log"
log "done in $(( ($(date +%s) - START) / 60 )) min"
mkdir -p "$OUT/kernels"
cp "$KURN_CACHE_DIR"/gpu/cuda/*.cu "$OUT/kernels/" 2>/dev/null
tar czf "$HERE/$NAME.tar.gz" -C "$HERE" "$NAME"
log "results: $HERE/$NAME.tar.gz ($(du -h "$HERE/$NAME.tar.gz" | cut -f1))"
echo "Copy $NAME.tar.gz into the artifacts folder (e.g. the artifacts/ folder) and "
