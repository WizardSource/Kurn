#!/usr/bin/env bash
# KURN GPU kit v2: every GPU check in one command, one report. On a Linux CUDA machine (A100 = sm_80, the measured
# target; B200 / RTX 50 run it too):
#
#   ./run_kit.sh              full (~3 h on an A100): toolchain preflight, llama.cpp ggml-cuda build, attention builds +
#                             GPU correctness + decode matrix (1K-32K) vs llama.cpp / FlashInfer / FlashAttention-2 +
#                             ablations, the GEMM/GEMV matmul kit (all formats incl. MXFP4 / NVFP4) vs ggml-cuda and
#                             cuBLAS, the GPU pytest subset
#   QUICK=1 ./run_kit.sh      (~1 h): quick shapes, attention at 1K and 16K, QUICK matmul kit
#   DRYRUN=1 ./run_kit.sh     any Linux box, no GPU: the same steps compile-only / on the CPU emulator
#
# Output: kurn-kit-results-<host>-<date>/report.md (the single report) and .tar.gz next to this script.
# Exit status 0 only if every correctness step passed.
#
# Containment: runs from wherever this folder was copied; writes only inside it (results, kurn's build cache, the ggml
# build, TMPDIR, CUDA / Python / Triton / FlashInfer caches) and checks that at the end (containment.txt). No network
# (llama.cpp's ggml sources are bundled), no root, no pip installs, no clock or power-limit changes.
#
# Options (environment):
#   KURN_CUDA_INCLUDE=/p KURN_CUDA_LIB=/p   CUDA runtime outside nvcc's toolkit (':'-lists ok); CUDA_HOME / CUDA_PATH,
#                                           NVCC_APPEND_FLAGS / NVCC_PREPEND_FLAGS and KURN_NVCC are honored too
#   LLAMA_CPP_DIR=/path      build this llama.cpp checkout's ggml instead of the bundled one (built inside this folder)
#   NO_LLAMA=1               no llama.cpp at all (no ggml-cuda baselines)
#   TORCH_PYTHON=python3     interpreter with torch + flashinfer and/or flash_attn for those baselines (if importable)
#   NO_MATMUL=1 NO_PYTEST=1 NO_TORCH=1   skip a step
#   FORMATS=q8_0,mxfp4       matmul formats (default: all ten)
#   CUDA_VISIBLE_DEVICES=N   pick the GPU
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
NAME="kurn-kit-results-$(hostname -s)-$STAMP"
OUT="$HERE/$NAME"
mkdir -p "$OUT" "$OUT/attn" "$OUT/pytest" "$OUT/llama"
QUICK=${QUICK:-0}
DRYRUN=${DRYRUN:-0}
START=$(date +%s)
FAILS=0

# ---------------------------------------------------------------- containment: everything below writes inside $HERE
touch "$OUT/.start"
export TMPDIR="$HERE/tmp" TEMP="$HERE/tmp" TMP="$HERE/tmp"
mkdir -p "$TMPDIR"
export KURN_CACHE_DIR="$HERE/kurn-cache"
export PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX="$HERE/tmp/pycache"
export CUDA_CACHE_PATH="$HERE/tmp/cuda-jit-cache" XDG_CACHE_HOME="$HERE/tmp/xdg-cache" CCACHE_DIR="$HERE/tmp/ccache"
export TORCH_EXTENSIONS_DIR="$HERE/tmp/torch-ext" TRITON_CACHE_DIR="$HERE/tmp/triton" FLASHINFER_WORKSPACE_BASE="$HERE/tmp/flashinfer"
export MPLCONFIGDIR="$HERE/tmp/mpl" PIP_NO_INDEX=1 PIP_CACHE_DIR="$HERE/tmp/pip"

log() { echo "[$(date -u +%T)] $*" | tee -a "$OUT/run.log"; }
# step NAME LOGFILE CMD...: run, record {step, result, min, log} in steps.jsonl; REQUIRED=1 counts a failure
step() {
  local name=$1 lf=$2; shift 2
  local t0; t0=$(date +%s)
  log "== $name"
  "$@" > "$OUT/$lf" 2>&1
  local rc=$?
  local res="ok"; [ $rc != 0 ] && res="FAILED (exit $rc)"
  [ $rc != 0 ] && [ "${REQUIRED:-1}" = "1" ] && FAILS=$((FAILS + 1))
  tail -2 "$OUT/$lf" | sed 's/^/   /' | tee -a "$OUT/run.log"
  echo "{\"step\": \"$name\", \"result\": \"$res\", \"min\": $(( ($(date +%s) - t0 + 30) / 60 )), \"log\": \"$lf\"}" >> "$OUT/steps.jsonl"
  return $rc
}
skip() { log "== $1: skipped ($2)"; echo "{\"step\": \"$1\", \"result\": \"skipped: $2\", \"log\": \"\"}" >> "$OUT/steps.jsonl"; }

if [ "$(uname -s)" != "Linux" ]; then echo "This kit needs Linux (with an NVIDIA GPU, or DRYRUN=1)." >&2; exit 1; fi
for d in "${CUDA_HOME:-}" /usr/local/cuda /usr/local/cuda-*; do
  if [ -n "$d" ] && [ -x "$d/bin/nvcc" ] && ! command -v nvcc >/dev/null; then export PATH="$d/bin:$PATH"; fi
done
[ -n "${KURN_NVCC:-}" ] || command -v nvcc >/dev/null || { log "nvcc not found: install the CUDA toolkit (or set CUDA_HOME / KURN_NVCC)"; exit 1; }
export KURN_NVCC=${KURN_NVCC:-$(command -v nvcc)}
PY=${PYTHON:-python3}
"$PY" -c 'import sys; assert sys.version_info >= (3, 9)' || { log "python3 >= 3.9 needed"; exit 1; }
export PYTHONPATH="$HERE/kurn/src${PYTHONPATH:+:$PYTHONPATH}"
K() { "$PY" -m kurn "$@"; }
GPU=0
if command -v nvidia-smi >/dev/null && nvidia-smi -L 2>/dev/null | grep -q GPU; then GPU=1; fi
if [ "$GPU" = "0" ] && [ "$DRYRUN" = "0" ]; then log "no NVIDIA GPU found: dry run (compile + CPU emulator only)"; DRYRUN=1; fi
MODE=$([ "$DRYRUN" = "1" ] && echo dryrun || ([ "$QUICK" = "1" ] && echo quick || echo full))
log "kurn GPU kit v2 ($MODE) -> $OUT"

{
  uname -a; head -3 /etc/os-release 2>/dev/null; lscpu 2>/dev/null | grep -E "Model name|^CPU\(s\)"; free -g | head -2
  "$KURN_NVCC" --version | tail -2; g++ --version 2>/dev/null | head -1; cmake --version 2>/dev/null | head -1; "$PY" --version
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"; nvidia-smi 2>/dev/null || echo "nvidia-smi: none"
  nvidia-smi -q -d CLOCK,POWER,PERFORMANCE,ECC 2>/dev/null | head -120
} > "$OUT/system.txt" 2>&1
GPUNAME=$(nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv,noheader 2>/dev/null | head -1)
[ -n "$GPUNAME" ] && log "GPU: $GPUNAME"
KVER=$(K --version 2>&1 | head -1)

# ---------------------------------------------------------------- 1. toolchain
NR=(); [ "$DRYRUN" = "1" ] && NR=(--no-run)
if ! step "CUDA toolchain preflight (nvcc + runtime headers/libs: compile, link, run)" toolchain.txt K gpu doctor "${NR[@]}" --json "$OUT/toolchain.json"; then
  log "stopping before any build: fix the CUDA toolchain as shown in toolchain.txt, then re-run"
  K gpu kit-report "$OUT" > /dev/null 2>&1
  tar czf "$HERE/$NAME.tar.gz" -C "$HERE" "$NAME"; exit 1
fi
CUBLAS=1
REQUIRED=0 step "cuBLAS check (the matmul harness links cuBLAS)" toolchain_cublas.txt K gpu doctor --cublas "${NR[@]}" || CUBLAS=0
ARCH=sm_80
[ "$DRYRUN" = "0" ] && ARCH=$("$PY" -c 'from kurn.gpu.harness import detect_arch; print(detect_arch() or "sm_80")')
NOGPU=(); [ "$DRYRUN" = "1" ] && NOGPU=(--no-gpu)
K gpu attn archs "${NOGPU[@]}" > "$OUT/archs.json" 2> "$OUT/archs_warnings.txt" || FAILS=$((FAILS + 1))
[ -s "$OUT/archs_warnings.txt" ] && sed 's/^/note: /' "$OUT/archs_warnings.txt" | tee -a "$OUT/run.log"
IFS='|' read -r TIER HOW FATB <<< "$("$PY" -c "
from kurn.gpu import attn as A
print(A.tier_for('$ARCH'), A.run_mode('$ARCH'), ' '.join(A.target_archs()), sep='|')" 2>/dev/null)"
RAN=$([ "$DRYRUN" = "1" ] && echo none || echo "$ARCH")
[ "$DRYRUN" = "1" ] && HOW="dry run"
echo "{\"ran_on\": \"$RAN\", \"tier\": \"$TIER\", \"how\": \"$HOW\", \"fatbin\": \"$FATB\", \"dryrun\": $DRYRUN}" > "$OUT/arch.json"
cp "$OUT/arch.json" "$OUT/attn/arch.json"
log "ran on: $RAN (tier $TIER, $HOW); fatbin SASS: $FATB"
echo "{\"mode\": \"$MODE\", \"host\": \"$(hostname -s)\", \"gpu\": \"${GPUNAME:-none}\", \"kurn\": \"$KVER\"}" > "$OUT/kit.json"

# ---------------------------------------------------------------- 2. llama.cpp ggml-cuda (bundled sources, no network)
LLAMA=""
if [ "${NO_LLAMA:-0}" = "1" ]; then
  skip "llama.cpp ggml-cuda build" "NO_LLAMA=1"
else
  if [ -n "${LLAMA_CPP_DIR:-}" ]; then SRC=$LLAMA_CPP_DIR; else SRC="$HERE/ggml-src"; fi
  # a view inside the kit: the sources' ggml/ headers + our build dir (nothing is written into LLAMA_CPP_DIR)
  VIEW="$HERE/llama-view"; mkdir -p "$VIEW"; ln -sfn "$SRC/ggml" "$VIEW/ggml"
  if [ ! -d "$SRC/ggml/include" ]; then
    skip "llama.cpp ggml-cuda build" "no ggml sources at $SRC (bundled ggml-src missing? set LLAMA_CPP_DIR)"
  elif REQUIRED=0 step "llama.cpp ggml-cuda build for $ARCH (bundled ggml, $(cat "$SRC/COMMIT" 2>/dev/null || echo "$SRC"); 10-30 min)" llama/build_stdout.txt \
      K gpu ggml-build --src "$SRC" --build "$VIEW/build-kurn" --arch "$ARCH" --log "$OUT/llama/build.log" --json "$OUT/llama/build.json"; then
    LLAMA=$VIEW
  fi
fi

# ---------------------------------------------------------------- 3. attention
AQ=(); [ "$QUICK" = "1" ] && AQ=(--quick)
step "attention builds: covering sets of all three tiers as fatbins ($FATB + PTX), registers, spills" attn/attn_ptxas.txt K gpu attn ptxas --all
if [ "$DRYRUN" = "1" ]; then
  step "attention numerics on the CPU emulator (sm_80 covering set + every tier's defaults)" attn/attn_verify_emu.txt \
    sh -c "'$PY' -m kurn gpu attn verify --all --tier sm_80 --quick && '$PY' -m kurn gpu attn verify --defaults --quick"
  for a in $FATB; do step "attention harness compiles for $a" "attn/harness_$a.txt" K gpu attn harness --arch "$a"; done
  if [ -n "$LLAMA" ]; then
    step "llama.cpp flash-attention bench compiles" attn/ggml_bench_build.txt "$PY" -c "from kurn.gpu import attn as A; print(A.build_ggml_bench('$LLAMA'))"
  fi
else
  HARNESS=$(K gpu attn harness --arch "$ARCH" 2> "$OUT/attn/harness_build.txt" | tail -1)
  if [ -z "$HARNESS" ]; then
    log "attention harness build failed (attn/harness_build.txt)"; FAILS=$((FAILS + 1))
  else
    step "attention correctness on the GPU: covering set of tier $TIER x awkward shapes vs float64" attn/attn_verify_gpu.txt \
      K gpu attn verify --all --gpu --arch "$ARCH" "${AQ[@]}"
    if [ "$CUBLAS" = "1" ]; then
      GH=$(K gpu harness --arch "$ARCH" 2> "$OUT/roofline_build.txt" | tail -1)
      [ -n "$GH" ] && REQUIRED=0 step "HBM roofline (pure-read bandwidth, launch overhead)" roofline_stdout.txt \
        sh -c "'$PY' -m kurn gpu roofline --harness '$GH' > '$OUT/roofline.json'"
    else
      skip "HBM roofline" "cuBLAS missing: the report uses 90% of the nominal bandwidth"
    fi
    LA=(); [ -n "$LLAMA" ] && LA=(--llama "$LLAMA")
    step "attention decode matrix: kurn defaults vs llama.cpp flash attention (+ ablations at 1K / 16K)" attn/attn_matrix_stdout.txt \
      K gpu attn matrix --results "$OUT/attn" --harness "$HARNESS" "${AQ[@]}" "${LA[@]}" --ablate 1024,16384
    if [ "${NO_TORCH:-0}" = "1" ]; then
      skip "FlashInfer / FlashAttention-2 baselines" "NO_TORCH=1"
    else
      CTX=1024,4096,16384,32768; [ "$QUICK" = "1" ] && CTX=1024,16384
      REQUIRED=0 step "FlashInfer / FlashAttention-2 decode baselines (if ${TORCH_PYTHON:-python3} has them; nothing installed)" attn/torch_stdout.txt \
        "${TORCH_PYTHON:-python3}" "$HERE/bench_attn_torch.py" --out "$OUT/attn/attn_baselines.jsonl" --contexts "$CTX" \
        $([ "$QUICK" = "1" ] && echo "--secs 0.2 --reps 3")
    fi
    cp "$OUT/roofline.json" "$OUT/attn/" 2>/dev/null
  fi
fi

# ---------------------------------------------------------------- 4. matmul kit (GEMM / GEMV, all formats incl. MXFP4 / NVFP4)
if [ "${NO_MATMUL:-0}" = "1" ]; then
  skip "GEMM/GEMV matmul kit" "NO_MATMUL=1"
elif [ "$CUBLAS" = "0" ] && [ "$DRYRUN" = "0" ]; then
  skip "GEMM/GEMV matmul kit" "cuBLAS headers/library not found (see toolchain_cublas.txt; KURN_CUDA_INCLUDE / KURN_CUDA_LIB)"
else
  mkdir -p "$OUT/matmul"
  step "GEMM/GEMV matmul kit (run_gpu_check.sh: correctness, tune, matrix vs ggml-cuda / cuBLAS; formats ${FORMATS:-all incl. mxfp4,nvfp4})" matmul/kit_stdout.txt \
    env KIT_OUT="$OUT/matmul" NO_ATTN=1 NO_TAR=1 NO_CLONE=1 NO_KERNELS=1 LLAMA_CPP_DIR="${LLAMA}" NO_LLAMA=$([ -n "$LLAMA" ] && echo 0 || echo 1) \
    bash "$HERE/run_gpu_check.sh"
  if [ "$DRYRUN" = "0" ] && grep -q "FAIL" "$OUT/matmul/verify_gpu.txt" 2>/dev/null; then FAILS=$((FAILS + 1)); log "matmul GPU correctness has failures"; fi
fi

# ---------------------------------------------------------------- 5. GPU pytest subset
if [ "${NO_PYTEST:-0}" = "1" ]; then
  skip "GPU pytest subset" "NO_PYTEST=1"
else
  if ! "$PY" -c "import pytest" 2>/dev/null; then  # bundled pure-Python wheels, unpacked inside the kit (no pip, no network)
    "$PY" -c "
import glob, zipfile
for w in glob.glob('$HERE/wheels/*.whl'):
    zipfile.ZipFile(w).extractall('$HERE/tmp/pydeps')" 2>> "$OUT/pytest/pytest.txt"
    export PYTHONPATH="$PYTHONPATH:$HERE/tmp/pydeps"
  fi
  step "GPU pytest subset (tests/test_gpu_*.py: spec, codegen, emulator, nvcc/ptxas builds, toolchain, reports)" pytest/pytest.txt \
    sh -c "cd '$HERE/kurn' && '$PY' -m pytest tests -q -p no:cacheprovider --basetemp='$HERE/tmp/pytest' -o cache_dir='$HERE/tmp/pytest-cache' --junitxml='$OUT/pytest/junit.xml'"
fi

# ---------------------------------------------------------------- 6. report, containment, archive
log "== containment: files written outside $HERE since the start (should be none)"
{
  find "${HOME:-/nonexistent}" -xdev -maxdepth 4 -newer "$OUT/.start" -type f 2>/dev/null | grep -v "^$HERE/" | grep -v "/\.bash_history$" | head -50
  find /tmp /var/tmp -xdev -maxdepth 2 -newer "$OUT/.start" -user "$(id -u)" 2>/dev/null | grep -v "^$HERE" | head -50
} > "$OUT/containment.txt"
if [ -s "$OUT/containment.txt" ]; then log "$(wc -l < "$OUT/containment.txt") path(s) outside the kit changed during the run (containment.txt; other processes can cause this)"; else log "none"; fi
K gpu kit-report "$OUT" > /dev/null 2> "$OUT/report_err.txt" || log "report generation failed (report_err.txt)"
log "== done in $(( ($(date +%s) - START) / 60 )) min: $([ "$FAILS" = 0 ] && echo "all checks passed" || echo "$FAILS step(s) FAILED")"
"$PY" - "$OUT" "$KURN_CACHE_DIR" <<'EOF'  # sources of the attention kernels the matrix timed (others: `kurn gpu attn gen`)
import glob, json, os, shutil, sys
out, cache = sys.argv[1:3]
want = set()
for f in ("attn/attn_matrix.jsonl", "attn/attn_ablation.jsonl"):
    p = os.path.join(out, f)
    if os.path.exists(p):
        want |= {json.loads(ln).get("config") for ln in open(p) if ln.strip()} - {None}
if want:
    os.makedirs(os.path.join(out, "kernels"), exist_ok=True)
    for cu in glob.glob(os.path.join(cache, "gpu", "cuda", "attn_*.cu")):
        with open(cu) as fh:
            head = fh.readline()
        if any(head.rstrip().endswith(w) for w in want):
            shutil.copy(cu, os.path.join(out, "kernels"))
EOF
rm -f "$OUT/.start"
tar czf "$HERE/$NAME.tar.gz" -C "$HERE" "$NAME"
log "report: $OUT/report.md"
log "results: $HERE/$NAME.tar.gz ($(du -h "$HERE/$NAME.tar.gz" | cut -f1))"
echo "Copy $NAME.tar.gz into your local artifacts/ folder (or wherever you keep run outputs)."
[ "$FAILS" = 0 ]
