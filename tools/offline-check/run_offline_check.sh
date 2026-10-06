#!/usr/bin/env bash
# kurn offline check: run on any Linux machine (bare metal, WSL2, AMD, ARM server). Checks this copy of kurn on this CPU and
# writes one archive, kurn-offline-results-<host>-<date>.tar.gz, in the current directory (or $KURN_OFFLINE_OUT).
#
#   tools/offline-check/run_offline_check.sh            kurn verify --all --strict + roofline + short tune sweeps + AMX test
#   FULL=1 tools/offline-check/run_offline_check.sh     also the full pytest suite
#   QUICK=1 tools/offline-check/run_offline_check.sh    smoke test: verify only the example specs instead of --all
#
# Needs: Linux, python3 >= 3.9, gcc or clang (KURN_CC picks one, e.g. KURN_CC=clang or KURN_CC=gcc-12).
# Installs nothing outside the output directory. Without network access it runs kurn from the source tree (PYTHONPATH)
# when `pip install` cannot fetch its build backend. Close other heavy programs while it runs: timings are recorded.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
KURN_ROOT=$(cd "$HERE/../.." && pwd)
BASE=${KURN_OFFLINE_OUT:-$PWD}
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUT="$BASE/kurn-offline-results-$(hostname -s)-$STAMP"
mkdir -p "$OUT"
log() { echo "[$(date -u +%T)] $*" | tee -a "$OUT/run.log"; }

if [ "$(uname -s)" != "Linux" ]; then
  echo "This check needs Linux (or WSL2 on Windows). kurn's harness uses Linux-only APIs (futex, thread pinning)." >&2
  exit 1
fi
if [ ! -f "$KURN_ROOT/src/kurn/__init__.py" ]; then
  echo "run this script from inside a kurn source tree (expected $KURN_ROOT/src/kurn)" >&2
  exit 1
fi

log "== system"
{
  uname -a
  head -3 /etc/os-release 2>/dev/null
  echo "virt: $(systemd-detect-virt 2>/dev/null || echo unknown)"
  lscpu
  echo; grep -m1 -o -w -E 'flags.*' /proc/cpuinfo | tr ' ' '\n' | grep -E '^(avx2|avx512f|avx512_vnni|avx_vnni|avx512_bf16|avx512vbmi|amx_tile|amx_int8|amx_bf16|asimddp|i8mm|sve|sme)$' | sort | tr '\n' ' '
  echo; free -g
  echo; for c in ${KURN_CC:-} gcc clang; do command -v "$c" > /dev/null && "$c" --version | head -1; done
  as --version 2>/dev/null | head -1
  python3 --version
  echo "autogroup: $(cat /proc/sys/kernel/sched_autogroup_enabled 2>/dev/null)"
} > "$OUT/system.txt" 2>&1
head -5 "$OUT/system.txt"

log "== install kurn ($KURN_ROOT) into $BASE/kurn-venv"
PY=python3
if python3 -m venv "$BASE/kurn-venv" > "$OUT/pip.txt" 2>&1 \
   && "$BASE/kurn-venv/bin/python" -m pip install -q "$KURN_ROOT" >> "$OUT/pip.txt" 2>&1; then
  PY="$BASE/kurn-venv/bin/python"
else
  log "pip install failed (offline or no python3-venv; see pip.txt): running kurn from the source tree"
  export PYTHONPATH="$KURN_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
  [ -x "$BASE/kurn-venv/bin/python" ] && PY="$BASE/kurn-venv/bin/python"
fi
export KURN_CACHE_DIR="$BASE/kurn-cache"
K() { "$PY" -m kurn "$@"; }
K --version | tee -a "$OUT/run.log"
K targets > "$OUT/targets.txt" 2>&1
awk '$5=="native"{print $3}' "$OUT/targets.txt" | sort -u | tr '\n' ' ' | sed 's/^/native targets: /' | tee -a "$OUT/run.log"; echo

log "== correctness: kurn verify --all --strict (every legal config this CPU can run, vs the exact reference)"
if [ "${QUICK:-0}" = "1" ]; then
  for e in "$KURN_ROOT"/examples/*.kurn; do K verify "$e" --strict; done > "$OUT/verify_all.txt" 2>&1
else
  ( time K verify --all --strict ) > "$OUT/verify_all.txt" 2>&1
fi
grep "^skip" "$OUT/verify_all.txt" | sort -u | tee -a "$OUT/run.log"
tail -3 "$OUT/verify_all.txt" | tee -a "$OUT/run.log"

NT=$(nproc)
log "== roofline (peak read bandwidth: widest loads, best of 1/2/4/8 streams and of several passes), 1 and $NT threads"
K roofline --threads 1 > "$OUT/roofline_1t.txt" 2>&1
K roofline --threads "$NT" > "$OUT/roofline_${NT}t.txt" 2>&1
cat "$OUT/roofline_1t.txt" "$OUT/roofline_${NT}t.txt" | tee -a "$OUT/run.log"

log "== short tune sweeps (energy objective, median of 3 interleaved rounds; hot = cache-resident 1 thread, cold = DRAM all threads)"
FLAGS=$(grep -m1 -w flags /proc/cpuinfo 2>/dev/null; grep -m1 -w Features /proc/cpuinfo 2>/dev/null)
has() { echo "$FLAGS" | grep -qw "$1"; }
mkdir -p "$OUT/specs" "$OUT/tune"
spec() {  # name, then spec lines
  local n=$1; shift
  printf '%s\n' "$@" > "$OUT/specs/$n.kurn"
}
H2=$((NT / 2)); [ "$H2" -lt 1 ] && H2=1
if has avx512_vnni; then
  spec q8_0_vnni16 "op gemv" "weights q8_0" "target avx512_vnni" "layout vnni16" "tune align=packed,64 rows=1,2,4,8 prefetch=0,8,16"
  spec q4_K_i16 "op gemv" "weights q4_K" "target avx512_vnni" "layout i16" "tune rows=1,2,4 prefetch=0,8"
  spec q4_K_native "op gemv" "weights q4_K" "target avx512_vnni" "layout native" "tune rows=1,2,4 prefetch=0,4,8"
  spec q4_0_i16 "op gemv" "weights q4_0" "target avx512_vnni" "layout i16" "tune rows=1,2,4 prefetch=0,8"
  spec q1_0_i16 "op gemv" "weights q1_0" "target avx512_vnni" "layout i16" "tune rows=1,2,4 prefetch=0,8"
fi
if has avx_vnni && ! has avx512_vnni; then
  spec q8_0_avx2vnni "op gemv" "weights q8_0" "target avx2_vnni" "layout native" "tune rows=1,2,4,8 prefetch=0,8,16"
  spec q4_0_i8 "op gemv" "weights q4_0" "target avx2_vnni" "layout i8" "tune rows=1,2,4 prefetch=0,8"
fi
if has avx2; then
  spec q8_0_avx2 "op gemv" "weights q8_0" "target avx2" "layout native" "act inline" "tune rows=1,2,4,8 prefetch=0,8,16"
fi
if has asimddp; then
  spec q8_0_neon "op gemv" "weights q8_0" "target neon" "tune rows=1,2,4,8 prefetch=0,8"
fi
for s in "$OUT"/specs/*.kurn; do
  [ -e "$s" ] || continue
  n=$(basename "$s" .kurn)
  if ! K check "$s" > "$OUT/tune/${n}_check.txt" 2>&1; then log "skip $n (not legal here)"; continue; fi
  log "tune $n hot (1 thread)"
  K tune "$s" threads=1 --regime hot --secs 0.3 --rounds 3 -o "$OUT/tune/${n}_hot.csv" > "$OUT/tune/${n}_hot.txt" 2>&1
  log "tune $n cold ($NT and $H2 threads)"
  K tune "$s" "threads=$NT,$H2" --regime cold --secs 0.7 --rounds 3 -o "$OUT/tune/${n}_cold.csv" > "$OUT/tune/${n}_cold.txt" 2>&1
  grep -A1 "best by energy" "$OUT/tune/${n}_cold.txt" | tail -1 | sed "s/^ */  $n cold best: /" | tee -a "$OUT/run.log"
  grep -h "^ranking:" "$OUT/tune/${n}_hot.txt" "$OUT/tune/${n}_cold.txt" | sed "s/^/  $n /" | tee -a "$OUT/run.log"
done

if has amx_tile; then
  log "== AMX tile-state test (does this host preserve AMX tile data across thread switches?)"
  CC=${KURN_CC:-$(command -v gcc || command -v clang)}
  if $CC -O2 -mamx-tile -mamx-int8 -pthread "$HERE/amx_pin.c" -o "$OUT/amx_pin" 2> "$OUT/amx_build.txt"; then
    {
      echo "expected on a correct host: 0 wrong in every row"
      for mode in "8 1 1:8 threads pinned, yield" "8 1 0:8 threads unpinned, yield" "8 0 1:8 threads pinned, spin" "2 1 0:2 threads unpinned, yield"; do
        args=${mode%%:*}; label=${mode#*:}
        printf '%s: ' "$label"
        # shellcheck disable=SC2086
        "$OUT/amx_pin" $args | awk -F'[ /]' '{b+=$3} END{print b, "wrong of", NR*200000}'
      done
    } > "$OUT/amx_test.txt" 2>&1
    tee -a "$OUT/run.log" < "$OUT/amx_test.txt"
  else
    log "AMX test build failed (old compiler or assembler? try KURN_CC=clang), see amx_build.txt"
  fi
else
  log "== no AMX on this CPU: AMX test skipped"
fi

if [ "${FULL:-0}" = "1" ]; then
  log "== full pytest suite"
  "$PY" -m pip install -q pytest pytest-xdist numpy >> "$OUT/pip.txt" 2>&1 || log "could not install pytest/numpy; using what is installed"
  JOBS=(); "$PY" -c "import xdist" 2> /dev/null && JOBS=(-n "$NT")
  ( cd "$KURN_ROOT" && time "$PY" -m pytest -q "${JOBS[@]}" ) > "$OUT/pytest.txt" 2>&1
  tail -3 "$OUT/pytest.txt" | tee -a "$OUT/run.log"
fi

tar -C "$BASE" -czf "$OUT.tar.gz" "$(basename "$OUT")"
log "== done: $OUT.tar.gz"
