#!/usr/bin/env bash
# Run CMD with this workstream's own untimed background jobs (pids listed in $PIDS, default
# /tmp/compress-bg.pids) stopped, resuming them on exit. Use inside benchlock.sh:
#   benchlock.sh paused.sh CMD ARGS...
PIDS=${PIDS:-/tmp/compress-bg.pids}
PAT=${PAT:-compress/(lowrank_model|lowrank_curve|e8p_model|e8p_tensor_err|mix_err)\.py|kurn mix (profile|plan|quantize)}
p=$(echo $(cat "$PIDS" 2>/dev/null) $(pgrep -u "$(id -u)" -f "$PAT"))
[ -n "$p" ] && kill -STOP $p 2>/dev/null
trap '[ -n "$p" ] && kill -CONT $p 2>/dev/null' EXIT
# ggml (OpenMP build) may run AMX, whose tile data this VM does not context-switch: pin 1:1.
export OMP_PROC_BIND=${OMP_PROC_BIND:-close} OMP_PLACES=${OMP_PLACES:-threads}
echo "load at start: $(cut -d' ' -f1-3 /proc/loadavg)"
"$@"
