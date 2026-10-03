#!/usr/bin/env bash
# Run a timed measurement (or anything that loads a model / uses > 2 GB RAM) with the
# machine to itself: an exclusive lock shared by every worker on this VM, plus a log
# line with the load average when the lock was taken, so contaminated runs can be found.
#   benchlock.sh CMD ARGS...
exec 9>/tmp/kurn-bench.lock
flock 9 || { echo "benchlock: could not take the lock" >&2; exit 1; }
printf '%s start load=%s pid=%s cmd=%s\n' "$(date -u +%FT%T)" "$(cut -d' ' -f1-3 /proc/loadavg | tr ' ' /)" "$$" "$*" >> /tmp/kurn-bench-lock.log
"$@"
rc=$?
printf '%s end   rc=%s pid=%s\n' "$(date -u +%FT%T)" "$rc" "$$" >> /tmp/kurn-bench-lock.log
exit $rc
