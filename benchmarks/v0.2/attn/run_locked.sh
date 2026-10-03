#!/usr/bin/env bash
# One locked measurement chunk (<= ~10 min each): run_locked.sh NAME matrix.py-args...
# Output: results/NAME.log (+ the CSVs matrix.py writes). Needs PYTHONPATH/KURN_CACHE_DIR set.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
NAME=$1
shift
mkdir -p "$HERE/results"
exec "$HERE/../benchlock.sh" /opt/kenv/bin/python "$HERE/matrix.py" "$@" > "$HERE/results/$NAME.log" 2>&1
