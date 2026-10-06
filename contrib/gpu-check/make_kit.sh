#!/usr/bin/env bash
# Build kurn-gpu-check.zip from this checkout: KURN's sources (no tests or benchmarks) + the kit scripts.
#   contrib/gpu-check/make_kit.sh OUT_DIR
# Every file in the zip must stay under 1 MB (the user's file sync is unreliable for large files);
# if the zip itself exceeds 1 MB it is also split into self-contained part zips.
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
OUTDIR=$(mkdir -p "${1:-.}" && cd "${1:-.}" && pwd)
STAGE=$(mktemp -d)
K="$STAGE/kurn-gpu-check"
mkdir -p "$K/kurn/examples"
cp "$HERE/run_gpu_check.sh" "$HERE/run_attn_check.sh" "$HERE/README.txt" "$HERE/bench_marlin.py" "$K/"
cp "$ROOT/pyproject.toml" "$ROOT/README.md" "$ROOT/LICENSE" "$ROOT/CHANGELOG.md" "$K/kurn/"
cp -r "$ROOT/src" "$K/kurn/"
cp -r "$ROOT/examples/gpu" "$K/kurn/examples/"
find "$K" \( -name __pycache__ -o -name '*.egg-info' \) -type d -prune -exec rm -rf {} +
find "$K" -name '*.pyc' -type f -exec rm -f {} +
chmod +x "$K/run_gpu_check.sh" "$K/run_attn_check.sh"
big=$(find "$K" -type f -size +1000k)
if [ -n "$big" ]; then echo "files over 1 MB: $big" >&2; exit 1; fi
rm -f "$OUTDIR/kurn-gpu-check.zip" "$OUTDIR"/kurn-gpu-check-part-*.zip
(cd "$STAGE" && zip -qr -X "$OUTDIR/kurn-gpu-check.zip" kurn-gpu-check)
size=$(stat -c %s "$OUTDIR/kurn-gpu-check.zip")
if [ "$size" -gt 1000000 ]; then  # self-contained parts, each a complete zip of a subset of files
  (cd "$STAGE" && find kurn-gpu-check -type f | sort > files.txt &&
   split -n l/2 -d files.txt part- &&
   for p in part-0*; do zip -qr -X "$OUTDIR/kurn-gpu-check-$p.zip" -@ < "$p"; done)
fi
ls -la "$OUTDIR"/kurn-gpu-check*.zip
echo "files: $(cd "$STAGE" && find kurn-gpu-check -type f | wc -l), largest: $(find "$K" -type f -printf '%s %P\n' | sort -n | tail -1)"
rm -rf "$STAGE"
