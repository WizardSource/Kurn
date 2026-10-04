#!/usr/bin/env bash
# Build kurn-<version>.zip (unpacks to kurn-<version>/) from the committed tree, then check the archive is self-contained:
# it holds every tracked file, and inside the unpacked copy the docs-path test passes and pytest collects with no errors.
#
#   tools/make_release.sh [OUT.zip]          (PYTHON=python3.9 to check with another interpreter)
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
VER=$(sed -n 's/^version = "\(.*\)"/\1/p' pyproject.toml)
OUT=$(realpath -m "${1:-$PWD/kurn-$VER.zip}")
PY=${PYTHON:-python3}
if ! git diff --quiet HEAD -- . || [ -n "$(git ls-files --others --exclude-standard -- .)" ]; then
  echo "uncommitted or untracked files under $ROOT; commit them first" >&2
  exit 1
fi
for need in contrib/gpu-check/run_gpu_check.sh contrib/gpu-check/make_kit.sh contrib/gpu-check/README.txt; do
  if [ -z "$(git ls-files -- "$need")" ]; then
    echo "release is missing $need (the hand-run GPU kit must ship with the package)" >&2
    exit 1
  fi
done
PREFIX=$(git rev-parse --show-prefix)
rm -f "$OUT"
git -C "$(git rev-parse --show-toplevel)" archive --format=zip --prefix="kurn-$VER/" -o "$OUT" "HEAD${PREFIX:+:$PREFIX}"

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
unzip -q "$OUT" -d "$TMP"
COPY="$TMP/kurn-$VER"
[ -x "$COPY/contrib/gpu-check/run_gpu_check.sh" ] || { echo "archive lacks an executable contrib/gpu-check/run_gpu_check.sh" >&2; exit 1; }
if ! diff <(git ls-files | sort) <(cd "$COPY" && find . -type f | sed 's|^\./||' | sort); then
  echo "archive does not match the tracked files" >&2
  exit 1
fi
(cd "$COPY" && "$PY" -m pytest -q -p no:cacheprovider tests/test_docs_paths.py)
(cd "$COPY" && "$PY" -m pytest -q -p no:cacheprovider --collect-only > "$TMP/collect.txt" 2>&1) || {
  grep -E "^ERROR|Error" "$TMP/collect.txt" >&2
  echo "pytest collection failed inside the archive" >&2
  exit 1
}
tail -1 "$TMP/collect.txt"
echo "$OUT: $(git ls-files | wc -l) files, $(stat -c %s "$OUT") bytes"
sha256sum "$OUT"
