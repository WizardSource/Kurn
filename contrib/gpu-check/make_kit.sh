#!/usr/bin/env bash
# Build the hand-run GPU kit from this checkout:
#   contrib/gpu-check/make_kit.sh OUT_DIR
#   KIT_GGML=/path/to/llama.cpp  also bundle llama.cpp's ggml sources (run_kit.sh builds ggml-cuda from them, offline)
#   KIT_WHEELS=/path/to/wheels   also bundle pure-Python wheels (pytest + deps) for boxes without pytest
#   KIT_NAME=kurn-gpu-check-attn zip name prefix (the folder inside is always kurn-gpu-check/)
# Every file and every zip stays under 1 MB (the user's file sync is unreliable for large files); all zips unzip into
# the same kurn-gpu-check/ folder:
#   NAME.zip               the scripts, KURN's sources, examples and the GPU test subset
#   NAME-ggml-NN.zip       the bundled ggml sources, in self-contained parts (KIT_GGML)
#   NAME-wheels.zip        the wheels (KIT_WHEELS)
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
OUTDIR=$(mkdir -p "${1:-.}" && cd "${1:-.}" && pwd)
NAME=${KIT_NAME:-kurn-gpu-check}
STAGE=$(mktemp -d)
K="$STAGE/kurn-gpu-check"
mkdir -p "$K/kurn/tests"
cp "$HERE/run_kit.sh" "$HERE/run_gpu_check.sh" "$HERE/run_attn_check.sh" "$HERE/README.txt" "$HERE/bench_marlin.py" "$HERE/bench_attn_torch.py" "$K/"
cp "$ROOT/pyproject.toml" "$ROOT/README.md" "$ROOT/LICENSE" "$ROOT/CHANGELOG.md" "$K/kurn/"
cp -r "$ROOT/src" "$ROOT/examples" "$K/kurn/"
# the GPU-relevant tests (test_gpu_kit.py checks this packaging from a full checkout, so it stays out)
cp "$ROOT/tests/conftest.py" "$K/kurn/tests/"
for t in "$ROOT"/tests/test_gpu_*.py; do [ "$(basename "$t")" = test_gpu_kit.py ] || cp "$t" "$K/kurn/tests/"; done
mkdir -p "$K/kurn/tests/golden" && cp -r "$ROOT/tests/golden/gpu" "$K/kurn/tests/golden/"
find "$K" \( -name __pycache__ -o -name '*.egg-info' \) -type d -prune -exec rm -rf {} +
find "$K" -name '*.pyc' -type f -exec rm -f {} +
chmod +x "$K/run_kit.sh" "$K/run_gpu_check.sh" "$K/run_attn_check.sh"

G="$STAGE/ggml/kurn-gpu-check/ggml-src"
if [ -n "${KIT_GGML:-}" ]; then  # what ggml-cuda needs: ggml/ minus the other backends, plus a wrapper CMakeLists.txt
  mkdir -p "$G/ggml/src"
  (cd "$KIT_GGML" && cp -r --parents ggml/CMakeLists.txt ggml/cmake ggml/include ggml/src/CMakeLists.txt ggml/src/ggml-cuda \
     ggml/src/ggml-cpu "$G/" && cp ggml/src/*.c ggml/src/*.cpp ggml/src/*.h ggml/src/*.in "$G/ggml/src/" 2>/dev/null; true)
  cp "$KIT_GGML/LICENSE" "$G/LICENSE"
  (cd "$KIT_GGML" && git log -1 --format='%h %cs %s' 2>/dev/null || echo unknown) > "$G/COMMIT"
  printf 'cmake_minimum_required(VERSION 3.14)\nproject(kurn_ggml C CXX)\nadd_subdirectory(ggml)\n' > "$G/CMakeLists.txt"
fi
if [ -n "${KIT_WHEELS:-}" ]; then
  mkdir -p "$STAGE/wheels/kurn-gpu-check/wheels" && cp "$KIT_WHEELS"/*.whl "$STAGE/wheels/kurn-gpu-check/wheels/"
fi
big=$(find "$STAGE" -type f -size +1000k)
if [ -n "$big" ]; then echo "files over 1 MB: $big" >&2; exit 1; fi
rm -f "$OUTDIR/$NAME.zip" "$OUTDIR/$NAME"-part-*.zip "$OUTDIR/$NAME"-ggml-*.zip "$OUTDIR/$NAME-wheels.zip"
(cd "$STAGE" && zip -qr -X "$OUTDIR/$NAME.zip" kurn-gpu-check)
if [ "$(stat -c %s "$OUTDIR/$NAME.zip")" -gt 1000000 ]; then echo "$NAME.zip is over 1 MB" >&2; exit 1; fi
if [ -d "$STAGE/ggml" ]; then  # self-contained parts: the fewest that each stay under 1 MB
  for n in 2 3 4 5 6 8; do
    rm -f "$OUTDIR/$NAME"-ggml-*.zip
    (cd "$STAGE/ggml" && find kurn-gpu-check -type f | sort > ../files.txt && rm -f ../part-* && split -n "l/$n" -d ../files.txt ../part- &&
     for p in ../part-*; do zip -qr -X "$OUTDIR/$NAME-ggml-${p##*-}.zip" -@ < "$p"; done)
    ok=1; for z in "$OUTDIR/$NAME"-ggml-*.zip; do [ "$(stat -c %s "$z")" -lt 1000000 ] || ok=0; done
    [ "$ok" = 1 ] && break
  done
fi
[ -d "$STAGE/wheels" ] && (cd "$STAGE/wheels" && zip -qr -X "$OUTDIR/$NAME-wheels.zip" kurn-gpu-check)
ls -la "$OUTDIR/$NAME"*.zip
echo "files: $(cd "$STAGE" && find . -type f | wc -l), largest: $(find "$STAGE" -type f -printf '%s %P\n' | sort -n | tail -1)"
rm -rf "$STAGE"
