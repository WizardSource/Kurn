#!/usr/bin/env bash
# Build a private libggml-cpu.so with the kurn attn hook (kattn_hook.inc) from the baseline
# llama.cpp tree (same commit, same local AMX fix, same flags) without touching that tree.
# Stock llama.cpp binaries pick it up through LD_LIBRARY_PATH (they use RUNPATH):
#   LD_LIBRARY_PATH=$OUT/build/bin KURN_ATTN_LIB=/path/attn_f16_d128.so llama-bench -fa 1 ...
set -euo pipefail
SRC=${LLAMA_SRC:-$HOME/src/llama.cpp}
OUT=${OUT:-/tmp/llama-kattn}
HERE=$(cd "$(dirname "$0")" && pwd)
rm -rf "$OUT/src"
mkdir -p "$OUT/src"
tar -C "$SRC" --exclude=./build --exclude=./.git --exclude=./models -cf - . | tar -C "$OUT/src" -xf -
OPS="$OUT/src/ggml/src/ggml-cpu/ops.cpp"
cp "$HERE/kattn_hook.inc" "$OUT/src/ggml/src/ggml-cpu/kattn_hook.inc"
python3 - "$OPS" <<'EOF'
import sys
p = sys.argv[1]
s = open(p).read()
anchor = "void ggml_compute_forward_flash_attn_ext(\n        const ggml_compute_params * params,\n        ggml_tensor * dst) {\n"
assert s.count(anchor) == 1, "flash_attn_ext entry point not found"
s = s.replace(anchor, '#include "kattn_hook.inc"\n\n' + anchor + "    if (kurn_attn::try_run(params, dst)) {\n        return;\n    }\n")
open(p, "w").write(s)
EOF
cmake -S "$OUT/src" -B "$OUT/build" -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DGGML_NATIVE=ON \
    -DCMAKE_C_FLAGS=-mno-avx512fp16 -DCMAKE_CXX_FLAGS=-mno-avx512fp16 -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_OPENSSL=OFF >/dev/null
cmake --build "$OUT/build" --target ggml-cpu -j "${JOBS:-3}" >/dev/null
ls -l "$OUT/build/bin/libggml-cpu.so"
