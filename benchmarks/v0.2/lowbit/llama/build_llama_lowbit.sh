#!/usr/bin/env bash
# Local llama.cpp measurement build with the kurn lowbit hook (not the integration patch: that is
# WS-H's integration/llama.cpp/). Copies the mainline checkout (same revision + local fixes as the
# baseline build), inserts kurn_lowbit_hook.h into ggml-cpu's MUL_MAT after the activation barrier,
# generates the kernels and builds with the baseline's flags.
#   build_llama_lowbit.sh [variants.json]      -> $DST/build/bin (default DST=/tmp/kurn-lowbit/llama.cpp)
# Run untimed (nice); the copy keeps its own build directory.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
SRC=${LLAMA:-$HOME/src/llama.cpp}
DST=${DST:-/tmp/kurn-lowbit/llama.cpp}
CFG=${1:-}
mkdir -p "$DST"
find "$DST" -mindepth 1 -maxdepth 1 ! -name build -exec rm -rf {} +
(cd "$SRC" && tar cf - --exclude=./build --exclude=./.git .) | (cd "$DST" && tar xf -)
CPU=$DST/ggml/src/ggml-cpu
"${PYTHON:-/opt/kenv/bin/python}" "$HERE/gen_lowbit_sources.py" "$CPU" ${CFG:+--config "$CFG"}
cp "$HERE/kurn_lowbit_hook.h" "$CPU/"
"${PYTHON:-/opt/kenv/bin/python}" - "$DST" <<'EOF'
import sys
L = sys.argv[1]
p = f"{L}/ggml/src/ggml-cpu/CMakeLists.txt"
s = open(p).read()
anchor = """    elseif (GGML_SYSTEM_ARCH STREQUAL "x86")
        message(STATUS "x86 detected")
        list(APPEND GGML_CPU_SOURCES"""
assert anchor in s
s = s.replace(anchor, """    elseif (GGML_SYSTEM_ARCH STREQUAL "x86")
        message(STATUS "x86 detected")
        file(GLOB KURN_LOWBIT_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/ggml-cpu/kurn_lowbit_*.c")
        list(APPEND GGML_CPU_SOURCES ${KURN_LOWBIT_SOURCES})
        list(APPEND GGML_CPU_SOURCES""", 1)
open(p, "w").write(s)
p = f"{L}/ggml/src/ggml-cpu/ggml-cpu.c"
s = open(p).read()
anchor = "static void ggml_compute_forward_mul_mat_one_chunk("
assert anchor in s
s = s.replace(anchor, """#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)
#include "kurn_lowbit_hook.h"
#endif

""" + anchor, 1)
call = """    ggml_barrier(params->threadpool);

#if GGML_USE_LLAMAFILE"""
assert call in s
s = s.replace(call, """    ggml_barrier(params->threadpool);

#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)
    if (kurn_lowbit_mul_mat(params, dst, (src1->type == vec_dot_type) ? src1->data : params->wdata)) {
        return;
    }
#endif

#if GGML_USE_LLAMAFILE""", 1)
open(p, "w").write(s)
EOF
cmake -S "$DST" -B "$DST/build" -G Ninja -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DGGML_NATIVE=ON -DLLAMA_CURL=OFF \
    -DCMAKE_C_FLAGS=-mno-avx512fp16 -DCMAKE_CXX_FLAGS=-mno-avx512fp16 > "$DST/cmake.log"
cmake --build "$DST/build" -j "${JOBS:-3}" --target llama-bench llama-perplexity llama-cli llama-quantize > "$DST/build.log" 2>&1 \
    || { tail -40 "$DST/build.log"; exit 1; }
grep -c "warning" "$DST/build.log" || true
echo "$DST/build/bin"
