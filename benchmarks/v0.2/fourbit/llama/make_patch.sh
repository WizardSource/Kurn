#!/usr/bin/env bash
# Apply the fourbit measurement hook to a clean llama.cpp checkout (4ebdf2c) and write
# ggml-kurn-fourbit.patch next to this script.
#   make_patch.sh LLAMA_CPP_DIR [config.json]
# Same hook as integration/llama.cpp (kurn_hook.h: GGML_KURN=1, weights in the plain CPU
# buffer, ne11 <= 8), with the formats and configs of config.json.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
INTEG=$(cd "$HERE/../../../../integration/llama.cpp" && pwd)
L=$1; CFG=${2:-$HERE/config.json}
CPU=$L/ggml/src/ggml-cpu
PY=${PY:-/opt/kenv/bin/python}
rm -f "$CPU"/kurn_*.c
"$PY" "$HERE/gen_fourbit_sources.py" "$CPU" --config "$CFG"
sed 's|        case GGML_TYPE_Q2_0: return 64;|&\n        case GGML_TYPE_NVFP4: return 64;|' "$INTEG/kurn_hook.h" > "$CPU/kurn_hook.h"
grep -q GGML_TYPE_NVFP4 "$CPU/kurn_hook.h"
"$PY" - "$L" <<'EOF'
import sys
L = sys.argv[1]
p = f"{L}/ggml/src/ggml-cpu/CMakeLists.txt"
s = open(p).read()
if "KURN_SOURCES" not in s:
    a = """    elseif (GGML_SYSTEM_ARCH STREQUAL "x86")
        message(STATUS "x86 detected")
        list(APPEND GGML_CPU_SOURCES"""
    assert a in s
    s = s.replace(a, """    elseif (GGML_SYSTEM_ARCH STREQUAL "x86")
        message(STATUS "x86 detected")
        file(GLOB KURN_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/ggml-cpu/kurn_*.c")  # kurn kernels (AVX-512 VNNI builds use them)
        list(APPEND GGML_CPU_SOURCES ${KURN_SOURCES})
        list(APPEND GGML_CPU_SOURCES""", 1)
    open(p, "w").write(s)
p = f"{L}/ggml/src/ggml-cpu/ggml-cpu.c"
s = open(p).read()
if "kurn_hook.h" not in s:
    anchor = "static void ggml_compute_forward_mul_mat_one_chunk("
    s = s.replace(anchor, """#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)
#include "kurn_hook.h"
#endif

""" + anchor, 1)
    call = """    ggml_barrier(params->threadpool);

#if GGML_USE_LLAMAFILE"""
    assert call in s
    s = s.replace(call, """    ggml_barrier(params->threadpool);

#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)
    if (kurn_mul_mat(params, dst, (src1->type == vec_dot_type) ? src1->data : params->wdata)) {
        return;
    }
#endif

#if GGML_USE_LLAMAFILE""", 1)
    open(p, "w").write(s)
EOF
for f in "$CPU"/kurn_*.c; do sed -i '1i #if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)' "$f"; echo '#endif' >> "$f"; done
cd "$L" && git add -N ggml/src/ggml-cpu/kurn_* && git diff -- ggml/src/ggml-cpu/CMakeLists.txt ggml/src/ggml-cpu/ggml-cpu.c ggml/src/ggml-cpu/kurn_* > "$HERE/ggml-kurn-fourbit.patch"
echo "wrote $HERE/ggml-kurn-fourbit.patch ($(wc -l < "$HERE/ggml-kurn-fourbit.patch") lines)"
