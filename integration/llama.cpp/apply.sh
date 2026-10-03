#!/usr/bin/env bash
# Add the KURN extra buffer type to a llama.cpp checkout (tested on 4ebdf2c).
#   apply.sh LLAMA_CPP_DIR [--config tuned.json] [--only q8_0,q4_0] [--patch OUT.patch]
# - copies ggml-kurn/kurn-buft.{cpp,h} into ggml/src/ggml-cpu/kurn/
# - generates the kurn kernels for every registry format ggml knows (gen_ggml_sources.py)
# - registers the buffer type first in ggml-cpu's extra buffer list and adds the sources to CMake
# Re-running regenerates the kernels (e.g. with a new --config); the edits are idempotent.
# --patch writes `git diff` of the checkout (relative to its HEAD) to OUT.patch.
# Needs a Python with kurn importable: $KURN_PYTHON (default: /opt/kenv/bin/python, else python3),
# with this repository's kurn/src first on PYTHONPATH.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
L=$(cd "$1" && pwd); shift
GEN_ARGS=(); PATCH=""
while [ $# -gt 0 ]; do
  case $1 in
    --config|--only) GEN_ARGS+=("$1" "$2"); shift 2;;
    --patch) PATCH=$2; shift 2;;
    *) echo "unknown argument $1" >&2; exit 2;;
  esac
done
PY=${KURN_PYTHON:-$( [ -x /opt/kenv/bin/python ] && echo /opt/kenv/bin/python || echo python3 )}
CPU=$L/ggml/src/ggml-cpu
mkdir -p "$CPU/kurn"
cp "$HERE/ggml-kurn/kurn-buft.cpp" "$HERE/ggml-kurn/kurn-buft.h" "$CPU/kurn/"
PYTHONPATH="$HERE/../../src${PYTHONPATH:+:$PYTHONPATH}" "$PY" "$HERE/gen_ggml_sources.py" "$CPU/kurn" \
  --ggml-h "$L/ggml/include/ggml.h" "${GEN_ARGS[@]}"
"$PY" - "$L" <<'EOF'
import sys
L = sys.argv[1]
p = f"{L}/ggml/src/ggml-cpu/CMakeLists.txt"
s = open(p).read()
if "ggml-cpu/kurn" not in s:
    anchor = "    ggml_add_backend_library(${GGML_CPU_NAME})\n"
    assert anchor in s, "CMakeLists anchor not found"
    s = s.replace(anchor, anchor + """
    # kurn extra buffer type (kurn/integration/llama.cpp); compiled out without AVX-512 VNNI
    file(GLOB GGML_CPU_KURN_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/ggml-cpu/kurn/*.c" "${CMAKE_CURRENT_SOURCE_DIR}/ggml-cpu/kurn/*.cpp")
    list (APPEND GGML_CPU_SOURCES ${GGML_CPU_KURN_SOURCES})
""", 1)
    open(p, "w").write(s)
elif "GLOB GGML_CPU_KURN_SOURCES CONFIGURE_DEPENDS" not in s:  # checkouts patched before CONFIGURE_DEPENDS
    s = s.replace("file(GLOB GGML_CPU_KURN_SOURCES ", "file(GLOB GGML_CPU_KURN_SOURCES CONFIGURE_DEPENDS ", 1)
    open(p, "w").write(s)
p = f"{L}/ggml/src/ggml-cpu/ggml-cpu.cpp"
s = open(p).read()
if "kurn-buft.h" not in s:
    inc = '#include "traits.h"\n'
    assert inc in s, "ggml-cpu.cpp include anchor not found"
    s = s.replace(inc, inc + '#include "kurn/kurn-buft.h"\n', 1)
    anchor = "        std::vector<ggml_backend_buffer_type_t> bufts;\n\n#if defined(__AMX_INT8__) && defined(__AVX512VNNI__)\n"
    assert anchor in s, "extra buffer list anchor not found"
    s = s.replace(anchor, """        std::vector<ggml_backend_buffer_type_t> bufts;

        // first: weights kurn has kernels for go to KURN; the rest falls through to AMX / CPU_REPACK
        if (ggml_backend_cpu_kurn_buffer_type()) {
            bufts.push_back(ggml_backend_cpu_kurn_buffer_type());
        }

#if defined(__AMX_INT8__) && defined(__AVX512VNNI__)
""", 1)
    open(p, "w").write(s)
EOF
if [ -n "$PATCH" ]; then
  (cd "$L" && git add -N ggml/src/ggml-cpu/kurn && git diff HEAD -- ggml) > "$PATCH"
  echo "wrote $PATCH ($(wc -l < "$PATCH") lines)"
fi
echo "kurn buffer type applied to $L; rebuild with: cmake --build build -j"
