#!/usr/bin/env bash
# kurn memory system, llama.cpp side (tested on 4ebdf2c, after ../apply.sh, which installs ggml-kurn-mem into ggml).
#   memsys/apply.sh LLAMA_CPP_DIR
# - llama-memsys.patch:
#   - tags: KV cache buffers -> kv, graph compute buffers -> compute, weight buffers -> weights (the log line and the
#     summary say what backs each: hugetlb 1G / 2M, THP, 4K)
#   - KV placement on multi-node hosts: with several streams (llama-server slots, unified KV off) stream s of every
#     layer's cache lives on node s % n_nodes, and kurn attention runs that stream on the node's threads
#   - common args: --hugepages auto|1g|2m|thp|4k|off (GGML_KURN_HUGEPAGES), --kurn-numa auto|on|off (GGML_KURN_NUMA);
#     ggml's --numa turns kurn's NUMA placement off unless --kurn-numa on / GGML_KURN_NUMA=on
#   - llama-bench --energy: joules per repetition and tokens per joule (CPU RAPL, GPU NVML; "unavailable" where the
#     host has no counters or they do not advance) and a `kurn mem` line per test on stderr
# - copies kurn-energy.h next to llama-bench.cpp
# Idempotent. Rebuild with: cmake --build build -j
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
L=$(cd "$1" && pwd)
if [ ! -f "$L/ggml/include/ggml-kurn-mem.h" ]; then
  echo "run integration/llama.cpp/apply.sh first (it installs ggml-kurn-mem into ggml)" >&2
  exit 1
fi
cp "$HERE/kurn-energy.h" "$L/tools/llama-bench/"
if grep -q "ggml_kurn_mem_tag_scope" "$L/src/llama-kv-cache.cpp"; then
  echo "llama-memsys.patch already applied"
else
  git -C "$L" apply "$HERE/llama-memsys.patch"
fi
echo "kurn memsys (llama.cpp side) applied to $L; rebuild with: cmake --build build -j"
