#!/bin/bash
# KV cache entropy study: dump real llama.cpp KV caches (state files), measure their entropy under simple models and
# the blog's predictor scheme, and kvz's actual ratio and speed.
#   LLAMA=~/src/llama-kurn MODELS=~/models ./run.sh
# Needs a llama.cpp build with apply.sh, k4c/apply.sh and kvz/apply.sh, Qwen3-1.7B-Q8_0 / Qwen3-8B-Q8_0 /
# Qwen3-8B-Q4_K_M GGUFs (the 1.7B Q4_K_M predictor is requantized from Q8_0), python with numpy, scipy, zstandard.
# Writes dumps/ (~70 GB) and results/; results/summary.md has the tables.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
L=${LLAMA:?set LLAMA to the llama.cpp checkout}
M=${MODELS:?set MODELS to the GGUF directory}
export KV_DUMPS=${KV_DUMPS:-$PWD/dumps} KV_RESULTS=${KV_RESULTS:-$PWD/results}
mkdir -p "$KV_DUMPS" "$KV_RESULTS" corpus
export GGML_K4C_EXACT_MB=16384  # k4c keys in the state are the exact f16 values for every group

g++ -O2 -std=c++17 -I"$L/include" -I"$L/ggml/include" "$HERE/kvdump.cpp" -L"$L/build/bin" -lllama -lggml -lggml-base \
    -Wl,-rpath,"$L/build/bin" -o kvdump
g++ -O2 -march=native -std=c++17 -DKVZ_BENCH_ZSTD "$HERE/../../../integration/llama.cpp/kvz/kvz-bench.cpp" \
    "$HERE/../../../integration/llama.cpp/kvz/llama-kvz.cpp" -lzstd -lpthread -o kvz-bench
[ -f "$M/Qwen3-1.7B-Q4_K_M-rq.gguf" ] || "$L/build/bin/llama-quantize" --allow-requantize "$M/Qwen3-1.7B-Q8_0.gguf" "$M/Qwen3-1.7B-Q4_K_M-rq.gguf" Q4_K_M 8

# corpora: WikiText-2 (test: evaluation, train: calibration) and llama.cpp's own sources (code)
python3 - <<'EOF'
from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq
for split in ("test", "train"):
    p = hf_hub_download("Salesforce/wikitext", f"wikitext-2-raw-v1/{split}-00000-of-00001.parquet", repo_type="dataset")
    open(f"corpus/wiki-{split}.txt", "w").write("".join(pq.read_table(p).column("text").to_pylist()))
EOF
cat "$L"/src/llama-model.cpp "$L"/src/llama-context.cpp "$L"/src/llama-kv-cache.cpp "$L"/tools/server/server-context.cpp > corpus/code-eval.txt
cat "$L"/ggml/src/ggml.c "$L"/ggml/src/ggml-backend.cpp > corpus/code-calib.txt

dump() { # tag model corpus ntok ctk ctv
    [ -f "$KV_DUMPS/$1.seq" ] || ./kvdump "$M/$2" "corpus/$3.txt" "$4" "$5" "$6" "$KV_DUMPS/$1.seq" 8
}
for m in 1.7b:Qwen3-1.7B:32768 8b:Qwen3-8B:16384; do
    IFS=: read tag name n <<< "$m"
    for c in wiki-test code-eval; do
        dump $tag-$c-f16   $name-Q8_0.gguf $c $n f16 f16
        dump $tag-$c-q8    $name-Q8_0.gguf $c $n q8_0 q8_0
        dump $tag-$c-k4cq4 $name-Q8_0.gguf $c $n k4c q4_0
        dump $tag-$c-k4cq8 $name-Q8_0.gguf $c $n k4c q8_0
    done
    pred=$name-Q4_K_M.gguf; [ $tag = 1.7b ] && pred=$name-Q4_K_M-rq.gguf
    for c in wiki-test code-eval wiki-train code-calib; do
        dump $tag-$c-pred $pred $c 8192 f16 f16
    done
    for c in wiki-train code-calib; do
        dump $tag-$c-f16 $name-Q8_0.gguf $c 8192 f16 f16
    done
done
dump 8b32k-wiki-test-f16 Qwen3-8B-Q8_0.gguf wiki-test 32768 f16 f16

for d in "$KV_DUMPS"/{1.7b,8b}-{wiki-test,code-eval}-{f16,q8,k4cq4,k4cq8}.seq "$KV_DUMPS"/8b32k-wiki-test-f16.seq; do
    t=$(basename "$d" .seq)
    [ -f "$KV_RESULTS/$t.json" ] || python3 "$HERE/analyze.py" "$d" "$KV_RESULTS/$t.json"
    m=${t%%-*}; f=${t##*-}; c=${t#*-}; c=${c%-*}
    for p in 4096 8192 16384 32768; do
        [ "$m" = 8b ] && [ $p = 32768 ] && continue  # 8B dumps hold 16K tokens
        echo "== $m $c $f $p" >> "$KV_RESULTS/kvz-ratios.txt"
        ./kvz-bench "$d" 8 $p 3 >> "$KV_RESULTS/kvz-ratios.txt"
    done
done
for m in 1.7b 8b; do for c in wiki-test code-eval; do python3 "$HERE/predictor.py" $m $c; done; done
python3 "$HERE/summarize.py"
