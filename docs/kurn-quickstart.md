# kurn quickstart (merged tree, 0.3.0.dev3 + attention, KV-format, verify-width and GPU series)

Every command below was run in this order on a fresh checkout on 2026-10-06: a new directory, a new venv, a fresh llama.cpp clone, and models downloaded from Hugging Face. The machine was an 8-vCPU Emerald Rapids KVM guest with Ubuntu 24.04, gcc 13.3 and cmake 3.28. Step 7 (GPU) could only be dry-run, because this machine has no GPU.

## 1. Prerequisites

- **CPU:** x86-64 with AVX-512 F/BW and AVX-512 VNNI. kurn's llama.cpp buffer type and its engine need these. AVX-512 BF16 and AMX are optional; they are only used by the attention kernel and by opt-in paths.
  ```sh
  grep -o -w -E 'avx512f|avx512bw|avx512_vnni|avx512_bf16|amx_tile' /proc/cpuinfo | sort -u
  ```
- **Toolchain:** gcc ≥ 12 (13.3 tested), cmake ≥ 3.21, git, Python ≥ 3.9 with venv (`sudo apt install python3-venv build-essential cmake git`).
- **Sapphire, Emerald or Granite Rapids:** always build llama.cpp with `-mno-avx512fp16`. Native builds without it hit a ggml AVX512-FP16 bug that turns output into garbage (see `/workspace/llama-cpp-fp16-bug.md`).

## 2. Build kurn

```sh
mkdir kurn-work && cd kurn-work
unzip /path/to/kurn-merged.zip            # artifacts/kurn-merged.zip -> ./Kurn-gpu
cd Kurn-gpu && python3 -m venv .venv && . .venv/bin/activate
pip install -e '.[gguf]' huggingface_hub
kurn --version && kurn targets           # lists what this host can build and run
cd ..
```

Optional full test suite: `pip install pytest pytest-xdist numpy && pytest -q -n 8`. It takes about 9 minutes on 8 vCPUs; result here was 9,205 passed, 35 skipped. nvcc, an aarch64 cross gcc and qemu-user un-skip the GPU-compile and NEON tests.

## 3. Build llama.cpp with the KURN buffer type (pinned commit `4ebdf2c`)

```sh
git clone https://github.com/ggml-org/llama.cpp llama-kurn && git -C llama-kurn checkout 4ebdf2c
Kurn-gpu/integration/llama.cpp/apply.sh llama-kurn                # KURN buffer type (weights repacked at load)
Kurn-gpu/integration/llama.cpp/spec-width/apply.sh llama-kurn     # verify-width selector for speculative decoding
cd llama-kurn
CC=gcc CXX=g++ cmake -B build -DGGML_NATIVE=ON -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_FLAGS=-mno-avx512fp16 -DCMAKE_CXX_FLAGS=-mno-avx512fp16
cmake --build build -j"$(nproc)"      # ~2 min for the main targets on 8 vCPUs
cd ..
```

No runtime flag is needed: KURN registers ahead of ggml's AMX and CPU_REPACK buffers and takes every weight it has kernels for. Those formats are Q8_0, Q4_0, IQ4_NL, Q4_K, Q2_0, TQ2_0 and Q1_0, plus MXFP4 and NVFP4. Other types fall back to ggml. For example, the Q6_K tensors inside a Q4_K_M file go to ggml.

## 4. Chat with a GGUF model

```sh
hf download Qwen/Qwen3-1.7B-GGUF Qwen3-1.7B-Q8_0.gguf --local-dir models
hf download Qwen/Qwen3-0.6B-GGUF Qwen3-0.6B-Q8_0.gguf --local-dir models   # draft model for step 4c
B=llama-kurn/build/bin
```

**a. Interactive chat / one-shot:**
```sh
$B/llama-cli -m models/Qwen3-1.7B-Q8_0.gguf -t 8 -fa on -ctk q8_0 -ctv q8_0          # interactive
$B/llama-cli -m models/Qwen3-1.7B-Q8_0.gguf -t 8 -fa on -p "Name three primary colors. /no_think" -n 64 -st --simple-io
```

**b. OpenAI-compatible server:**
```sh
$B/llama-server -m models/Qwen3-1.7B-Q8_0.gguf -t 8 -fa on -ctk q8_0 -ctv q8_0 -c 8192 --port 8088 &
curl -s localhost:8088/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Say hello in French. /no_think"}],"max_tokens":32,"temperature":0}'
```

**KV cache formats.** In llama.cpp you get ggml's formats only: `-ctk/-ctv f16|q8_0|q4_0` (quantized KV needs `-fa on`). **kurn's k4c KV formats are not available in llama-cli or llama-server**, because ggml has no matching KV type. Today k4c runs only in kurn's own engine (step 5).

**c. Speculative decoding with the verify-width selector.** First measure the cost table, once per machine × model pair × thread count:
```sh
KURN_CALIB_OUT=qwen3-1.7b $B/kurn-spec-calib -m models/Qwen3-1.7B-Q8_0.gguf -md models/Qwen3-0.6B-Q8_0.gguf \
  -p "Write a Python function that checks whether a string is a palindrome." -n 128 -t 8 -td 8      # writes qwen3-1.7b.cost
```
Then run speculative decoding through it:
```sh
KURN_SPEC_WIDTH=qwen3-1.7b.cost $B/llama-speculative-simple -m models/Qwen3-1.7B-Q8_0.gguf \
  -md models/Qwen3-0.6B-Q8_0.gguf --spec-type draft-simple --spec-draft-n-max 15 \
  -p "Write a Python function that checks whether a string is a palindrome." -n 128 --temp 0 -t 8 -td 8 -fa on
```
- The log shows `kurn spec width: qwen3-1.7b.cost, k <= 15, mode policy` and a `verify widths (M:steps)` histogram.
- On this prompt it gave 47.8 tok/s with the selector, against 23.6 tok/s for a fixed k=15 draft.
- `--spec-type draft-simple` is required; without it the run fails with "failed to initialize speculative decoding".
- The selector only exists in `llama-speculative-simple`. llama-server and llama-cli can draft with fixed settings (`-md ... --spec-draft-n-max N`) but do not use the selector.

## 5. kurn's own engine (whole-model decode program)

```sh
. Kurn-gpu/.venv/bin/activate
kurn model models/Qwen3-1.7B-Q8_0.gguf -o qwen3-1.7b-engine        # ~6 s; MAX_CTX 2048
# longer context: python -c "from kurn.model.compile_model import compile_model; \
#   compile_model('models/Qwen3-1.7B-Q8_0.gguf', out='qwen3-1.7b-engine-8k', defines=('MAX_CTX=8192',))"
```

The engine takes and prints token ids. This small wrapper does the chat template and the detokenizing:
```sh
pip install tokenizers && hf download Qwen/Qwen3-1.7B tokenizer.json --local-dir models/qwen3-tok
cat > chat_engine.py <<'EOF'
import subprocess, sys
from tokenizers import Tokenizer
tok = Tokenizer.from_file("models/qwen3-tok/tokenizer.json")
prompt = f"<|im_start|>user\n{sys.argv[1]}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
ids = ",".join(map(str, tok.encode(prompt, add_special_tokens=False).ids))
out = subprocess.run(["./qwen3-1.7b-engine", "models/Qwen3-1.7B-Q8_0.gguf", "gen", "8", "64", ids],
                     capture_output=True, text=True).stdout
print(tok.decode([int(t) for t in out.split("gen:")[1].split("\n")[0].split()]))
print(out.strip().splitlines()[-1].split(" barriers")[0])
EOF
python chat_engine.py "Name three primary colors."                    # F16 KV
KURN_KV=k4c_q4 python chat_engine.py "Name three primary colors."     # k4c KV (also k4c_q8, q8_0, q4_0)
```

- Perplexity: `./qwen3-1.7b-engine MODEL ppl 8 CTX tokens.txt` (space-separated ids; `PPL_CHUNKS=n`).
- KL against an F16-KV run: `KURN_KLD=ref:FILE` for the reference, then `KURN_KLD=cmp:FILE` for the run being compared (`benchmarks/v0.2/kvformat/run_ppl.sh` does both).

## 6. Confirming kurn kernels are in use

1. **Load log.** Add `-v` to llama-cli or llama-server. Look for `KURN model buffer size = 1802.10 MiB` and `common_memory_breakdown_print: | - KURN`. With `GGML_KURN=0` the same model shows `AMX model buffer size` instead. `GGML_KURN_VERBOSE=1` logs every repacked tensor and every fallback decision.
2. **A/B switch.** `GGML_KURN=0` runs stock ggml paths in the same binary. Compare `llama-bench -m MODEL -p 512 -n 64` with and without it.
3. **Kernel self-test** (correctness and batch invariance of every KURN format):
   ```sh
   L=$PWD/llama-kurn
   gcc -O2 -I$L/ggml/include Kurn-gpu/integration/llama.cpp/test_kurn_buft.c -L$L/build/bin \
     -lggml -lggml-base -lggml-cpu -lm -Wl,-rpath,$L/build/bin -o test_kurn_buft
   ./test_kurn_buft smoke        # here: "smoke: 105 passed, 0 failed"; exit code = failures
   ```

## 7. A100 GPU attention check kit

```sh
unzip kurn-gpu-check-attn.zip && cd kurn-gpu-check && ./run_attn_check.sh     # QUICK=1 ~5-10 min; full ~20-30 min
```

- Needs an NVIDIA driver, CUDA 11.x/12.x (12.8+ also builds sm_120), python3 ≥ 3.9 and g++. It installs nothing.
- Output is `kurn-attn-results-<host>-<date>.tar.gz` with `attn_report.md`. Exit status 0 means every check passed. Copy the tarball into the artifacts folder's `artifacts/`.
- `DRYRUN=1` runs without a GPU (compile + CPU emulator only). That is all that could be run here: `QUICK=1 DRYRUN=1 ./run_attn_check.sh` on the published zip built 150 configs with 0 failures, the emulator numerics passed, and it exited 0 after 11 min.
- The same script is in the merged tree as `contrib/gpu-check/run_attn_check.sh`.

## Not production-ready yet

- **k4c KV:** only in kurn's engine. There is no llama.cpp/ggml path, so llama-server cannot use it. The engine's quantized KV formats need threads ≤ KV heads (8 on Qwen3). k4c_q4 saves 1.8x KV memory over Q8_0 but decodes about 11–14% slower on a quiet host.
- **Verify-width selector:** only in `llama-speculative-simple`, not in llama-server. The cost table is per machine, model pair and thread count. About 30% of speculative runs produce text that differs from plain greedy decoding; this is under investigation in the [CPU comparison](kurn-cpu-comparison.md).
- **The engine is a research tool:**
  - Qwen3 / OLMoE only, with all-Q8_0 or all-Q4_0 (`--pure`) weights; no Q4_K_M, no Llama.
  - Greedy decoding only, token ids in and out.
  - The prompt is processed one token at a time (no batched prefill).
  - 2048-token context unless built with `MAX_CTX`.
- **AMX paths are opt-in:**
  - `GGML_KURN_AMX=1` turns on Q8_0 prefill through AMX.
  - Packed K/V `kattn_pack` is an API only.
  - On some VMs, AMX tile state is not preserved across context switches; check a host with `integration/llama.cpp/tools/amx_check.sh`.
  - kurn's attention kernel is not wired into llama.cpp. The KURN buffer type covers matmuls only; there is a measurement hook in `benchmarks/v0.2/attn/llama/`.
- **GPU:** the attention op is verified on the CPU warp emulator only and has never run on silicon (the A100 kit run is pending). sm_120 / Blackwell is compile-only. There is no llama.cpp CUDA integration.
- **Packaging:** no wheels or releases. llama.cpp support is a patch set pinned to `4ebdf2c`; newer commits may need `apply.sh` fixes.

## 8. llama-server with kurn attention, k4c KV and the verify-width selector

Requires the [llama-integration patches](../artifacts/kurn-patches/llama-integration/) applied to the kurn tree (`git am` onto `kurn-merged.zip`). Verified on 2026-10-06 with a fresh llama.cpp clone, Qwen3-1.7B / Qwen3-0.6B / Qwen3-8B Q8_0, on the same 8-vCPU Emerald Rapids VM. The commands below run from the same directory as steps 3 and 4, with `B=llama-kurn/build/bin`.

This section replaces some items under "Not production-ready yet" above:
- k4c KV now works in llama.cpp.
- The selector now works in llama-server.
- kurn's attention is wired into llama.cpp, and `kattn_pack` is used there.
- The engine's 2048-token limit is gone.

**Build.** Same as step 3, plus two more scripts. The order matters: `apply.sh` comes first.
```sh
git clone https://github.com/ggml-org/llama.cpp llama-kurn && git -C llama-kurn checkout 4ebdf2c
Kurn-gpu/integration/llama.cpp/apply.sh llama-kurn              # KURN buffer type + kurn attention (FLASH_ATTN_EXT)
Kurn-gpu/integration/llama.cpp/spec-width/apply.sh llama-kurn   # verify-width selector (speculative-simple + llama-server)
Kurn-gpu/integration/llama.cpp/k4c/apply.sh llama-kurn          # k4c KV cache type
cd llama-kurn && CC=gcc CXX=g++ cmake -B build -DGGML_NATIVE=ON -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_FLAGS=-mno-avx512fp16 -DCMAKE_CXX_FLAGS=-mno-avx512fp16 && cmake --build build -j"$(nproc)" && cd ..   # ~4 min
```

**a. Attention.** No flag is needed. Flash attention is on by default on CPU, and every FA node kurn supports runs on kurn:
- KV: F16, BF16, Q8_0 or k4c.
- Head dims: 64, 128, 256 and 576/512.
- No ALiBi, softcap or sinks; anything else falls back to ggml.

To check it, run `GGML_KURN_VERBOSE=1 $B/llama-completion ...`. At exit it prints `kurn fa: N FLASH_ATTN_EXT nodes on kurn, 0 on ggml`; `GGML_KURN_FA=0` gives stock ggml.

| Variable | Effect |
|---|---|
| `GGML_KURN_FA_MODE=exact` | Batch-invariant attention. Speculative and cached outputs equal plain decoding bit for bit. Decode is slower at long context (tg64 at depth 4096 on 1.7B: 25.5 vs 46.5 tok/s; ggml FA 22.0). |
| `GGML_KURN_AMX=1` | AMX tile engine. pp512 at depth 4096 is 2.62x ggml's FA instead of 1.81x. It carries the AMX caveat of step 6. |

**b. k4c KV:**
```sh
$B/llama-server -m models/Qwen3-1.7B-Q8_0.gguf -t 8 -ctk k4c_q4 -c 8192 --port 8088 --slot-save-path slots &
# -ctk k4c_q4 is the same as -ctk k4c -ctv q4_0. k4c_q8 means -ctv q8_0.
```
- **Memory:** KV is 1.8x smaller than Q8_0.
- **Accuracy** on Qwen3-1.7B (WikiText-2, KL vs F16 KV):
  - k4c_q4: 0.016, +0.10% perplexity.
  - k4c_q8: 0.010.
  - llama.cpp's own `-ctk q4_0 -ctv q4_0`: 0.32, +29% perplexity.
- **What works:** prompt caching, `/slots/N?action=save|restore`, multiple slots and speculative decoding.
- **What doesn't:** context shift and `--cache-reuse`, because K-shift is off for k4c.
- **Requirements:** V must be q4_0 or q8_0, and the model must not use MLA.

**c. Speculative decoding with the selector in llama-server.** First measure the cost table, as in step 4c, using the same flags as the server:
```sh
KURN_CALIB_OUT=qwen3-8b $B/kurn-spec-calib -m models/Qwen3-8B-Q8_0.gguf -md models/Qwen3-0.6B-Q8_0.gguf \
  --spec-type draft-simple -p "Write a Python function that checks whether a string is a palindrome." -n 128 -t 8 -td 8
$B/llama-server -m models/Qwen3-8B-Q8_0.gguf -md models/Qwen3-0.6B-Q8_0.gguf --spec-type draft-simple \
  --spec-draft-n-max 15 --spec-width qwen3-8b.cost -t 8 -td 8 --port 8088 &
```
- Each request logs `kurn spec width: verify widths (M:steps) = ...`.
- On Qwen3-8B + 0.6B (8 prompts × 256 tokens) it gave 21.7 tok/s. No draft gave 13.6, and the best fixed draft length (k=7) gave 19.6.
- On Qwen3-1.7B + 0.6B, speculation does not pay, because the draft costs about 40% of a target step. The selector gave 54.7 tok/s against 55.5 without a draft; fixed k=3/7/15 gave 50.4/45.3/33.2.
- Add `GGML_KURN_FA_MODE=exact` to make speculative output identical to non-speculative output. On 1.7B, 0 of 32 runs differed, against 22 of 32 with ggml's attention.

**d. kurn's engine:** `kurn model ...` engines now size their context from the request, and `KURN_CTX=n` asks for more. No `MAX_CTX` rebuild is needed.

**Checks:**
```sh
gcc -O2 -Illama-kurn/ggml/include Kurn-gpu/integration/llama.cpp/k4c/test_k4c.c -L$B -lggml -lggml-base -lggml-cpu -lm \
  -Wl,-rpath,$PWD/$B -o test_k4c && ./test_k4c        # "k4c: 25 passed, 0 failed"
$B/test-backend-ops -o FLASH_ATTN_EXT -b CPU            # kurn vs ggml's reference: 5312/5312 passed
```
