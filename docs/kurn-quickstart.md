# kurn quickstart (0.3.0.dev3 full tree)

This checkout is 0.3.0.dev3 plus, in this order:

- **merged:** attention, KV formats, verify width, GPU phase 1;
- **llama-integration:** kurn attention, k4c KV and the selector in llama.cpp;
- **amx-prefill:** AMX-BF16 prefill and AMX attention by default;
- **verify-kernels:** exact-width prefetching verify kernels and native Q6_K/Q5_K kernels;
- **gpu-blackwell:** sm_100/sm_120 tiers, FP8 KV, MXFP4/NVFP4 on the GPU;
- **0058:** GPU builds find the CUDA runtime even when it lives outside nvcc's toolkit (split installs), with a fail-fast preflight.
- **0059–0061:** GPU attention A100 phase 2. Q8_0 KV dequantizes in registers, splits are short, MLA runs on 4 warps, and BF16 q is split hi + lo. This is driven by the first A100 run; speed still needs the A100.
- **0062:** GPU kit v2. `run_kit.sh` runs everything in one command and writes one report, with llama.cpp / FlashInfer / FlashAttention-2 attention baselines.

Each step below was run in order on 2026-10-07, on a fresh directory with a new venv, a fresh llama.cpp clone and models downloaded from Hugging Face. The machine was an 8-vCPU Emerald Rapids KVM guest (AMX, no GPU) with Ubuntu 24.04, gcc 13.3 and cmake 3.28. Step 7 ran as a dry run: CUDA 12.8 compilers, no GPU.

Against llama.cpp `4ebdf2c`, the three `apply.sh` scripts in step 3 are the supported integration path.

## 1. Prerequisites

- **CPU:** x86-64 with AVX-512 F/BW and AVX-512 VNNI. kurn's llama.cpp buffer type and its engine need these.
  ```sh
  grep -o -w -E 'avx512f|avx512bw|avx512_vnni|avx512_bf16|amx_tile' /proc/cpuinfo | sort -u
  ```
  With AMX (Sapphire Rapids and later), prefill and attention use AMX by default; see step 6.
- **Toolchain:** gcc ≥ 12 (13.3 tested), cmake ≥ 3.21, git, Python ≥ 3.9 with venv (`sudo apt install python3-venv build-essential cmake git`).
- **Sapphire, Emerald or Granite Rapids:** always build llama.cpp with `-mno-avx512fp16`. Native builds without it hit a ggml AVX512-FP16 bug that turns output into garbage.
- **Optional, for the GPU kit dry run without a GPU (step 7):** CUDA ≥ 12.8. Only the compilers are needed; the full toolkit works as well. On Ubuntu 24.04, from NVIDIA's apt repository:
  ```sh
  wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
  sudo dpkg -i cuda-keyring_1.1-1_all.deb && sudo apt-get update
  sudo apt-get install cuda-nvcc-12-8 cuda-cudart-dev-12-8 cuda-crt-12-8 cuda-cuobjdump-12-8 libcublas-dev-12-8
  export PATH=/usr/local/cuda-12.8/bin:$PATH
  ```
  The kit needs `cuobjdump`, and the GEMM harness needs `cublas_v2.h`.
- **CUDA runtime outside the toolkit (split install).** nvcc alone is not enough: the builds need `cuda_runtime.h`, `cuda_fp16.h` and `libcudart` (and `cublas_v2.h` / `libcublas` for the GEMM harness).
  - Some boxes keep these outside the toolkit dir, e.g. nvcc in `/usr/local/cuda-12.8` with the headers and libraries in a monorepo third-party dir that has `include_no_implicit/` and `lib/`.
  - Both kit scripts start with a preflight (`kurn gpu doctor`). It compiles, links and (on a GPU) runs a tiny `.cu` with the exact nvcc and flags the builds use.
  - If that fails, it stops in seconds with the paths it searched, instead of failing hundreds of builds. Point kurn at the runtime:
  ```sh
  KURN_CUDA_INCLUDE=/path/to/cuda/include_no_implicit KURN_CUDA_LIB=/path/to/cuda/lib QUICK=1 ./run_gpu_check.sh
  ```
  - Both variables take `:`-separated lists.
  - `CUDA_HOME=/path/to/cuda` also works: kurn probes `include/`, `include_no_implicit/`, `lib64/` and `lib/` under it.
  - So does `NVCC_APPEND_FLAGS="-I... -L..."`. It is passed through to nvcc unchanged; kurn only adds an rpath.
  - `KURN_NVCC` picks the compiler. `KURN_CUDA_NO_PROBE=1` stops kurn from picking up headers from another installed CUDA version.
  - Check without building anything: `PYTHONPATH=src python3 -m kurn gpu doctor --cublas`.

## 2. Build kurn

```sh
git clone https://github.com/WizardSource/Kurn.git && cd Kurn
python3 -m venv .venv && . .venv/bin/activate
pip install -e '.[gguf]' huggingface_hub
kurn --version && kurn targets           # "kurn 0.3.0.dev3", then what this host can build and run
```

Optional full test suite:

```sh
pip install pytest pytest-xdist numpy
KURN_LLAMA_CPP=$PWD/../llama-kurn pytest -q -n 8   # after step 3, so the llama.cpp tests run too
```

- Result here, with CUDA 12.8 on `PATH`: **11,995 passed, 62 skipped, 0 failed**, in about 10 minutes on 8 vCPUs.
- Without nvcc, the GPU-compile tests skip: 11,950 passed, 107 skipped.
- An aarch64 cross gcc and qemu-user also un-skip the NEON tests.

## 3. Build llama.cpp (pinned commit `4ebdf2c`)

The three scripts add, in this order:

1. the KURN buffer type (weights repacked at load; verify kernels; native Q6_K/Q5_K) and kurn attention as `FLASH_ATTN_EXT`;
2. the verify-width selector (`llama-speculative-simple`, `llama-server`, `kurn-spec-calib`);
3. the k4c KV cache type.

```sh
git clone https://github.com/ggml-org/llama.cpp llama-kurn && git -C llama-kurn checkout 4ebdf2c
integration/llama.cpp/apply.sh llama-kurn
integration/llama.cpp/spec-width/apply.sh llama-kurn
integration/llama.cpp/k4c/apply.sh llama-kurn
cd llama-kurn
CC=gcc CXX=g++ cmake -B build -DGGML_NATIVE=ON -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_FLAGS=-mno-avx512fp16 -DCMAKE_CXX_FLAGS=-mno-avx512fp16
cmake --build build -j"$(nproc)"      # 138 s here, 0 warnings
cd ..
```

- **Run the scripts with the venv active**, so the kernel generators can import kurn.
- **Alternative:** `git -C llama-kurn apply /path/to/llama.cpp-kurn.patch` on the fresh `4ebdf2c` checkout gives the same tree.
- **Optional per-node profiler** (`GGML_OPPROF=1`): `git -C llama-kurn apply integration/llama.cpp/tools/opprof.patch`. It is not part of the scripts or the combined patch.

No runtime flag is needed. KURN registers ahead of ggml's AMX and CPU_REPACK buffers and takes every weight it has kernels for:

- Q8_0, Q4_0, IQ4_NL, Q4_K, Q2_0, TQ2_0, Q1_0, MXFP4 and NVFP4, repacked;
- Q6_K and Q5_K, kept in ggml's layout (AMX CPUs).

Other types fall back to ggml.

## 4. Chat with a GGUF model

```sh
hf download Qwen/Qwen3-1.7B-GGUF Qwen3-1.7B-Q8_0.gguf --local-dir models
hf download Qwen/Qwen3-0.6B-GGUF Qwen3-0.6B-Q8_0.gguf --local-dir models   # draft model for 4c and 4d
B=llama-kurn/build/bin
```

**a. Interactive chat / one-shot:**
```sh
$B/llama-cli -m models/Qwen3-1.7B-Q8_0.gguf -t 8                     # interactive
$B/llama-cli -m models/Qwen3-1.7B-Q8_0.gguf -t 8 -p "Name three primary colors. /no_think" -n 64 -st --simple-io
```
Here: prompt 491 tok/s, generation 61 tok/s.

**b. OpenAI-compatible server:**
```sh
$B/llama-server -m models/Qwen3-1.7B-Q8_0.gguf -t 8 -ctk q8_0 -ctv q8_0 -c 8192 --port 8088 &
curl -s localhost:8088/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Say hello in French. /no_think"}],"max_tokens":32,"temperature":0}'
```
This returns `"Bonjour!"`.

**c. Speculative decoding with the verify-width selector in `llama-speculative-simple`.** First measure the cost table, once per machine × model pair × thread count:
```sh
KURN_CALIB_OUT=qwen3-1.7b $B/kurn-spec-calib -m models/Qwen3-1.7B-Q8_0.gguf -md models/Qwen3-0.6B-Q8_0.gguf \
  --spec-type draft-simple -p "Write a Python function that checks whether a string is a palindrome." -n 128 -t 8 -td 8
KURN_SPEC_WIDTH=qwen3-1.7b.cost $B/llama-speculative-simple -m models/Qwen3-1.7B-Q8_0.gguf \
  -md models/Qwen3-0.6B-Q8_0.gguf --spec-type draft-simple --spec-draft-n-max 15 \
  -p "Write a Python function that checks whether a string is a palindrome." -n 128 --temp 0 -t 8 -td 8
```
- The cost table here: verify 1/2/4/8 = 15.7 / 16.5 / 18.9 / 23.8 ms. With the exact-width kernels it is no longer a staircase.
- The run logs `kurn spec width: qwen3-1.7b.cost, k <= 15, mode policy`, then a `verify widths (M:steps)` histogram (here `1:1 2:24 3:6 4:15 6:2 8:3`), at 53.2 tok/s.
- `--spec-type draft-simple` is required.

**d. llama-server with the selector, k4c KV and exact mode:**
```sh
GGML_KURN_FA_MODE=exact $B/llama-server -m models/Qwen3-1.7B-Q8_0.gguf -md models/Qwen3-0.6B-Q8_0.gguf \
  --spec-type draft-simple --spec-draft-n-max 15 --spec-width qwen3-1.7b.cost -ctk k4c_q4 -t 8 -td 8 -c 8192 --port 8088 &
```
- **Logs:** each request logs `kurn spec width: verify widths (M:steps) = ...`.
- **`--spec-width TABLE`:** one policy per slot, which keeps learning across the slot's requests.
- **`-ctk k4c_q4`:** k4c keys with Q4_0 values (`k4c_q8`: Q8_0 values).
  - KV is 1.8× smaller than Q8_0.
  - Prompt cache, slot save/restore, multiple slots and speculative decoding work.
  - Context shift and `--cache-reuse` do not, because K-shift is off for k4c.
- **`GGML_KURN_FA_MODE=exact`:** batch-invariant attention and matmuls, so speculative output equals non-speculative output. On Qwen3-8B Q8_0, 24 of 24 speculative runs matched the no-draft text.
- **Without `-md`**, the server runs normally.
- **Qwen3-8B Q8_0 + 0.6B draft** (8 prompts × 256 tokens):
  - selector 30.1 tok/s;
  - fixed k=7 29.7;
  - no draft 14.4;
  - stock llama.cpp's best fixed k 18.8.

  See [verify-kernel results](kurn-verify-kernel-results.md).

## 5. kurn's own engine (whole-model decode program)

```sh
. .venv/bin/activate
kurn model models/Qwen3-1.7B-Q8_0.gguf -o qwen3-1.7b-engine        # a few seconds
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
python chat_engine.py "Name three primary colors."                    # F16 KV: 66.6 tok/s here
KURN_KV=k4c_q4 python chat_engine.py "Name three primary colors."     # k4c KV: 65.6 tok/s (also k4c_q8, q8_0, q4_0)
```

- **Context:** sized from the request; `KURN_CTX=n` asks for more.
- **Perplexity:** `./qwen3-1.7b-engine MODEL ppl 8 CTX tokens.txt` (space-separated ids; `PPL_CHUNKS=n`).

## 6. Confirming kurn kernels are in use

1. **Load log.**
   - Add `-v` to llama-cli or llama-server and look for `KURN model buffer size = ...`.
   - `GGML_KURN_VERBOSE=1` logs every repacked tensor, each weight's prefill path, and, at exit, how many FLASH_ATTN_EXT nodes ran on kurn and how many AMX steps the preemption guard recomputed.
2. **A/B switch.** `GGML_KURN=0` runs stock ggml paths in the same binary.
3. **Kernel self-tests:**
   ```sh
   L=$PWD/llama-kurn
   gcc -O2 -I$L/ggml/include integration/llama.cpp/test_kurn_buft.c -L$L/build/bin \
     -lggml -lggml-base -lggml-cpu -lm -Wl,-rpath,$L/build/bin -o test_kurn_buft
   ./test_kurn_buft smoke        # "smoke: 112 passed, 0 failed"; exit code = failures
   ./test_kurn_buft native       # native-layout Q6_K / Q5_K: "native: 34 passed, 0 failed" (also with GGML_KURN_EXACT=1)
   gcc -O2 -I$L/ggml/include integration/llama.cpp/k4c/test_k4c.c -L$L/build/bin -lggml -lggml-base -lggml-cpu -lm \
     -Wl,-rpath,$L/build/bin -o test_k4c && ./test_k4c                   # "k4c: 25 passed, 0 failed"
   $L/build/bin/test-backend-ops -o FLASH_ATTN_EXT -b CPU                # 5312/5312 passed
   ```
   - `smoke`, `quick` and `full` check every format's GEMV and verify kernels against ggml's `vec_dot`.
   - They also check batch invariance: every column of every batch width must equal that column computed alone.
   - They run in exact mode automatically.

**Switches:**

| Variable | Effect |
|---|---|
| `GGML_KURN=0` | stock ggml everywhere (KURN buffer and kurn attention off) |
| `GGML_KURN_AMX=0` | no AMX anywhere: verify-kernel prefill, f32 attention engine. Use it on VMs whose AMX state you do not trust. Every AMX step is timed and recomputed after a preemption-length gap, but `integration/llama.cpp/tools/amx_check.sh` is the test for a host |
| `GGML_KURN_EXACT=1` | batch-invariant matmuls (Q8_0 prefill on the guarded AMX-INT8 kernel, other formats on the verify kernels, Q6_K/Q5_K on the native kernels); implied by `GGML_KURN_FA_MODE=exact` |
| `GGML_KURN_FA_MODE=exact` | batch-invariant attention as well: speculative and cached outputs equal plain decoding bit for bit; slower long-context decode |
| `GGML_KURN_FA=0` | ggml's flash attention only |
| `GGML_KURN_NATIVE=0` | Q6_K / Q5_K back to ggml's buffers |

## 7. GPU check kit (A100 / B200 / RTX 50)

**Kit v2, one command:** [kurn-gpu-check-attn.zip](../artifacts/kurn-gpu-check-attn.zip) plus its `-ggml-00/01/02` and `-wheels` zips. Unzip all of them into one folder, then:

```sh
mkdir -p ~/kurn-kit && cd ~/kurn-kit && for z in /path/to/artifacts/kurn-gpu-check-attn*.zip; do unzip -q -o "$z"; done
cd kurn-gpu-check && QUICK=1 ./run_kit.sh        # ~1 h on an A100; ./run_kit.sh = full (~3 h); DRYRUN=1 = no GPU
```

On a split install (nvcc in the toolkit, runtime headers/libs elsewhere):

```sh
KURN_NVCC=/usr/local/cuda-12.8/bin/nvcc \
KURN_CUDA_INCLUDE=/path/to/cuda/include_no_implicit \
KURN_CUDA_LIB=/path/to/cuda/lib \
QUICK=1 ./run_kit.sh
```

- **What runs:**
  - the toolchain preflight;
  - llama.cpp's ggml-cuda, built inside the folder from bundled sources pinned in `ggml-src/COMMIT` (needs cmake + g++);
  - attention fatbins for sm_80 / sm_100 / sm_120, GPU correctness and the HBM roofline;
  - the decode matrix against llama.cpp's CUDA flash attention on the same data and timing, plus FlashInfer / FlashAttention-2 if `TORCH_PYTHON` already has them, and phase-2 ablations at 1K and 16K;
  - the GEMM/GEMV matmul kit, all ten formats incl. MXFP4 / NVFP4;
  - the GPU pytest subset.
- **Output:** `kurn-kit-results-<host>-<date>/report.md` is the single report: toolchain, steps, and what the box cannot test (on an A100: FP8 KV and sm_100 / sm_120 runtime). It has kurn vs baselines against the stated target, Q8_0 vs F16, ablations, matmul wins / losses, pytest results and the llama.cpp build. A `.tar.gz` of the folder sits next to it. Copy that into `artifacts/`.
- **Contained:** it writes only inside the kit folder (TMPDIR and every cache are redirected there) and checks this at the end (`containment.txt`). No network, no root, no pip.
- **Skip steps:** `NO_LLAMA=1`, `NO_MATMUL=1`, `NO_PYTEST=1`, `NO_TORCH=1`.
- **Dry run here:** `QUICK=1 DRYRUN=1 ./run_kit.sh` ran from an unzipped copy on a simulated split install (CUDA 12.9 nvcc without headers, runtime via `KURN_CUDA_INCLUDE` / `KURN_CUDA_LIB`; system python without pytest). It took 23 min:
  - ggml-cuda built in 4 min;
  - 586 attention fatbin configs, 0 failures; emulator 161 + 44 configs, 0 failures;
  - every harness compiled, including the llama.cpp bench;
  - matmul: 500 configs ptxas and emulator, 0 failures;
  - the pytest subset passed.

The older single-purpose scripts are still in the kit:

```sh
contrib/gpu-check/make_kit.sh $PWD          # -> kurn-gpu-check.zip (KIT_GGML=llama.cpp KIT_WHEELS=dir for the extra zips)
unzip kurn-gpu-check.zip && cd kurn-gpu-check && ./run_attn_check.sh   # QUICK=1 ~5-10 min; full ~20-30 min
```

- **Needs:** an NVIDIA driver, CUDA 11.x/12.x (12.8+ also builds sm_100 and sm_120 SASS), python3 ≥ 3.9 and g++. It installs nothing.
- **First step: the CUDA preflight.** It logs the resolved toolchain (nvcc, header and library dirs, versions, how each was found) to `run.log` and `toolchain.json`; `archs.json` has it too.
  - It warns when the headers' version differs from nvcc's, and when the driver is older than the runtime.
  - If the runtime isn't found, it stops before any build with the fix. On a split install, use `KURN_CUDA_INCLUDE` / `KURN_CUDA_LIB` as in step 1.
- **Output:** `kurn-attn-results-<host>-<date>.tar.gz` with `attn_report.md`. Exit status 0 means every check passed, and the report says which arch ran and whether it used native SASS or PTX JIT. Copy the tarball into your local `artifacts/` folder.
- **`DRYRUN=1`** runs without a GPU: compile plus CPU emulator.
  - Here, with CUDA 12.8 from the packages in step 1, `QUICK=1 DRYRUN=1 ./run_attn_check.sh` built 537 fatbin configurations for sm_80, sm_100 and sm_120 with 0 failures.
  - It compiled the harness for all three archs, and the CPU emulator passed 150 + 44 configurations.
  - It exited 0 after 16 min.
- **Also in the tree:** the same script as `contrib/gpu-check/run_attn_check.sh`; `run_gpu_check.sh` covers the GEMM/GEMV side.

## Not production-ready yet

- **AMX on VMs:**
  - AMX is now on by default where it is usable: prefill from 32 columns (16 for Q6_K/Q5_K), and attention.
  - Some KVM guests drop AMX tile data on preemption. On the dev VM, stock ggml's AMX buffer returned different results in up to 9 of 39 repeated runs.
  - kurn's AMX paths time every step and recompute the ones that spanned a gap. Run `tools/amx_check.sh` on a new host, and use `GGML_KURN_AMX=0` when in doubt.
- **Batch invariance by default:** default prefill (AMX-BF16) is not batch invariant with decode. Exact mode is, at lower prefill speed (about 0.65× on Q8_0).
- **Fast-mode attention:** about 3–4 of 8 speculative runs differ from plain greedy text at near-tie tokens. Exact mode removes this.
- **k4c KV:**
  - no K-shift (context shift and `--cache-reuse`);
  - CPU only, V must be q4_0 or q8_0, no MLA.
- **The engine is a research tool:**
  - Qwen3 / OLMoE only, with all-Q8_0 or all-Q4_0 weights.
  - Greedy only, token ids in and out.
  - No batched prefill.
- **GPU:**
  - Every GPU path (attention, FP8 KV, MXFP4/NVFP4) is verified on the CPU warp emulator and compiled for sm_80/sm_100/sm_120.
  - None has run on silicon yet; the A100 kit run is pending.
  - There is no llama.cpp CUDA integration.
- **Packaging:**
  - no wheels or releases;
  - llama.cpp support is pinned to `4ebdf2c` (scripts or `llama.cpp-kurn.patch`), and newer commits may need `apply.sh` fixes;
  - the cost table is per machine, model pair and thread count.
