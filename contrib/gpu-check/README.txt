KURN GPU check kit (KURN 0.3.0.dev3: tensor-core engine, tuned GEMV defaults, GPU attention)

Kit v2: everything in one command, one report (recommended):

    mkdir kit && cd kit && for z in /path/to/kurn-gpu-check*.zip; do unzip -q -o "$z"; done && cd kurn-gpu-check
    QUICK=1 ./run_kit.sh          # ~1 h on an A100
    ./run_kit.sh                  # full, ~3 h
    DRYRUN=1 ./run_kit.sh         # any Linux box, no GPU: everything compile-only / on the CPU emulator

  Unzip every part into the same folder: the main zip (scripts, KURN, tests), the -ggml-NN parts (llama.cpp's ggml
  sources, pinned in ggml-src/COMMIT) and -wheels (pytest, used only if python3 lacks it).
  Steps, each logged and recorded in steps.jsonl:
    1. CUDA toolchain preflight (`kurn gpu doctor`, with the split-install variables below), cuBLAS check
    2. llama.cpp ggml-cuda build from the bundled sources (or LLAMA_CPP_DIR), built inside this folder with kurn's CUDA
       resolution (`kurn gpu ggml-build`), so split installs work. Needs cmake and g++. No network.
    3. attention: fatbins for sm_80 / sm_100 / sm_120 (ptxas, spills); GPU correctness of the covering set; HBM roofline;
       decode matrix (Llama-3-8B / Qwen3-1.7B / MLA x 1K-32K x F16 / BF16 / Q8_0) with llama.cpp's CUDA flash attention
       on every cell (same data, same cold-KV timing); ablations of each phase-2 change at 1K and 16K; FlashInfer /
       FlashAttention-2 decode if TORCH_PYTHON (default python3) already has torch plus flashinfer / flash_attn
    4. the GEMM/GEMV matmul kit (run_gpu_check.sh): GPU correctness, tune, matrix vs ggml-cuda and cuBLAS for all ten
       formats incl. MXFP4 / NVFP4 (dequant-in-register, runs on sm_80)
    5. the GPU pytest subset (kurn/tests/test_gpu_*.py)
  Output: kurn-kit-results-<host>-<date>/report.md - the single report: toolchain, steps, what this box cannot test
  (on an A100: FP8 KV and sm_100 / sm_120 runtime - compile + emulator only), attention kurn vs baselines against the
  stated target, Q8_0 vs F16, ablations, matmul wins/losses incl. MXFP4 / NVFP4, pytest summary, llama.cpp build -
  plus a .tar.gz of the whole folder. Exit status 0 only if every correctness step passed.
  Containment: writes only inside this folder (results, kurn-cache/, llama-view/ build, tmp/ for TMPDIR and the CUDA /
  Python / Triton / FlashInfer caches), checked at the end (containment.txt). No network, no root, no pip installs.
  Options: NO_LLAMA=1, NO_MATMUL=1, NO_PYTEST=1, NO_TORCH=1, TORCH_PYTHON=..., FORMATS=..., LLAMA_CPP_DIR=... (its ggml
  is built here, nothing is written into the checkout), plus the CUDA variables below.

Attention only (one command, ~20-30 min on an A100; QUICK=1 for ~5-10 min):

    unzip kurn-gpu-check.zip && cd kurn-gpu-check && ./run_attn_check.sh

  Runs on A100 (sm_80), B200/GB200 (sm_100) and RTX 50 (sm_120): every build carries SASS for all three (CUDA >= 12.8)
  plus PTX, and the report says which arch and tier it ran on and whether that was native SASS or JIT.
  The archs are decided by this box's nvcc: ones it can't build are skipped with a note (those GPUs JIT the PTX);
  KURN_GPU_ARCHS=sm_80,sm_120 chooses them explicitly (then an unbuildable arch is an error). archs.json records it.
  It builds every attention config, checks each one on the GPU against a float64 reference on
  awkward shapes (GQA 1-8, MLA, causal/mask/non-causal, splits, head-major caches, rows that see no key), then runs a
  decode matrix (Llama-3-8B / Qwen3-1.7B / MLA x 1K-32K context x F16/BF16/Q8_0 KV). Output:
  kurn-attn-results-<host>-<date>.tar.gz with attn_report.md. Exit status 0 only if everything is correct.

Full GEMM/GEMV + attention kit:

Run on a Linux machine with an NVIDIA GPU (A100, H100/H200, B200, L40, RTX 30/40/50 ...):

    unzip kurn-gpu-check.zip && cd kurn-gpu-check
    ./run_gpu_check.sh            # ~1.5 h on an A100: GPU correctness, roofline, tune sweep per format, full matrix
    QUICK=1 ./run_gpu_check.sh    # ~25 min: smaller tune, matrix at batch 1 and 16, 3 rounds
    DRYRUN=1 ./run_gpu_check.sh   # any Linux box, no GPU: compile + CPU-emulator checks + packaging

Needs: NVIDIA driver, CUDA toolkit 12.x (nvcc; 12.8+ also builds the B200 / sm_100 and RTX 50 / sm_120 code), python3 >= 3.9, g++ (KURN runs from the bundled source; no pip, no network needed except for llama.cpp).

CUDA runtime headers and libraries: both scripts start with a preflight (`kurn gpu doctor`) that compiles, links and
(on a GPU) runs a tiny .cu with the exact nvcc and flags the builds use. If nvcc is installed but cuda_runtime.h,
cuda_fp16.h, libcudart (and for run_gpu_check.sh cublas_v2.h / libcublas) live somewhere else - a "split" install, e.g.
a monorepo third_party dir with include_no_implicit/ and lib/ - it stops in seconds with the paths it searched. Fix:

    KURN_CUDA_INCLUDE=/path/to/cuda/include_no_implicit KURN_CUDA_LIB=/path/to/cuda/lib QUICK=1 ./run_gpu_check.sh

  - KURN_CUDA_INCLUDE / KURN_CUDA_LIB take ':'-separated lists; kurn adds -I / -L and an rpath for the libraries.
  - CUDA_HOME=/path/to/cuda also works: kurn probes CUDA_HOME/{include,include_no_implicit,targets/*/include} and
    {lib64,lib}. CUDA_PATH likewise. Without any setting kurn also probes nvcc's own targets/*/{include,lib} and sibling
    /usr/local/cuda-* installs (same version as nvcc first), and warns when it uses a directory it discovered;
    KURN_CUDA_NO_PROBE=1 turns that probing off (explicit settings and nvcc's own toolkit only).
  - NVCC_APPEND_FLAGS="-I... -L..." / NVCC_PREPEND_FLAGS are passed through to nvcc unchanged (kurn only adds an rpath).
  - KURN_NVCC=/path/to/nvcc picks the compiler (default: CUDA_HOME/bin, PATH, /usr/local/cuda*/bin).
  The resolved toolchain (nvcc, header and library dirs, versions, how each was found) is in run.log, toolchain.json and
  archs.json. Warnings: headers whose CUDART_VERSION differs from nvcc's version; a driver older than the runtime.
For the llama.cpp (ggml-cuda) competitor: git + cmake + network, or LLAMA_CPP_DIR=/path/to/llama.cpp.
Marlin is measured only if PyTorch with vLLM or the `marlin` package is already installed (MARLIN_PYTHON=...).
Installs nothing outside this folder. No root, no clock locking, no power-limit changes.
Pick the GPU with CUDA_VISIBLE_DEVICES=N. Close other GPU jobs: timings and board energy are recorded.

What it measures, per format (Q8_0, Q4_0, IQ4_NL, Q4_K, Q2_0, TQ2_0, Q1_0, E8P) and batch (1, 4, 16, 64, 256):
tokens/s of a Llama-3-8B layer's matmuls x 32 layers, % of measured HBM bandwidth, % of the int8 tensor-core peak,
NVML joules per token and exactness vs an exact reference, for KURN and every installed competitor
(ggml-cuda, cuBLAS FP16, cuBLAS INT8 as a speed reference, Marlin), interleaved, 5 rounds, mean and spread.
Every format is tuned before the matrix; report.md shows KURN's default and tuned kernels side by side.
Attention (new): GPU correctness of the covering set against a float64 reference, then a decode matrix (Llama-3-8B,
Qwen3-1.7B and MLA shapes x 1K-32K context x F16/BF16/Q8_0 KV, cold KV larger than L2) in attn_matrix.jsonl.
A KURN win needs a > 2 sigma margin over the best competitor; dispatch.json (kernel_for) keeps the competitor's
kernel everywhere else.

Output: kurn-gpu-results-<host>-<date>.tar.gz (report.md, dispatch.json, raw JSON lines, logs, generated kernels).
Copy it into your local artifacts/ folder (or wherever you keep run outputs).

Split download: if kurn-gpu-check.zip arrives in parts (kurn-gpu-check-part-00.zip ...), each part is a complete zip;
unzip them all into the same folder.
