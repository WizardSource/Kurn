KURN GPU check kit (KURN 0.3.0.dev3: tensor-core engine, tuned GEMV defaults, GPU attention)

Attention only (one command, ~20-30 min on an A100; QUICK=1 for ~5-10 min):

    unzip kurn-gpu-check.zip && cd kurn-gpu-check && ./run_attn_check.sh

  It builds every attention config (sm_80 + sm_120 fatbins), checks each one on the GPU against a float64 reference on
  awkward shapes (GQA 1-8, MLA, causal/mask/non-causal, splits, head-major caches, rows that see no key), then runs a
  decode matrix (Llama-3-8B / Qwen3-1.7B / MLA x 1K-32K context x F16/BF16/Q8_0 KV). Output:
  kurn-attn-results-<host>-<date>.tar.gz with attn_report.md. Exit status 0 only if everything is correct.

Full GEMM/GEMV + attention kit:

Run on a Linux machine with an NVIDIA GPU (A100, H100/H200, B200, L40, RTX 30/40/50 ...):

    unzip kurn-gpu-check.zip && cd kurn-gpu-check
    ./run_gpu_check.sh            # ~1.5 h on an A100: GPU correctness, roofline, tune sweep per format, full matrix
    QUICK=1 ./run_gpu_check.sh    # ~25 min: smaller tune, matrix at batch 1 and 16, 3 rounds
    DRYRUN=1 ./run_gpu_check.sh   # any Linux box, no GPU: compile + CPU-emulator checks + packaging

Needs: NVIDIA driver, CUDA toolkit 12.x (nvcc; 12.8+ also builds the RTX 50 / sm_120 code), python3 >= 3.9, g++ (KURN runs from the bundled source; no pip, no network needed except for llama.cpp).
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
Copy it into the Project store (e.g. the artifacts/ folder you see on your Mac) and tell the coordinator.

Split download: if kurn-gpu-check.zip arrives in parts (kurn-gpu-check-part-00.zip ...), each part is a complete zip;
unzip them all into the same folder.
