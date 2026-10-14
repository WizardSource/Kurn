# KV cache entropy: how much can lossless coding take out of kurn's KV caches?

This directory measures the entropy of real llama.cpp KV caches in kurn's formats. It compares four things:

- simple entropy models;
- zstd;
- kvz, the coder in `integration/llama.cpp/kvz`;
- the "speculative KV coding" scheme from [fergusfinn.com/blog/kv-entropy-coder](https://fergusfinn.com/blog/kv-entropy-coder/).

The scheme codes the target's KV against a predictor model's KV, with μ taken from the predictor and σ per channel.

`run.sh` reproduces everything. It needs `LLAMA=` (a llama.cpp checkout with apply.sh, k4c/apply.sh and kvz/apply.sh) and `MODELS=` (a GGUF directory).

## What is measured

- **Caches:** Qwen3-1.7B and Qwen3-8B, Q8_0 weights, kurn attention. `kvdump.cpp` prefills the text and saves
  sequence 0 with `llama_state_seq_save_file`.
  - Cache types: F16, Q8_0, k4c_q4 and k4c_q8.
  - Context: 1.7B at 32K; 8B at 16K, plus F16 at 32K. Every shorter prefix is the same cache, so 4K-32K come from
    one run.
  - Texts: WikiText-2 test (prose) and llama.cpp's own C++ sources (code).
  - Quantized K / V caches are stored after llama.cpp's Hadamard rotation (K 128-point, V 64-point); k4c keys are not.
  - k4c keys appear in the state as their exact f16 values. The in-memory k4c bytes are rebuilt with an exact numpy
    port of `ggml-k4c.c`.
- **Entropy** (`analyze.py`, `kvent.py`): bits per element of the stored format, under three kinds of model.
  - Order-0 bytes: the blog's "bytewise entropy".
  - Adaptive (KT) code lengths per channel, which count the model cost: f16 high byte per channel, low byte per high
    byte or per channel; Q8_0 / Q4_0 / k4c codes per channel, plus their scales.
  - A previous-token predictor per channel.
- **Speculative KV coding** (`predictor.py`):
  - μ is the KV of the same model with Q4_K_M weights on the same tokens.
  - σ² is the per-(layer, kv, channel) residual variance on WikiText-2 train + other C++ sources (8K tokens each).
  - Symbols are coded under the bin-integrated 0.95 N(μ, σ²) + 0.03 N(μ, 9σ²) + 0.02 empirical-marginal mixture.
  - For the rotated caches, μ and σ are taken in the rotated basis.
- **kvz:** actual ratio and speed (`kvz-bench`, bit-exact round trip of every block).

## Results (`results/summary.md` has every row)

Bits per element, with the ratio against the stored format in parentheses.

| cache (32K / 16K tokens) | stored | order-0 bytes | best static model | **kvz** | zstd -3 | speculative (Q4_K_M predictor) |
|---|---|---|---|---|---|---|
| F16 K | 16 | 13.8-14.0 | 13.33-13.36 (1.20x) | **13.36-13.40 (1.19-1.20x)** | 1.07x | 10.0-10.8 (1.48-1.59x) |
| F16 V | 16 | 13.6 | 13.47-13.54 (1.18-1.19x) | **13.48-13.56 (1.18-1.19x)** | 1.10-1.12x | 10.8-11.4 (1.41-1.48x) |
| Q8_0 K | 8.5 | 8.05-8.12 | 7.17-7.51 (1.13-1.19x) | **7.17-7.50 (1.13-1.19x)** | 1.02-1.03x | 4.4-5.0 (1.69-1.92x) |
| Q8_0 V | 8.5 | 8.03-8.04 | 7.93-7.99 (1.06-1.07x) | **8.00-8.04 (1.06x)** | 1.06x | 5.4-6.0 (1.41-1.56x) |
| k4c keys (in-memory) | 5 | 4.66 | 4.61-4.63 (1.08x) | n/a (states hold f16 keys) | 0.99-1.02x | 2.1-2.8 (1.82-2.37x) |
| Q4_0 V (k4c_q4) | 4.5 | 4.15 | 4.12-4.14 (1.09x) | **4.13-4.15 (1.09x)** | 1.07-1.10x | 1.75-2.26 (1.99-2.57x) |

- Ratios hardly move between 4K and 32K tokens, between prose and code, or between 1.7B and 8B. Q8_0 K is the
  exception: 1.18x on 1.7B, 1.14x on 8B.
- **F16 is not BF16.** The blog's "~11 bits" is for bf16, whose 8-bit exponent field compresses. F16's 10-bit mantissa
  leaves a near-random low byte, so the floor is about 13.3 bits.
- **Quantized formats are near saturation**, as the blog says for FP4/FP8: 4-9% for Q8_0 V, Q4_0 and k4c.
- **The exact keys in a k4c state:** f16 given its k4c group costs 9.2 bits. So k4c codes plus that residual (13.8)
  cost more than the f16 alone.
- **Speculative coding** gains 1.4-2.6x over the stored format, more on low-bit formats and on the bigger model, in
  line with the blog's trend. But decoding needs the predictor's prefill.
  - On this CPU, Q4_K_M prefill is no cheaper than the Q8_0 target: llama-bench 1390 vs 1317 tok/s (1.7B, pp8192),
    424 vs 411 (8B, pp4096).
  - In exact mode, which bit-identical μ requires, it is slower: 539 vs 720.
  - A restore by the predictor therefore costs at least a full target re-prefill. In exact mode that re-prefill
    reproduces the cache bit for bit with nothing stored.

## Speed (8-vCPU Emerald Rapids, `results/kvz-speed.txt`)

kvz with 8 threads, on 32K-token blocks (8B Q8_0 / k4c at 16K):

| block | encode | decode | 1 thread enc / dec |
|---|---|---|---|
| F16 K / V | 1.5-1.6 GB/s | 1.7-1.9 GB/s | 0.31 / 0.38 GB/s |
| Q8_0 K | 1.1-1.2 GB/s | 1.6 GB/s | 0.24 / 0.27 GB/s |
| Q8_0 V | 1.25-1.4 GB/s | 1.8-2.1 GB/s | 0.29 / 0.36 GB/s |
| Q4_0 V | 0.8 GB/s | 0.9 GB/s | 0.16 / 0.17 GB/s |

zstd -3 runs at 0.8-1.4 GB/s on 8 threads for 1.02-1.12x.
