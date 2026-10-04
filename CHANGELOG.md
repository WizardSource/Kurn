# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed
- Compile identical source/target requests once per parallel search batch while benchmarking every runtime configuration separately.

### Fixed
- Cache repeated legality-probe validations locally without changing draw weights or the random stream.
- Aggregate energy-delay products per repetition and reject incomplete or invalid measurements during tuning.
- Preserve confirmed refinement winners, validate plan reuse, and publish compiler outputs atomically.
- Bound search parameters and avoid redundant sampling work without merging runtime settings.

## [0.2.1] - 2026-10-03

### Fixed (weak 4-bit / Q8_0 end to end in the llama.cpp buffer type)
- Q4_0 decode through the KURN buffer type was 0.83-0.92x stock llama.cpp while its kernel tied ggml. Causes, measured
  per op: 4 dynamic row chunks per thread (15-30% of op bandwidth), two weight streams per core (rows=2), and the older
  `mask` unpack. Now `pair` with 4 row groups, guided partitioning in whole passes, shared activation prep and a
  one-record GEMV for remainders. Qwen3-1.7B Q4_0 decode 75.8 vs 71.1 tok/s stock (AMX buffer), prefill 1.53x.
- Q8_0 decode (0.97x stock): `rows=8` on the i16 records (v0.1 vnni16's pass width) plus the partitioning fix.
- `gen_ggml_sources.py --config` overrides of a tuned key no longer raise a duplicate-keyword error.
- Q4_K_M end to end is now a resolved win: decode 1.32x stock (AMX buffer), 1.39x stock built without AMX; prefill 1.28x.
- 8-core DRAM roofline on a Qwen3-8B-shaped graph: Q4_0 79%, Q4_K 73%, Q8_0 91% of 128.7 GB/s (the 4-bit >= 90% goal is
  still not met).

### Added
- `pair` verify kernels up to 8 columns (one accumulator chain beyond 4 row-group x column pairs).
- `ilv` schedule key (record interleave across row groups, `kurn.ext.q4fix`); measured slower than separate streams and
  not used by default.
- Generated `_xprep` / `_xprep_bytes` / `_packed_x` entry points (`xprep` codegen flag) for every generic lowering.
- Per-tensor fallback in the buffer type (`GGML_KURN_FALLBACK*`), `GGML_KURN_PROFILE` phase timing,
  `integration/llama.cpp/tools/opbench.c` (model-shaped MUL_MAT graphs per buffer type) and
  `benchmarks/v0.2/q4fix/cmp.py` (interleaved A/B rounds with 2-sigma verdicts).

## [0.2.0] - 2026-10-03

Built as eight parallel workstreams on one integration branch. Every kernel is checked against an exact reference; lossy
formats carry a stated tolerance plus perplexity/KLD checks. Measured results, including what did not pay off, are in the
project's v0.2 results document; per-workstream scripts and CSVs are under `benchmarks/v0.2/`.

### Added
- **Extension mechanism** (`kurn.hooks`, `kurn.ext`): modules register formats, recipes, kernels, layouts, schedule keys,
  ops, CLI commands, test generators and golden cases without editing `spec.py` or `kernels.py`. 1,590 x86 + 8 NEON legal
  configurations (v0.1: 102 + 8).
- **Recipe lowering ("what" vs "how")** (`generic.py`): one recipe per format, lowered to interleaved lane-per-row layouts
  (`i16` AVX-512, `i8` AVX2-VNNI), with algorithm keys `unpack` (mask, LUT, `mask16`, `pair`, `perm`), `correction`
  (activation-sum seed, weight sums, Q4_K `dpmin` min precompute), `scales` and `accum`.
- **CuTe-style layout algebra** (`layout.py`) and a layout-driven lowering (`sched.py`, `layout=composed`): shape/stride
  composition, complement, divide/product tiling, swizzles. Repack code and kernel addressing are generated from the same
  layout expression; vnni16/i16/i8 are reproduced bit for bit. Tiling (`kpanel`, `rpanel`) and pipelining (`stages`,
  `pfhint`, `pfgran`) keys.
- **Staged schedule search** (`kurn search`): search spaces derived from the registry; rediscovers the hand-designed Q4_K v2
  algorithm with no hints.
- **Plan cache** (`kurn plan build/show/lookup`): tuned winners per machine fingerprint x GGUF tensor shape, reused at load.
- **Formats:** Q4_0, IQ4_NL, MXFP4, NVFP4 (with quantizers), Q2_K, Q2_0, TQ1_0, TQ2_0, Q1_0 (Bonsai), E8P lattice codebook.
- **Low-bit kernels:** exact int16 lookup-table GEMVs (T-MAC style: group size, bit-serial planes, mirror consolidation,
  base-3 TQ1_0), add/subtract-only ternary kernels, Q2_K `k16`.
- **Multi-token verify op** (`op verify`, 2–8 activation columns, same per-column arithmetic as the GEMV).
- **Attention op** (`kurn attn`): tiled online-softmax attention (f32 FMA, AVX-512 BF16, AMX-BF16 with a guard against VMs
  that lose AMX tile state), split-KV decode with log-sum-exp merge, causal tile skipping, F16/BF16/Q8_0 KV, MLA latent KV.
- **Compressed weights:** `kurn mix` (sensitivity-predictor-guided per-tensor mixed precision with stock ggml types), E8P
  codebook with LUT GEMV, Huffman entropy-coded GEMV, LoRC low-rank residual kernels.
- **Runtime** (`src/kurn/runtime/`): persistent pinned pool; static, balanced, split-K and work-stealing partitioners; spin,
  spin-then-futex and yield waits; cache-cliff and expert-prefetch benchmarks.
- **Whole-model engine v2** (`kurn model`): one compiled decode step per fixed model (Qwen3 dense, OLMoE MoE; Q8_0 or
  Q4_0), schedule fixed at load and replayed per token, fused GEMV epilogues, per-sync-kind wait policies.
- **llama.cpp integration:** a KURN extra buffer type (`integration/llama.cpp/apply.sh`) that repacks once at load (no
  second weight copy), covers `MUL_MAT` and `MUL_MAT_ID`, is batch-invariant through the verify kernels, and leaves AMX
  prefill opt-in (`GGML_KURN_AMX=1`). An end-to-end harness (`benchmarks/v0.2/e2e/`) for decode/prefill tok/s, J/token
  proxy, perplexity and RSS.
- `benchmarks/v0.2/benchlock.sh`: exclusive lock for timed runs on shared machines.

### Changed
- `legal_configs` enumerates extension keys; open-ended composed layouts register covering sets instead.
- The v0.1 llama.cpp hook patch is kept for reference; the buffer type replaces it.

### Known limitations
- Measured on one 8-vCPU Emerald Rapids VM (Linux). ARM kernels only run under qemu; macOS, Windows and AMD are untested.
- AMX is not reliable on that VM (tile data is not preserved across context switches), so AMX paths are opt-in.
- 4-bit decode is compute-bound when cache-resident (two `vpdpbusd` per 64-byte load); see the results document for DRAM.

## [0.1.0] - 2026-10-02

First public release, extracted from an internal research prototype (`kdsl`).

### Added
- Registries for formats (`formats.py`), targets (`targets.py`) and kernels (`kernels.py`), with a table-driven harness, so new
  formats and kernels (v0.2: 4-bit repacked layouts, 2-bit/ternary LUT kernels) drop in without touching the CLI, validation or
  tests; `tests/test_registry.py` checks that Python and C sides agree.
- Spec language (`.kurn`): flat `key value` files with closed, target-dependent value sets; `tune` lines declaring search spaces;
  errors that list the legal values; detection of tune values that are illegal in every combination.
- Formats `q8_0`, `q4_K` (weights) and `q8_0`, `q8_K` (activations), byte-compatible with ggml, with a pure-Python reference.
- Lowering rules: Q8_0 GEMV (scalar, AVX2, AVX-VNNI, AVX-512 VNNI incl. the `vnni16` interleaved layout, NEON), Q4_K GEMV
  (scalar, AVX2, AVX-VNNI, AVX-512 VNNI), Q8_0 GEMM (scalar, AVX-512 VNNI register tiles, AMX tiles). 110 legal configurations.
- `kurn` CLI: `check`, `gen` (with `--embed PREFIX` for pasting into ggml), `build`, `verify` (`--space`, `--all`, `--strict`),
  `tune` (energy proxy / speed / EDP objectives, static platform power, Pareto front, CSV), `roofline`, `targets`.
- Standalone benchmark harness (no ggml dependency): random valid ggml blocks, exact double-precision reference, NaN-filled outputs,
  pinned thread pool with spin or futex barriers, hot/cold regimes, serial-gap modelling, bandwidth roofline. Builds on x86-64 and
  AArch64.
- NEON kernels cross-compiled and run under `qemu-user` for correctness on x86 hosts.
- Content-addressed build cache keyed by generated source, flags and compiler.
- Optional ggml-linked harness (`contrib/ggml-harness`) and a llama.cpp integration patch (`integration/llama.cpp`).
- Tests: spec validation, golden codegen, every legal config vs the Python reference (random and extreme data), harness failure
  detection, tuner, CLI. CI: lint, tests on x86-64 (Python 3.9 and 3.12), NEON cross-compile + qemu.

### Changed (relative to the prototype)
- `threads` is validated as 1..256 instead of 1..host CPU count, so specs validate the same on every machine; the tuner skips
  points with more threads than CPUs.
- The vnni16 kernel's inner broadcast variable is renamed (`xv` → `xk`) so generated code is clean under `-Wshadow`, as llama.cpp
  builds it. No change in behaviour.
- NEON is compiled with `-mcpu=neoverse-n1` (GCC and Clang spelling) and the cross compiler is auto-detected or set with
  `KURN_CROSS_CC`.
