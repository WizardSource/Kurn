# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased] - 0.3.0.dev3

### Added: aligned activation layout for the dp4a GEMV (`xlayout split`), contributed by the user
- `xlayout blocks|split` for the q8_0-activation formats (Q8_0, Q4_0, IQ4_NL, Q2_0, Q1_0). `split` reads the activations as an
  aligned int8 plane [M][K] plus a float scale plane [M][K/32], with 16-byte vector loads instead of 16-bit loads from ggml's
  34-byte blocks. The q8_K formats (Q4_K, TQ2_0, E8P) stay on `blocks`.
- Ported from 0.3.0.dev0 onto dev2's restructured GEMV. The per-column activation pointers became dev2's 32-bit column offsets,
  and the per-column scale pointers are derived from them (scale offset = plane offset / 32), so no registers were added.
- `split` is the default for those formats: on the default kernels it cuts global loads 3-5x and hot-loop instructions
  14-40% (SASS). `xlayout` is in the kit's GEMV tune space.

### Fixed: spilling kernels in the search space
- **GEMV.** The block-size rule allowed 512 threads (a 128-register cap) for every 2-7 column kernel and 1024 threads (a
  64-register cap) for most 1-column ones, and `cols * unroll` up to 16 for Q1_0/Q2_0/Q4_0. Under those caps ptxas spills
  Q1_0 at 4 columns, and split activations raise pressure further (16-byte loads hold four registers each). Adding the
  `xlayout` key changed which random configs the covering sets sample, and that surfaced a Q1_0 4-column kernel with
  `xlayout blocks` at 512 threads (32 bytes of spill). Some of the other spilling configs the old rule allowed are older
  than `xlayout`.
- The GEMV limit is now measured, not estimated. Uncapped register counts swing by +-90 with `tpr` alone, so no estimate
  predicts where ptxas spills. `kurn/gpu/data/gemv_threads.json` holds, for every (weights, xlayout, sub, cols, unroll), the
  largest `tpr * rpb * max(1, minb)` that compiles without spill or stack on sm_80/90/100. Each entry is the worst case
  over every layout, mins, unpack, `tpr` (8-128) and `minb` (1, 2, 4); `minb` matters because ptxas schedules
  `__launch_bounds__(128, 4)` differently from `(512, 1)`. `tools/gemv_threads.py` regenerates the table. The old rules
  stay, so this only removes configs: 2,452 of 77,418 (3.2%), mostly Q1_0 (1,872), Q2_0 (354) and Q8_0 (166). Every
  default keeps at least 4x headroom.
- **GEMM.** Fresh covering seeds found engine tiles the register estimate let through. All of them had 64-row warp tiles
  with 2+ n8 tiles and 2+ k-tiles per stage, and each thread staged 12+ 16-byte `cp.async` chunks per stage: Q4_0, Q2_0,
  TQ2_0 and E8P, with 1-4 warps. This class sits at the 255-register ceiling and the estimate cannot separate its spills
  from clean tiles, so it is now an explicit rule. The rule removes 2,484 of 88,668 tiles (2.8%), and every default and
  matrix tile is unchanged.
- Validation: ptxas covering sweeps on sm_80/90/100 over 40 seeds, about 8,800 GEMV and 11,200 GEMM configs, with 0 spills;
  `kurn gpu ptxas --defaults --strict` is clean.

### Fixed: optimizer selection, plan reuse, and sampling
- Aggregate energy-delay products per repetition and reject incomplete or invalid measurements during tuning.
- Preserve confirmed refinement winners, validate plan reuse, and publish compiler outputs atomically.
- Bound search parameters and avoid redundant sampling work without merging runtime settings.

## 0.3.0.dev2

### Changed (first A100 run: correctness held, speed below the competition)
- **`op gemm` is now a tensor-core engine for every batch size and all 8 formats.** It replaces the int8 GEMM, which reached at
  most 5% of tensor-core peak and 0.11-0.41x of ggml-cuda.
  - Arithmetic: f16 `mma.sync.m16n8k16` with f32 accumulation. Weights are repacked once into fragment order, and each lane's
    16 bytes dequantize in registers to small integers that are exact in f16:
    - Q4 via the 0x6400 magic number;
    - IQ4_NL via a `prmt` lookup;
    - Q8 via byte permutes;
    - 2-bit and 1-bit via shifted masks and a power-of-two FMA;
    - E8P via a shared-memory codebook with sign flips.
  - Block scales are applied in f32 after the MMA.
  - Data movement:
    - weights and scales go through a multi-stage `cp.async` pipeline;
    - activations sit in XOR-swizzled shared memory and are read with `ldmatrix.x4`;
    - f32 activations are rounded to f16 inside the kernel (`xin=f32`, one launch per matmul, no quantization kernel) or
      converted once and streamed (`xin=f16`).
  - Tiles go up to 128x128 with 64x32 warp tiles.
  - Split-K uses a deterministic serial fixup and is sized from the SM count by default.
  - Exact against the f16-rounded activations; `kg_act()` tells harnesses which activation path a kernel uses.
- **dp4a GEMV defaults:**
  - the split layout with lane loads sized to 16 bytes and 4 units in flight (2 for Q8_0/Q4_K/TQ2_0);
  - an unrolled main loop without per-unit bounds checks, so loads issue ahead of the math;
  - 32-bit column offsets. The default Q4_K multi-column kernel no longer sits at 255 registers (138-168, no spills).
- **Spilling configs are not legal.** A ptxas-calibrated register estimate and launch-bounds checks prune them; every covering
  config compiles without spills or stack on sm_80/90/100. Kernels emit `__launch_bounds__(threads, >= 1)` (ptxas otherwise
  targets 128 registers and spills).
- **Benchmarks and kit:**
  - the matrix races `default-*` and `tuned-*` KURN kernels per batch range;
  - the kit tunes every format before the matrix;
  - `report.md` shows default vs tuned side by side;
  - `kurn gpu sass` reports the SASS instruction mix (tensor-core MMA, ldmatrix, cp.async, local memory, hot loop).
- **Release:** `tools/make_release.sh` fails if `contrib/gpu-check` is missing.

## 0.3.0.dev1

Includes everything in 0.2.2 (CPU release: clang `--strict` helper pruning, optional numpy, toolchain probe, self-contained
release), merged into the CUDA development line. The CUDA modules (`kurn.gpu`) do not need numpy.

### Changed (measurement)
- **`kurn roofline` is now a peak-read probe.** Before, `bench --bw` timed one pass of a loop that compiled to mixed
  64/128/256-bit loads, with one stream per thread, and under-reported bandwidth: about 2x on a Xeon 8339HC, 25% on this VM.
  - It now uses the widest vector loads (AVX-512, AVX2, NEON; else 64-bit) into 4 independent accumulators per thread, with
    all threads pinned.
  - Each pass is barrier-timed; the result is the best pass, with the median also shown. Without `--streams`, it measures 1, 2,
    4 and 8 interleaved streams per thread and reports the best.
  - This VM, 8 threads, same session: DRAM 140–144 GB/s (old default probe 103–108); L2 1.40–1.61 TB/s (old 0.75–1.09).
- **`kurn tune` ranks on medians of interleaved rounds** (`--rounds`, default 3). It re-measures the leaders (`--keep`,
  `--budget`) until their order is stable, reports each result's spread, and warns ("ranking: NOT resolved", suggesting
  `--secs` / `--rounds`) when the spread is too large to rank. `--rounds 1 --keep 1` gives the old single-run behaviour.

### Fixed (cross targets without a cross toolchain)
- `kurn verify` no longer counts NEON configurations as failures when the AArch64 cross compiler is missing. Like the
  assembler probe, it skips the target once, says why, and names the package to install
  (`apt install gcc-aarch64-linux-gnu`, or set `KURN_CROSS_CC`).
- If the cross compiler is present but qemu is missing, configurations are still compiled (so `--strict` warnings are caught)
  and are reported as "compiled, not run" with the qemu hint (`apt install qemu-user`, or set `KURN_QEMU`). They are counted
  separately from passes.
- `run_mode` reasons now name the missing piece and how to install it.

### Fixed (CUDA emulator portability, found by the 0.2.2 clang runs)
- The CPU emulator now probes the host C++ compiler once (`kurn.gpu.toolchain.cxx_problem`). If it can't build a C++17
  program, it reports why (typically clang++ on Ubuntu selecting a GCC installation without libstdc++ headers) and the
  emulator checks are skipped rather than failing. `kurn gpu targets` shows the emulator status.
- Emulator builds with clang++ no longer fail `-Werror` on `#pragma unroll` loops clang can't unroll at -O1
  (`-Wno-pass-failed`, clang only).

### Added: CUDA backend (`target cuda`, `kurn.gpu`), stages 0-2 of the GPU plan; no GPU measurements yet
- **Code generation:** CUDA C++ for the following, every file implementing `kurn_gpu.h`:
  - decode GEMV (dp4a) on Q8_0, Q4_0, IQ4_NL, Q4_K, Q2_0, TQ2_0, Q1_0 and E8P, in `native` and `split` layouts, with
    multi-column small-batch variants;
  - an int8 tensor-core GEMM (`mma.sync.m16n8k32`, sync / register double-buffer / 2- and 3-stage `cp.async` pipelines) for
    Q8_0, Q4_0 and IQ4_NL;
  - activation quantizers matching ggml's reference rounding;
  - device repack kernels.
- **CPU warp emulator** (`kurn_cuemu.h`): fibers with real barrier, warp-shuffle and `mma.sync` fragment semantics, deferred
  `cp.async`, alignment and guard-page checks, randomized schedules. Every generated kernel runs on it against an exact C
  reference (`kurn_gpu_ref.h`, tied to `kurn.formats`).
- **nvcc/ptxas checks** for sm_80 / sm_90 / sm_100: registers, shared memory, spills and static occupancy. SASS is checked for
  IMMA and LDGSTS.
- **GPU harness** (`bench_gpu.cu`): measured HBM and tensor-core roofline, CUDA-graph timing with cold weights, NVML board
  energy, cuBLAS FP16/INT8 baselines, and llama.cpp ggml-cuda MUL_MAT as an in-process competitor on identical bytes.
- **Benchmark matrix:** formats × batch 1/4/16/64/256 on Llama-3-8B layer shapes, with interleaved rounds.
- **Report:** a 2-sigma win rule, wins/ties/losses, and `dispatch.json`. `kernel_for()` falls back to the competitor's kernel
  wherever KURN does not win.
- **Tuning:** energy-ranked GPU tuning (NVML joules).
- **Kit:** the hand-run kit `contrib/gpu-check/` (`run_gpu_check.sh`, `make_kit.sh`, optional Marlin script).
- **Integration:** `kurn.hooks.TARGET_BACKENDS` routes `kurn check|gen|build|verify|tune` to a backend by the spec's target.
- **CLI:** `kurn gpu ...` commands.

## [0.2.2] - 2026-10-03

Portability release, from a run of 0.2.1 on a second Linux machine. No kernel performance changes.

### Fixed
- `kurn verify --all --strict` with clang: the generators emitted every `static inline` helper (`f16f`, `bits8x4`, `ld64`,
  `ld16`, ...) whether or not a kernel used it. Clang's `-Wunused-function` (GCC stays quiet) turned that into errors on
  most AVX-VNNI and many AVX-512 configs. `generate()` now drops the helpers a kernel doesn't call.
- Q4_K AVX-512 native-layout GEMV (the `act`/`accum`/`scales`/`correction` algorithm space and the v2 kernel) summed the
  min-correction vector over the undefined upper half of `_mm512_castps256_ps512`. GCC happened to zero it. Clang didn't
  at `rows=4`, which gave relative errors of 0.08-0.1. The kernel now zero-extends explicitly.
- numpy was a hidden dependency: five modules (`codebook`, `entropy`, `latent`, `lowrank`, `mixed`) imported it at load time,
  so 9 test files failed to collect without it. It is now the optional extra `kurn[compress]`, with `kurn[gguf]` for GGUF
  input. The modules import numpy lazily, and their tests use `pytest.importorskip`.
- A toolchain that can't build a target (binutils < 2.36 can't assemble AVX-VNNI) failed every config of that target. kurn
  now probes each target once. It reports "toolchain can't assemble target X; set KURN_CC" and skips the target in
  `kurn verify`, `kurn targets` and the tests. The harness falls back from `-march=native` to portable flags.
- Two more clang-only build errors, found by running the full suite with `KURN_CC=clang`. First, the attention kernel's
  shared helpers are now marked unused, because each engine uses only some of them (`kurn attn verify --strict`). Second,
  the fused engine epilogue now passes `_mm512_roundscale_ps` a literal rounding mode, which clang requires.
- Parallel builds of identical kernel sources could compile a half-written file. Sources are now written atomically.
- Tests that import benchmark harnesses (`benchmarks/v0.2/e2e`) or the llama.cpp integration skip cleanly when those
  directories are absent, instead of failing to load.

### Added
- `tools/offline-check/`: the one-command host check (verify, roofline, tune sweeps, AMX test), shipped in the package. It runs
  on the enclosing tree and falls back to the source tree when offline.
- `tools/make_release.sh`: builds the release zip from the committed tree. It checks that the archive holds every tracked
  file, that the docs-path test passes, and that pytest collects with no errors inside the unpacked copy.
- Tests: clang `-Werror` over every golden file and every legal config (`test_codegen_golden.py`), every path README.md and
  CONTRIBUTING.md reference (`test_docs_paths.py`), and the toolchain probe (`test_toolchain_probe.py`).
- CI: `kurn verify --all --strict` with clang, and a test job without numpy.

### Changed
- The code base is `ruff format`-clean, as CONTRIBUTING.md requires.

### Fixed
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
