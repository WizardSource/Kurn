# kurn

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**A tool that writes the math AI models run on.** You describe a kernel in a short spec file; kurn generates it, checks it
against an exact reference and tunes it for speed or energy. CPU: 8-, 4-, 2- and 1-bit GEMV/verify, attention, compressed
weights and a compiled whole-model decode engine. CUDA (new, 0.3 development line): decode GEMV for eight ggml formats and an
int8 tensor-core GEMM, verified without a GPU and packaged as a hand-run benchmark kit.

You write a kernel as a flat spec file of about ten `key value` lines. kurn turns it into plain, dependency-free C with intrinsics for
scalar, AVX2, AVX-VNNI, AVX-512 VNNI, AMX or NEON. The kernels read ggml's block formats (Q8_0, Q4_0, Q4_K, IQ4_NL, MXFP4, NVFP4,
Q2_K, Q2_0, TQ1_0, TQ2_0, Q1_0) and repack them into layouts composed for the target's dot-product instructions. kurn checks
every generated kernel against an exact reference. Its autotuner sweeps the search space the spec declares and ranks configurations
by an energy proxy, by speed, or by energy × delay.

```text
kernel   q8_0_gemv_vnni16
op       gemv               # decode: Q8_0 weights x Q8_0 activation vector
weights  q8_0
target   avx512_vnni
layout   vnni16             # 16 weight rows interleaved per block, one vpdpbusd feeds 16 rows
rows     8
tune     align=packed,64 rows=1,2,4,8 prefetch=0,8,16 threads=4,6,8
```

```sh
kurn gen    examples/q8_0_gemv_vnni16.kurn -o q8_gemv.c                  # 187 lines of C intrinsics
kurn verify examples/q8_0_gemv_vnni16.kurn                               # numerical check vs reference
kurn tune   examples/q8_0_gemv_vnni16.kurn --regime cold --objective energy
```

Status: **0.3.0.dev3, alpha research code.** It contains the 0.2.2 CPU release plus the CUDA backend. CPU results come
from one machine (see [Results](#results) and its caveats). The CUDA backend has **no GPU measurements yet** (see
[CUDA backend](#cuda-backend-target-cuda)).

## Why

A study of the same CPU inference kernels written in eight languages (C, Rust, Zig, ISPC, Halide, Mojo, Julia, Triton-CPU) found
that **the language did not matter for speed or energy. What mattered was whether the code issued the right instructions
(VNNI/AMX) on the right weight layout with a few schedule choices.** Implementations that issued the same instruction stream
landed within run-to-run noise of each other, in time and in energy proxy. Portable code without VNNI was 2–4x slower.

kurn makes exactly those things first-class and nothing else:

- **Formats are types.** `q8_0` and `q4_K` carry their activation format (`q8_0`, `q8_K`), byte layout and exact semantics.
- **Schedules are explicit keys** with closed sets of legal values: `layout`, `align`, `rows`, `cols`, `act`, `prefetch`.
- **Targets are one word.** The same spec retargets to 5 ISAs.
- **Runtime knobs that move energy** sit in the same spec: `threads` and the barrier `wait` policy.
- **Agent- and tuner-friendly.** Specs are flat with no nesting, and an invalid value fails with the list of legal ones. An agent
  or the tuner explores by editing one token, and every point is checked automatically.

kurn is deliberately a thin generator plus tuner over C, in the spirit of ggml's repack templates or XNNPACK/LIBXSMM microkernel
generators. It is not a general-purpose language or optimizer: every (op × format × target) lowering is a hand-written template.

## Install

Requirements: Linux, Python ≥ 3.9, and GCC or Clang. The core (spec language, code generation, `verify`, `tune`, attention)
has no Python dependencies.

```sh
pip install git+https://github.com/<org>/kurn       # or, from a checkout: pip install .
kurn targets                                        # what this host can build and run
```

Optional Python extras:
- `pip install 'kurn[compress]'` adds numpy, which the compressed-weight modules need (`kurn.codebook`, `kurn.entropy`,
  `kurn.lowrank`, `kurn.mixed`, `kurn.latent`), as do the MXFP4/NVFP4 quantizers in `kurn.mx`. Without it, those modules
  still import, and the first call raises an error naming the extra. Tests that need numpy are skipped.
- `pip install 'kurn[gguf]'` adds numpy and the `gguf` reader for `kurn mix`, `kurn model` and planning from GGUF files.
- `pip install -e '.[dev]'` adds pytest, ruff and numpy for development.

**Choosing the compiler.** kurn compiles with `$KURN_CC` (default: `$CC`, then gcc, cc, clang). Before building for a
target, it compiles one probe instruction for that target. If the compiler or assembler can't handle it, `kurn verify`,
`kurn targets` and the tests report `toolchain can't assemble target X; set KURN_CC` and skip that target instead of failing
each configuration. The usual cause is binutils older than 2.36 (Debian 11, for example), which can't assemble AVX-VNNI
(`avx2_vnni`). Fix it with `KURN_CC=clang` (clang uses its own assembler) or a newer gcc with newer binutils. Both GCC and
Clang build every target warning-free under `kurn verify --all --strict`.

Optional tools:
- `aarch64-linux-gnu-gcc` (or `zig cc`) to cross-compile NEON kernels on x86.
- `qemu-user` to also *run* them there, for correctness only. Emulated timings are meaningless.
- `apt install gcc-aarch64-linux-gnu qemu-user` covers both on Debian/Ubuntu.
- Without them, `kurn verify` doesn't fail on NEON configurations:
  - with no cross compiler, it skips the `neon` target and says which package to install;
  - with a cross compiler but no qemu, it still compiles them (so `--strict` warnings are caught) and reports them as
    "compiled, not run", with the qemu hint.

## Quickstart

```sh
kurn check  examples/q8_0_gemv_vnni16.kurn                 # validate; print resolved config and tune-space size
kurn check  examples/q8_0_gemv_vnni16.kurn rows=3          # -> rows=3 not allowed ...: expected one of [1, 2, 4, 8]
kurn gen    examples/q8_0_gemv_vnni16.kurn -o k.c          # emit C (implements kurn.h)
kurn gen    examples/q8_0_gemv_vnni16.kurn target=avx2 layout=native   # same spec, another ISA
kurn gen    examples/q8_0_gemv_vnni16.kurn --embed kurn_ -o kurn_q8_0_gemv.inc   # static + prefixed, paste into ggml
kurn build  examples/q4_K_gemv.kurn                        # compile to a cached .so; prints the path
kurn verify examples/q4_K_gemv.kurn --space                # check every config in the spec's tune space
kurn verify --all --strict                                 # every legal config, compiled with -Wall -Wextra -Wshadow -Werror
kurn tune   examples/q4_K_gemv.kurn --regime cold --objective energy -o results.csv
kurn tune   examples/moe_expert_gemv.kurn --static-w 10 --bench-args "--K 2048 --N 512 --serial-us 20"
kurn roofline --threads 8                                  # measured peak DRAM and L2 read bandwidth
```

`key=value` arguments override spec keys. For `kurn tune`, `key=v1,v2` overrides a dimension of the tune space.

## Spec reference

A spec is UTF-8 text with one `key value` pair per line. `#` starts a comment. Keys may appear in any order, but duplicates are
errors. Spec files use the `.kurn` extension by convention.

| key | legal values | meaning |
|---|---|---|
| `kernel` | any word | name, used in output and logs (default `<weights>_<op>_<target>`) |
| `op` | `gemv`, `gemm` | `gemv`: weights × one activation vector (decode); `gemm`: weights × M activation rows (prefill) |
| `weights` | `q8_0`, `q4_K` (`gemm`: `q8_0`) | weight format; implies the activation format (`q8_0` → `q8_0`, `q4_K` → `q8_K`) |
| `target` | see [targets](#supported-formats-and-targets) | instruction set to generate for |
| `layout` | `native`; `vnni16` (q8_0 gemv on avx512_vnni) | `native`: ggml blocks as stored. `vnni16`: 16 rows interleaved per block, weights pre-biased (+128) so one 64-byte load feeds `vpdpbusd` for 16 rows; repacked once by `kq8_gemv_prepare`. Same bytes as Q8_0 |
| `align` | `packed`; `64` (vnni16 only) | `packed`: vnni16 blocks are exactly 16 Q8_0 blocks (544 B). `64`: pad to 576 B for aligned loads |
| `rows` | gemv q8_0: 1, 2, 4, 8; gemv q4_K: 1, 2, 4; gemm: 2, 4, 6; amx: 1, 2 | weight rows per register pass (vnni16: 16-row groups per pass; amx: n-tiles) |
| `cols` | gemm: 2, 4, 6; amx: 1, 2; gemv: 1 | activation rows per register tile (amx: m-tiles). avx512_vnni gemm needs rows × cols ≤ 24 |
| `act` | `once`, `inline` (q8_0 gemv on avx2_vnni / avx512_vnni); avx2 q8_0: `inline` only; otherwise `once` | activation prep: once per call and reused by all rows, or inline per row |
| `prefetch` | 0, 2, 4, 8, 16, 32 (0 only for gemm and scalar) | software prefetch distance in blocks |
| `threads` | 1..256 (default: min(8, CPUs)) | runtime: worker threads (harness/tuner) |
| `wait` | `spin`, `sleep` | runtime: barrier wait. `spin` forever (ggml-like) or spin briefly then futex-sleep |
| `tune` | `key=v1,v2 ...` | search space for `kurn tune` / `kurn verify --space`; any schedule key, runtime key, or `target` |

Illegal combinations (`act=inline` with `vnni16`, `align=64` with `native`, more than 24 avx512 GEMM accumulators) are rejected with
a reason. Tune values that are illegal in every combination (typos) are rejected too. Legal points that do not apply to a particular
combination are skipped.

## Supported formats and targets

| op | weights | targets | layouts and keys |
|---|---|---|---|
| `gemv` (decode) | `q8_0`, `q4_K` | `scalar`, `avx2`, `avx2_vnni`, `avx512_vnni` (+ `neon` for q8_0) | `native`, `vnni16` (q8_0), `i16` / `i8` interleaved, `composed` (layout algebra); `unpack`, `correction`, `scales`, `accum`, rows, prefetch |
| `gemv` | `q4_0`, `iq4_nl`, `mxfp4`, `nvfp4` | `scalar`, `avx2_vnni`, `avx512_vnni` | `i16` / `i8`, `composed` (f16-scale formats); `unpack` mask / LUT / `mask16` / `pair` / `perm` |
| `gemv` | `q2_K`, `q2_0`, `tq1_0`, `tq2_0`, `q1_0` | `scalar`, `avx512_vnni` (+ `avx2_vnni` for some) | `i16`, `lut` (exact int16 T-MAC tables: group size, bit-serial, mirror, base-3), `addsub`, `k16` (q2_K) |
| `gemv` | `e8p` (E8 lattice codebook) | `scalar`, `avx512_vnni` | codebook LUT decode; rows, prefetch |
| `verify` (2–8 tokens) | every recipe format | `avx2_vnni`, `avx512_vnni` | `i16` / `i8`, `composed`; cols 2/4/8; same per-column arithmetic as the GEMV |
| `gemm` (prefill) | `q8_0` | `scalar`, `avx512_vnni`, `amx` | register tiles; AMX tiles |
| `attn` (`kurn attn`) | F16 / BF16 / Q8_0 KV | `avx512_vnni` (f32 FMA, AVX-512 BF16), `amx` (AMX-BF16) | tile sizes, split-KV, KV format; online softmax, causal tile skipping, GQA, MLA latent KV |

That is 1,590 legal x86 GEMV/verify/GEMM configurations plus 8 NEON (`kurn verify --all`); composed layouts and attention have
their own covering sets and checkers.

| target | compiler flags | needs (to run) |
|---|---|---|
| `scalar` | `-O3 -march=x86-64-v3` (`-O3` on non-x86 hosts) | x86-64-v3 or any non-x86 host |
| `avx2` | `-O3 -march=x86-64-v3` | AVX2, FMA, F16C |
| `avx2_vnni` | `+ -mavxvnni` | AVX-VNNI (Alder Lake+, Zen 5) |
| `avx512_vnni` | `-O3 -march=x86-64-v4 -mavx512vnni` | AVX-512 F/BW/VL + VNNI (Ice Lake+, Zen 4+) |
| `amx` | `+ -mamx-tile -mamx-int8` | AMX-INT8 (Sapphire Rapids+), Linux ≥ 5.16 |
| `neon` | `-O3 -mcpu=neoverse-n1` | ARMv8.2 dot product + fp16 |

**Generated code** implements the C ABI in [`src/kurn/data/kurn.h`](src/kurn/data/kurn.h): `kq8_gemv`, `kq4k_gemv`, `kq8_gemm`,
optional `*_prepare` / `*_packed` repacking hooks and `kern_thread_init`. Each call computes a contiguous slice of output rows,
so any thread pool can split work. Block structs are byte-identical to ggml's `block_q8_0`, `block_q4_K` and `block_q8_K`.

**Limits:**
- K ≤ 32768, because per-call activation buffers live on the stack.
- Q8_0 avx512_vnni kernels with `layout native` need K to be a multiple of 64.
- int8 values must be in [-127, 127], which is what ggml's quantizers produce.
- The harness and tuner are Linux-only (futex, `sched_setaffinity`). `gen` works anywhere.

Not implemented: NEON for the 4-bit and lower formats, SVE, AMX-FP16, and a Windows/macOS harness. AMX paths are opt-in, because
the development VM does not preserve AMX tile state across context switches (see Results).

**Extending kurn.** Formats, targets and kernels each live in a registry:
- `formats.py`: layout, activation pairing and Python reference.
- `targets.py`: flags, architecture and required CPU features.
- `kernels.py`: (op, weights) → targets, lowering, `kurn.h` entry point and harness name.

Spec validation, the CLI, `verify`, `tune` and the tests are all driven by these registries. A new kernel needs:
- a registry entry;
- a lowering in `codegen.py`;
- its entry point in `kurn.h`;
- a table row in `bench.c`.

`tests/test_registry.py` fails until every piece is in place. See [CONTRIBUTING.md](CONTRIBUTING.md#adding-a-format-or-kernel).

Since 0.2, new work goes in its own module under `src/kurn/ext/`, which is imported automatically. It registers formats, recipes,
kernels, layouts, schedule keys, ops and CLI commands through [`kurn.hooks`](src/kurn/hooks.py), so it needs no edits to
`spec.py` or `kernels.py`.

## How verification works

Correctness is mechanical, at three levels:

1. **Independent Python reference, every legal config** (`tests/test_numerics.py`).
   - `kurn.formats.reference_dot` implements each format's exact semantics in pure Python, with integer sums per (sub-)block and
     double-precision scaling.
   - Every legal configuration the host CPU can execute is compiled, loaded through `ctypes`, and compared on two data sets:
     random blocks, and extreme blocks (all ±127, all-ones nibbles and scales) that would expose saturating-arithmetic bugs.
   - Each case runs once over all rows and once split into 16-row slices, as threads would split it. N = 37 exercises row tails.
2. **The bundled harness checks every run** (`src/kurn/data/bench.c`; used by `kurn verify` and `kurn tune`).
   - It generates random valid ggml blocks and computes an exact double-precision scalar reference in C.
   - Output buffers are pre-filled with NaN, so unwritten rows fail.
   - `kurn verify` uses awkward sizes: N = 100, M = 40, and 3 threads, so slices are uneven.
   - The tuner drops any configuration whose relative error exceeds 1e-2 and reports the error of the rest. On the reference
     machine every kernel was within 5e-7.
   - On x86 hosts with a cross compiler and `qemu-user`, NEON kernels are run under emulation: correctness only, not speed.
   - Tests also feed the harness deliberately wrong kernels to make sure it fails them.
3. **Against ggml itself (optional)**.
   - [`contrib/ggml-harness`](contrib/ggml-harness) is the original ggml-linked harness: data from ggml's own quantizers, reference
     from ggml's own `vec_dot`, plus ggml's graph paths (including its AMX buffer) as baselines.
   - `kurn tune --harness path/to/bench_ggml` uses it. The published results were measured this way.

Code generation is also covered by golden tests (`tests/golden/`) and by compiling all 110 configurations with
`-Wall -Wextra -Wshadow -Werror`.

## How the energy-aware tuner works

`kurn tune` takes the cartesian product of the spec's `tune` line (plus any `key=v1,v2` overrides). It skips illegal points with
the reason, then generates, compiles (cached by a hash of the generated source and flags), checks and times each configuration in
the harness.

- **Noise-robust ranking.** A single short run ranks configurations by whatever else the machine was doing at that moment. On a
  busy server, one 1 s run per configuration ranked a VNNI kernel below AVX2 (median 0.90x), while 3 s runs had it winning all
  six rounds. So `kurn tune` works like this:
  - It measures every configuration `--rounds` times (default 3) for `--secs` each (default 1 s), in interleaved rounds whose
    order rotates, so slow drift affects all candidates alike.
  - It ranks on the **median**.
  - It re-measures the top `--keep` configurations (default 3) in extra interleaved rounds until their order holds for two
    rounds in a row, or `--budget` seconds (default 30) are spent.
  - Each result reports its round count and spread (interquartile range ÷ median).
  - If the top two are closer than their spread, or a leader spreads more than `--max-spread` (default 10%), the tuner prints
    **"ranking: NOT resolved"**. In that case use longer runs (`--secs 3`), more rounds (`--rounds 5`), or a quieter machine.
  - `--rounds 1 --keep 1` reproduces the old single-run behaviour.

  On this VM, ranking AVX2 against AVX-512 VNNI Q8_0 GEMV (8 threads, DRAM) with six busy processes running, single 1 s runs put
  AVX2 first in 5 of 6 tries. That's wrong: quiet, VNNI is 1.12x faster, resolved after 5 rounds. The new tuner flagged every
  busy-machine ranking as not resolved instead of reporting a winner.

- **Regimes.** `hot` keeps each thread's weight slice cache-resident, which measures code generation. `cold` (the default) rotates
  through more than 1.2 GB of weights so every call streams from DRAM, which is the realistic decode case.
- **Energy proxy.** `energy = busy CPU-seconds × 5.47 W per core + wall-seconds × static_w`. The first term is a per-core share of
  a 350 W, 64-core server. It works on any machine, without RAPL or PMU counters. Because every worker spins at the barrier, it is
  proportional to time × threads, and that is what lets the tuner trade cores for latency. `--static-w W` charges platform power
  (DRAM, uncore, the rest of the box) per wall-second. Under the pure busy-core proxy, fewer threads can look cheaper even when
  slower. A platform term usually flips that back.
- **Objectives.** `--objective energy` (default), `speed`, or `edp` (energy × delay). The tuner prints the top configurations and
  the **time/energy Pareto front**, and `-o` writes every result to CSV.
- **Runtime knobs.** `threads` and `wait` (`spin` vs futex `sleep`) are tuned like schedule keys. `--bench-args "--serial-us 20"`
  adds a single-threaded gap after each call, modelling MoE router/top-k work between barriers.

The proxy cannot see differences in energy per instruction or DRAM energy. If you have RAPL, compare its readings against the
proxy before trusting energy rankings.

## CUDA backend (`target cuda`)

A spec with `target cuda` goes to `kurn.gpu` (registered through `kurn.hooks.TARGET_BACKENDS`), which emits one
self-contained CUDA C++ file per configuration implementing [`kurn_gpu.h`](src/kurn/gpu/data/kurn_gpu.h). The same file compiles
with nvcc and, with `-DKURN_EMU`, as host C++ against a CPU warp emulator.

| op | formats | method | keys |
|---|---|---|---|
| `gemm`: the tensor-core engine, any batch size (decode 1-8 with `bn` 8, up to 256+) | all 8: Q8_0, Q4_0, IQ4_NL, Q4_K, Q2_0, TQ2_0, Q1_0, E8P | f16 `mma.sync.m16n8k16`, f32 accumulate; weights repacked once into fragment order and dequantized in registers to small integers that are exact in f16; block scales applied in f32 after the MMA; `cp.async` multi-stage pipeline; swizzled shared memory + `ldmatrix`; f32 activations rounded to f16 inside the kernel (`xin f32`, one launch per matmul) or once by `kg_quant` (`xin f16`); deterministic split-K | `bm`, `bn`, `wm`, `wn`, `bk`, `stages`, `splitk` (0 = from the SM count), `xin`, `minb` |
| `gemv` (decode, dp4a) | the same 8 | dp4a over 32-value chunks with q8 activations, warp-shuffle reduction; default: split layout, 16-byte weight loads, 4 units in flight, and aligned activations (`xlayout split`, contributed by the user) for the q8_0-activation formats | `layout`, `tpr`, `rpb`, `sub`, `unroll`, `cols`, `xlayout`, `minb`, `mins` (Q4_K), `unpack` (Q1_0) |

**Why f16 tensor cores rather than int8.** With ggml's per-32-value block scales, an int8 MMA kernel must convert every int32
result to float and apply two scales per output and block. On A100 that conversion runs at a quarter of the tensor-core rate,
which caps such kernels well below peak (ggml's int8 MMQ reaches 12% of int8 peak at batch 256). The engine instead dequantizes
weights to small integers, which are exact in f16, and multiplies f16 activations. That leaves one FFMA per output and block,
needs no activation quantization launch, and stays exact against the f16-rounded activations. At batch 1 the same kernel streams
the repacked weights with 16-byte `cp.async` copies, with 3-5 stages in flight. Weight tiles with spills or local memory are not
legal configurations: a ptxas-calibrated register estimate prunes them, and the tuner drops anything ptxas reports as spilling.

Only sm_80 features are used, so the code runs on Ampere, Ada, Hopper and Blackwell. wgmma / tcgen05 / TMA / FP8 / FP4 paths are
planned, not built.

```sh
kurn check  examples/gpu/q8_0_gemm_cuda.kurn            # validate; tune-space size
kurn gen    examples/gpu/tq2_0_gemv_cuda.kurn -o k.cu    # emit CUDA C++
kurn verify examples/gpu/e8p_gemv_cuda.kurn --space      # numerics on the CPU emulator (no GPU needed)
kurn gpu ptxas --all --archs sm_80,sm_90,sm_100          # nvcc/ptxas: registers, shared memory, spills, occupancy
kurn gpu sass --defaults                                 # SASS mix: tensor-core MMA, ldmatrix, cp.async, local memory, hot loop
kurn gpu build examples/gpu/q4_0_gemm_cuda.kurn --archs sm_80,sm_90
kurn gpu harness --llama ~/src/llama.cpp                 # GPU harness, optionally linked with ggml-cuda as a competitor
kurn gpu verify --all --gpu                              # on a GPU: every covering config vs the exact reference
kurn gpu tune examples/gpu/q4_0_gemv_cuda.kurn --objective energy   # NVML board energy, not a proxy
kurn gpu matrix --results out/ && kurn gpu report out/   # benchmark matrix, wins/ties/losses, dispatch.json
kurn gpu dispatch q4_0 1 --table out/dispatch.json       # what kernel_for() picks
```

**Verification without a GPU.**
- **CPU warp emulator** ([`kurn_cuemu.h`](src/kurn/gpu/data/kurn_cuemu.h)): every CUDA thread is a fiber. It provides
  `__syncthreads`, warp shuffles, `dp4a`, the PTX fragment layouts of `mma.sync`, and `cp.async` copies that are deferred until
  their `wait_group`. Vector loads are alignment-checked and buffers end at a guard page. A randomized schedule (`KEMU_SEED`)
  exposes shared-memory races. Tests inject the bug classes it is meant to catch (missing barriers, a missing `cp.async` wait,
  a misaligned load, a dropped shuffle, dropped work) and require each one to fail.
- **Exact reference:** a C double-precision reference ([`kurn_gpu_ref.h`](src/kurn/gpu/data/kurn_gpu_ref.h)) that tests tie to
  `kurn.formats`. The quantizers must match ggml's reference quantizers byte for byte.
- **nvcc/ptxas:** each config is compiled for sm_80, sm_90 and sm_100, recording register use and spills, with a static
  occupancy estimate.

**On a GPU** (the hand-run kit `contrib/gpu-check/`, built into `kurn-gpu-check.zip` by `make_kit.sh`), the harness
([`bench_gpu.cu`](src/kurn/gpu/data/bench_gpu.cu)) works as follows:
- **Correctness:** every KURN config is checked against the exact reference.
- **Roofline:** HBM read bandwidth and the int8 `mma.sync` / dp4a peaks are measured.
- **Benchmark matrix:** formats × batch 1/4/16/64/256 on a Llama-3-8B layer's matmuls. KURN runs against ggml-cuda (MUL_MAT on
  identical bytes), cuBLAS FP16, cuBLAS INT8 (speed reference only) and Marlin (if installed). Implementations are interleaved,
  5 rounds each, with NVML joules per token.
- **Win rule:** a KURN win needs a > 2σ margin over the best competitor. `kernel_for()` returns the competitor's kernel
  everywhere else.

The plan, competitor analysis and success and kill criteria are in the Project's KURN GPU plan document (kept outside this
repository).

## Embedding in ggml / llama.cpp

`kurn gen SPEC --embed PREFIX` drops the `kurn.h` include (the host code base provides the block structs), prefixes every symbol,
and makes the entry points `static`. [`integration/llama.cpp`](integration/llama.cpp) has a prototype patch that hooks the
generated vnni16 Q8_0 decode kernel into ggml's `mul_mat`. It applies to llama.cpp `4ebdf2c` and compiles. See its README for
usage and caveats.

## Results

### 0.2.x

Measured on one 8-vCPU Emerald Rapids VM. Every comparison is interleaved within one session. "ggml best" is the fastest stock
path on that Xeon (AMX buffer, CPU_REPACK or `vec_dot`). Full tables, including what did not pay off, are in the project's
v0.2 results document; the scripts and CSVs are under `benchmarks/v0.2/`.

| item | kurn 0.2 vs ggml best | vs kurn 0.1 |
|---|---|---|
| Q4_K GEMV, cache-resident, 1 core (composed layout / `pair`+`dpmin`) | 1.92x / 2.57x | 3.05x / 4.12x |
| Q4_K GEMV, DRAM, 8 cores (i16) | 1.48x | 1.37x |
| IQ4_NL / MXFP4 / NVFP4 GEMV, DRAM, 8 cores | 2.1x / 1.9x / 2.2x | new |
| Q4_0 GEMV, DRAM, 8 cores | 1.03x (0.2.1, 8B-shaped graph: 1.055x, 79% of roofline) | new |
| Q8_0 GEMV, DRAM, 8 cores (vnni16, 90% of the 128.7 GB/s roofline) | 1.17x | unchanged |
| Decode end to end in llama.cpp (KURN buffer type): Ternary-Bonsai-8B / Bonsai-1.7B / OLMoE Q8_0 / IQ4_NL | 8.8x / 2.04x / 1.38x / 1.34x | new |
| 0.2.1: Qwen3-1.7B Q4_0 / Q4_K_M / Q8_0, OLMoE Q4_0 decode vs stock built without AMX (vs stock AMX buffer) | 1.68x / 1.39x / 2.11x / 1.66x (1.07x / 1.32x / 1.04x / 1.49x) | 0.2.0 → 0.2.1: 1.19x / 1.13x / 1.08x / 1.17x |
| Tiled attention, prefill per layer, 8B shapes, ≥ 8K context | 7.7–10.3x (flash attention) | new |
| Whole-step engine, Qwen3-1.7B Q8_0 / Q4_0, OLMoE Q8_0 | 1.17x / 1.48x / 1.20x tok/s | new |
| Mixed precision (`kurn mix`) at 3.2–4.0 bpw | 33–49% lower KL divergence than stock quants at equal bits | new |

Not met or did not pay: 4-bit decode reaches 1.6–1.7x Q8_0 instead of the 1.89x byte ratio, and MoE spin-wait stays at 28–30%.
Activation skipping, entropy coding, expert prefetch, producer/consumer threads and batch-1 fused epilogues measurably don't pay.

**AMX on virtual machines.** On the development VM, AMX tile data is not preserved across context switches, so concurrent or
migrating AMX threads silently compute wrong results. kurn's AMX paths are therefore opt-in, and its attention kernel guards
each AMX block. Check a host with [`tools/offline-check/run_offline_check.sh`](tools/offline-check/README.md) (or
`integration/llama.cpp/tools/amx_ctx.c`).

### 0.1.0

These were measured during development of the prototype: one 8-vCPU Intel Emerald Rapids KVM guest (AVX-512, VNNI, AMX; 320 MB L3),
using the ggml-linked harness. Medians of 3 interleaved repetitions; 222/222 runs correct against ggml. Roofline: 128.7 GB/s on
8 cores and 18.3 GB/s on 1 (multi-stream read test). K = 4096.

**Roofline note (0.3.0.dev1).** `kurn roofline` was rewritten as a peak-read probe, because the old loop compiled to mixed
64/128/256-bit loads and timed a single pass:
- it now uses the widest vector loads, all threads pinned, the best of 1/2/4/8 streams per thread, and the best of several
  barrier-timed passes;
- on this VM it reads 140–144 GB/s on 8 cores, against 103–108 GB/s from the old default probe in the same session;
- re-measured against it, the vnni16 Q8_0 kernel below runs at 98% of roofline on 8 cores (137 GB/s), so no kernel exceeds
  100%.

The "~101%" in the table is relative to the old 128.7 GB/s figure. On 1 core, the same kernel measures 103% of the new probe
with the harness's default 1.2 GB cold set: part of that set stays in the 320 MB L3. With a 2.4 GB set (`--bench-args
"--footprint 2400"`) it measures 97%. Compare % of roofline with a cold set several times the L3 size.

| kernel, regime | kurn (tuned config) | ggml `vec_dot` | ggml AMX path | best hand-written kernel (8-language study) |
|---|---|---|---|---|
| Q8_0 GEMV, cache-resident, 1 core | **7.0 µs** (vnni16, align 64, rows 8) | 33.0 µs | 16.4 µs | 19.6 µs (Mojo) |
| Q8_0 GEMV, DRAM, 1 core | **1,032 µs, 95% of roofline** (vnni16, packed) | 2,538 µs | 1,444 µs | 1,729 µs (C) |
| Q8_0 GEMV, DRAM, 8 cores | **138 µs, 130 GB/s, ~101% of roofline** (vnni16, packed) | 328 µs | 188 µs | 215 µs (Mojo) |
| Q4_K GEMV, cache-resident, 1 core | 22.6 µs | 27.2 µs | **17.5 µs** | 21.5 µs (Rust) |
| Q4_K GEMV, DRAM, 8 cores | **151 µs** (rows 4, prefetch 4) | 206 µs | 188 µs | 196 µs (Rust) |
| Q8_0 GEMM, M = 128, 8 cores | **5.56 TOPS** (AMX, 2×1 tiles) | 0.80 TOPS (default path) | 2.62 TOPS | 5.77 TOPS (C, AMX, hand-written) |
| Q8_0 GEMV, DRAM, 8 cores, **AVX2 target** | **167 µs** (rows 8, prefetch 16) | 328 µs (also AVX2) | – | – |

End to end in llama.cpp (Qwen3-1.7B Q8_0 decode, 8 threads, prototype patch):

| path | tok/s |
|---|---|
| ggml plain `vec_dot`, `--repack 0` (the path CPUs without AMX run) | 28–29 |
| kurn vnni16 inside ggml's dynamic 64-row chunks | 45–47 |
| **kurn vnni16, static per-thread rows** | **59–61** |
| ggml AMX buffer (default on this Xeon) | 57–59 |

What the numbers say:
- **Kernel level.** The generated Q8_0 decode kernel is 1.36x ggml's best path from DRAM, where it sits at the bandwidth limit,
  and 2.35x when cache-resident.
- **End to end.** It matches ggml's AMX path on this Xeon, because both are bandwidth-bound across the whole model. It is
  **2.1x** the plain path that CPUs without AMX use. Perplexity was identical to ggml's AMX path (4.3023 vs 4.3023).
- **What the tuner found.**
  - Padding depends on the regime: 64-byte alignment wins cache-resident, packed wins from DRAM.
  - Prefetch helps Q4_K from DRAM.
  - With the same instruction set, a better schedule made the AVX2 target 1.96x ggml's AVX2 `vec_dot` from DRAM.
- **Energy vs speed.**
  - For single kernels, the speed winner was also the energy winner, except where the busy-core proxy traded cores for time:
    4 threads used 4% less proxy energy than 8 at 1.9x the latency. Charging 10 W of platform power flipped that back.
  - In an MoE-shaped microbenchmark with 20–50 µs serial gaps, 2 sleeping threads used 2.8–4x less proxy energy than 8 spinning
    threads, at 12–45% more latency.
  - End to end on a 35B-A3B MoE, sleep and/or fewer threads saved only 7–16% proxy energy and cost 29–44% speed.

**Caveats. Read these before citing numbers.**
- **Energy is a CPU-time proxy, not a measurement.** The VM exposed no RAPL or PMU counters.
- **One VM, one CPU generation.**
  - AVX2 builds were timed on the AVX-512 host, not on an AVX2-only CPU.
  - The non-AMX end-to-end benefit was measured with `--repack 0` on an AMX machine, not on a Zen 4/5 or Ice Lake CPU.
  - NEON kernels have never run on ARM hardware. They pass the numerical check under qemu emulation only.
- **Tuning used one measurement per configuration.** Only the winners were re-validated with 3 repetitions.
- **The llama.cpp hook is a prototype.** It repacks lazily, so Q8_0 weights exist twice in memory. (0.2 replaces it with an
  extra buffer type that repacks at load.)
- **Cache-resident Q4_K was still 1.3x slower than ggml's AMX path in 0.1.** (0.2's repacked layouts fix this; see above.)
- **The table predates this package.** It was measured with the ggml-linked harness on the prototype generator. The bundled
  harness uses its own random data and reference, and the table was not re-measured with it. The `xv` → `xk` variable rename in
  vnni16 (a `-Wshadow` fix) is the only change to generated code since then.

## Roadmap

- **GPU:**
  - first measurements with the hand-run kit (A100, then B200);
  - ggml-cuda integration through `kernel_for()`;
  - a `q8_1` activation variant;
  - Hopper/Blackwell batched paths (wgmma / tcgen05, TMA, FP8 / NVFP4) via generated CUTLASS kernels;
  - a vLLM custom op.

- **Validation off this VM:** AMD Zen 4/5, an AVX2-only laptop, real ARM (Graviton / Grace), and bare-metal AMX.
  [`tools/offline-check/`](tools/offline-check/README.md) packages the checks.
- **ARM:** NEON and SVE lowerings for the recipe formats; macOS support for the harness and runtime.
- **Search:** add WS-B's `pair` / `dpmin` algorithm keys to the composed-layout search space, so the tuner can find the best
  hand-expanded Q4_K kernel.
- **Speculative verify:** an activation-in-lanes or AMX verify kernel for M ≥ 4, which is compute-bound today.
- **Integration:** upstream the ggml buffer type; E8P and attention in the buffer type; engine support for more architectures.
- RAPL-backed energy measurement where available.

## Python API

```python
import kurn

spec, space = kurn.load("examples/q8_0_gemv_vnni16.kurn")
cfg = kurn.resolve(spec, {"rows": 4})  # validates; raises kurn.SpecError listing legal values
c_src = kurn.generate(cfg)
inc = kurn.embed(c_src, "my_")

from kurn.toolchain import build
from kurn.harness import check
from kurn.tune import tune

so = build(cfg)
print(check(so, cfg)["relerr"])
results, pareto = tune(spec, {"rows": [4, 8], "threads": [4, 8]}, regime="cold", objective="energy")
```

## Repository layout

```text
src/kurn/            spec.py (language + validation), formats.py / targets.py / kernels.py (registries), hooks.py,
                     codegen.py and generic.py (lowerings), layout.py + sched.py (layout algebra, composed lowering),
                     plan.py (plan cache), lowbit.py, mx.py, codebook.py, mixed.py, entropy.py, lowrank.py, epilogue.py,
                     toolchain.py, harness.py, tune.py (tuner + staged search), cli.py
src/kurn/ext/        workstream extensions (auto-imported), registering through kurn.hooks
src/kurn/attention/  attention op (spec, codegen, checker, bench, tuner); data/attn_kernel.c, data/kurn_attn.h
src/kurn/runtime/    persistent pool, partitioners (static / balanced / split-K / steal), wait primitives
src/kurn/model/      whole-model decode engine (engine.c v2, engine_v1.c), compile_model.py
src/kurn/data/       kurn.h (kernel ABI), bench.c (standalone harness)
src/kurn/gpu/        target cuda: spec.py, codegen.py (CUDA C++), emu.py + data/kurn_cuemu.h (CPU warp emulator),
                     toolchain.py (nvcc, ptxas, occupancy), harness.py + data/bench_gpu.cu, tune.py, matrix.py,
                     report.py (2-sigma win rule), dispatch.py (kernel_for), kit.py, cli.py (`kurn gpu ...`)
benchmarks/v0.2/     per-workstream benchmark scripts, results CSVs, benchlock.sh, final/ (coordinator re-measurement)
tools/               offline-check/ (one-command correctness, roofline, tune and AMX check for a new host),
                     make_release.sh (builds the release zip and checks it is self-contained)
examples/            q8_0_gemv_vnni16.kurn, q4_K_gemv.kurn, q8_0_gemm_amx.kurn, moe_expert_gemv.kurn; gpu/*.kurn
tests/               spec validation, registry consistency, golden codegen, numerics vs reference, harness, CLI
integration/         llama.cpp: KURN extra buffer type (apply.sh), the 0.1 vnni16 hook patch, checkers, AMX test
contrib/             ggml-linked harness (check against ggml's own kernels); gpu-check/ (hand-run GPU kit)
```

Environment variables (details in [`src/kurn/toolchain.py`](src/kurn/toolchain.py)):

| variable | meaning |
|---|---|
| `KURN_CC` | C compiler for host targets, e.g. `KURN_CC=clang` or `KURN_CC=gcc-12` (default: `$CC`, then gcc, cc, clang) |
| `KURN_CROSS_CC` | AArch64 cross compiler for `neon` on x86 (default: `aarch64-linux-gnu-gcc`, then `zig cc`) |
| `KURN_QEMU` | command prefix that runs AArch64 binaries on x86 (default: `qemu-aarch64 -L /usr/aarch64-linux-gnu`) |
| `KURN_CACHE_DIR` | build cache (default: `~/.cache/kurn`) |
| `KURN_NVCC` | nvcc for `target cuda` (default: `nvcc` on PATH, then `/usr/local/cuda*/bin/nvcc`) |
| `KURN_CXX` | host C++ compiler for the CUDA CPU emulator (default: `$CXX`, then g++, clang++) |
| `KEMU_SEED` | randomize the CUDA emulator's thread schedule (exposes shared-memory races) |
| `KURN_GPU_DISPATCH` | `dispatch.json` used by `kurn.gpu.dispatch.kernel_for()` |

## Contributing and license

See [CONTRIBUTING.md](CONTRIBUTING.md). Licensed under the [MIT License](LICENSE) (copyright: kurn contributors). The quantization block layouts follow
[ggml](https://github.com/ggml-org/ggml) (MIT) for binary compatibility.
