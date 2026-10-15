# Hypercomplex linear layers: compact quaternion and complex weights (CPU)

A quaternion layer stores 4 reals per weight where the equivalent real layer stores 16, and a complex layer 2 instead
of 4. `kurn-hc.cpp` computes the Hamilton / complex product directly from the compact weights, so memory-bound decode
streams 4x / 2x fewer weight bytes, while every output is bit-identical to kurn's real Q8_0 kernels on the expanded
matrix. Results: `docs/kurn-matmul-opt.md` (project store), section "Compact quaternion and complex layers".

## Layouts

- **Vectors (block-planar):** groups of 32 numbers, each stored component by component: element (k, c) at
  `(k / 32) * 32 D + c * 32 + k % 32`. Layer outputs use the same layout, so layers chain.
- **`KURN_HC_Q8`:** one fp16 scale per (row, component, block of 32 inputs), Q8_0-style. Per row group of 16 outputs
  and input block, the D component records are contiguous, each a kurn i16 Q8_0 record (16 scales, then 8 lines of
  64 bytes holding 4 values of each of the 16 rows, +128 bias). That is 1.0625 bytes per real parameter, or
  4.25 bytes per quaternion weight against 17 for the expansion.
- **`KURN_HC_F32`:** per row group and input, D x 16 floats.
- **Real expansion:** output component o takes input component q through weight component `p = o ^ q` with
  sign `S[o][q]` (quaternion `[[+,-,-,-],[+,+,-,+],[+,+,+,-],[+,-,+,+]]`, complex `[[+,-],[+,+]]`). With
  block-planar rows and columns, every 32-value block of an expanded row is one component of one compact row,
  possibly negated, so a `KURN_HC_Q8` layer expands to an ordinary Q8_0 matrix with the same scales
  (`kurn_hc_expand`).

## Kernels

| columns | kernel |
|---|---|
| 1-8 | GEMV / verify: per input block, the int32 dots of every (weight component, input component) pair are computed from shared loads. The loop runs input-component-outer: since `p = o ^ q`, each finished dot is scaled and accumulated straight into its output (`fma(float(s * isum), d_w * d_x, acc)`), in the order kurn's real Q8_0 GEMV uses on the expansion. One weight stream per row group carries all components (2 row groups per pass for quaternion decode, work-stealing schedule) |
| 9-31, or any width in exact mode | the same kernels over 8-column groups and L2-sized row chunks |
| 32+ (AMX CPUs, not exact) | AMX-BF16 GEMM on the expansion, built per 128-row L2 panel straight from the compact records (kurn-buft.cpp's per-record dequantization and tile loop with its preemption guard) |

Every path is bit-identical to kurn's real kernels on the expanded Q8_0 matrix at the same width and mode (GEMV,
verify, AMX-INT8 / verify-kernel exact prefill, AMX-BF16 prefill): `hc_bench check`. Activations are quantized with
ggml's `quantize_row_q8_0`, reproduced bit for bit (`hc_bench quant`).

**Gauss 3-multiplication (complex):** not used. On the int8 paths it loses: `a + b` and `xr + xi` need 9 bits, which
halves VNNI's products per instruction, and a shared scale for `a + b` breaks the per-component Q8_0 scales and the
bit-exact expansion. On the compute-bound AMX-BF16 prefill it would cut tile work to 3/4; see the results doc for
the accuracy cost.

## Build and run

```sh
L=~/src/llama-kurn                                   # llama.cpp with the KURN buffer type (apply.sh), built
g++ -O3 -march=native -mno-avx512fp16 -fopenmp -c kurn-hc.cpp -o kurn-hc.o
gcc -O2 -march=native -mno-avx512fp16 -I. -I$L/ggml/include hc_bench.c kurn-hc.o -L$L/build/bin -lggml -lggml-base \
    -lggml-cpu -lm -lstdc++ -fopenmp -Wl,-rpath,$L/build/bin -o hc_bench
export OMP_PROC_BIND=close OMP_PLACES=cores OMP_WAIT_POLICY=active
./hc_bench quant                                     # activation quantizer == ggml's
./hc_bench check 4 q8 1024 1024 "1 2 3 5 8 9 16 32 64"   # quaternion; GGML_KURN_EXACT=1 for exact mode; D=2 complex
./hc_bench bench 4 q8 3072 1024 1 8 9 40 compact     # vs ... expanded (same layers through ggml's KURN buffer)
./hc_bench mlp 4 q8 1024 3072 8 1 8 9 check          # synthetic quaternion MLP: compact vs expanded, output and time
```

`hc_bench check` also verifies the expansion against the Hamilton / complex product in double, and runs the
Freivalds oracle (bigVerify-style, as `test_kurn_buft freivalds`) on the AMX-BF16 outputs. `tests/test_hypercomplex.py`
covers the algebra and layout references (`kurn.hypercomplex`) and, with `KURN_LLAMA_CPP`, the compiled checks.

AMX-BF16 prefill runs only where kurn's KURN buffer type would use AMX: it follows `GGML_KURN_AMX=0` and kurn's AMX
self-test (`kurn_amx_check()` in libggml-cpu), so compact and expanded layers never run different kernels.

Switches: `KURN_HC_AMX=0` (no AMX-BF16 prefill), `KURN_HC_AMX_MIN` (32), `KURN_HC_G` (row groups per pass),
`KURN_HC_PO=1` (weight-component-outer kernel, for comparison).

## What ggml / GGUF integration needs

- **Types:** a GGUF tensor type (or a KV flag on a Q8_0 / F32 tensor) marking a weight as a compact quaternion /
  complex layer with its D and the component-major layout; the converter writes `W[p, n, k]`.
- **Op:** either a new `GGML_OP_MUL_MAT_HC` (src0 compact weights, src1 block-planar activations, op param D) or a
  `MUL_MAT` on a compact type whose `ne` describe the real shape (D Nq x D Kq) so graphs and shapes stay real. The
  second keeps model graphs unchanged: the KURN buffer type takes the tensor (`supports_op`, `buf_init_tensor`
  repacking into `KURN_HC_Q8` records), and `compute_forward` calls `kurn_hc_matmul`'s kernels with ggml's
  threadpool instead of OpenMP.
- **Layout contract:** activations must be block-planar. Elementwise ops are layout-agnostic; norms over the hidden
  dimension are permutation-invariant. Only rope and attention head splits care, so a model either uses
  block-planar hidden states throughout (permuting q/k/v rows at conversion time) or inserts a permute at those
  points.
- **Backends without kurn:** a reference expansion at load (`kurn_hc_expand` to Q8_0) keeps any ggml backend working,
  at 4x / 2x the memory.
