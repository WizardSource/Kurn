# ggml-linked harness (optional)

`bench_ggml.c` is the harness the published kurn results were measured with. It has the same command line and CSV output as the
bundled harness (`src/kurn/data/bench.c`), with two differences:

- **Data and reference come from ggml.** Weights are produced by ggml's own quantizers, and the reference is ggml's own `vec_dot`.
  It checks kurn kernels against exactly what llama.cpp computes.
- **ggml baselines are built in.** `--impl ggml` runs ggml's `vec_dot`, the path used with `--no-repack`. `--impl ggml-graph-cpu`,
  `ggml-graph-amx` and `ggml-graph-repack` run a full ggml `MUL_MAT` graph on that buffer type (activation quantization plus
  ggml's own threading).

## Build

Against a llama.cpp checkout built with shared libraries:

```sh
L=~/src/llama.cpp     # cmake -B build -DGGML_NATIVE=ON && cmake --build build -j --target ggml ggml-cpu
gcc -O3 -march=native -I src/kurn/data -I $L/ggml/include contrib/ggml-harness/bench_ggml.c -o bench_ggml \
    -L$L/build/bin -lggml -lggml-base -lggml-cpu -Wl,-rpath,$L/build/bin -lpthread -ldl -lm
```

It needs an AVX-512 host: the bandwidth test uses AVX-512 intrinsics directly.

## Use

```sh
./bench_ggml --impl ggml --kernel q8gemv --regime cold --threads 8                       # ggml baseline
./bench_ggml --impl $(kurn build examples/q8_0_gemv_vnni16.kurn) --kernel q8gemv --regime cold --threads 8
kurn tune examples/q8_0_gemv_vnni16.kurn --harness ./bench_ggml --regime cold -o tune.csv
```

`BENCH_CACHE=dir` caches the generated data, which saves re-quantizing 1.2 GB per process in the cold regime.
`--dump DIR` writes the weights, activations and reference for out-of-process runners.
