# Contributing to kurn

Thanks for your interest. kurn is small on purpose: a closed spec language, hand-written lowering templates, and a harness that
checks everything. Contributions that keep it that way are very welcome: new lowerings, new formats, bug reports with a failing
spec, and measurements on CPUs we have not tested (AVX2-only desktops, Zen 4/5, Ice Lake, real ARM hardware).

## Development setup

```sh
python -m venv .venv && . .venv/bin/activate
pip install -e '.[dev]'
pytest -q                        # ~1 minute; skips what the host CPU cannot run
ruff check . && ruff format --check .
kurn verify --all --strict       # every legal config, -Wall -Wextra -Wshadow -Werror
```

For NEON on an x86 machine, install an AArch64 cross compiler and qemu-user (Debian/Ubuntu:
`apt install gcc-aarch64-linux-gnu qemu-user`). `kurn targets` shows what your host can build and run.

## Golden files

`tests/golden/*.c` pin the generated code for representative configurations. If you change a lowering on purpose:

```sh
KURN_UPDATE_GOLDEN=1 pytest tests/test_codegen_golden.py
git diff tests/golden            # review every changed line
```

## Adding a format or kernel

Formats, targets and kernels are registries; everything downstream (spec validation, CLI, `verify`, `tune`, tests) reads them.
`tests/test_registry.py` fails until all the pieces below exist, so use it as the checklist. Take a 2-bit LUT GEMV
(`op gemv`, `weights q2_K`) as the example:

1. **Format** (`src/kurn/formats.py`): a `Format(...)` entry with block size, bytes per block, activation format and field layout,
   plus a pure-Python `_ref_<name>(wblocks, xblocks)` giving the exact dot product. Keep layouts byte-compatible with ggml when the
   format comes from ggml. A repacked layout of an existing format (like `vnni16`) is a `layout` value, not a new format.
2. **Lowering** (`src/kurn/codegen.py`): a function `(target, config) -> C source` implementing the kernel's `kurn.h` entry point.
   Unroll register tiles in Python so the C compiler sees constant indices. List any new file-scope helper names in
   `EMBED_HELPERS` so `--embed` prefixes them.
3. **Kernel** (`src/kurn/kernels.py`): a `Kernel(op, weights, targets, lower, entry, bench, doc)` entry.
4. **C side**: declare the entry point (and optional `<entry>_prepare` / `<entry>_packed`) and block struct in
   `src/kurn/data/kurn.h`. Add a `format_desc` (random valid blocks) and a `kernel_desc` row with an exact double-precision
   reference to `src/kurn/data/bench.c`.
5. **Schedule keys** (`src/kurn/spec.py`): extend the `SCHEDULE` value functions and `INVALID_COMBOS` for the new kernel's
   legal values. A new key (e.g. a LUT group size) goes in `SCHEDULE`, `DEFAULTS`, and in `CODEGEN_KEYS` if it changes the C.
6. **Tests**: a block generator in `tests/conftest.py`, a golden case in `tests/test_codegen_golden.py`, and the new count in
   `tests/test_spec.py::test_legal_config_count`. `tests/test_numerics.py` picks up every legal config automatically.
7. Run `kurn verify --all --strict` on hardware that has the targets, and say in the PR which machine you used.

A new **target** is a `Target(...)` entry in `src/kurn/targets.py` (flags, architecture, required `/proc/cpuinfo` flags, intrinsics
header), plus lowerings for the kernels that support it, listed in their `targets`.

## Performance claims

Numbers in PRs and docs should say:
- the CPU model and core count, and whether it is a VM;
- the regime (`hot` or `cold`), threads, and number of repetitions;
- whether energy is the CPU-time proxy or a measurement such as RAPL.

Re-run baselines in the same session; run-to-run noise on shared machines is often 5–10%.

## Pull requests

- One logical change per PR, with tests. CI must pass (lint, x86 tests, NEON cross-compile + qemu).
- Describe user-visible changes in `CHANGELOG.md` under *Unreleased*.
- By submitting a contribution you agree that it is licensed under the MIT License, the project license.
