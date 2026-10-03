"""The registries (formats, targets, kernels) and the C side (kurn.h, bench.c) must agree.
These tests are the checklist for adding a format, kernel or target."""

import re

import pytest

from kurn.formats import FORMATS
from kurn.kernels import KERNELS
from kurn.spec import OPS, SCHEDULE, TARGETS
from kurn.targets import TARGETS as TARGET_INFO
from kurn.toolchain import data_path

from conftest import BLOCKS
from test_codegen_golden import GOLDEN

HEADER = open(data_path("kurn.h")).read()
BENCH = open(data_path("bench.c")).read()
BENCH_ROWS = {m[0]: m for m in re.findall(r'\{"(\w+)", "(\w+)", &F_(\w+), &F_(\w+), (\w+), ([01]), "(\w+)"\}', BENCH)}
BENCH_FORMATS = set(re.findall(r'static const format_desc F_\w+ = \{"(\w+)"', BENCH))


@pytest.mark.parametrize("key", sorted(KERNELS), ids=lambda k: "-".join(k))
def test_kernel_is_fully_registered(key):
    k = KERNELS[key]
    assert k.weights in FORMATS and FORMATS[k.weights].reference, "weight format needs a Python reference"
    assert k.act in FORMATS, "activation format must be registered"
    assert set(k.targets) <= set(TARGET_INFO), "unknown target"
    assert re.search(rf"\b{k.entry}\(|_DECL\({k.entry}\)", HEADER), f"declare {k.entry} in kurn.h"
    row = BENCH_ROWS.get(k.bench)
    assert row, f"add a kernel_desc row for --kernel {k.bench} to bench.c"
    assert row[1] == f"{k.weights}_{k.op}" and row[5] == str(int(k.op in ("gemm", "verify"))) and row[6] == k.entry
    assert {k.weights, k.act} <= BENCH_FORMATS, "add a format_desc to bench.c"
    for fmt in (k.weights, k.act):
        assert fmt in BLOCKS, f"add a {fmt} block generator to tests/conftest.py"
    assert any((g["op"], g["weights"]) == key for g in GOLDEN.values()), "add a golden case"


def test_spec_tables_are_derived_from_registry():
    assert TARGETS == {key: k.targets for key, k in KERNELS.items()}
    assert {(op, w) for op, ws in OPS.items() for w in ws} == set(KERNELS)


def test_every_target_has_flags_and_schedule_values():
    for name, t in TARGET_INFO.items():
        assert t.flags and t.arch in ("x86_64", "aarch64", "any")
        for key, k in KERNELS.items():
            if name in k.targets:
                for allowed in SCHEDULE.values():
                    assert allowed(k.op, k.weights, name), (name, key)


def test_bits_per_weight():
    assert FORMATS["q8_0"].bits_per_weight == 8.5 and FORMATS["q4_K"].bits_per_weight == 4.5
