"""Golden tests: generated C for representative configs must match tests/golden/
byte for byte. After an intentional codegen change, regenerate with

    KURN_UPDATE_GOLDEN=1 pytest tests/test_codegen_golden.py

and review the diff. Also checks that every legal config compiles warning-free."""

import os
import shutil
import subprocess

import pytest

from kurn import hooks
from kurn.kernels import embed, generate
from kurn.spec import legal_configs, load, resolve
from kurn.toolchain import BuildError, cc_for, compile_source, data_path

from conftest import EXAMPLES, ROOT

GOLDEN_DIR = ROOT / "tests" / "golden"

G = {"op": "gemv", "weights": "q8_0"}
K4 = {"op": "gemv", "weights": "q4_K"}
MM = {"op": "gemm", "weights": "q8_0"}
GOLDEN = {
    "q8_0_gemv_scalar": {**G, "target": "scalar"},
    "q8_0_gemv_avx2_rows4": {**G, "target": "avx2", "rows": 4},
    "q8_0_gemv_avx2_vnni_once_rows8_pf16": {**G, "target": "avx2_vnni", "rows": 8, "prefetch": 16},
    "q8_0_gemv_avx512_once_rows4_pf8": {**G, "target": "avx512_vnni", "rows": 4, "prefetch": 8},
    "q8_0_gemv_avx512_inline_rows2": {**G, "target": "avx512_vnni", "rows": 2, "act": "inline"},
    "q8_0_gemv_avx512_vnni16_packed_rows8": {**G, "target": "avx512_vnni", "layout": "vnni16", "rows": 8},
    "q8_0_gemv_avx512_vnni16_align64_rows4_pf8": {**G, "target": "avx512_vnni", "layout": "vnni16", "align": 64, "rows": 4,
                                                  "prefetch": 8},
    "q8_0_gemv_neon_rows4_pf8": {**G, "target": "neon", "rows": 4, "prefetch": 8},
    "q4_K_gemv_scalar": {**K4, "target": "scalar"},
    "q4_K_gemv_avx2_rows4_pf4": {**K4, "target": "avx2", "rows": 4, "prefetch": 4},
    "q4_K_gemv_avx2_vnni_rows2": {**K4, "target": "avx2_vnni", "rows": 2},
    "q4_K_gemv_avx512_rows4_pf4": {**K4, "target": "avx512_vnni", "rows": 4, "prefetch": 4},
    "q8_0_gemm_scalar": {**MM, "target": "scalar"},
    "q8_0_gemm_avx512_rows6_cols4": {**MM, "target": "avx512_vnni", "rows": 6, "cols": 4},
    "q8_0_gemm_amx_rows2_cols1": {**MM, "target": "amx", "rows": 2, "cols": 1},
    # v0.2 generic recipe lowering: interleaved (i16 / i8), lookup-table (l32) and multi-token verify
    "q4_0_gemv_avx512_i16_rows2_pf8": {"op": "gemv", "weights": "q4_0", "target": "avx512_vnni", "layout": "i16",
                                       "rows": 2, "prefetch": 8},
    "q4_0_gemv_scalar": {"op": "gemv", "weights": "q4_0", "target": "scalar"},
    "iq4_nl_gemv_avx2_vnni_i8_rows2": {"op": "gemv", "weights": "iq4_nl", "target": "avx2_vnni", "layout": "i8", "rows": 2},
    "q2_0_gemv_avx512_l32_rows2": {"op": "gemv", "weights": "q2_0", "target": "avx512_vnni", "layout": "l32", "rows": 2},
    "tq2_0_gemv_avx512_i16_rows1": {"op": "gemv", "weights": "tq2_0", "target": "avx512_vnni", "layout": "i16", "rows": 1},
    "q1_0_gemv_avx512_l32_rows1": {"op": "gemv", "weights": "q1_0", "target": "avx512_vnni", "layout": "l32", "rows": 1},
    "q8_0_verify_avx512_i16_cols4": {"op": "verify", "weights": "q8_0", "target": "avx512_vnni", "layout": "i16", "cols": 4},
    "q4_K_verify_avx512_i16_cols2": {"op": "verify", "weights": "q4_K", "target": "avx512_vnni", "layout": "i16", "cols": 2},
    "q4_0_verify_avx2_vnni_i8_cols4": {"op": "verify", "weights": "q4_0", "target": "avx2_vnni", "layout": "i8", "cols": 4},
    "iq4_nl_verify_avx512_i16_cols8": {"op": "verify", "weights": "iq4_nl", "target": "avx512_vnni", "layout": "i16",
                                       "cols": 8},
    "q2_0_verify_avx512_i16_cols2": {"op": "verify", "weights": "q2_0", "target": "avx512_vnni", "layout": "i16", "cols": 2},
    "tq2_0_verify_avx512_i16_cols4": {"op": "verify", "weights": "tq2_0", "target": "avx512_vnni", "layout": "i16",
                                      "cols": 4},
    "q1_0_verify_avx512_i16_cols2": {"op": "verify", "weights": "q1_0", "target": "avx512_vnni", "layout": "i16", "cols": 2},
}  # fmt: skip
GOLDEN.update(hooks.GOLDEN)


def _check_golden(name, src):
    path = GOLDEN_DIR / f"{name}.c"
    if os.environ.get("KURN_UPDATE_GOLDEN"):
        GOLDEN_DIR.mkdir(exist_ok=True)
        path.write_text(src)
    assert path.exists(), f"missing golden file {path}; run with KURN_UPDATE_GOLDEN=1"
    assert src == path.read_text(), f"generated code for {name} changed; review and run with KURN_UPDATE_GOLDEN=1"


@pytest.mark.parametrize("name", sorted(GOLDEN))
def test_golden(name):
    _check_golden(name, generate(resolve(GOLDEN[name])))


def test_golden_embed():
    spec, _ = load(EXAMPLES / "q8_0_gemv_vnni16.kurn")
    _check_golden("embed_q8_0_gemv_vnni16", embed(generate(resolve(spec)), "kurn_"))


def test_no_stale_golden_files():
    expected = {f"{n}.c" for n in GOLDEN} | {"embed_q8_0_gemv_vnni16.c"}
    assert {p.name for p in GOLDEN_DIR.glob("*.c")} == expected


def test_generation_is_deterministic():
    for cfg in GOLDEN.values():
        c = resolve(cfg)
        assert generate(c) == generate(dict(c))


def _compilable(target):
    try:
        cc_for(target)
        return True
    except BuildError:
        return False


@pytest.mark.parametrize(
    "c",
    [c for c in legal_configs()],
    ids=lambda c: "-".join(str(c[k]) for k in ("weights", "op", "target", "layout", "align", "rows", "cols", "act", "prefetch")),
)
def test_every_legal_config_compiles_without_warnings(c):
    if not _compilable(c["target"]):
        pytest.skip(f"no compiler for {c['target']}")
    compile_source(generate(c), c["target"], stem="warn", extra_flags=("-Wall", "-Wextra", "-Wshadow", "-Werror"), shared=False)


def test_embedded_code_exports_no_symbols(tmp_path):
    if shutil.which("nm") is None:
        pytest.skip("nm not available")
    spec, _ = load(EXAMPLES / "q8_0_gemv_vnni16.kurn")
    src = '#include "kurn.h"\n' + embed(generate(resolve(spec)), "kurn_") + "\nvoid *use(void) { return (void *)kurn_kq8_gemv_packed; }\n"
    obj = compile_source(src, "avx512_vnni", out_dir=str(tmp_path), stem="embed", shared=False)
    syms = subprocess.run(["nm", "-g", "--defined-only", obj], capture_output=True, text=True, check=True).stdout.split()
    assert syms[-1] == "use" and len(syms) == 3, syms  # only the test's own `use` symbol is global


def test_header_is_bundled():
    assert os.path.exists(data_path("kurn.h")) and os.path.exists(data_path("bench.c"))
