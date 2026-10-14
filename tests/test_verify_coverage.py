"""`kurn verify`'s default check must reach every kernel's main loop.

Regression: the default check ran one shape, N=200 over at most 3 threads (~67 rows per thread). A rows=8
vnni16 Q8_0 GEMV processes 8 groups of 16 rows per pass, so only its tail code ran, and a bug planted in
the main loop passed `kurn verify` (found by the RSI loop's mutation control)."""

import pytest

from kurn.harness import CHECK_THREADS, check, check_shapes, main_loop_rows
from kurn.kernels import generate
from kurn.spec import resolve
from kurn.toolchain import build, compile_source, run_mode

SITE = "_mm512_loadu_si512(blk0->q + 0), xk)"
MUTANT = "_mm512_loadu_si512(blk0->q + 0), _mm512_setzero_si512())"


def vnni16(rows):
    return resolve({"op": "gemv", "weights": "q8_0", "target": "avx512_vnni", "layout": "vnni16", "rows": rows,
                    "align": "packed", "threads": 8})


def _shape_n(shape):
    return int(shape[shape.index("--N") + 1])


@pytest.mark.parametrize("cfg", [
    {"op": "gemv", "weights": "q8_0", "target": "avx512_vnni", "layout": "vnni16", "rows": 8},
    {"op": "gemv", "weights": "q8_0", "target": "avx512_vnni", "layout": "i16", "rows": 8},
    {"op": "gemv", "weights": "q4_K", "target": "avx512_vnni", "rows": 4},
    {"op": "gemv", "weights": "q8_0", "target": "avx2", "rows": 8},
    {"op": "gemm", "weights": "q8_0", "target": "avx512_vnni", "rows": 6, "cols": 4},
], ids=lambda c: f"{c['weights']}-{c['op']}-{c.get('layout', 'native')}-r{c['rows']}")
def test_default_shapes_cover_a_full_pass_per_thread(cfg):
    c = resolve({**cfg, "threads": 8})
    shapes = check_shapes(c)
    assert len(shapes) == 2
    t = min(CHECK_THREADS, c["threads"])
    # thread chunks are aligned to 16 rows; each still holds a full pass
    assert _shape_n(shapes[1]) // t - 16 >= main_loop_rows(c)
    assert _shape_n(shapes[0]) == (100 if c["op"] == "gemm" else 200)


@pytest.mark.parametrize("rows", [1, 2, 4, 8])
def test_planted_main_loop_bug_is_caught(tmp_path, rows):
    mode, why = run_mode("avx512_vnni")
    if mode != "native":
        pytest.skip(why or "avx512_vnni not native here")
    c = vnni16(rows)
    src = generate(c)
    assert SITE in src
    mutant = compile_source(src.replace(SITE, MUTANT, 1), c["target"], str(tmp_path), "mutant")
    good = check(build(c), c)
    bad = check(mutant, c)
    assert good["check"] == "ok", good
    assert bad["check"] == "FAIL", bad
    if rows == 8:
        # the old default shape alone misses it: the bug only lives in the 8-group main loop
        assert check(mutant, c, extra=check_shapes(c)[0])["check"] == "ok"
