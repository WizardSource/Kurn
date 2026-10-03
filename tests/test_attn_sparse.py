"""kurn.sparse: sparse SwiGLU FFN rows vs the harness's float64 reference of the same quantized
math (exact path: <= 1e-5 relative), for forced active sets and self-selected (threshold) rows."""

import pytest

from kurn import attention, sparse

pytestmark = pytest.mark.skipif(not attention.runnable("avx512")[0], reason="needs AVX-512")


@pytest.mark.parametrize(
    "mode,kw",
    [
        ("dense", {}),
        ("gate", {"density": 0.3}),
        ("gate", {"density": 0.0}),
        ("gate", {"thr": 0.05}),
        ("pred", {"density": 0.7, "rank": 64}),
        ("pred", {"thr": 0.02, "rank": 128}),
    ],
)
@pytest.mark.parametrize("threads", [1, 3])
def test_sparse_ffn_matches_reference(mode, kw, threads):
    row = sparse.bench(mode, d=512, dff=1000, threads=threads, secs=0, **kw)
    assert row["check"] == "ok", row
    if "density" in kw:
        assert abs(row["density"] - kw["density"]) < 0.08


def test_prefetch_variants_and_lib():
    for pf in (0, 4):
        assert sparse.bench("gate", density=0.5, d=256, dff=512, threads=2, secs=0, prefetch=pf)["check"] == "ok"
    assert sparse.build_lib().endswith(".so")


def test_bad_mode():
    with pytest.raises(ValueError):
        sparse.bench("nope")
