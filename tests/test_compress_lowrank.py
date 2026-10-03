"""kurn.lowrank: weighted residual SVD, int8 factors, bit accounting."""

import numpy as np
import pytest

from kurn import lowrank as lr


def test_full_rank_recovers_residual():
    rng = np.random.default_rng(0)
    W = rng.standard_normal((24, 40))
    Wq = np.round(W * 2) / 2
    h = rng.lognormal(0, 1, 40)
    U, V = lr.factors(lr.weighted_svd(W - Wq, h), 24)
    np.testing.assert_allclose(Wq + U @ V, W, atol=1e-10)


def test_weighted_svd_beats_unweighted_under_weighted_error():
    rng = np.random.default_rng(1)
    W = rng.standard_normal((64, 128)) + np.outer(rng.standard_normal(64), rng.standard_normal(128))
    Wq = np.round(W * 2) / 2
    h = rng.lognormal(0, 2, 128)
    Uw, Vw = lr.factors(lr.weighted_svd(W - Wq, h), 8)
    Uu, Vu = lr.factors(lr.weighted_svd(W - Wq, None), 8)
    assert lr.rel_err(W, Wq + Uw @ Vw, h) < lr.rel_err(W, Wq + Uu @ Vu, h)


def test_rank_curve_monotone_and_bits():
    rng = np.random.default_rng(2)
    W = rng.standard_normal((128, 256)) + 3 * np.outer(rng.standard_normal(128), rng.standard_normal(256))
    Wq = np.round(W)
    curve = lr.rank_curve(W, Wq, None, (0, 4, 16, 64))
    errs = [e for _, _, e in curve]
    assert all(b <= a * 1.02 for a, b in zip(errs, errs[1:]))  # int8 factor rounding may add a hair
    assert curve[2][1] == pytest.approx((8 * 16 * 384 + 32 * (128 + 16)) / (128 * 256))
    What, ((uq, us), (vq, vs)) = lr.lorc(W, Wq, None, 16)
    assert uq.dtype == np.int8 and vq.shape == (16, 256)
    assert lr.rel_err(W, What) == pytest.approx(curve[2][2])
    spec = lr.residual_spectrum(W, Wq)
    assert spec[-1] == pytest.approx(1.0) and np.all(np.diff(spec) >= -1e-12)
