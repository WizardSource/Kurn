"""The autotuner's confirmation gate (tune._confirm / tune.paired_gate) on synthetic timing noise.

The old rule (median of 3 interleaved repetitions must beat the incumbent by > min_gain) admitted 2 of 10
real A/A trials on a shared VM. The paired-CI gate must keep A/A admissions near zero while still
admitting real wins."""

import math
import random
import statistics

import pytest

from kurn import spec, tune

BASE = {"op": "gemv", "weights": "q8_0", "target": "avx512_vnni", "layout": "vnni16", "rows": 8, "threads": 8}


def noisy(monkeypatch, means, sigma, seed, outlier=0.0):
    """Fake measurements: lognormal noise around a per-candidate mean, plus rare slow outliers."""
    rng = random.Random(seed)

    def measure(c, so, *args):
        t = means[so] * math.exp(rng.gauss(0, sigma))
        if outlier and rng.random() < outlier:
            t *= 1.5
        return {"us": t, "cpu_us": t, "energy_uJ": t, "edp": t * t, "GBps": 1.0, "relerr": 0.0}

    monkeypatch.setattr(tune, "_measure", measure)


def pair():
    return ({"c": spec.resolve(BASE), "so": "a"}, {"c": spec.resolve({**BASE, "rows": 4}), "so": "b"})


def old_rule(means, sigma, rng, min_gain=0.02):
    a = [means["a"] * math.exp(rng.gauss(0, sigma)) for _ in range(3)]
    b = [means["b"] * math.exp(rng.gauss(0, sigma)) for _ in range(3)]
    return statistics.median(b) < statistics.median(a) * (1 - min_gain)


def test_aa_false_admissions(monkeypatch):
    means, sigma, trials = {"a": 100.0, "b": 100.0}, 0.10, 300
    admitted = 0
    for seed in range(trials):
        noisy(monkeypatch, means, sigma, seed, outlier=0.05)
        a, b = pair()
        tune._confirm(a, b, "hot", "us", 0.1, (), None, 0, 0.02)
        admitted += b["_gate"] == 1
    rng = random.Random(0)
    old = sum(old_rule(means, sigma, rng) for _ in range(trials))
    assert admitted <= trials * 0.01, admitted
    assert old >= trials * 0.2, old  # the rule it replaces


def test_real_win_is_admitted(monkeypatch):
    admitted = 0
    for seed in range(100):
        noisy(monkeypatch, {"a": 100.0, "b": 90.0}, 0.03, seed)
        a, b = pair()
        tune._confirm(a, b, "hot", "us", 0.1, (), None, 0, 0.02)
        admitted += b["_gate"] == 1
    assert admitted >= 95


def test_win_below_min_gain_is_not_admitted(monkeypatch):
    admitted = 0
    for seed in range(100):
        noisy(monkeypatch, {"a": 100.0, "b": 99.0}, 0.01, seed)
        a, b = pair()
        tune._confirm(a, b, "hot", "us", 0.1, (), None, 0, 0.02)
        admitted += b["_gate"] == 1
    assert admitted <= 5


def test_clear_loss_is_rejected_early(monkeypatch):
    noisy(monkeypatch, {"a": 100.0, "b": 130.0}, 0.03, 1)
    a, b = pair()
    tune._confirm(a, b, "hot", "us", 0.1, (), None, 0, 0.02)
    assert b["_gate"] == 0 and b["_pairs"] == tune.CONFIRM_MIN_PAIRS


def test_gate_is_deterministic_and_three_valued():
    r = [0.95, 0.97, 0.96, 0.99, 0.94, 0.96]
    assert tune.paired_gate(r, 0.02) == tune.paired_gate(list(r), 0.02)
    assert tune.paired_gate([0.80] * 6, 0.02)[0] == 1
    assert tune.paired_gate([1.10] * 6, 0.02)[0] == 0
    assert tune.paired_gate([0.90, 1.05, 0.95, 1.02, 0.97, 1.0], 0.02)[0] is None


@pytest.mark.parametrize("bad", ["min_pairs", "max_pairs"])
def test_pair_limits_validated(monkeypatch, bad):
    noisy(monkeypatch, {"a": 1.0, "b": 1.0}, 0.0, 0)
    a, b = pair()
    kw = {"min_pairs": 0} if bad == "min_pairs" else {"min_pairs": 6, "max_pairs": 4}
    with pytest.raises(spec.SpecError):
        tune._confirm(a, b, "hot", "us", 0.1, (), None, 0, 0.02, **kw)
