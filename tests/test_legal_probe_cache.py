"""Legality-probe reuse must not alter sampling or the reported fraction."""

import random
from unittest.mock import Mock

import pytest

from kurn import tune
from kurn.spec import SpecError

BASE = {"op": "gemv", "weights": "q8_0", "target": "avx2", "threads": 1}


def uncached_fraction(spec, active, rng):
    """The original per-draw calculation, retained as a reference."""
    legal = 0
    for _ in range(2000):
        try:
            tune.resolve(spec, {k: rng.choice(v) for k, v in active.items()})
            legal += 1
        except SpecError:
            pass
    return legal / 2000


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("active", [
    {"rows": [1, 2, 4, 8], "prefetch": [0, 8]},
    {"rows": [1, 3, 4], "prefetch": [0, 3, 8]},
    {"rows": [3, 5]},
    {"rows": [1, 1, 1, 3]},
    {"wait": ["spin", "sleep"], "threads": [1, 2], "rows": [1, 2]},
])
def test_fraction_and_rng_match_reference(seed, active):
    old, new = random.Random(seed), random.Random(seed)
    assert tune._probe_legal_fraction(BASE, active, new) == uncached_fraction(BASE, active, old)
    assert new.getstate() == old.getstate()


def test_repeated_draws_keep_their_weight_and_cache_false(monkeypatch):
    class Draws:
        def __init__(self):
            self.calls = 0

        def choice(self, _):
            self.calls += 1
            return 1 if self.calls <= 1999 else 3

    wrapped = Mock(wraps=tune.resolve)
    monkeypatch.setattr(tune, "resolve", wrapped)
    rng = Draws()
    assert tune._probe_legal_fraction(BASE, {"rows": [1, 3]}, rng) == 1999 / 2000
    assert wrapped.call_count == 2
    assert rng.calls == 2000

    wrapped.reset_mock()
    assert tune._probe_legal_fraction(BASE, {"rows": [3]}, random.Random(0)) == 0
    assert wrapped.call_count == 1


def test_valid_cache_is_local_to_one_call(monkeypatch):
    wrapped = Mock(wraps=tune.resolve)
    monkeypatch.setattr(tune, "resolve", wrapped)
    assert tune._probe_legal_fraction(BASE, {"rows": [1]}, random.Random(0)) == 1
    assert wrapped.call_count == 1
    wrapped.side_effect = SpecError("registry changed between calls")
    assert tune._probe_legal_fraction(BASE, {"rows": [1]}, random.Random(0)) == 0
    assert wrapped.call_count == 2


def test_cache_does_not_cross_specs():
    active = {"prefetch": [8]}
    assert tune._probe_legal_fraction(BASE, active, random.Random(0)) == 1
    assert tune._probe_legal_fraction({**BASE, "target": "scalar"}, active, random.Random(0)) == 0


def test_unexpected_exception_preserves_draw_position(monkeypatch):
    class Draws:
        def __init__(self):
            self.calls = 0

        def choice(self, _):
            self.calls += 1
            return 1 if self.calls < 5 else 2

    resolve = tune.resolve

    def bad(spec, overrides):
        if overrides["rows"] == 2:
            raise RuntimeError("unexpected resolver failure")
        return resolve(spec, overrides)

    monkeypatch.setattr(tune, "resolve", bad)
    for fn in (tune._probe_legal_fraction, uncached_fraction):
        rng = Draws()
        with pytest.raises(RuntimeError, match="unexpected resolver"):
            fn(BASE, {"rows": [1, 2]}, rng)
        assert rng.calls == 5


def test_all_distinct_draws_are_still_checked(monkeypatch):
    class Draws:
        def __init__(self):
            self.calls = 0

        def choice(self, _):
            self.calls += 1
            return self.calls

    def check(_spec, overrides):
        if overrides["value"] % 2:
            raise SpecError("odd")
        return {}

    wrapped = Mock(side_effect=check)
    monkeypatch.setattr(tune, "resolve", wrapped)
    assert tune._probe_legal_fraction({}, {"value": range(1, 2001)}, Draws()) == 0.5
    assert wrapped.call_count == 2000


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("spec,space,n", [
    (BASE, {"rows": [1, 2, 4, 8], "prefetch": [0, 8]}, 8),
    ({**BASE, "target": "avx512_vnni"},
     {"layout": ["native", "vnni16"], "align": ["packed", 64],
      "rows": [1, 2, 4, 8], "act": ["once", "inline"], "prefetch": [0, 8]}, 12),
    (BASE, {"rows": [1, 2], "threads": [1, 2], "wait": ["spin", "sleep"]}, 8),
    ({**BASE, "target": "scalar"}, {"rows": [1, 2, 4, 8]}, 8),
    ({**BASE, "rows": 3}, {"rows": [1, 2]}, 2),
])
def test_full_sampler_preserves_records_and_rng(monkeypatch, seed, spec, space, n):
    actual_rng, expected_rng = random.Random(seed), random.Random(seed)
    actual = tune.sample_legal(spec, space, n, actual_rng)
    monkeypatch.setattr(tune, "_probe_legal_fraction", uncached_fraction)
    expected = tune.sample_legal(spec, space, n, expected_rng)
    assert actual == expected
    assert actual_rng.getstate() == expected_rng.getstate()
    assert [tune._source_key(c) for _, c in actual[0]] == [tune._source_key(c) for _, c in expected[0]]


def test_empty_active_path_does_not_probe(monkeypatch):
    probe = Mock(side_effect=AssertionError("empty active case needs no random probe"))
    monkeypatch.setattr(tune, "_probe_legal_fraction", probe)
    result, active, fraction = tune.sample_legal(BASE, {}, 1, random.Random(0))
    assert len(result) == 1
    assert active == {}
    assert fraction == 1
    probe.assert_not_called()


@pytest.mark.parametrize("seed", range(5))
def test_large_space_uses_original_probe_loop(monkeypatch, seed):
    space = {"rows": [1, 2, 4, 8], "prefetch": [0, 2, 4, 8, 16, 32],
             "threads": list(range(1, 129)), "wait": ["spin", "sleep"]}
    probe = Mock(side_effect=AssertionError("large space must use the original loop"))
    monkeypatch.setattr(tune, "_probe_legal_fraction", probe)
    actual_rng = random.Random(seed)
    result, active, fraction = tune.sample_legal(BASE, space, 24, actual_rng)
    assert len(result) == 24 and tune.space_size(active) >= 2000
    assert fraction == 1
    probe.assert_not_called()


@pytest.mark.parametrize("size,cached", [(1999, True), (2000, False)])
def test_cache_cardinality_boundary(monkeypatch, size, cached):
    # Repeated supplied values are intentional: weighting must still be per draw.
    values = [1] * (size // 2) + [2] * (size - size // 2)
    original = tune._probe_legal_fraction
    probe = Mock(wraps=original)
    monkeypatch.setattr(tune, "_probe_legal_fraction", probe)
    result, active, fraction = tune.sample_legal(BASE, {"rows": values}, 2, random.Random(0))
    assert len(result) == 2 and len(active["rows"]) == size and fraction == 1
    assert bool(probe.call_count) == cached
