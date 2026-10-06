"""Noise-robust tuning: interleaved rounds ranked on the median, refinement of the leaders until their order is stable,
and a warning (with a --secs / --rounds hint) when the spread is too large to rank. Uses a fake harness."""

import random

import pytest

from kurn import tune as T
from kurn.spec import load

from conftest import EXAMPLES

TRUE_US = {1: 100.0, 2: 108.0, 4: 130.0, 8: 160.0}  # rows -> true time per call


@pytest.fixture
def fake_harness(monkeypatch):
    state = {"noise": 0.01, "calls": [], "rng": random.Random(7), "outlier": None}

    def bench(so, c, regime, secs, extra, harness):
        state["calls"].append(c["rows"])
        us = TRUE_US[c["rows"]] * (1 + state["rng"].gauss(0, state["noise"]))
        if state["outlier"] and len(state["calls"]) == state["outlier"][0]:
            us *= state["outlier"][1]  # one disturbed measurement (another process woke up)
        return {"us": us, "cpu_us": us * c["threads"], "GBps": 1e3 / us, "GOPs": 1.0, "relerr": 1e-7, "check": "ok"}

    monkeypatch.setattr(T, "bench", bench)
    monkeypatch.setattr(T, "build", lambda c: f"fake-{c['rows']}.so")
    return state


def _spec():
    spec, _ = load(EXAMPLES / "q8_0_gemv_vnni16.kurn")
    return spec


SPACE = {"rows": [1, 2, 4, 8], "prefetch": [0], "threads": [1]}


def test_interleaved_rounds_and_median_ranking(fake_harness):
    fake_harness["outlier"] = (1, 3.0)  # the very first measurement (rows=1) is 3x slow
    res, _ = T.tune(_spec(), SPACE, objective="speed", secs=0.01, rounds=3, keep=2, log=lambda s: None)
    calls = fake_harness["calls"]
    assert calls[:4] == [1, 2, 4, 8] and calls[4:8] == [2, 4, 8, 1] and calls[8:12] == [4, 8, 1, 2]  # rotated rounds
    assert [r["rows"] for r in res] == [1, 2, 4, 8]  # the median ignores the outlier; one 1 s run would rank rows=1 last
    assert res[0]["rounds"] >= 3 and res[0]["warnings"] == []
    assert abs(res[0]["us"] - 100) < 3


def test_leaders_refined_until_stable(fake_harness):
    logs = []
    res, _ = T.tune(_spec(), SPACE, objective="speed", secs=0.01, rounds=3, keep=2, log=logs.append)
    assert res[0]["rounds"] == res[1]["rounds"] >= 3 + T.STABLE_ROUNDS  # only the top 2 got extra rounds
    assert res[2]["rounds"] == res[3]["rounds"] == 3
    assert any("order stable" in s for s in logs)


def test_noise_too_large_to_rank_warns(fake_harness):
    fake_harness["noise"] = 0.25
    logs = []
    res, _ = T.tune(_spec(), SPACE, objective="speed", secs=0.01, rounds=3, keep=3, budget=0.0, log=logs.append)
    warns = res[0]["warnings"]
    assert warns and any("--secs" in w and "--rounds" in w for w in warns)
    assert any(s.startswith("warning:") for s in logs)


def test_single_round_is_the_old_behaviour(fake_harness):
    res, _ = T.tune(_spec(), SPACE, objective="speed", secs=0.01, rounds=1, keep=1, log=lambda s: None)
    assert len(fake_harness["calls"]) == 4 and all(r["rounds"] == 1 for r in res)


def test_rounds_must_be_positive(fake_harness):
    with pytest.raises(T.SpecError):
        T.tune(_spec(), SPACE, rounds=0, log=lambda s: None)
