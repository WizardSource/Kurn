"""Hybrid device routing: cpu / gpu / auto / hybrid (CPU and GPU together)."""

import json

from kurn.gpu.kit import config_string
from kurn.gpu.spec import resolve
from kurn.hybrid import plan, route

CFG_Q4 = config_string(resolve({"op": "gemv", "weights": "q4_0"}))


def _table(tmp_path, cells):
    path = tmp_path / "dispatch.json"
    path.write_text(json.dumps({
        "device": "Synthetic",
        "arch": "sm_80",
        "rule": "test",
        "cells": cells,
    }))
    return str(path)


def test_route_cpu_mode_ignores_gpu(tmp_path):
    table = _table(tmp_path, {"q4_0": {"1": {"impl": "kurn", "verdict": "win", "config": CFG_Q4}}})
    r = route("q4_0", 1, mode="cpu", table=table, gpu_ok=True)
    assert r["device"] == "cpu" and r["mode"] == "cpu"


def test_route_auto_uses_gpu_on_win(tmp_path):
    table = _table(tmp_path, {"q4_0": {"1": {"impl": "kurn", "verdict": "win", "config": CFG_Q4}}})
    r = route("q4_0", 1, mode="auto", arch="sm_80", table=table, gpu_ok=True)
    assert r["device"] == "gpu" and r["impl"] == "kurn" and r["config"]["weights"] == "q4_0"


def test_route_auto_falls_back_to_cpu_on_stock(tmp_path):
    table = _table(tmp_path, {"q8_0": {"1": {"impl": "stock", "verdict": "tie"}}})
    r = route("q8_0", 1, mode="auto", arch="sm_80", table=table, gpu_ok=True)
    assert r["device"] == "cpu"


def test_route_no_gpu_device(tmp_path):
    table = _table(tmp_path, {"q4_0": {"1": {"impl": "kurn", "verdict": "win", "config": CFG_Q4}}})
    r = route("q4_0", 1, mode="gpu", table=table, gpu_ok=False)
    assert r["device"] == "cpu" and "no CUDA" in r["reason"]


def test_hybrid_plan_splits_cpu_and_gpu(tmp_path):
    table = _table(tmp_path, {
        "q4_0": {"1": {"impl": "kurn", "verdict": "win", "config": CFG_Q4}},
        "q8_0": {"1": {"impl": "stock", "verdict": "tie"}},
    })
    ops = [
        {"fmt": "q4_0", "batch": 1, "K": 4096, "name": "attn_q"},
        {"fmt": "q8_0", "batch": 1, "K": 4096, "name": "ffn_down"},
        {"fmt": "q4_0", "batch": 1, "K": 4096, "name": "ffn_up"},
    ]
    p = plan(ops, mode="hybrid", arch="sm_80", table=table, gpu_ok=True)
    assert p["overlap"] is True
    assert {e["name"] for e in p["gpu"]} == {"attn_q", "ffn_up"}
    assert {e["name"] for e in p["cpu"]} == {"ffn_down"}


def test_hybrid_respects_min_k(tmp_path, monkeypatch):
    table = _table(tmp_path, {"q4_0": {"1": {"impl": "kurn", "verdict": "win", "config": CFG_Q4}}})
    monkeypatch.setenv("KURN_HYBRID_MIN_K", "2048")
    r = route("q4_0", 1, mode="hybrid", arch="sm_80", table=table, K=512, gpu_ok=True)
    assert r["device"] == "cpu" and "KURN_HYBRID_MIN_K" in r["reason"]
