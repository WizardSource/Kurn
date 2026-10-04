"""Staged search (tune.search) and the plan cache (kurn.plan)."""

import json
import random

import pytest

from kurn import hooks, plan, spec, toolchain, tune
from kurn.cli import main


def _native(t="avx512_vnni"):
    return toolchain.run_mode(t)[0] == "native"


def test_search_space_has_no_auto_or_sentinels():
    s = tune.search_space("gemv", "q4_K", "avx512_vnni")
    for k, vals in s.items():
        assert "auto" not in vals
        assert hooks.AUTO_VALUES.get(k, "auto") not in vals
    assert {"layout", "unpack", "correction", "scales", "accum", "rows", "lanes", "plane", "rgroup"} <= set(s)
    assert "threads" not in s and "wait" not in s


def test_sampler_is_legal_distinct_and_covers_layouts():
    sp = {"op": "gemv", "weights": "q4_K", "target": "avx512_vnni"}
    s = tune.search_space("gemv", "q4_K", "avx512_vnni")
    cs, active, frac = tune.sample_legal(sp, s, 40, random.Random(1))
    assert len(cs) == 40 and 0 < frac < 1
    from kurn.kernels import generate

    assert len({generate(c) for _, c in cs}) == 40
    assert {c["layout"] for _, c in cs} == {"native", "i16", "composed"}
    assert {"layout", "rows", "plane", "rgroup", "correction"} <= set(active) <= set(s)


def test_sampler_respects_fixed_keys():
    sp = {"op": "gemv", "weights": "q4_K", "target": "avx512_vnni", "layout": "native"}
    s = tune.search_space("gemv", "q4_K", "avx512_vnni", exclude=("layout",))
    cs, active, _ = tune.sample_legal(sp, s, 20, random.Random(2))
    assert all(c["layout"] == "native" for _, c in cs)
    assert set(active) <= {"rows", "act", "prefetch", "correction", "scales", "accum"}


@pytest.mark.skipif(not _native(), reason="needs avx512_vnni")
def test_search_runs_and_ranks():
    sp = {"op": "gemv", "weights": "q4_0", "target": "avx512_vnni", "threads": 1}
    s = {"layout": ["i16", "composed"], "rows": [1, 2], "rgroup": [1, 2]}
    res, st = tune.search(
        sp, s, "hot", "energy", n0=6, keep=2, secs0=0.01, secs=0.02, max_moves=1, extra=["--K", "512", "--N", "64"], log=lambda m: None
    )
    assert res and all(r["relerr"] < 1e-4 for r in res)
    assert [r["energy_uJ"] for r in res[1:]] == sorted(r["energy_uJ"] for r in res[1:])
    assert st["builds"] >= 4 and st["measurements"] >= 4 and st["space_raw"] == 8


def test_fingerprint_is_stable_and_complete():
    a, b = plan.fingerprint(), plan.fingerprint()
    assert a == b and plan.fingerprint_id(a) == plan.fingerprint_id(b)
    assert {"cpu", "flags", "caches", "cores", "cc", "kurn"} <= set(a)
    c = {**a, "cores": a["cores"] + 1}
    assert plan.fingerprint_id(c) != plan.fingerprint_id(a)


def test_plan_roundtrip_and_lookup(tmp_path):
    p = plan.Plan(str(tmp_path / "p.json"))
    c = spec.resolve({"op": "gemv", "weights": "q4_0", "target": "avx512_vnni", "layout": "composed", "rows": 2})
    k = plan.entry_key("gemv", "q4_0", 2048, 1024, "cold", 8)
    p.entries[k] = {
        "config": {x: c[x] for x in spec.CODEGEN_KEYS},
        "us": 1.0,
        "GBps": 1.0,
        "default_us": 2.0,
        "winner": "search0",
        "tune_s": 1.0,
    }
    p.save()
    got = plan.lookup("gemv", "q4_0", 2048, 1024, "cold", 8, path=str(tmp_path / "p.json"))
    assert got["layout"] == "composed" and got["rows"] == 2 and got["threads"] == 8
    assert plan.lookup("gemv", "q4_0", 2048, 999, "cold", 8, path=str(tmp_path / "p.json")) is None
    assert json.load(open(tmp_path / "p.json"))["entries"][k]["winner"] == "search0"


def _tiny_gguf(path):
    gguf = pytest.importorskip("gguf")
    import numpy as np

    w = gguf.GGUFWriter(str(path), "qwen3")
    q4_0 = np.zeros((64, 2048 // 32 * 18), dtype=np.uint8)
    w.add_tensor("blk.0.attn_q.weight", q4_0, raw_dtype=gguf.GGMLQuantizationType.Q4_0)
    w.add_tensor("blk.0.attn_norm.weight", np.ones(2048, dtype=np.float32))
    w.add_tensor("output.weight", np.zeros((32, 2048), dtype=np.float16))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def test_model_shapes(tmp_path):
    m = tmp_path / "tiny.gguf"
    _tiny_gguf(m)
    shapes = {s[0]: s[1:] for s in plan.model_shapes(str(m))}
    assert shapes["blk.0.attn_q.weight"] == ("Q4_0", "q4_0", 2048, 64)
    assert shapes["output.weight"][1] is None  # F16: not a kurn format
    assert "blk.0.attn_norm.weight" not in shapes


@pytest.mark.skipif(not _native(), reason="needs avx512_vnni")
def test_plan_build_show_lookup_cli(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("KURN_CACHE_DIR", str(tmp_path / "cache"))
    m = tmp_path / "tiny.gguf"
    _tiny_gguf(m)
    p, st = plan.build_plan(str(m), threads=2, regime="hot", n0=3, top=1, secs=0.02, reps=1, refine=False, log=lambda s: None)
    assert st["tuned"] == 1 and len(p.entries) == 1
    ((k, e),) = p.entries.items()
    assert k == "gemv/q4_0/K2048xN64/hot2" and e["config"]["weights"] == "q4_0"
    p2, st2 = plan.build_plan(str(m), threads=2, regime="hot", n0=3, top=1, secs=0.02, reps=1, refine=False, log=lambda s: None)
    assert st2["tuned"] == 0  # reused
    cfgs = plan.for_model(str(m), "hot", 2)
    assert list(cfgs) == ["blk.0.attn_q.weight"]
    rc = plan.reuse_cost(str(m), "hot", 2)
    assert rc["tensors"] == 1 and rc["kernels"] == 1
    assert main(["plan", "show"]) == 0
    assert "gemv/q4_0/K2048xN64/hot2" in capsys.readouterr().out
    assert main(["plan", "lookup", "gemv", "q4_0", "2048", "64", "--regime", "hot", "--threads", "2"]) == 0
    assert main(["plan", "lookup", "gemv", "q4_0", "2048", "65", "--regime", "hot", "--threads", "2"]) == 1
