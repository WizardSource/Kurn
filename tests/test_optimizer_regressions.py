"""Regression tests for tuning, plan reuse, and benchmark execution."""
import json
import math
import random
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from kurn import harness, kernels, plan, spec, toolchain, tune

BASE = {"op": "gemv", "weights": "q8_0", "target": "scalar", "threads": 1}


def row(us=10.0, cpu_us=None, **kw):
    return {"check": "ok", "us": us, "cpu_us": us if cpu_us is None else cpu_us,
            "GBps": 1.0, "GOPs": 1.0, "relerr": 1e-7, **kw}


def plan_fixture(monkeypatch, streams):
    alt = spec.resolve({**BASE, "rows": 2})
    monkeypatch.setattr(plan, "search", Mock(return_value=([{"config": {k: v for k, v in alt.items() if k != "act_format"}}],
                        {"builds": 0, "measurements": 0})))
    monkeypatch.setattr(plan, "build", lambda c: str(c["rows"]))
    its = {4 if k == 1 else k: iter(v) for k, v in streams.items()}
    def bench(so, c, *a):
        item = next(its[c["rows"]])
        if isinstance(item, Exception):
            raise item
        return item
    monkeypatch.setattr(plan, "bench", bench)


def run_plan(**kw):
    return plan.tune_entry("gemv", "q8_0", 64, 33, threads=1, target="scalar",
                           reps=3, refine=False, log=lambda _: None, **kw)


@pytest.mark.parametrize("objective,winner", [("speed", "search0"), ("energy", "search0"), ("edp", "default")])
def test_plan_objective_uses_paired_run_products(monkeypatch, objective, winner):
    plan_fixture(monkeypatch, {1: [row(1, 10), row(10, 1), row(11, 11)], 2: [row(6)] * 3})
    result = run_plan(objective=objective)
    assert result["winner"] == winner
    assert result["reps"] == 3
    expected_edp = 10 if winner == "default" else 36
    assert result["edp"] == pytest.approx(expected_edp * tune.PROXY_W_PER_CORE)


@pytest.mark.parametrize("bad", [row(check="FAIL"), harness.HarnessError("failed"), row(us=float("nan")),
                                   row(cpu_us=float("inf")), row(relerr=0.1), row(check="UNKNOWN")])
def test_plan_never_averages_away_one_failed_repetition(monkeypatch, bad):
    plan_fixture(monkeypatch, {1: [row(10)] * 3, 2: [row(1), bad, row(1)]})
    result = run_plan(objective="speed")
    assert result["winner"] == "default"
    assert "search0" not in result["candidates"]


def test_plan_all_failed_is_error(monkeypatch):
    plan_fixture(monkeypatch, {1: [row(check="FAIL")] * 3, 2: [row(check="FAIL")] * 3})
    with pytest.raises(harness.HarnessError):
        run_plan()


def test_plan_rounds_hot_slice_up(monkeypatch):
    plan_fixture(monkeypatch, {1: [row()] * 3, 2: [row()] * 3})
    run_plan()
    args = plan.search.call_args.kwargs["extra"]
    assert args[args.index("--N") + 1] == "48"


def test_plan_static_power_is_used_at_both_fidelities(monkeypatch):
    plan_fixture(monkeypatch, {1: [row(10, 1)] * 3, 2: [row(1, 3)] * 3})
    result = run_plan(objective="energy", static_w=10)
    assert result["winner"] == "search0"
    assert result["static_w"] == 10
    assert plan.search.call_args.kwargs["static_w"] == 10


@pytest.mark.parametrize("kwargs", [{"objective": "invalid"}, {"reps": 0}, {"threads": 0}, {"top": 0},
                                    {"K": 0}, {"N": -1}, {"secs": float("nan")}, {"static_w": -1}])
def test_plan_invalid_arguments_fail_before_search(monkeypatch, kwargs):
    search = Mock(side_effect=AssertionError("must reject before search"))
    monkeypatch.setattr(plan, "search", search)
    args = dict(op="gemv", fmt="q8_0", K=64, N=32, threads=1, target="scalar", reps=3)
    args.update(kwargs)
    with pytest.raises(spec.SpecError):
        plan.tune_entry(**args)
    search.assert_not_called()


def saved_plan(tmp_path, config=None):
    path = tmp_path / "p.json"
    fp = {"machine": "a"}
    p = plan.Plan(path=str(path), fp=fp)
    c = spec.resolve(config or BASE)
    p.entries[plan.entry_key("gemv", "q8_0", 64, 16, "hot", 1)] = {
        "config": {k: c[k] for k in spec.CODEGEN_KEYS}, "objective": "energy"}
    p.save()
    return path, fp


def test_explicit_path_matching_fingerprint_reuses(tmp_path):
    path, fp = saved_plan(tmp_path)
    assert plan.Plan(path=str(path), fp=fp).lookup("gemv", "q8_0", 64, 16, "hot", 1)


def test_explicit_path_mismatch_rejects_without_overwrite(tmp_path):
    path, fp = saved_plan(tmp_path)
    before = path.read_bytes()
    with pytest.raises(spec.SpecError, match="fingerprint"):
        plan.Plan(path=str(path), fp={**fp, "machine": "b"})
    assert path.read_bytes() == before


@pytest.mark.parametrize("mutation", ["id", "entries", "models", "invalid_json", "not_object"])
def test_plan_metadata_validation(tmp_path, mutation):
    path, fp = saved_plan(tmp_path)
    data = json.loads(path.read_text())
    if mutation == "invalid_json":
        path.write_text("{")
    else:
        if mutation == "not_object":
            data = []
        else:
            data[mutation] = "invalid"
        path.write_text(json.dumps(data))
    with pytest.raises(spec.SpecError):
        plan.Plan(path=str(path), fp=fp)


def test_plan_key_cannot_relabel_a_different_format(tmp_path):
    path, fp = saved_plan(tmp_path, {**BASE, "weights": "q4_K"})
    with pytest.raises(spec.SpecError):
        plan.Plan(path=str(path), fp=fp).lookup("gemv", "q8_0", 64, 16, "hot", 1)


def test_plan_request_objective_change_retunes(tmp_path, monkeypatch):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"fixture only: model reader mocked")
    monkeypatch.setattr(plan, "fingerprint", lambda: {"machine": "fixture"})
    monkeypatch.setattr(plan, "model_shapes", lambda _: [("w", "Q8_0", "q8_0", 64, 16)])
    def entry(*a, **kw):
        return {"config": spec.resolve(BASE), "objective": kw.get("objective", "energy"), "static_w": 0, "reps": kw.get("reps", 3),
                "us": 1, "GBps": 1, "winner": "default", "tune_s": 0, "default_us": 1}
    tuner = Mock(side_effect=entry)
    monkeypatch.setattr(plan, "tune_entry", tuner)
    path = str(tmp_path / "plan.json")
    for objective, reps, expected in [("energy", 3, 1), ("energy", 3, 0), ("energy", 5, 1), ("speed", 5, 1)]:
        _, stats = plan.build_plan(str(model), path=path, threads=1, objective=objective, reps=reps, log=lambda _: None)
        assert stats["tuned"] == expected
    assert tuner.call_count == 3


@pytest.mark.parametrize("changes", [dict(us=float("nan")), dict(us=0), dict(us=-1), dict(cpu_us=float("inf")),
                                     dict(cpu_us=-1), dict(GBps=float("nan")), dict(relerr=float("nan")),
                                     dict(relerr=0.01), dict(check="UNKNOWN"), dict(us=1e308, cpu_us=1e308)])
def test_measure_rejects_unusable_rows(monkeypatch, changes):
    monkeypatch.setattr(tune, "bench", lambda *a: row(**changes))
    assert tune._measure(spec.resolve(BASE), "fake.so", "hot", 0.1, (), None, 0) is None


@pytest.mark.parametrize("kwargs", [{"eta": 1}, {"eta": 0}, {"keep": 0}, {"n0": 0}, {"secs": 0},
                                    {"secs0": float("inf")}, {"min_gain": 1}, {"max_moves": -1}, {"jobs": 0},
                                    {"static_w": float("nan")}])
def test_search_rejects_invalid_parameters_before_sampling(monkeypatch, kwargs):
    sampler = Mock(side_effect=AssertionError("must reject before sampling"))
    monkeypatch.setattr(tune, "sample_legal", sampler)
    with pytest.raises(spec.SpecError):
        tune.search(BASE, {"rows": [1]}, **kwargs)
    sampler.assert_not_called()


def test_finalist_harness_error_does_not_abort_other_candidates(monkeypatch):
    configs = [spec.resolve({**BASE, "rows": r}) for r in (1, 2)]
    monkeypatch.setattr(tune, "sample_legal", lambda *a: ([(dict(rows=r), c) for r,c in zip((1,2),configs)], {"rows":[1,2]},1))
    monkeypatch.setattr(tune, "_build_all", lambda *a: {0: "1", 1: "2"})
    calls = {1: 0, 2: 0}
    def measure(c, *a):
        calls[c["rows"]] += 1
        if c["rows"] == 1 and calls[1] == 2:
            raise harness.HarnessError("final repetition failed")
        return {"us": 1., "cpu_us": 1., "energy_uJ": 5.47, "edp": 5.47, "GBps": 1., "relerr": 0.}
    monkeypatch.setattr(tune, "_measure", measure)
    result, _ = tune.search(BASE, {"rows":[1,2]}, n0=2, keep=2, refine=False, log=lambda _:None)
    assert len(result) == 1 and result[0]["config"]["rows"] == 2


def test_confirmation_one_failure_disqualifies(monkeypatch):
    a = {"c": spec.resolve(BASE), "so": "a"}
    b = {"c": spec.resolve({**BASE,"rows":2}), "so": "b"}
    outputs = iter([{"us":10}, {"us":1}, {"us":10}, None, {"us":10}, {"us":1}])
    monkeypatch.setattr(tune, "_measure", lambda *args: next(outputs))
    am, bm = tune._confirm(a,b,"hot","us",.01,(),None,0)
    assert am == 10 and math.isinf(bm)


@pytest.mark.parametrize("runtime,values", [("threads",[1,2]), ("wait",["spin","sleep"])])
def test_search_source_dedup_preserves_runtime_distinctions(runtime, values):
    result, active, _ = tune.sample_legal(BASE, {runtime:values}, 2, random.Random(0))
    assert len(result) == 2
    assert {c[runtime] for _, c in result} == set(values)
    assert runtime in active


def test_sampler_generates_duplicate_resolved_configuration_once(monkeypatch):
    generate = Mock(wraps=kernels.generate)
    monkeypatch.setattr(kernels, "generate", generate)
    result, _, _ = tune.sample_legal(BASE, {"rows": [1,2]}, 8, random.Random(0))
    # Scalar rows=1/2 lower to the same C: retain the original semantic dedup.
    assert len(result) == 1
    assert generate.call_count == 2


def test_concurrent_compile_uses_unique_temporary_outputs(tmp_path, monkeypatch):
    barrier = threading.Barrier(2)
    names = []
    def fake_run(args, **kw):
        tmp = Path(args[args.index("-o") + 1])
        names.append(str(tmp))
        tmp.write_bytes(b"compiled fixture")
        barrier.wait(timeout=5)
        return SimpleNamespace(returncode=0, stderr="")
    monkeypatch.setattr(toolchain.subprocess, "run", fake_run)
    out = str(tmp_path / "kernel.so")
    with ThreadPoolExecutor(2) as pool:
        futures = [pool.submit(toolchain._compile, ["fixture-cc"], out) for _ in range(2)]
        for future in futures:
            future.result()
    assert len(set(names)) == 2
    assert Path(out).read_bytes() == b"compiled fixture"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["kernel.so"]


def test_compile_failure_cleans_temporary_output(tmp_path, monkeypatch):
    def fake_run(args, **kw):
        Path(args[-1]).write_bytes(b"partial")
        return SimpleNamespace(returncode=1, stderr="intentional compiler failure")
    monkeypatch.setattr(toolchain.subprocess,"run",fake_run)
    with pytest.raises(toolchain.BuildError):
        toolchain._compile(["fixture-cc"], str(tmp_path/"out.so"))
    assert list(tmp_path.iterdir()) == []


def csv_text(**changes):
    fields = dict(zip(harness.CSV_COLUMNS, ["x","q8_0_gemv","hot","1","64","32","1","10","0.001","0.001",
                          "100","1","1","547","1","0","ok","0"]))
    fields.update(changes)
    return ",".join(str(fields[k]) for k in harness.CSV_COLUMNS)+"\n"


@pytest.mark.parametrize("kind", ["exit", "malformed", "calls_zero", "timeout", "no_file", "unknown_status"])
def test_harness_failure_is_typed_and_scratch_removed(tmp_path, monkeypatch, kind):
    scratch = tmp_path/"scratch"
    scratch.mkdir()
    monkeypatch.setattr(harness, "scratch_dir", lambda: str(scratch))
    monkeypatch.setattr(harness, "harness_command", lambda *a:["fake-harness"])
    def fake_run(args, **kw):
        path = Path(args[args.index("--csv")+1])
        if kind == "timeout":
            raise subprocess.TimeoutExpired(args, 1)
        if kind != "no_file":
            path.write_text("broken" if kind=="malformed" else csv_text(**({"calls":0} if kind=="calls_zero" else
                                                    {"check":"UNKNOWN"} if kind=="unknown_status" else {})))
        return SimpleNamespace(returncode=2 if kind=="exit" else 0, stdout="", stderr="fixture failure")
    monkeypatch.setattr(harness.subprocess, "run", fake_run)
    with pytest.raises(harness.HarnessError):
        harness.bench("x", spec.resolve(BASE))
    assert not scratch.exists()


def test_harness_numeric_failure_remains_inspectable(tmp_path, monkeypatch):
    scratch = tmp_path/"scratch"
    scratch.mkdir()
    monkeypatch.setattr(harness,"scratch_dir",lambda:str(scratch))
    monkeypatch.setattr(harness,"harness_command",lambda *a:["fake-harness"])
    def fake_run(args,**kw):
        Path(args[args.index("--csv")+1]).write_text(csv_text(check="FAIL",relerr="inf"))
        return SimpleNamespace(returncode=1,stdout="",stderr="")
    monkeypatch.setattr(harness.subprocess,"run",fake_run)
    assert harness.bench("x",spec.resolve(BASE))["check"] == "FAIL"
    assert not scratch.exists()


@pytest.mark.parametrize("available,fmt,expected", [
    ({"scalar", "avx2"}, "q8_0", "avx2"),
    ({"scalar", "avx2"}, "q4_K", "avx2"),
    ({"scalar", "avx2"}, "q4_0", "scalar"),
    ({"scalar", "avx2_vnni", "avx512_vnni"}, "q8_0", "avx512_vnni"),
    ({"scalar", "neon"}, "q8_0", "neon"),
])
def test_default_target_respects_pair_and_native_features(monkeypatch, available, fmt, expected):
    monkeypatch.setattr(plan, "run_mode", lambda t: ("native", "") if t in available else (None, "missing"))
    assert plan.default_target("gemv", fmt) == expected


def test_plan_lookup_zero_threads_is_not_a_default(tmp_path):
    path, fp = saved_plan(tmp_path)
    with pytest.raises(spec.SpecError):
        plan.Plan(path=str(path), fp=fp).lookup("gemv", "q8_0", 64, 16, "hot", 0)


def test_empty_sampler_space_is_rejected():
    with pytest.raises(spec.SpecError):
        tune.sample_legal(BASE, {"rows": []}, 4, random.Random(0))


def test_sampler_reuses_partial_validation_without_caching_final_configs(monkeypatch):
    resolve = Mock(wraps=tune.resolve)
    monkeypatch.setattr(tune, "resolve", resolve)
    cs, _, _ = tune.sample_legal(BASE, {"rows": [1,2,4,8]}, 27, random.Random(0))
    assert len(cs) == 1
    # 4 unique partial checks + 540 final resolves + 1 exact null probe.
    assert resolve.call_count == 545


@pytest.mark.parametrize("challenger_valid", [False, True])
def test_unconfirmed_neighbors_cannot_be_exported(monkeypatch, challenger_valid):
    configs = {r: spec.resolve({**BASE, "rows": r}) for r in (1, 2, 4)}
    incumbent = {"ov": {"rows": 1}, "c": configs[1], "so": "1",
                 **tune.measurement(row(10))}
    monkeypatch.setattr(tune, "_source_key", lambda c: c["rows"])
    monkeypatch.setattr(tune, "_build_all", lambda cs, *a: {i: str(c["rows"]) for i, c in enumerate(cs)})
    monkeypatch.setattr(tune, "_measure", lambda c, *a: tune.measurement(row({2: 1, 4: 2}[c["rows"]])))

    def confirm(a, b, *args):
        a.update(tune.measurement(row(10)), _valid=True)
        b.update(tune.measurement(row(12)), _valid=challenger_valid)
        return 10, 12 if challenger_valid else math.inf

    monkeypatch.setattr(tune, "_confirm", confirm)
    result = tune._coordinate_descent(
        BASE, {"rows": [1, 2, 4]}, [incumbent], "hot", "us", .01, .02,
        1, (), None, 0, 1, lambda _: None, {"builds": 0, "measurements": 0},
    )
    assert [r["c"]["rows"] for r in result] == [1]


def test_search_serialization_preserves_confirmed_winner(monkeypatch):
    configs = [spec.resolve({**BASE, "rows": r}) for r in (1, 2)]
    monkeypatch.setattr(tune, "sample_legal", lambda *a: (
        [(dict(rows=r), c) for r, c in zip((1, 2), configs)], {"rows": [1, 2]}, 1,
    ))
    monkeypatch.setattr(tune, "_build_all", lambda *a: {0: "1", 1: "2"})
    monkeypatch.setattr(tune, "_measure", lambda *a: tune.measurement(row(10)))

    def refine(_spec, _space, best, *args):
        best[0].update(tune.measurement(row(20)))
        best[1].update(tune.measurement(row(1)))
        return best

    monkeypatch.setattr(tune, "_coordinate_descent", refine)
    result, _ = tune.search(BASE, {"rows": [1, 2]}, n0=2, keep=2, log=lambda _: None)
    assert result[0]["config"]["rows"] == 1


def test_invalid_incumbent_is_not_exported(monkeypatch):
    incumbent = {"ov": {"rows": 1}, "c": spec.resolve({**BASE, "rows": 1}), "so": "1",
                 **tune.measurement(row(10))}
    monkeypatch.setattr(tune, "_source_key", lambda c: c["rows"])
    monkeypatch.setattr(tune, "_build_all", lambda *a: {0: "2"})
    monkeypatch.setattr(tune, "_measure", lambda *a: tune.measurement(row(1)))

    def confirm(a, b, *args):
        a["_valid"] = b["_valid"] = False
        return math.inf, math.inf

    monkeypatch.setattr(tune, "_confirm", confirm)
    with pytest.raises(harness.HarnessError, match="incumbent"):
        tune._coordinate_descent(
            BASE, {"rows": [1, 2]}, [incumbent], "hot", "us", .01, .02,
            1, (), None, 0, 1, lambda _: None, {"builds": 0, "measurements": 0},
        )


@pytest.mark.parametrize("seed", range(5))
def test_native_edp_plan(seed):
    if toolchain.run_mode("avx2")[0] != "native":
        pytest.skip("AVX2 unavailable")
    entry = plan.tune_entry(
        "gemv", "q8_0", 128, 33, regime="hot", threads=1, target="avx2",
        objective="edp", n0=3, top=1, secs=.01, reps=3, jobs=1,
        seed=seed, refine=False, log=lambda _: None,
    )
    assert entry["winner"] in entry["candidates"]
    assert math.isfinite(entry["edp"]) and entry["edp"] > 0
    assert entry["relerr"] < 1e-5
