"""Build sharing must never merge runtime measurements or unequal compile inputs."""

import hashlib
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Lock

import pytest

from kurn import toolchain, tune
from kurn.spec import resolve

BASE = {"op": "gemv", "weights": "q8_0", "target": "avx2"}


def config(**overrides):
    return resolve(BASE, {"threads": 1, **overrides})


def fake_compile(monkeypatch):
    calls = []
    guard = Lock()

    def compile_one(src, target, out_dir=None, stem="kernel", extra_flags=(), shared=True):
        with guard:
            calls.append((src, target, stem))
        return f"{stem}-{hashlib.sha256(src.encode()).hexdigest()}.so"

    monkeypatch.setattr(toolchain, "compile_source", compile_one)
    return calls


@pytest.mark.parametrize("jobs", [0, 1, 2, 4])
@pytest.mark.parametrize("repetitions", [1, 2, 5])
def test_runtime_aliases_share_only_the_build(monkeypatch, jobs, repetitions):
    calls = fake_compile(monkeypatch)
    cs = [config(threads=t, wait=w) for t in (1, 2) for w in ("spin", "sleep")] * repetitions
    before = [dict(c) for c in cs]
    paths = tune._build_all(cs, jobs, lambda _: None)
    assert len(calls) == 1
    assert list(paths) == list(range(len(cs)))
    assert len(set(paths.values())) == 1
    assert cs == before


@pytest.mark.parametrize("jobs", [1, 4])
def test_distinct_generated_sources_are_not_merged(monkeypatch, jobs):
    calls = fake_compile(monkeypatch)
    cs = [config(rows=r) for r in (1, 2, 1, 4, 2)]
    paths = tune._build_all(cs, jobs, lambda _: None)
    assert len(calls) == 3
    assert list(paths) == [0, 1, 2, 3, 4]
    assert paths[0] == paths[2]
    assert paths[1] == paths[4]
    assert len(set(paths.values())) == 3


@pytest.mark.parametrize("changed", ["target", "weights", "op"])
def test_compile_identity_includes_target_and_stem(monkeypatch, changed):
    calls = fake_compile(monkeypatch)
    monkeypatch.setattr(toolchain, "generate", lambda c: "identical C fixture")
    first = config()
    second = {**first, changed: "different"}
    paths = tune._build_all([first, second], 2, lambda _: None)
    assert len(calls) == 2
    assert paths[0] != paths[1]


def test_custom_builder_keeps_original_contract(monkeypatch):
    calls = []

    def custom(c):
        calls.append(c["threads"])
        return f"custom-{c['threads']}.so"

    monkeypatch.setattr(tune, "build", custom)
    monkeypatch.setattr(toolchain, "generate", lambda c: pytest.fail("custom builder was bypassed"))
    paths = tune._build_all([config(threads=1), config(threads=2)], 1, lambda _: None)
    assert calls == [1, 2]
    assert paths == {0: "custom-1.so", 1: "custom-2.so"}


def test_failed_alias_group_does_not_hide_other_results(monkeypatch):
    calls, messages = [], []

    def compile_one(src, target, out_dir=None, stem="kernel", extra_flags=(), shared=True):
        calls.append(src)
        if src == "bad":
            raise toolchain.BuildError("deliberate failure\ncompiler detail")
        return "good.so"

    monkeypatch.setattr(toolchain, "generate", lambda c: "bad" if c["rows"] == 1 else "good")
    monkeypatch.setattr(toolchain, "compile_source", compile_one)
    paths = tune._build_all([config(rows=1), config(rows=2), config(rows=1)], 2, messages.append)
    assert paths == {1: "good.so"}
    assert sorted(calls) == ["bad", "good"]
    assert sorted(messages) == ["FAIL build 0: deliberate failure", "FAIL build 2: deliberate failure"]


def test_generation_build_error_is_reported_per_candidate(monkeypatch):
    calls = fake_compile(monkeypatch)
    messages = []

    def generate(c):
        if c["threads"] == 2:
            raise toolchain.BuildError("generation failed")
        return "good"

    monkeypatch.setattr(toolchain, "generate", generate)
    paths = tune._build_all([config(threads=1), config(threads=2)], 2, messages.append)
    assert list(paths) == [0]
    assert len(calls) == 1
    assert messages == ["FAIL build 1: generation failed"]


@pytest.mark.parametrize("err", [RuntimeError, ValueError])
def test_unexpected_errors_are_not_silenced(monkeypatch, err):
    def broken(*args, **kwargs):
        raise err("unexpected")

    monkeypatch.setattr(toolchain, "compile_source", broken)
    with pytest.raises(err, match="unexpected"):
        tune._build_all([config()], 1, lambda _: None)


def test_failure_is_retried_by_a_later_batch(monkeypatch):
    calls = []

    def compile_one(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise toolchain.BuildError("transient")
        return "recovered.so"

    monkeypatch.setattr(toolchain, "compile_source", compile_one)
    cs = [config(threads=1), config(threads=2)]
    assert tune._build_all(cs, 2, lambda _: None) == {}
    assert tune._build_all(cs, 2, lambda _: None) == {0: "recovered.so", 1: "recovered.so"}
    assert len(calls) == 2


def test_separate_batches_keep_independent_lifetimes(monkeypatch):
    calls = fake_compile(monkeypatch)
    cs = [config(), config(threads=2)]
    tune._build_all(cs, 2, lambda _: None)
    tune._build_all(cs, 2, lambda _: None)
    assert len(calls) == 2  # persistent artifact reuse stays in compile_source


def test_unrelated_compile_requests_still_run_in_parallel(monkeypatch):
    barrier = Barrier(2)

    def compile_one(src, target, out_dir=None, stem="kernel", extra_flags=(), shared=True):
        barrier.wait(timeout=5)
        return src + ".so"

    monkeypatch.setattr(toolchain, "generate", lambda c: str(c["rows"]))
    monkeypatch.setattr(toolchain, "compile_source", compile_one)
    assert tune._build_all([config(rows=1), config(rows=2)], 2, lambda _: None) == {0: "1.so", 1: "2.so"}


def test_empty_batch_does_not_generate_or_compile(monkeypatch):
    calls = fake_compile(monkeypatch)
    assert tune._build_all([], 4, lambda _: None) == {}
    assert calls == []


def test_generator_input_and_mapping_order(monkeypatch):
    calls = fake_compile(monkeypatch)
    cs = (config(rows=r) for r in (1, 2, 1, 4, 2))
    paths = tune._build_all(cs, 2, lambda _: None)
    assert list(paths) == [0, 1, 2, 3, 4]
    assert len(calls) == 3


def test_concurrent_batches_do_not_share_in_memory_results(monkeypatch):
    calls = fake_compile(monkeypatch)
    with ThreadPoolExecutor(2) as executor:
        futures = [executor.submit(tune._build_all, [config(), config(threads=2)], 2, lambda _: None) for _ in range(2)]
        results = [future.result() for future in futures]
    assert results[0] == results[1]
    assert len(calls) == 2


@pytest.mark.parametrize("seed", range(5))
def test_search_measures_each_runtime_configuration(monkeypatch, seed):
    calls = fake_compile(monkeypatch)
    measurements = []

    def observe(so, c, regime, secs, extra, harness):
        measurements.append((c["threads"], c["wait"], secs))
        us = float(10 + c["threads"] + (c["wait"] == "sleep"))
        return {"check": "ok", "us": us, "cpu_us": us, "GBps": 1.0, "relerr": 0.0}

    monkeypatch.setattr(tune, "bench", observe)
    results, stats = tune.search(
        BASE, {"threads": [1, 2], "wait": ["spin", "sleep"]},
        n0=4, keep=4, jobs=4, seed=seed, refine=False, secs0=0.01, secs=0.02, log=lambda _: None,
    )
    assert len(calls) == 1
    assert len(results) == 4
    assert stats["builds"] == 4
    assert stats["measurements"] == len(measurements) == 8
    for threads in (1, 2):
        for wait in ("spin", "sleep"):
            assert measurements.count((threads, wait, 0.01)) == 1
            assert measurements.count((threads, wait, 0.02)) == 1


@pytest.mark.parametrize("seed", range(5))
def test_native_shared_builds_keep_each_runtime_check(tmp_path, monkeypatch, seed):
    from kurn.harness import bench

    if toolchain.run_mode("avx2")[0] != "native":
        pytest.skip("native AVX2 is unavailable")
    try:
        toolchain.host_cc()
    except toolchain.BuildError:
        pytest.skip("native C compiler is unavailable")
    monkeypatch.setenv("KURN_CACHE_DIR", str(tmp_path))
    compiled = []
    original_compile = toolchain._compile

    def counted_compile(args, path):
        compiled.append(path)
        return original_compile(args, path)

    monkeypatch.setattr(toolchain, "_compile", counted_compile)
    cs = [config(rows=r, threads=t, wait=w) for r in (1, 2) for t in (1, 2) for w in ("spin", "sleep")]
    paths = tune._build_all(cs, 4, lambda _: None)
    assert len(compiled) == 2
    assert len(paths) == 8
    assert len(set(paths.values())) == 2
    for i, c in enumerate(cs):
        row = bench(paths[i], c, "hot", 0, ["--N", "37", "--K", "512", "--seed", str(seed + 1)])
        assert tune.measurement(row) is not None
        assert row["relerr"] < 1e-5
