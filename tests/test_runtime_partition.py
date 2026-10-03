"""kurn runtime (src/kurn/runtime): partitioners, fix-up reduction, wait primitives, rtbench."""

import subprocess
import sys

import pytest

from kurn import runtime
from kurn.harness import bench
from kurn.spec import SCHEDULE, resolve
from kurn.toolchain import build, run_mode

pytestmark = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="the runtime uses futex / sched_setaffinity")


def _target():
    for t in ("avx512_vnni", "avx2", "scalar"):
        if run_mode(t)[0] == "native":
            return t
    pytest.skip("no natively runnable x86 target")


@pytest.fixture(scope="module")
def kernel():
    return runtime.default_kernel(_target())


def test_selftest_coverage_fixup_and_waits():
    out = subprocess.run([runtime.build_selftest(), "4"], capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr
    assert out.stdout.startswith("ok")


@pytest.mark.parametrize("part", runtime.PARTITIONERS)
@pytest.mark.parametrize("threads", [1, 3, 4])
@pytest.mark.parametrize("workload", ["tiny-moe", "tiny-mixed"])
def test_rtbench_partitioners_are_exact(kernel, part, threads, workload):
    row = runtime.rtbench(kernel, workload, part, threads, regime="hot", secs=0)
    assert row["check"] == "ok" and row["relerr"] <= 1e-5, row["stdout"]


@pytest.mark.parametrize("threads", [1, 3])
def test_rtbench_barrier_per_matrix_terminates(kernel, threads):
    row = runtime.rtbench(kernel, "tiny-moe", "static", threads, regime="hot", secs=0.05, extra=["--sync-each"], timeout=60)
    assert row["check"] == "ok" and row["tokens"] >= 1


@pytest.mark.parametrize("ksplit", [2, 4])
@pytest.mark.parametrize("part", ["streamk", "balanced", "static"])
def test_split_k_within_tolerance(kernel, part, ksplit):
    row = runtime.rtbench(kernel, "tiny-mixed", part, 3, regime="hot", secs=0, extra=["--ksplit", ksplit, "--tile", 64])
    assert row["check"] == "ok" and row["relerr"] <= 1e-5, row["stdout"]


def _dump(kernel, tmp_path, part, threads, extra=()):
    f = tmp_path / f"{part}_{threads}_{len(extra)}.bin"
    runtime.rtbench(kernel, "tiny-mixed", part, threads, regime="hot", secs=0, extra=["--dump", f, *extra])
    return f.read_bytes()


def test_streamk_is_deterministic_across_thread_counts(kernel, tmp_path):
    ks = ["--ksplit", "4", "--tile", "32"]
    ref = _dump(kernel, tmp_path, "streamk", 1, ks)
    for t in (2, 3, 4):
        assert _dump(kernel, tmp_path, "streamk", t, ks) == ref
    assert _dump(kernel, tmp_path, "balanced", 3, ks) == ref  # same left-to-right K-tile fold


def test_row_partitioners_are_bitwise_identical(kernel, tmp_path):
    ref = _dump(kernel, tmp_path, "static", 1)
    for part in ("static", "balanced", "steal", "ggml"):
        assert _dump(kernel, tmp_path, part, 3) == ref, part


@pytest.mark.parametrize("wait", ["futex", "hybrid:50", "umwait"])
def test_rtbench_wait_policies(kernel, wait):
    row = runtime.rtbench(kernel, "tiny-moe", "balanced", 3, regime="hot", secs=0.05, extra=["--wait", wait, "--serial-us", 20])
    assert row["check"] == "ok" and row["tokens"] >= 1


def test_rtbench_prefetch_and_gaps(kernel):
    row = runtime.rtbench(kernel, "tiny-moe", "balanced", 3, regime="hot", secs=0.05,
                          extra=["--prefetch", "wait", "--gaps", "10,5", "--pf-predict", "0.5"])  # fmt: skip
    assert row["check"] == "ok" and row["prefetch"] == "wait"


def test_rtbench_read_impl_sweep():
    row = runtime.rtbench("read", "sweep", "balanced", 2, secs=0.05, extra=["--footprint", 8])
    assert row["GBps"] > 0 and row["footprint_mb"] == 0


@pytest.mark.parametrize("threads", [2, 4])
def test_single_op_workload_terminates(threads):
    # one op per token: the stop token must not be read as "stop now" by a thread still finishing the previous token
    for _ in range(5):
        row = runtime.rtbench("read", "sweep", "steal", threads, secs=0.01, extra=["--footprint", 8], timeout=60)
        assert row["tokens"] >= 1


@pytest.mark.parametrize(("part", "threads", "producers"), [("static", 3, 1), ("balanced", 4, 2), ("static", 2, 1)])
@pytest.mark.parametrize("workload", ["tiny-moe", "tiny-mixed"])
def test_rtbench_producers_are_exact(kernel, part, threads, producers, workload):
    row = runtime.rtbench(kernel, workload, part, threads, regime="hot", secs=0.05, extra=["--producers", producers])
    assert row["check"] == "ok" and row["relerr"] <= 1e-5, row["stdout"]


@pytest.mark.parametrize("extra", [["--producers", "3"], ["--producers", "1", "--rotate"], ["--producers", "1", "--prefetch", "wait"]])
def test_rtbench_producers_rejects_bad_combinations(kernel, extra):
    with pytest.raises(RuntimeError):
        runtime.rtbench(kernel, "tiny-moe", "static", 3, regime="hot", secs=0, extra=extra)


def test_rtbench_reports_spin_per_thread(kernel):
    row = runtime.rtbench(kernel, "tiny-moe", "steal", 3, regime="hot", secs=0.05)
    per = [float(v) for v in row["spin_pct_per_thread"].split(";")]
    assert len(per) == 3 and all(0 <= v <= 100 for v in per)


def test_rtbench_rotate_is_exact(kernel):
    row = runtime.rtbench(kernel, "tiny-mixed", "balanced", 3, regime="hot", secs=0.05, extra=["--rotate"])
    assert row["check"] == "ok" and row["tokens"] >= 1


def test_ext_wait_values_reach_the_harness():
    t = _target()
    allowed = SCHEDULE["wait"]("gemv", "q8_0", t)
    assert {"spin", "sleep", "futex", "hybrid:2000"} <= set(allowed)
    c = resolve({"op": "gemv", "weights": "q8_0", "target": t, "threads": 2, "wait": "hybrid:2000"})
    row = bench(build(c), c, "hot", 0.05, ["--N", "64", "--serial-us", "5"])
    assert row["check"] == "ok"
