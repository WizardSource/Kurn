"""The bundled harness: its own reference check, failure detection, NEON under
qemu, and a small tuning run."""

import csv

import pytest

from kurn.harness import bench, check
from kurn.spec import load, resolve
from kurn.toolchain import build, compile_source, run_mode
from kurn.tune import pareto_front, tune

from conftest import EXAMPLES


def _runnable(target, modes=("native",)):
    mode, why = run_mode(target)
    if mode not in modes:
        pytest.skip(why or f"{target} runs under {mode}")


@pytest.mark.parametrize(
    "cfg",
    [
        {"op": "gemv", "weights": "q8_0", "target": "scalar"},
        {"op": "gemv", "weights": "q8_0", "target": "avx2", "rows": 8, "prefetch": 16},
        {"op": "gemv", "weights": "q4_K", "target": "avx2_vnni", "rows": 4, "prefetch": 4},
        {"op": "gemv", "weights": "q8_0", "target": "avx512_vnni", "layout": "vnni16", "rows": 8},
        {"op": "gemv", "weights": "q4_K", "target": "avx512_vnni", "rows": 4, "prefetch": 4},
        {"op": "gemm", "weights": "q8_0", "target": "avx512_vnni", "rows": 6, "cols": 4},
        {"op": "gemm", "weights": "q8_0", "target": "amx", "rows": 2, "cols": 1},
    ],
    ids=lambda c: f"{c['weights']}-{c['op']}-{c['target']}",
)
def test_harness_check_passes(cfg):
    _runnable(cfg["target"])
    c = resolve(cfg)
    row = check(build(c), c)
    assert row["check"] == "ok" and row["relerr"] < 1e-5, row


@pytest.mark.parametrize("rows", [1, 4, 8])
def test_neon_under_qemu(rows):
    _runnable("neon", modes=("native", "qemu"))
    c = resolve({"op": "gemv", "weights": "q8_0", "target": "neon", "rows": rows, "prefetch": 8})
    row = check(build(c), c)
    assert row["check"] == "ok", row


BROKEN = """#include "kurn.h"
void kq8_gemv(const void *W, const void *x, float *y, int64_t K, int64_t r0, int64_t r1) {
    (void)W; (void)x; (void)K;
    for (int64_t r = r0; r < r1; r++) y[r] = (r == 7) ? 1.0f : 0.0f;  /* wrong */
}
"""


def test_harness_rejects_wrong_kernel():
    _runnable("scalar")
    so = compile_source(BROKEN, "scalar", stem="broken")
    row = bench(so, resolve({"op": "gemv", "weights": "q8_0", "target": "scalar", "threads": 1}), "hot", 0, ["--N", "64"])
    assert row["check"] == "FAIL"


def test_harness_rejects_unwritten_rows():
    _runnable("scalar")
    src = BROKEN.replace("for (int64_t r = r0; r < r1; r++) y[r] = (r == 7) ? 1.0f : 0.0f;  /* wrong */", "(void)y; (void)r0; (void)r1;")
    so = compile_source(src, "scalar", stem="broken_unwritten")
    row = bench(so, resolve({"op": "gemv", "weights": "q8_0", "target": "scalar", "threads": 1}), "hot", 0, ["--N", "64"])
    assert row["check"] == "FAIL" and row["relerr"] == float("inf")


def test_cold_regime_and_wait_policy():
    _runnable("avx2")
    c = resolve({"op": "gemv", "weights": "q8_0", "target": "avx2", "threads": 2, "wait": "sleep"})
    row = bench(build(c), c, "cold", 0.05)
    assert row["check"] == "ok" and row["us"] > 0 and row["cpu_us"] > 0 and row["GBps"] > 0


def test_tune_small_space(tmp_path):
    _runnable("avx2")
    spec, _ = load(EXAMPLES / "q4_K_gemv.kurn")
    spec["target"] = "avx2"
    out = tmp_path / "tune.csv"
    logs = []
    res, front = tune(spec, {"rows": [1, 2], "prefetch": [0, 4], "threads": [1]}, regime="hot", secs=0.05, out=str(out),
                      log=logs.append)  # fmt: skip
    assert len(res) == 4 and front and front == pareto_front(res)
    assert res == sorted(res, key=lambda r: r["energy_uJ"])
    rows = list(csv.DictReader(out.open()))
    assert len(rows) == 4 and {"rows", "prefetch", "us", "energy_uJ", "relerr"} <= set(rows[0])


def test_pareto_front():
    rs = [{"us": 1, "energy_uJ": 10}, {"us": 2, "energy_uJ": 5}, {"us": 3, "energy_uJ": 7}, {"us": 4, "energy_uJ": 1}]
    assert [r["us"] for r in pareto_front(rs)] == [1, 2, 4]


def test_static_power_term_changes_ranking():
    from kurn.tune import energy_uj

    slow_few = {"us": 190.0, "cpu_us": 760.0}  # 4 threads, 1.9x the latency
    fast_many = {"us": 100.0, "cpu_us": 800.0}  # 8 threads
    assert energy_uj(slow_few) < energy_uj(fast_many)
    assert energy_uj(slow_few, static_w=10) > energy_uj(fast_many, static_w=10)


def test_roofline_probe():
    from kurn.harness import bandwidth

    bw = bandwidth(2, detail=True)  # default: best of 1, 2, 4 and 8 streams
    assert set(bw) == {"dram", "dram_streams", "dram_median", "l2", "l2_streams", "l2_median"}
    assert bw["dram_streams"] in (1, 2, 4, 8) and 0 < bw["dram_median"] <= bw["dram"] < bw["l2"]
    fixed = bandwidth(2, streams=3, detail=True)
    assert fixed["dram_streams"] == 3 and fixed["l2_streams"] == 3
