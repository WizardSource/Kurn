"""Benchmark-matrix report: the 2-sigma win rule, wins/ties/losses, dispatch.json and kernel_for()'s
fallback, on synthetic harness output (no GPU)."""

import json
import random

import pytest

from kurn.gpu import report
from kurn.gpu.dispatch import kernel_for, parse_config
from kurn.gpu.kit import config_string
from kurn.gpu.matrix import MODEL
from kurn.gpu.spec import resolve

SHAPES = MODEL["shapes"]
CFG = {f: config_string(resolve({"op": "gemv", "weights": f})) for f in ("q4_0", "q8_0", "tq2_0", "q1_0")}


def rows_for(fmt, m, impl, us, sd, rounds=5, seed=0, status="ok", config=None, joules=1e-4):
    rng = random.Random(seed)
    out = []
    for n, k in SHAPES:
        out.append({"kind": "check", "fmt": fmt, "N": n, "K": k, "M": m, "impl": impl, "config": config or impl, "status": status,
                    "relerr_exact": 1e-7 if impl.startswith("kurn") else None, "relerr_model": 5e-3, "copies": 4})  # fmt: skip
        if status != "ok":
            continue
        for r in range(rounds):
            t = us * (n * k) / (4096 * 4096) * (1 + rng.gauss(0, sd))
            out.append({"kind": "sample", "fmt": fmt, "N": n, "K": k, "M": m, "impl": impl, "round": r, "us": t, "joules": joules,
                        "watts": 300.0, "calls": 1000, "bytes": n * k * 0.56, "ops": 2.0 * n * k * m})  # fmt: skip
    return out


@pytest.fixture
def results(tmp_path):
    rows = []
    rows += rows_for("q4_0", 1, "kurn:gemv", 10.0, 0.01, config=CFG["q4_0"])  # clear win: 1.3x, ~1% noise
    rows += rows_for("q4_0", 1, "ggml", 13.0, 0.01, seed=1)
    rows += rows_for("q4_0", 1, "cublas-fp16", 30.0, 0.01, seed=2)
    rows += rows_for("q8_0", 1, "kurn:gemv", 10.0, 0.05, config=CFG["q8_0"])  # 3% faster but 5% noise: tie
    rows += rows_for("q8_0", 1, "ggml", 10.3, 0.05, seed=3)
    rows += rows_for("q8_0", 16, "kurn:cols", 20.0, 0.01, config=CFG["q8_0"])  # loss
    rows += rows_for("q8_0", 16, "ggml", 12.0, 0.01, seed=4)
    rows += rows_for("tq2_0", 1, "kurn:gemv", 6.0, 0.01, config=CFG["tq2_0"])  # ggml-cuda has no TQ2_0 kernel
    rows += rows_for("tq2_0", 1, "ggml", 0, 0, status="not supported by this ggml backend (llama.cpp would run this MUL_MAT on the CPU)")
    rows += rows_for("tq2_0", 1, "cublas-fp16", 30.0, 0.01, seed=5)
    rows += rows_for("q1_0", 1, "kurn:gemv", 5.0, 0.01, rounds=2, config=CFG["q1_0"])  # too few rounds
    rows += rows_for("q1_0", 1, "ggml", 9.0, 0.01, rounds=2, seed=6)
    with open(tmp_path / "matrix.jsonl", "w") as fh:
        fh.write("\n".join(json.dumps(r) for r in rows) + "\n")
    (tmp_path / "info.json").write_text(json.dumps({"name": "Synthetic A100", "cc": 80, "sms": 108, "l2_bytes": 40 << 20,
                                                    "nominal_bw_gbs": 2039.0, "nvml_energy": True}))  # fmt: skip
    (tmp_path / "roofline.json").write_text(json.dumps({"hbm_read_gbs": 1800.0, "int8_mma_sync_tops": 550.0}))
    return tmp_path


def test_verdict_rule():
    assert report.verdict(10, 0.01, 13, 0.01, 5, 5)[0] == "win"
    assert report.verdict(10, 0.05, 10.3, 0.05, 5, 5)[0] == "tie"
    assert report.verdict(20, 0.01, 12, 0.01, 5, 5)[0] == "loss"
    assert report.verdict(5, 0.01, 9, 0.01, 2, 5)[0] == "unresolved"
    # exactly at the 2-sigma line is not a win
    assert report.verdict(10, 0.03, 10 * (1 + 2 * (2 * 0.03**2) ** 0.5), 0.03, 5, 5)[0] == "tie"


def test_report_cells_and_dispatch(results):
    md, cells = report.write(str(results), ["q4_0", "q8_0", "tq2_0", "q1_0"])
    assert cells[("q4_0", 1)]["v_same"] == "win" and cells[("q4_0", 1)]["dispatch"] == "kurn"
    assert cells[("q4_0", 1)]["v_overall"] == "win"  # ggml beats cuBLAS fp16, so ggml is the overall best too
    assert cells[("q8_0", 1)]["v_same"] == "tie" and cells[("q8_0", 1)]["dispatch"] == "stock"
    assert cells[("q8_0", 16)]["v_same"] == "loss" and cells[("q8_0", 16)]["dispatch"] == "stock"
    assert cells[("tq2_0", 1)]["dispatch"] == "kurn" and "only GPU kernel" in cells[("tq2_0", 1)]["v_same"]
    assert cells[("tq2_0", 1)]["v_overall"] == "win"  # vs cuBLAS fp16
    assert cells[("q1_0", 1)]["v_same"] == "unresolved" and cells[("q1_0", 1)]["dispatch"] == "stock"
    assert cells[("q4_0", 4)]["v_same"] == "unmeasured" and cells[("q4_0", 4)]["dispatch"] == "stock"
    assert "## Wins, ties and losses" in md and "Synthetic A100" in md and "unmeasured" in md
    t = json.loads((results / "dispatch.json").read_text())
    assert t["arch"] == "sm_80" and t["cells"]["q4_0"]["1"]["impl"] == "kurn" and t["cells"]["q8_0"]["16"]["impl"] == "stock"


def test_tokens_per_second(results):
    rows, _ = report.load(str(results))
    agg = report.aggregate(rows)
    s = agg[("q4_0", 1, "kurn:gemv")]
    step_us = sum(10.0 * n * k / (4096 * 4096) for n, k in SHAPES) * MODEL["layers"]
    assert s["rounds"] == 5 and abs(s["tok_s"] / (1e6 / step_us) - 1) < 0.02
    assert abs(s["J_tok"] - 4 * 1e-4 * MODEL["layers"]) < 1e-9


def test_kernel_for_falls_back(results):
    report.write(str(results), ["q4_0", "q8_0", "tq2_0", "q1_0"])
    table = str(results / "dispatch.json")
    k = kernel_for("q4_0", 1, "sm_80", table)
    assert k["impl"] == "kurn" and k["config"]["weights"] == "q4_0" and k["config"]["op"] == "gemv"
    assert kernel_for("q4_0", 1, "sm_90", table)["impl"] == "stock"  # other GPU
    assert kernel_for("q4_0", 2, "sm_80", table)["impl"] == "stock"  # between 1 (win) and 4 (unmeasured)
    assert kernel_for("q8_0", 1, "sm_80", table)["impl"] == "stock"  # tie
    assert kernel_for("q8_0", 64, "sm_80", table)["impl"] == "stock"
    assert kernel_for("tq2_0", 1, "sm_80", table)["impl"] == "kurn"
    assert kernel_for("e8p", 1, "sm_80", table)["impl"] == "stock"
    assert kernel_for("q4_0", 1)["impl"] == "stock"  # no table at all


def test_parse_config_roundtrip():
    for c in (resolve({"op": "gemv", "weights": "q4_K", "layout": "split", "mins": "dp4a"}),
              resolve({"op": "gemm", "weights": "q4_0", "bm": 128, "bn": 64, "wm": 4, "wn": 2, "xin": "f16", "stages": 3})):  # fmt: skip
        assert parse_config(config_string(c)) == c


def test_default_vs_tuned_table(tmp_path):
    rows = rows_for("q4_0", 1, "kurn:default-gemv", 13.0, 0.01, config=CFG["q4_0"])
    rows += rows_for("q4_0", 1, "kurn:tuned-mma8", 10.0, 0.01, config=CFG["q4_0"], seed=3)
    rows += rows_for("q4_0", 1, "ggml", 12.0, 0.01, seed=4)
    (tmp_path / "matrix.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    md, cells = report.write(str(tmp_path), ["q4_0"])
    assert "## Default vs tuned KURN kernels" in md
    line = next(ln for ln in md.splitlines() if ln.startswith("| q4_0 | 1 | default-gemv"))
    assert "tuned-mma8" in line and "1.30x" in line
    assert cells[("q4_0", 1)]["kurn"]["impl"] == "kurn:tuned-mma8" and cells[("q4_0", 1)]["v_same"] == "win"


def test_dry_report_is_all_unmeasured(tmp_path):
    md, cells = report.write(str(tmp_path), ["q4_0", "tq2_0"], dry=True)
    assert all(c["v_overall"] == "unmeasured" and c["dispatch"] == "stock" for c in cells.values())
    assert "No GPU measurements" in md
    assert json.loads((tmp_path / "dispatch.json").read_text())["cells"]["q4_0"]["1"]["impl"] == "stock"
