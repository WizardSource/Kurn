"""kurn.energy (RAPL / NVML counters, unavailable on VMs) and the GPU kit's energy section."""

import json
import os

import pytest

from kurn import energy
from kurn.gpu import kitreport


def _zone(root, name, label, uj, rng=262143328850):
    d = root / name
    d.mkdir()
    (d / "name").write_text(label + "\n")
    (d / "energy_uj").write_text(f"{uj}\n")
    (d / "max_energy_range_uj").write_text(f"{rng}\n")
    return d


@pytest.fixture
def powercap(tmp_path, monkeypatch):
    root = tmp_path / "powercap"
    root.mkdir()
    monkeypatch.setattr(energy, "POWERCAP", str(root))
    monkeypatch.setattr(energy, "PERF_PMU", str(tmp_path / "no-pmu"))
    return root


def test_powercap_package_and_dram(powercap):
    pkg = _zone(powercap, "intel-rapl:0", "package-0", 1_000_000)
    _zone(powercap, "intel-rapl:0:0", "core", 5)  # inside the package: not added
    dram = _zone(powercap, "intel-rapl:0:1", "dram", 2_000_000)
    m = energy.Meter(gpu=False)
    assert m.cpu_src == "rapl-powercap:package-0,dram"
    assert m.cpu_joules() == 0.0
    (pkg / "energy_uj").write_text("3500000\n")
    (dram / "energy_uj").write_text("2500000\n")
    assert m.cpu_joules() == pytest.approx(3.0)


def test_powercap_wraparound(powercap):
    pkg = _zone(powercap, "intel-rapl:0", "package-0", 900, rng=1000)
    m = energy.Meter(gpu=False)
    (pkg / "energy_uj").write_text("100\n")  # wrapped: +200 uJ
    assert m.cpu_joules() == pytest.approx(200e-6)


def test_stuck_counter_is_unavailable(powercap):
    _zone(powercap, "intel-rapl:0", "package-0", 0)  # a VM's RAPL: present, never advances
    m = energy.Meter(gpu=False)
    m.check(0.05)
    d = m.describe()["cpu"]
    assert not d["available"] and "did not advance" in d["reason"]


def test_no_rapl_reason(powercap):
    d = energy.Meter(gpu=False).describe()["cpu"]
    assert not d["available"] and "no RAPL interface" in d["reason"]


def test_between():
    rows = [{"t": 0.0, "gpu_j": [0.0], "cpu_j": 0.0}, {"t": 1.0, "gpu_j": [100.0], "cpu_j": 50.0},
            {"t": 2.0, "gpu_j": [300.0], "cpu_j": 80.0}]  # fmt: skip
    e = energy.between(rows, 0.5, 2.5)
    assert e["seconds"] == 1.0 and e["gpu_j"] == 200.0 and e["gpu_w"] == 200.0 and e["cpu_j"] == 30.0
    assert energy.between(rows, 1.5, 1.9) is None


def test_run_reports_unavailable(powercap, capsys):
    assert energy.main(["run", "--tokens", "5", "--", "true"]) == 0
    assert "unavailable" in capsys.readouterr().err


def test_kit_energy_section(tmp_path):
    out = tmp_path
    (out / "attn").mkdir()
    meta = {"meta": {"gpu": {"available": True, "source": "nvml:1 GPU(s)", "names": ["NVIDIA A100-SXM4-80GB"], "reason": None},
                     "cpu": {"available": False, "source": None, "reason": "not readable"}}, "interval": 0.5}  # fmt: skip
    rows = [meta] + [{"t": 100.0 + i, "gpu_j": [60.0 * i]} for i in range(20)]
    (out / "energy_samples.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (out / "energy_marks.json").write_text(json.dumps({"idle": [100.0, 104.0]}))
    (out / "steps.jsonl").write_text(json.dumps({"step": "attention decode matrix", "result": "ok", "min": 0, "log": "x",
                                                 "t0": 105.0, "t1": 115.0}) + "\n")  # fmt: skip
    cells = [{"model": "llama3-8b", "kv": kv, "nkv": c, "nq": 1, "impl": impl, "status": "ok", "uJ": uj}
             for kv, c, impl, uj in (("f16", 1024, "kurn", 10.0), ("f16", 32768, "kurn", 200.0), ("f16", 1024, "ggml-cuda", 12.0))]  # fmt: skip
    (out / "attn" / "attn_matrix.jsonl").write_text("\n".join(json.dumps(r) for r in cells if r["impl"] == "kurn") + "\n")
    (out / "attn" / "attn_baselines.jsonl").write_text(json.dumps(cells[2]) + "\n")
    md = kitreport.energy_report(str(out))
    assert "A100" in md and "unavailable (not readable)" in md
    assert "Idle board power before the first step: 60.0 W" in md
    assert "| attention decode matrix | 10 | 600 | 60.0 | unavailable |" in md
    assert "| kurn | f16 | 0.32 | 6.40 |" in md  # 32 layers x 10 uJ = 0.32 mJ; 32 x 200 uJ = 6.4 mJ
    assert "| ggml-cuda | f16 | 0.38 | - |" in md


def test_kit_energy_section_without_capture(tmp_path):
    md = kitreport.energy_report(str(tmp_path))
    assert "No energy capture" in md and "No per-call energy" in md


def test_kit_energy_section_nothing_available(tmp_path, monkeypatch):
    # what the sampler writes on a box without NVML or readable RAPL (a dry run on a VM)
    monkeypatch.setattr(energy, "POWERCAP", str(tmp_path / "none"))
    monkeypatch.setattr(energy, "PERF_PMU", str(tmp_path / "none"))
    assert energy.main(["sample", "--out", str(tmp_path / "energy_samples.jsonl"), "--interval", "0.01"]) == 0
    md = kitreport.energy_report(str(tmp_path))
    assert "- GPU: unavailable (" in md and "- CPU: unavailable (no RAPL interface" in md


def test_run_kit_has_energy_capture():
    p = os.path.join(os.path.dirname(__file__), "..", "contrib", "gpu-check", "run_kit.sh")
    s = open(p).read()
    assert "energy sample --out" in s and "NO_ENERGY" in s and '\\"t0\\": $t0f' in s
