"""Cost-aware speculative verify width (kurn.specwidth) and its C++ twin
(integration/llama.cpp/spec-width/kurn-spec-width.h)."""

import random
import shutil
import struct
import subprocess
import sys

import pytest

from kurn import specwidth as sw

from conftest import ROOT

INTEG = ROOT / "integration" / "llama.cpp"
HEADER = INTEG / "spec-width" / "kurn-spec-width.h"

# 8-column staircase shaped like kurn on Qwen3-8B: GEMV, vfy2, vfy4 (3-4), vfy8 (5-8), then 8 + remainder
STAIR = [100.0, 130.0, 150.0, 152.0, 178.0, 179.0, 180.0, 181.0, 230.0, 245.0, 260.0, 262.0, 280.0, 281.0, 282.0, 283.0]
DRAFT = [8.0] * 16


def table(v=STAIR, d=DRAFT):
    return sw.CostTable(list(v), list(d))


def f32(x):
    return struct.unpack("f", struct.pack("f", x))[0]


def test_kernel_staircase():
    assert [sw.kernel_cols(m) for m in range(1, 9)] == [1, 2, 4, 4, 8, 8, 8, 8]
    assert sw.kernel_passes(8) == [8]
    assert sw.kernel_passes(9) == [8, 1]
    assert sw.kernel_passes(12) == [8, 4]
    assert sw.kernel_passes(19) == [8, 8, 4]
    with pytest.raises(ValueError):
        sw.kernel_cols(9)


def test_buft_keys_match_the_generator():
    sys.path.insert(0, str(INTEG))
    gen = pytest.importorskip("gen_ggml_sources")
    assert sw.BUFT_DEFAULT_KEYS == gen.DEFAULT_KEYS
    assert sw.BUFT_TUNED_KEYS == gen.TUNED_KEYS
    assert sw.VFY_COLS == gen.VFY_COLS


def test_buft_configs_resolve():
    c = sw.buft_configs("q8_0", 8)
    assert c[1]["op"] == "gemv" and c[1]["rows"] == 8
    assert [c[k]["cols"] for k in (2, 4, 8)] == [2, 4, 8]
    assert all(c[k]["rows"] == 1 and c[k]["layout"] == "i16" for k in (2, 4, 8))


def test_cost_table_roundtrip(tmp_path):
    t = sw.CostTable(STAIR[:5], DRAFT[:5], ["test table"])
    p = tmp_path / "t.cost"
    t.save(p)
    u = sw.CostTable.load(p)
    assert u.verify_ms == pytest.approx(t.verify_ms) and u.draft_ms == pytest.approx(t.draft_ms)
    assert u.meta == ["test table"]
    p.write_text("verify 1 10\nverify 3 12\n")
    with pytest.raises(ValueError):
        sw.CostTable.load(p)


def test_cap_lands_on_kernel_boundaries():
    """With free drafts, the cap never ends on a padded width (M = 3, 5, 6, 7): those cost as much as 4 or 8."""
    for alpha in [i / 20 for i in range(1, 20)]:
        p = sw.WidthPolicy(table(d=[]), k_max=15, prior=1e9, alpha0=alpha)
        k = p.n_cap()
        assert k + 1 not in (3, 5, 6, 7), (alpha, k)
    hi = sw.WidthPolicy(table(), k_max=15, prior=1e9, alpha0=0.9)
    lo = sw.WidthPolicy(table(), k_max=15, prior=1e9, alpha0=0.3)
    assert hi.n_cap() >= 7 and lo.n_cap() <= 3


def test_truncate_backs_off_a_cost_step():
    """Nine confident drafts but a weak tail: verifying 10 tokens costs an extra pass, cut to 7 (M = 8)."""
    p = sw.WidthPolicy(table(), k_max=15)
    probs = [f32(x) for x in (0.999, 0.999, 0.995, 0.99, 0.99, 0.99, 0.99, 0.35, 0.3)]
    assert p.truncate(probs) == 7
    assert p.truncate(probs[:7] + [f32(0.97), f32(0.97)]) == 9
    assert p.truncate([f32(0.2)]) == 0


def test_keep_drafting_stops_on_low_confidence():
    p = sw.WidthPolicy(table(), k_max=15)
    assert p.keep_drafting([f32(0.999)])
    assert not p.keep_drafting([f32(0.999), f32(0.1)])
    assert not p.keep_drafting([f32(0.999)] * 15)


def test_observe_learns_acceptance():
    p = sw.WidthPolicy(table(), k_max=15)
    for _ in range(200):
        p.observe([f32(0.9995)] * 4, 1, 150.0, 2)  # confident drafts: first accepted, second rejected
    assert p.alpha() == pytest.approx(0.5, abs=0.02)
    assert p.acc(0.9995) == pytest.approx(0.5, abs=0.02)
    assert p.acc(0.4) == pytest.approx(0.4)  # untouched bin keeps its prior (bin centre)
    assert p.lam() == pytest.approx(2 / 150.0, rel=1e-6)
    for _ in range(200):
        p.observe([f32(0.9995)] * 4, 0, 150.0, 1)
    assert p.alpha() < 0.1 and p.n_cap() == 0


def test_simulate_accounts_every_token():
    rng = random.Random(1)
    trace = [(rng.random() < 0.7, f32(rng.random())) for _ in range(300)]
    for kw in ({"policy": "fixed", "width": 4}, {"policy": "fixed", "width": 16, "p_min": 0.5}, {"policy": "cap"}, {"policy": "policy"}):
        r = sw.simulate(table(), trace, **kw)
        assert r["tokens"] == len(trace)
        assert r["accepted"] + r["steps"] >= len(trace)
        assert sum(r["widths"].values()) == r["steps"]


def test_policy_beats_fixed_widths_on_phased_trace():
    """Alternating easy / hard phases: no single width fits both; the policy adapts."""
    rng = random.Random(7)
    trace = []
    for phase in range(12):
        a = 0.92 if phase % 2 == 0 else 0.25
        for _ in range(60):
            m = rng.random() < a
            trace.append((m, f32(min(0.9999, max(0.01, (0.97 if m else 0.5) + rng.gauss(0, 0.05))))))
    best_fixed = max(sw.simulate(table(), trace, "fixed", w)["tok_s"] for w in range(0, 16))
    assert sw.simulate(table(), trace, "policy", k_max=15)["tok_s"] > best_fixed


DRIVER = r"""
#include "kurn-spec-width.h"
#include <iostream>
#include <sstream>
int main(int, char ** argv) {
    kurn::verify_cost_table t;
    if (!t.load(argv[1])) return 2;
    kurn::width_policy w(t, atoi(argv[2]));
    std::string line;
    std::cout.precision(17);
    while (std::getline(std::cin, line)) {
        std::istringstream in(line);
        char op; in >> op;
        if (op == 'C') { std::cout << w.n_cap() << "\n"; continue; }
        int n_acc = 0, n_tok = 0; double ms = 0;
        if (op == 'O') in >> n_acc >> ms >> n_tok;
        std::vector<float> p; float x;
        while (in >> x) p.push_back(x);
        if (op == 'D') std::cout << (w.keep_drafting(p) ? 1 : 0) << "\n";
        else if (op == 'R') std::cout << w.truncate(p) << "\n";
        else if (op == 'O') { w.observe(p, n_acc, ms, n_tok); std::cout << w.alpha() << " " << w.lam() << "\n"; }
    }
}
"""


@pytest.mark.skipif(not shutil.which("g++"), reason="needs g++")
def test_cpp_header_matches_python(tmp_path):
    src = tmp_path / "drv.cpp"
    src.write_text(DRIVER)
    exe = tmp_path / "drv"
    subprocess.run(["g++", "-std=c++17", "-O1", "-Wall", "-Wextra", "-Werror", f"-I{HEADER.parent}", str(src), "-o", str(exe)], check=True)
    tpath = tmp_path / "t.cost"
    table().save(tpath)
    rng = random.Random(3)
    py = sw.WidthPolicy(sw.CostTable.load(tpath), k_max=12)
    script, expect = [], []
    for _ in range(400):
        k = rng.randrange(0, 13)
        probs = [f32(rng.choice([rng.random(), 1 - rng.random() ** 3 * 0.05])) for _ in range(k)]
        ps = " ".join(repr(p) for p in probs)
        script.append("C")
        expect.append(str(py.n_cap()))
        if k:
            script.append(f"D {ps}")
            expect.append("1" if py.keep_drafting(probs) else "0")
        script.append(f"R {ps}")
        expect.append(str(py.truncate(probs)))
        acc = rng.randrange(0, k + 1)
        ms = round(rng.uniform(80, 300), 3)
        script.append(f"O {acc} {ms} {acc + 1} {ps}")
        py.observe(probs, acc, ms, acc + 1)
        expect.append((py.alpha(), py.lam()))
    out = subprocess.run([str(exe), str(tpath), "12"], input="\n".join(script) + "\n", capture_output=True, text=True, check=True)
    got = out.stdout.split("\n")
    for i, e in enumerate(expect):
        if isinstance(e, tuple):
            a, lam = map(float, got[i].split())
            assert a == pytest.approx(e[0], rel=1e-12) and lam == pytest.approx(e[1], rel=1e-12), (i, script[i])
        else:
            assert got[i] == e, (i, script[i])
