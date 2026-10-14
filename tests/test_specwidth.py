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
    assert [sw.kernel_cols(m) for m in range(1, 9)] == [1, 2, 3, 4, 5, 6, 7, 8]
    assert sw.kernel_passes(8) == [8]
    assert sw.kernel_passes(9) == [8, 1]
    assert sw.kernel_passes(12) == [8, 4]
    assert sw.kernel_passes(19) == [8, 8, 3]
    with pytest.raises(ValueError):
        sw.kernel_cols(9)


def test_buft_keys_match_the_generator():
    sys.path.insert(0, str(INTEG))
    gen = pytest.importorskip("gen_ggml_sources")
    assert sw.BUFT_DEFAULT_KEYS == gen.DEFAULT_KEYS
    assert sw.BUFT_TUNED_KEYS == gen.TUNED_KEYS
    assert sw.VFY_COLS == gen.VFY_COLS
    assert sw.BUFT_VFY_KEYS == gen.VFY_KEYS


def test_buft_configs_resolve():
    c = sw.buft_configs("q8_0", 8)
    assert c[1]["op"] == "gemv" and c[1]["rows"] == 8
    assert [c[k]["cols"] for k in sw.VFY_COLS] == list(range(2, 9))
    assert all(c[k]["layout"] == "i16" and c[k]["rows"] * k <= 8 for k in sw.VFY_COLS)
    assert [c[k]["rows"] for k in (2, 3, 4, 8)] == [4, 2, 2, 1]  # the buffer type's measured schedule
    assert all(c[k]["pfgran"] == "line" for k in sw.VFY_COLS)


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


# --------------------------------------------------------------------------- retrieval arm
def selector(model_ms=8.0, has_model=True, has_retr=True, k_model=12, k_retr=15):
    return sw.ArmSelector(table(d=[model_ms] * 16), k_model, k_retr, has_model, has_retr)


def feed(sel, arm, rate, steps, rng, k=6):
    for _ in range(steps):
        n = sum(1 for _ in range(k) if rng.random() < rate) if arm else 0
        acc = 0
        while acc < k and acc < n:
            acc += 1
        probs = [None] * k if arm == sel.RETRIEVAL else ([f32(0.9)] * k if arm == sel.MODEL else [])
        sel.observe(arm, probs, acc if arm else 0, 150.0, (acc if arm else 0) + 1)


def test_retrieval_taken_when_the_draft_model_is_expensive():
    """A draft model costing ~40% of a target step (Qwen3-1.7B + 0.6B on CPU) loses to a free lookup."""
    sel = selector(model_ms=40.0)
    rng = random.Random(1)
    feed(sel, sel.RETRIEVAL, 0.8, 40, rng)
    assert sel.take_retrieval(16) >= 4


def test_cheap_accurate_model_beats_a_poor_lookup():
    sel = selector(model_ms=4.0)
    rng = random.Random(2)
    feed(sel, sel.MODEL, 0.9, 60, rng)
    feed(sel, sel.RETRIEVAL, 0.1, 60, rng, k=1)
    assert sel.take_retrieval(16) == 0
    assert sel.allow_model()


def test_unprofitable_model_is_not_run_but_probed():
    sel = selector(model_ms=60.0)
    rng = random.Random(3)
    feed(sel, sel.MODEL, 0.2, 30, rng)
    assert sel.model.best_value() <= sel.model.value_at(0, 0.0) + 1e-9
    runs = 0
    for _ in range(64):
        if sel.allow_model():
            runs += 1
            sel.observe(sel.MODEL, [f32(0.5)], 0, 150.0, 1)
        else:
            sel.observe(sel.NONE, [], 0, 150.0, 1)
    assert 1 <= runs <= 64 // sel.model.probe_every + 1


def test_catch_up_is_weighed_against_the_remaining_horizon():
    """Lazy sync: a draft model 900 prompt tokens behind (~600 ms of draft prefill) is not worth catching up for a
    short remainder, but is for a long one; probes never pay a large catch-up."""
    flat = sw.CostTable([100.0 + 2.0 * m for m in range(24)], [4.0] * 24)  # kurn-like near-flat verify costs
    sel = sw.ArmSelector(flat, 12, 23, True, True)
    rng = random.Random(4)
    feed(sel, sel.MODEL, 0.8, 40, rng)
    assert sel.allow_model()
    assert sel.allow_model(catchup_ms=1.0, horizon_tokens=400)
    assert not sel.allow_model(catchup_ms=600.0, horizon_tokens=20)
    assert sel.allow_model(catchup_ms=600.0, horizon_tokens=100000)
    assert sel.allow_model(catchup_ms=600.0, horizon_tokens=-1)  # unknown horizon: catch-up ignored
    idle = selector(model_ms=60.0)
    feed(idle, idle.MODEL, 0.1, 30, rng)
    for _ in range(idle.model.probe_every):
        idle.observe(idle.NONE, [], 0, 150.0, 1)
    assert idle.allow_model(catchup_ms=0.0, horizon_tokens=400)
    assert not idle.allow_model(catchup_ms=600.0, horizon_tokens=400)


def test_retrieval_only_without_a_draft_model():
    sel = selector(has_model=False)
    assert not sel.allow_model()
    assert sel.take_retrieval(16) > 0
    assert sel.take_retrieval(0) == 0
    assert selector(has_retr=False).take_retrieval(16) == 0


def test_shared_lambda_and_observe_split():
    sel = selector()
    sel.observe(sel.RETRIEVAL, [None] * 4, 3, 120.0, 4)
    sel.observe(sel.MODEL, [f32(0.95)] * 2, 1, 140.0, 2)
    assert sel.model.lam() == sel.retr.lam()
    w = sw.WidthPolicy(table(), k_max=12)
    v = sw.WidthPolicy(table(), k_max=12)
    w.observe([f32(0.9), f32(0.5)], 1, 100.0, 2)
    v.observe_acc([f32(0.9), f32(0.5)], 1)
    v.observe_time(100.0, 2)
    assert (w.alpha(), w.lam(), w.n_cap()) == (v.alpha(), v.lam(), v.n_cap())


ARM_DRIVER = r"""
#include "kurn-spec-width.h"
#include <iostream>
#include <sstream>
int main(int, char ** argv) {
    kurn::verify_cost_table t;
    if (!t.load(argv[1])) return 2;
    kurn::arm_selector s;
    s.init(t, atoi(argv[2]), atoi(argv[3]), atoi(argv[4]) != 0, atoi(argv[5]) != 0);
    std::string line;
    std::cout.precision(17);
    while (std::getline(std::cin, line)) {
        std::istringstream in(line);
        char op; in >> op;
        if (op == 'T') { int L; in >> L; std::cout << s.take_retrieval(L) << "\n"; continue; }
        if (op == 'A') { double c = 0; int h = -1; in >> c >> h; std::cout << (s.allow_model(c, h) ? 1 : 0) << "\n"; continue; }
        int arm, n_acc, n_tok; double ms; in >> arm >> n_acc >> ms >> n_tok;
        std::vector<float> p; float x;
        while (in >> x) p.push_back(x);
        s.observe((kurn::arm_selector::arm) arm, p, n_acc, ms, n_tok);
        std::cout << s.model.alpha() << " " << s.retr.alpha() << " " << s.lam() << "\n";
    }
}
"""


@pytest.mark.skipif(not shutil.which("g++"), reason="needs g++")
@pytest.mark.parametrize("has_model,has_retr,model_ms", [(1, 1, 8.0), (1, 1, 45.0), (0, 1, 8.0), (1, 0, 8.0)])
def test_cpp_arm_selector_matches_python(tmp_path, has_model, has_retr, model_ms):
    src = tmp_path / "arm.cpp"
    src.write_text(ARM_DRIVER)
    exe = tmp_path / "arm"
    subprocess.run(["g++", "-std=c++17", "-O1", "-Wall", "-Wextra", "-Werror", f"-I{HEADER.parent}", str(src), "-o", str(exe)], check=True)
    tpath = tmp_path / "t.cost"
    table(d=[model_ms] * 16).save(tpath)
    py = sw.ArmSelector(sw.CostTable.load(tpath), 12, 15, bool(has_model), bool(has_retr))
    rng = random.Random(11)
    script, expect = [], []
    for _ in range(300):
        L = rng.randrange(0, 17)
        script.append(f"T {L}")
        expect.append(str(py.take_retrieval(L)))
        c = rng.choice([0.0, round(rng.uniform(0, 400), 3)])
        h = rng.choice([-1, rng.randrange(1, 600)])
        script.append(f"A {c} {h}")
        expect.append("1" if py.allow_model(c, h) else "0")
        arm = rng.choice([0, 1, 2])
        k = 0 if arm == 0 else rng.randrange(1, 9)
        probs = [-1.0] * k if arm == 1 else [f32(rng.random()) for _ in range(k)]
        acc = rng.randrange(0, k + 1)
        ms = round(rng.uniform(80, 300), 3)
        script.append(f"O {arm} {acc} {ms} {acc + 1} " + " ".join(repr(p) for p in probs))
        py.observe(arm, [None if p < 0 else p for p in probs], acc, ms, acc + 1)
        expect.append((py.model.alpha(), py.retr.alpha(), py.lam()))
    out = subprocess.run([str(exe), str(tpath), "12", "15", str(has_model), str(has_retr)],
                         input="\n".join(script) + "\n", capture_output=True, text=True, check=True)
    got = out.stdout.split("\n")
    for i, e in enumerate(expect):
        if isinstance(e, tuple):
            vals = list(map(float, got[i].split()))
            assert vals == pytest.approx(list(e), rel=1e-12), (i, script[i])
        else:
            assert got[i] == e, (i, script[i])
