"""kurn.mixed: convex-hull allocation, recipes, GGUF write/read and the predictor on synthetic GGUFs."""

import itertools
import json

import pytest

from kurn import mixed

np = pytest.importorskip("numpy")

gguf = pytest.importorskip("gguf")

TYPES = {"Q4_0": 4.5, "Q5_0": 5.5, "Q8_0": 8.5}


def _prof(errs, params):
    return {
        "types": dict(TYPES),
        "tensors": {
            n: {"kind": n.split(".")[2], "layer": 0, "params": p, "rows": 1, "cols": p,
                "err": e, "abs_err": {q: v * p for q, v in e.items()}}
            for (n, e), p in zip(errs.items(), params)
        },
    }  # fmt: skip


def _random_prof(rng, n):
    errs = {}
    for i in range(n):
        base = rng.uniform(1e-3, 1e-1)
        # error falls ~4x per extra bit, with tensor-specific jitter
        errs[f"blk.{i}.ffn_down.weight"] = {q: base * 4.0 ** (4.5 - b) * rng.uniform(0.5, 1.5) for q, b in TYPES.items()}
    return _prof(errs, [int(rng.integers(1, 5)) * 1024 for _ in range(n)])


def test_hull_drops_dominated_and_concave_points():
    pts = [(1, 10.0, "a"), (2, 4.0, "b"), (3, 3.9, "c"), (4, 1.0, "d"), (5, 2.0, "e")]
    assert [p[2] for p in mixed._hull(pts)] == ["a", "b", "d"]


@pytest.mark.parametrize("seed", range(6))
def test_allocation_matches_brute_force_on_hull_points(seed):
    rng = np.random.default_rng(seed)
    prof = _random_prof(rng, 5)
    names = list(prof["tensors"])
    total = sum(v["params"] for v in prof["tensors"].values())
    for bpw in (4.6, 5.2, 6.0, 7.3):
        assign, got, obj = mixed.allocate(prof, bpw)
        assert got <= bpw + 1e-9
        assert got == pytest.approx(mixed.mix_bpw(prof, assign))
        best = min(
            sum(prof["tensors"][n]["err"][q] for n, q in zip(names, combo))
            for combo in itertools.product(TYPES, repeat=len(names))
            if sum(TYPES[q] * prof["tensors"][n]["params"] for n, q in zip(names, combo)) <= bpw * total
        )
        # greedy on the hull is optimal at the hull vertices and within one step otherwise
        assert obj >= best - 1e-12
        assert obj <= best * 4.0 + 1e-12


def test_allocation_extremes_and_fixed():
    prof = _random_prof(np.random.default_rng(1), 4)
    a, got, _ = mixed.allocate(prof, 4.5)
    assert set(a.values()) == {"Q4_0"} and got == pytest.approx(4.5)
    a, got, _ = mixed.allocate(prof, 9.0)
    assert set(a.values()) == {"Q8_0"}
    n0 = next(iter(prof["tensors"]))
    a, _, _ = mixed.allocate(prof, 9.0, fixed={n0: "Q4_0"})
    assert a[n0] == "Q4_0"
    a, _, _ = mixed.taalas_rule(prof, 6.0, low="Q4_0", high="Q8_0")
    assert set(a.values()) <= {"Q4_0", "Q8_0"}


def test_recipe_lines_are_anchored_and_skip_token_embd(tmp_path):
    assign = {"token_embd.weight": "Q6_K", "blk.1.attn_q.weight": "Q3_K", "blk.10.attn_q.weight": "Q6_K"}
    lines = mixed.recipe_lines(assign)
    assert lines == [r"^blk\.1\.attn_q\.weight$=q3_k", r"^blk\.10\.attn_q\.weight$=q6_k"]
    mixed.write_recipe(str(tmp_path / "r.txt"), assign, {"bpw": 4.0})
    assert json.load(open(tmp_path / "r.json"))["assign"] == assign


def _write_src(path, rng):
    w = gguf.GGUFWriter(str(path), arch="qwen3")
    w.add_block_count(2)
    ws = {}
    for i in range(2):
        W = rng.standard_normal((48, 256)).astype(np.float32)
        W[:, 7] *= 20  # outlier input channel
        ws[f"blk.{i}.ffn_down.weight"] = W
        w.add_tensor(f"blk.{i}.ffn_down.weight", W.astype(np.float16))
        w.add_tensor(f"blk.{i}.ffn_norm.weight", np.ones(256, dtype=np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return ws


def test_write_gguf_roundtrip_and_profile(tmp_path):
    rng = np.random.default_rng(0)
    src = tmp_path / "src.gguf"
    ws = _write_src(src, rng)
    quants = {}
    for q in ("Q4_0", "Q8_0"):
        quants[q] = str(tmp_path / f"{q}.gguf")
        mixed.write_gguf(str(src), quants[q], {n: W for n, W in ws.items()}, dtype=q)
    ts = mixed.tensors(quants["Q8_0"])
    assert mixed.type_name(ts["blk.0.ffn_down.weight"]) == "Q8_0"
    assert mixed.type_name(ts["blk.0.ffn_norm.weight"]) == "F32"
    W = ws["blk.1.ffn_down.weight"]
    assert np.linalg.norm(mixed.dequant(ts["blk.1.ffn_down.weight"]) - W) < 0.01 * np.linalg.norm(W)
    # lazy replacement keeps the KV metadata and the untouched tensors
    out = tmp_path / "f16.gguf"
    mixed.write_gguf(str(src), str(out), {"blk.0.ffn_down.weight": lambda: ws["blk.0.ffn_down.weight"] * 2})
    r = mixed.reader(str(out))
    assert r.fields["general.architecture"].contents() == "qwen3"
    t = mixed.tensors(r)
    np.testing.assert_allclose(mixed.dequant(t["blk.0.ffn_down.weight"]), 2 * ws["blk.0.ffn_down.weight"], rtol=1e-3, atol=1e-2)

    # raw copies of quantized / BF16 tensors keep type and data
    bf = tmp_path / "bf16.gguf"
    mixed.write_gguf(str(src), str(bf), dict(ws), dtype="BF16")
    out2 = tmp_path / "raw.gguf"
    mixed.write_gguf(str(bf), str(out2), {}, raw={"blk.1.ffn_down.weight": mixed.tensors(quants["Q4_0"])["blk.1.ffn_down.weight"]})
    t2 = mixed.tensors(str(out2))
    assert mixed.type_name(t2["blk.0.ffn_down.weight"]) == "BF16"
    bf0 = mixed.dequant(mixed.tensors(str(bf))["blk.0.ffn_down.weight"])
    np.testing.assert_array_equal(mixed.dequant(t2["blk.0.ffn_down.weight"]), bf0)
    assert mixed.type_name(t2["blk.1.ffn_down.weight"]) == "Q4_0"
    np.testing.assert_array_equal(mixed.dequant(t2["blk.1.ffn_down.weight"]),
                                  mixed.dequant(mixed.tensors(quants["Q4_0"])["blk.1.ffn_down.weight"]))  # fmt: skip

    prof = mixed.profile(str(src), quants, rows_out=str(tmp_path / "rows.npz"), log=lambda s: None)
    assert set(prof["tensors"]) == set(ws)
    for n, v in prof["tensors"].items():
        assert v["err"]["Q8_0"] < v["err"]["Q4_0"] < 0.05
        W = mixed.dequant(mixed.tensors(str(src))[n])
        D = mixed.dequant(mixed.tensors(quants["Q4_0"])[n]) - W
        assert v["err"]["Q4_0"] == pytest.approx(float((D * D).sum() / (W * W).sum()), rel=1e-6)
        assert v["feat"]["max_over_rms"] > 5
    row_assign, types, got = mixed.allocate_rows(prof, str(tmp_path / "rows.npz"), 6.5, group=16)
    assert got <= 6.5 + 1e-9 and len(row_assign["blk.0.ffn_down.weight"]) == 3
    comp = mixed.compose_rows(str(src), quants, row_assign, types, 16, str(tmp_path / "rows.gguf"))
    t = mixed.tensors(comp)["blk.0.ffn_down.weight"]
    ga = row_assign["blk.0.ffn_down.weight"]
    want = mixed.dequant(mixed.tensors(quants[types[ga[0]]])["blk.0.ffn_down.weight"])[:16]
    np.testing.assert_allclose(mixed.dequant(t)[:16], want, rtol=1e-3, atol=1e-3)


def test_cli_dispatch(tmp_path, capsys):
    from kurn import cli

    prof = _random_prof(np.random.default_rng(2), 3)
    p = tmp_path / "p.json"
    p.write_text(json.dumps(prof))
    assert cli.main(["mix", "plan", str(p), "--bpw", "6", "-o", str(tmp_path / "r.txt")]) == 0
    assert json.load(open(tmp_path / "r.json"))["bpw"] <= 6
    assert cli.main(["mix", "frontier", str(p), "--bpw", "5,7"]) == 0
    assert "bpw" in capsys.readouterr().out
    kinds = {v["kind"] for v in prof["tensors"].values()}
    w = ",".join(f"{k}=1000" for k in kinds) + ",nope=1"
    assert cli.main(["mix", "plan", str(p), "--bpw", "6", "--weights", w, "-o", str(tmp_path / "w.txt")]) == 0
    rw = json.load(open(tmp_path / "w.json"))
    assert rw["how"]["nope"] == 1 and rw["objective"] == pytest.approx(1000 * json.load(open(tmp_path / "r.json"))["objective"])


def test_ggml_quantize_matches_gguf_py_reference():
    import os

    if not os.path.exists(os.environ.get("KURN_LIBGGML", os.path.expanduser("~/src/llama.cpp/build/bin/libggml-base.so"))):
        pytest.skip("libggml-base.so not available")
    rng = np.random.default_rng(5)
    W = rng.standard_normal((37, 512)).astype(np.float32)
    for q in ("Q8_0", "Q4_0"):
        raw = mixed.ggml_quantize(W, q, threads=3)
        ref = gguf.quants.quantize(W, gguf.GGMLQuantizationType[q])
        np.testing.assert_array_equal(raw, ref)
    R = mixed.ggml_roundtrip(W, "Q4_K", np.ones(512))
    assert R.shape == W.shape and np.linalg.norm(R - W) < 0.15 * np.linalg.norm(W)


def test_quantize_inproc_writes_recipe_types(tmp_path):
    import os

    if not os.path.exists(os.environ.get("KURN_LIBGGML", os.path.expanduser("~/src/llama.cpp/build/bin/libggml-base.so"))):
        pytest.skip("libggml-base.so not available")
    src = tmp_path / "src.gguf"
    ws = _write_src(src, np.random.default_rng(6))
    names = sorted(ws)
    assign = {n: ("Q4_K" if i % 2 else "Q6_K") for i, n in enumerate(names)}
    rj = tmp_path / "r.json"
    rj.write_text(json.dumps({"assign": assign}))
    out = mixed.quantize_inproc(str(rj), str(src), str(tmp_path / "mix.gguf"), threads=2)
    t = mixed.tensors(out)
    for n, q in assign.items():
        assert mixed.type_name(t[n]) == q
        np.testing.assert_array_equal(np.asarray(t[n].data).view(np.uint8).reshape(-1),
                                      mixed.ggml_quantize(ws[n].astype(np.float16), q, threads=1).reshape(-1))  # fmt: skip
    assert mixed.type_name(t["blk.0.ffn_norm.weight"]) == "F32"
