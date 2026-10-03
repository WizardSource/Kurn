"""End-to-end llama.cpp harnesses (benchmarks/v0.2/e2e): output parsing, mode specs and the
fixed perplexity text, without running llama.cpp."""

import sys

from conftest import ROOT

E2E = ROOT / "benchmarks" / "v0.2" / "e2e"
sys.path.insert(0, str(E2E))
import run_e2e  # noqa: E402
import run_spec  # noqa: E402

PERPLEXITY_TAIL = """[1]8.1234,[2]9.8765,
Final estimate: PPL = 12.3456 +/- 0.78901
llama_perf_context_print: load time = 1.0 ms
"""

BENCH_OUT = """load_backend: loaded CPU backend
[
  {"model_type": "qwen3 1.7B Q8_0", "n_prompt": 0, "n_gen": 128, "avg_ts": 61.25}
]
"""


def test_ppl_value():
    assert run_e2e.ppl_value(PERPLEXITY_TAIL) == (12.3456, 0.78901)


def test_bench_json_skips_log_lines():
    assert run_e2e.bench_json(BENCH_OUT)[0]["avg_ts"] == 61.25


def test_parse_mode():
    name, (bindir, env, flags) = run_e2e.parse_mode("kurn-ub=/opt/llama/bin:GGML_KURN_AMX=0,GGML_KURN_CHUNKS=8:-ub 128")
    assert name == "kurn-ub" and bindir == "/opt/llama/bin"
    assert env == {"GGML_KURN_AMX": "0", "GGML_KURN_CHUNKS": "8"}
    assert flags == ["-ub", "128"]
    assert run_e2e.parse_mode("x=/b")[1] == ("/b", {}, [])


def test_default_modes():
    modes = run_e2e.default_modes("/stock", "/kurn")
    assert modes["plain"] == ("/stock", {}, ["--repack", "0"])
    assert modes["kurn"][1] == {"GGML_KURN_AMX": "0"}
    assert modes["kurn-amx"][1] == {"GGML_KURN_AMX": "1"}


def test_ppl_text_script_pins_hash():
    src = (E2E / "make_ppl_text.sh").read_text()
    assert "7265711fa147bb438d4eb0ec1e4a10c256053f09277fe84ae3e2911f43e5adfe" in src


def test_spec_parsing():
    err = """I encoded   16 tokens in    0.368 seconds, speed:   43.479 t/s
I decoded   25 tokens in    0.927 seconds, speed:   26.979 t/s
I n_draft   = 8
I n_predict = 25
I n_drafted = 22
I n_accept  = 20
"""
    v = run_spec.parse(err, "spec")
    assert v["n"] == 25 and v["n_drafted"] == 22 and v["n_accept"] == 20
    assert abs(v["tok_s"] - 25 / 0.927) < 1e-9
    greedy = """common_perf_print: prompt eval time =    53.93 ms /    16 tokens
common_perf_print:        eval time =   756.39 ms /    15 runs   (   50.43 ms per token,    19.83 tokens per second)
"""
    g = run_spec.parse(greedy, "greedy")
    assert g["n"] == 16 and abs(g["tok_s"] - 15 / 0.75639) < 1e-9
    assert run_spec.parse("common_perf_print:        eval time =     0.00 ms /     0 runs", "greedy")["n"] == 1


def test_spec_gen_text_strips_prompt_echo():
    p = run_spec.PROMPTS[0]
    out = "\n\n" + run_spec.chat(p) + "<think>\n\n</think>\n\nHello world\n\n\n"
    assert run_spec.gen_text(out, "spec", p) == "<think>\n\n</think>\n\nHello world"
    assert run_spec.gen_text("  Hello world\n", "greedy", p) == "Hello world"
