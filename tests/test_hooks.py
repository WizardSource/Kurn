"""kurn.hooks: an extension can add a layout with its own lowering without editing spec.py."""

from kurn import hooks
from kurn.kernels import generate
from kurn.spec import resolve


def test_extra_layout_value_and_lowering():
    def legal(op, f, t):
        return ("demo",) if (op, f, t) == ("gemv", "q8_0", "scalar") else ()

    hooks.extra_values("layout", legal)
    hooks.LOWERINGS["demo"] = lambda target, c: f"// demo {target} {c['entry']}\n"
    try:
        c = resolve({"op": "gemv", "weights": "q8_0", "target": "scalar", "layout": "demo"})
        assert generate(c) == "// demo scalar kq8_gemv\n"
    finally:
        hooks.EXTRA_VALUES["layout"].remove(legal)
        del hooks.LOWERINGS["demo"]
