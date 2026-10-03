"""`kernel_for()`: which kernel an integration should run for (format, batch) on this GPU.

The table comes from `kurn gpu report` (dispatch.json) on the same GPU model. KURN is chosen only
where the benchmark matrix showed a win over the best same-format competitor by more than 2 sigma
(or where ggml-cuda has no kernel for the format); every tie, loss, unmeasured batch, other GPU
or missing table returns "stock", i.e. the competitor's own kernel. An integrated build is
therefore never slower than stock on measured cells, and identical to stock elsewhere.

Table lookup order: the `table` argument (dict or path), $KURN_GPU_DISPATCH, none.
"""

import json
import os

from .spec import SpecError, resolve


def load_table(table=None):
    if isinstance(table, dict):
        return table
    path = table or os.environ.get("KURN_GPU_DISPATCH")
    if not path or not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def parse_config(s):
    """kg_config() string ("op=gemv weights=q4_0 arch=sm_80 ...") -> resolved config dict."""
    kv = {}
    for item in s.split():
        k, _, v = item.partition("=")
        kv[k] = int(v) if v.lstrip("-").isdigit() else v
    for k in ("mins", "unpack"):
        if kv.get(k) == "none":
            kv.pop(k)
    return resolve(kv)


def kernel_for(fmt, batch, arch=None, table=None):
    """-> {"impl": "kurn", "config": {...}, "measured_batch": b} or {"impl": "stock", "reason": ...}."""
    t = load_table(table)
    if not t:
        return {"impl": "stock", "reason": "no dispatch table (run the benchmark matrix on this GPU)"}
    if arch and t.get("arch") and t["arch"] != arch:
        return {"impl": "stock", "reason": f"table was measured on {t['arch']}, not {arch}"}
    cells = t.get("cells", {}).get(fmt)
    if not cells:
        return {"impl": "stock", "reason": f"no measurements for {fmt}"}
    measured = sorted(int(b) for b in cells)
    lo = max((b for b in measured if b <= batch), default=None)
    hi = min((b for b in measured if b >= batch), default=None)
    # between two measured batch sizes, both must be KURN wins; outside the range, the nearest one
    need = [b for b in {lo, hi} if b is not None]
    if any(cells[str(b)]["impl"] != "kurn" for b in need):
        b = next(b for b in need if cells[str(b)]["impl"] != "kurn")
        return {"impl": "stock", "reason": f"batch {b}: {cells[str(b)]['verdict']}"}
    pick = lo if lo is not None else hi
    try:
        cfg = parse_config(cells[str(pick)]["config"])
    except (SpecError, KeyError, TypeError) as e:
        return {"impl": "stock", "reason": f"unusable table entry: {e}"}
    return {"impl": "kurn", "config": cfg, "measured_batch": pick, "verdict": cells[str(pick)]["verdict"]}
