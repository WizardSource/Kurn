"""ppl_table.py TAG [GGUF_DIR]: KL / PPL / top-1 per model from /tmp/ppl-logs/ppl-TAG*-NAME.log, plus file sizes
(model + sidecar) and bits per weight over the linear layers (main + side channels)."""
import glob, os, re, sys

tag = sys.argv[1]
rows = []
for f in sorted(glob.glob(f"/tmp/ppl-logs/ppl-{tag}-*.log")) + sorted(glob.glob(f"/tmp/ppl-logs/ppl-{tag}-extra-*.log")):
    s = open(f).read()
    name = os.path.basename(f)[len(f"ppl-{tag}-"):-4]
    if name.startswith("extra-"):
        name = name[6:]
    m = lambda rx: (re.search(rx, s) or [None, None])
    ppl = re.search(r"Mean PPL\(Q\)\s+:\s+([\d.]+) ±\s+([\d.]+)", s)
    kld = re.search(r"Mean    KLD:\s+([\d.]+) ±\s+([\d.]+)", s)
    top = re.search(r"^Same top p:\s+([\d.]+) ±\s+([\d.]+)", s, re.M)
    mf = re.search(r"llama_perplexity: .*|main: .*", s)
    fn = re.search(r"llama_model_loader: loaded meta data with .* from (\S+)", s)
    size = None
    if fn:
        p = fn.group(1)
        p = p if os.path.isabs(p) else os.path.join("/tmp/gguf", p)
        if os.path.exists(p):
            size = os.path.getsize(p) + (os.path.getsize(p + ".oa.gguf") if os.path.exists(p + ".oa.gguf") else 0)
    if not kld:
        continue
    rows.append((name, size, float(ppl.group(1)), float(ppl.group(2)), float(kld.group(1)), float(kld.group(2)),
                 float(top.group(1)) if top else float("nan")))
print(f"| model | size MiB | PPL | KL vs F16 | same top-1 |")
print("|---|---|---|---|---|")
for n, sz, p, pe, k, ke, t in rows:
    print(f"| {n} | {sz / 2**20:.0f} | {p:.3f} ± {pe:.3f} | {k:.4f} ± {ke:.4f} | {t:.2f}% |" if sz else
          f"| {n} | ? | {p:.3f} | {k:.4f} | {t:.2f}% |")
