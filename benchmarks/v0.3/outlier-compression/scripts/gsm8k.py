"""gsm8k.py MODEL.gguf [N=50] [--think]: GSM8K test subset (first N problems) through llama-server (4 slots, greedy).
Prints accuracy and writes MODEL.gsm8k.json. Qwen3 runs with /no_think unless --think."""
import json, os, re, signal, subprocess, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
import pyarrow.parquet as pq

model = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 50
port = int(os.environ.get("PORT", "8099"))
B = os.environ.get("B", "/tmp/llama-kurn/build/bin")
rows = pq.read_table("/tmp/gsm8k/main/test-00000-of-00001.parquet").to_pylist()[:n]
srv = subprocess.Popen([f"{B}/llama-server", "-m", model, "-t", "8", "-np", "4", "-c", "8192", "--port", str(port),
                        "-ctk", "f16", "-ctv", "f16"], stdout=subprocess.DEVNULL, stderr=open(model + ".server.log", "w"))
try:
    for _ in range(600):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2)
            break
        except Exception:
            time.sleep(1)
    from gguf import GGUFReader
    qwen = "qwen3" in str(GGUFReader(model).fields["general.architecture"].contents()).lower()  # thinking off via /no_think

    def ask(r):
        q = r["question"] + "\nSolve step by step, then finish with a line 'The answer is N'." + (" /no_think" if qwen else "")
        body = json.dumps({"messages": [{"role": "user", "content": q}], "max_tokens": 640, "temperature": 0,
                           "seed": 1}).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", body,
                                     {"Content-Type": "application/json"})
        out = json.load(urllib.request.urlopen(req, timeout=1800))["choices"][0]["message"]["content"]
        m = re.findall(r"answer is[\s*$\\{boxed]*(-?[\d,]*\.?\d+)", out) or re.findall(r"(-?[\d,]*\.?\d+)", out)
        pred = m[-1].replace(",", "") if m else ""
        gold = r["answer"].split("####")[-1].strip().replace(",", "")
        try:
            ok = abs(float(pred) - float(gold)) < 1e-6
        except ValueError:
            ok = False
        return ok, pred, gold, out

    t0 = time.time()
    with ThreadPoolExecutor(4) as ex:
        res = list(ex.map(ask, rows))
    acc = sum(r[0] for r in res) / len(res)
    json.dump({"model": model, "n": len(res), "acc": acc, "seconds": time.time() - t0,
               "items": [{"pred": p, "gold": g, "ok": o, "text": t} for o, p, g, t in res]}, open(model + ".gsm8k.json", "w"))
    print(f"{os.path.basename(model)}\tGSM8K {acc * 100:.1f}% ({sum(r[0] for r in res)}/{len(res)})\t{time.time() - t0:.0f}s")
finally:
    srv.send_signal(signal.SIGINT)
    srv.wait(timeout=60)
