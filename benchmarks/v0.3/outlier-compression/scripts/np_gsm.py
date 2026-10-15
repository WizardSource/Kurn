"""np_gsm.py MODEL [N=8]: N concurrent GSM8K requests (4 slots); prints how many replies are empty"""
import json, os, subprocess, sys, time, urllib.request, signal
from concurrent.futures import ThreadPoolExecutor
import pyarrow.parquet as pq
model = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 8
port = 8094
srv = subprocess.Popen(["/tmp/llama-kurn/build/bin/llama-server", "-m", model, "-t", "8", "-np", "4", "-c", "8192",
                        "--port", str(port), "-ctk", "f16", "-ctv", "f16"], stdout=subprocess.DEVNULL,
                       stderr=open("/tmp/np_gsm.log", "w"))
for _ in range(300):
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2)
        break
    except Exception:
        time.sleep(1)
rows = pq.read_table("/tmp/gsm8k/main/test-00000-of-00001.parquet").to_pylist()[:n]


def ask(r):
    q = r["question"] + "\nSolve step by step, then finish with a line 'The answer is N'. /no_think"
    body = json.dumps({"messages": [{"role": "user", "content": q}], "max_tokens": int(os.environ.get("MAXT", "200")),
                       "temperature": 0, "seed": 1}).encode()
    o = json.load(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", body,
                                                                {"Content-Type": "application/json"})))
    m = o["choices"][0]["message"]
    if not (m.get("content") or "").strip():
        print("EMPTY:", json.dumps(o)[:1500], flush=True)
    return (m.get("content") or "")


with ThreadPoolExecutor(4) as ex:
    res = list(ex.map(ask, rows))
print(f"{sum(1 for t in res if not t.strip())}/{len(res)} empty;", [t[:40] for t in res[:4]])
srv.send_signal(signal.SIGINT)
srv.wait()
