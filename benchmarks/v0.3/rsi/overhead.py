import json, os, statistics, sys
sys.argv=['x']
import bench as Bm  # run from this directory with LLAMA_SERVER and MODELS set
os.environ["PORT"]="8095"
Bm.PORT=8095
size, n, prompts = Bm.WORKLOADS["think"]
log=open("logs/overhead.log","w")
p=Bm.start("Qwen3-1.7B-Q8_0.gguf", [], {}, log)
res={"off":[], "replay1":[], "replay2_rngpos":[]}
texts={}
try:
    for rnd in range(4):
        for name, extra in [("off",{}),("replay1",{"kurn_replay":1}),("replay2_rngpos",{"kurn_replay":2,"kurn_rng_position":True})]:
            out=Bm.post({"prompt":prompts[0],"n_predict":n,"temperature":0,"cache_prompt":False,"seed":1,**extra})
            res[name].append(out["timings"]["predicted_per_second"]); texts.setdefault(name,out["content"])
finally:
    Bm.stop(p)
print({k:[round(x,1) for x in v] for k,v in res.items()})
print({k:round(statistics.median(v),1) for k,v in res.items()}, "same text:", len(set(texts.values()))==1)
json.dump(res, open("overhead.json","w"))
