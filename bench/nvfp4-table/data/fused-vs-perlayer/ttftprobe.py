"""TTFT across context lengths. The fused context-KV GEMM runs once per prefill,
so that is where fusing it can pay -- not in decode."""
import json, os, re, sys, time, urllib.request, random
KEY=os.environ["LLM_KEY"]; BASE="http://127.0.0.1:8080"
def post(p,payload,timeout=600):
    rq=urllib.request.Request(BASE+p,data=json.dumps(payload).encode(),method="POST",
        headers={"Content-Type":"application/json","Authorization":f"Bearer {KEY}"})
    with urllib.request.urlopen(rq,timeout=timeout) as r: return json.loads(r.read())
corpus=re.sub(r'[\x00-\x1f]',' ',open("/opt/llm/corpus.txt",errors="ignore").read())
label,out=sys.argv[1],sys.argv[2]
res={}
for ctx_chars,name in ((4000,"~1k"),(32000,"~8k"),(120000,"~30k")):
    ts=[]
    for rep in range(3):
        # unique prefix per request: prefix caching must not shorten the prefill
        off=(rep*911_000+ctx_chars*7) % max(1,len(corpus)-ctx_chars-1)
        p=f"[{label} {rep} {random.random()}] " + corpus[off:off+ctx_chars]
        t=time.perf_counter()
        post("/v1/completions",{"model":"qwen38-27b","prompt":p,"max_tokens":1,"temperature":0})
        ts.append(time.perf_counter()-t)
    ts.sort(); res[name]={"median":round(ts[1],3),"all":[round(x,3) for x in ts]}
    print(f"  {label} | {name} Kontext: median {ts[1]:.3f}s   {[round(x,3) for x in ts]}")
json.dump({"label":label,"res":res},open(out,"w"),indent=1)
