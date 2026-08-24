"""Three sends of one exact-block prompt; the journal carries the trace."""
import json,os,re,sys,time,urllib.request
KEY=os.environ["LLM_KEY"]; BASE="http://127.0.0.1:8080"
def post(p,payload,t=600):
    rq=urllib.request.Request(BASE+p,data=json.dumps(payload).encode(),method="POST",
        headers={"Content-Type":"application/json","Authorization":f"Bearer {KEY}"})
    with urllib.request.urlopen(rq,timeout=t) as r: return json.loads(r.read())
def metrics():
    rq=urllib.request.Request(BASE+"/metrics",headers={"Authorization":f"Bearer {KEY}"})
    with urllib.request.urlopen(rq,timeout=20) as r: t=r.read().decode()
    g=lambda x: float(re.search(rf'^vllm:{x}\{{[^}}]*}} ([0-9.e+]+)',t,re.M).group(1))
    return g('prefix_cache_hits_total')
corpus=re.sub(r'[\x00-\x1f]',' ',open("/opt/llm/corpus.txt",errors="ignore").read())
ids=post("/tokenize",{"model":"qwen38-27b","prompt":corpus[1000000:1000000+60000]})["tokens"]
N=6400; k=N
for _ in range(14):
    s=post("/detokenize",{"model":"qwen38-27b","tokens":ids[:k]})["prompt"]
    got=post("/tokenize",{"model":"qwen38-27b","prompt":s})["count"]
    if got==N: break
    k+=N-got
print(f"  Prompt: {got} Token (Ziel {N})")
for i in (1,2,3):
    h0=metrics()
    post("/v1/completions",{"model":"qwen38-27b","prompt":s,"max_tokens":1,"temperature":0})
    print(f"  send {i}: hits +{int(metrics()-h0)}")
