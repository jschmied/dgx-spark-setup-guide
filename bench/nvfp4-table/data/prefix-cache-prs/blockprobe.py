"""Misst, wieviel cachebares Praefix ein Lauf zurueckgibt -- pro Blockgrenze.
Aufruf: blockprobe.py <label> <ausgabedatei>"""
import json,os,re,sys,time,urllib.request
KEY=os.environ["LLM_KEY"]; BASE="http://127.0.0.1:8080"
def post(p,payload,timeout=300):
    rq=urllib.request.Request(BASE+p,data=json.dumps(payload).encode(),method="POST",
        headers={"Content-Type":"application/json","Authorization":f"Bearer {KEY}"})
    with urllib.request.urlopen(rq,timeout=timeout) as r: return json.loads(r.read())
def metrics():
    rq=urllib.request.Request(BASE+"/metrics",headers={"Authorization":f"Bearer {KEY}"})
    with urllib.request.urlopen(rq,timeout=20) as r: t=r.read().decode()
    g=lambda x: float(re.search(rf'^vllm:{x}\{{[^}}]*}} ([0-9.e+]+)',t,re.M).group(1))
    return g('prefix_cache_queries_total'), g('prefix_cache_hits_total')
BLK=int(os.environ.get("BLK","1648"))
corpus=re.sub(r'[\x00-\x1f]',' ',open("/opt/llm/corpus.txt",errors="ignore").read())
OFF=int(os.environ.get("OFF","2000000"))
def toks(off):
    return post("/tokenize",{"model":"qwen38-27b","prompt":corpus[off:off+60000]})["tokens"]
def exact(n, ids):
    s=post("/detokenize",{"model":"qwen38-27b","tokens":ids[:n]})["prompt"]
    got=post("/tokenize",{"model":"qwen38-27b","prompt":s})["count"]
    k=n
    for _ in range(12):
        if got==n: return s
        k+=n-got
        s=post("/detokenize",{"model":"qwen38-27b","tokens":ids[:k]})["prompt"]
        got=post("/tokenize",{"model":"qwen38-27b","prompt":s})["count"]
    return None
label,out=sys.argv[1],sys.argv[2]
res=[]
# Jedes Ziel aus einem EIGENEN Korpusbereich -- sonst ist das zweite ein Praefix des ersten
for target, off in ((4*BLK, OFF), (4*BLK+700, OFF+2000000)):
    p=exact(target, toks(off))
    if p is None: res.append({"target":target,"error":"Kalibrierung fehlgeschlagen"}); continue
    runs=[]
    for i in (1,2,3):
        q0,h0=metrics(); t=time.perf_counter()
        post("/v1/completions",{"model":"qwen38-27b","prompt":p,"max_tokens":1,"temperature":0})
        dt=time.perf_counter()-t; q1,h1=metrics()
        runs.append({"n":i,"s":round(dt,2),"queries":int(q1-q0),"hits":int(h1-h0)})
    pred=(target//BLK-1)*BLK
    res.append({"target":target,"aligned":target%BLK==0,"vorhergesagt_mit_rueckstufung":pred,"runs":runs})
    print(f"  {label} | {target} Token (aligned={target%BLK==0}): "
          + " | ".join(f"#{r['n']} {r['s']}s hits={r['hits']}" for r in runs)
          + f"   [Formel: {pred}]")
json.dump({"label":label,"block":BLK,"res":res},open(out,"w"),indent=1)
