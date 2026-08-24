"""Three sends, then his separation test: parent once, sibling B immediately."""
import json,os,re,sys,urllib.request
KEY=os.environ["LLM_KEY"]; BASE="http://127.0.0.1:8080"
def post(p,payload,t=600):
    rq=urllib.request.Request(BASE+p,data=json.dumps(payload).encode(),method="POST",
        headers={"Content-Type":"application/json","Authorization":f"Bearer {KEY}"})
    with urllib.request.urlopen(rq,timeout=t) as r: return json.loads(r.read())
def hits():
    rq=urllib.request.Request(BASE+"/metrics",headers={"Authorization":f"Bearer {KEY}"})
    with urllib.request.urlopen(rq,timeout=20) as r: t=r.read().decode()
    return float(re.search(r'^vllm:prefix_cache_hits_total\{[^}]*} ([0-9.e+]+)',t,re.M).group(1))
tok=lambda s: post("/tokenize",{"model":"qwen38-27b","prompt":s})["tokens"]
detok=lambda i: post("/detokenize",{"model":"qwen38-27b","tokens":i})["prompt"]
corpus=re.sub(r'[\x00-\x1f]',' ',open("/opt/llm/corpus.txt",errors="ignore").read())
def exact(ids,n):
    k=n
    for _ in range(14):
        s=detok(ids[:k]); g=tok(s)
        if len(g)==n and g[:n]==ids[:n]: return s
        k+=n-len(g)
    return None
send=lambda p:(lambda h0: (post("/v1/completions",{"model":"qwen38-27b","prompt":p,"max_tokens":1,"temperature":0}), int(hits()-h0))[1])(hits())

A=tok(corpus[1000000:1000000+60000]); B=tok(corpus[4600000:4600000+60000])
p1=exact(A,6400); assert p1
print("  --- 3x derselbe Prompt ---")
for i in (1,2,3): print(f"  send {i}: hits +{send(p1)}")

print("  --- Trennungstest: Eltern EINMAL, dann Geschwister B (5000 geteilt) ---")
A2=tok(corpus[300000:300000+60000])
p2=exact(A2,6400); assert p2
print(f"  parent (1x): hits +{send(p2)}")
sib=detok(A2[:5000]+B[:1400]); g=tok(sib)
print(f"  sibling B ({len(g)} Token, Praefix ok={g[:5000]==A2[:5000]}): hits +{send(sib)}")
