"""Diverged-sibling probe: does a partially shared prefix hit?

Sends a parent prompt, then a sibling that shares exactly N leading tokens and
diverges after. kamb-code's predictions for a 6400-token parent, block 1600:
  shared 3400 -> base 0, PR #53479 1600
  shared 5000 -> base 0, PR #53479 3200
"""
import json, os, re, sys, time, urllib.request
KEY = os.environ["LLM_KEY"]; BASE = "http://127.0.0.1:8080"
def post(p, payload, timeout=300):
    rq = urllib.request.Request(BASE+p, data=json.dumps(payload).encode(), method="POST",
        headers={"Content-Type":"application/json","Authorization":f"Bearer {KEY}"})
    with urllib.request.urlopen(rq, timeout=timeout) as r: return json.loads(r.read())
def metrics():
    rq = urllib.request.Request(BASE+"/metrics", headers={"Authorization":f"Bearer {KEY}"})
    with urllib.request.urlopen(rq, timeout=20) as r: t = r.read().decode()
    g = lambda x: float(re.search(rf'^vllm:{x}\{{[^}}]*}} ([0-9.e+]+)', t, re.M).group(1))
    return g('prefix_cache_queries_total'), g('prefix_cache_hits_total')
tok  = lambda s: post("/tokenize",{"model":"qwen38-27b","prompt":s})["tokens"]
detok= lambda i: post("/detokenize",{"model":"qwen38-27b","tokens":i})["prompt"]

corpus = re.sub(r'[\x00-\x1f]',' ', open("/opt/llm/corpus.txt", errors="ignore").read())
# Each pair gets its OWN parent material. Sharing one parent across pairs lets the
# first sibling cache blocks the second one then hits, which fakes a base-line hit.
REGIONS = {3400: 300000, 5000: 2400000}
B = tok(corpus[4600000:4600000+60000])       # divergent tail material

def exact(ids, n):
    """A prompt whose re-tokenization is exactly ids[:n]."""
    k = n
    for _ in range(14):
        s = detok(ids[:k]); got = tok(s)
        if len(got) == n and got[:n] == ids[:n]: return s
        k += n - len(got)
    return None

label, out = sys.argv[1], sys.argv[2]
PARENT_N = 6400
res = []
for shared in (3400, 5000):
    off = REGIONS[shared]
    A = tok(corpus[off:off+60000])
    parent = exact(A, PARENT_N)
    assert parent, f"parent calibration failed for shared={shared}"
    # sibling: shared leading tokens from this pair's parent, then a divergent tail
    sib_ids = A[:shared] + B[:PARENT_N - shared]
    sib = detok(sib_ids); got = tok(sib)
    ok_shared = got[:shared] == A[:shared]
    # parent first (populate), then the sibling
    for _ in range(3):
        post("/v1/completions", {"model":"qwen38-27b","prompt":parent,"max_tokens":1,"temperature":0})
    q0,h0 = metrics(); t = time.perf_counter()
    post("/v1/completions", {"model":"qwen38-27b","prompt":sib,"max_tokens":1,"temperature":0})
    dt = time.perf_counter()-t; q1,h1 = metrics()
    res.append({"shared":shared,"sibling_tokens":len(got),"prefix_verified":ok_shared,
                "queries":int(q1-q0),"hits":int(h1-h0),"s":round(dt,2)})
    print(f"  {label} | shared {shared}: hits={int(h1-h0)}  queries={int(q1-q0)}  "
          f"{dt:.2f}s  prefix_ok={ok_shared}")
json.dump({"label":label,"parent":PARENT_N,"res":res}, open(out,"w"), indent=1)
