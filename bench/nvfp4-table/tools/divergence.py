#!/usr/bin/env python3
"""Teacher-forced logprob divergence against BF16 over LONG autoregressive runs.

Why teacher forcing: if each model generates freely, the texts diverge after a few tokens
and any later difference measures drift, not quantization. Instead BF16 generates one long
continuation and every candidate is scored on THAT exact token sequence.

Why token IDs rather than text: the same string can retokenize differently. The prompt is
tokenized once, the continuation once, and all candidates score the identical id list.

Two numbers per candidate from a single pass -- vLLM's prompt_logprobs returns both the
forced token's logprob and its rank:
  * NLL   -- mean negative logprob; the delta vs BF16 is the log perplexity ratio on real
             agent text
  * top1  -- how often the candidate would itself have picked BF16's token. A single wrong
             token derails a tool call, so this is the agentic-relevant number.
Both are also reported per quarter of the run, because the point of a long continuation is
to see whether the gap accumulates with depth.
"""
import argparse, json, os, statistics, sys, time, urllib.request

AP = argparse.ArgumentParser()
AP.add_argument("--mode", choices=["ref", "score"], required=True)
AP.add_argument("--label", required=True)
AP.add_argument("--prompts", default="/opt/llm/agent-prompts.json")
AP.add_argument("--ref", default="/opt/llm/divergence-ref.json")
AP.add_argument("--out", default=None)
AP.add_argument("--gen-tokens", type=int, default=768)
AP.add_argument("--max-ctx", type=int, default=32768)
AP.add_argument("--url", default="http://127.0.0.1:8080")
AP.add_argument("--model", default="qwen38-27b")
A = AP.parse_args()
H = {"Content-Type": "application/json",
     "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY','')}"}


def post(path, body, timeout=3600):
    req = urllib.request.Request(A.url + path, data=json.dumps(body).encode(), headers=H)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def tokenize(**kw):
    return post("/tokenize", {"model": A.model, **kw}, timeout=180)


def quarters(xs, fn):
    n = len(xs)
    return [fn(xs[i * n // 4:(i + 1) * n // 4] or [0]) for i in range(4)]


if A.mode == "ref":
    ref = []
    for p in json.load(open(A.prompts)):
        pid = tokenize(messages=p["messages"], add_generation_prompt=True)["tokens"]
        if len(pid) + A.gen_tokens > A.max_ctx:
            print(f"  {p['id'][:40]:<42} uebersprungen ({len(pid)} Prompt-Token)")
            continue
        t0 = time.time()
        # ignore_eos: without it greedy stops after a median of 144 tokens -- too short
        # to see whether the gap widens with depth. Text past the natural end is still
        # generated autoregressively, and that is where the divergences pile up.
        d = post("/v1/completions", {"model": A.model, "prompt": pid,
                                     "max_tokens": A.gen_tokens, "temperature": 0,
                                     "ignore_eos": True})
        cid = tokenize(prompt=d["choices"][0]["text"])["tokens"]
        if not cid:
            print(f"  {p['id'][:40]:<42} LEERE Fortsetzung — uebersprungen")
            continue
        ref.append({"id": p["id"], "prompt_ids": pid, "cont_ids": cid})
        print(f"  {p['id'][:40]:<42} Prompt {len(pid):>6}  Fortsetzung {len(cid):>4} Tok"
              f"  {time.time()-t0:5.1f}s")
    json.dump(ref, open(A.ref, "w"))
    print(f"\n  {len(ref)} Referenzlaeufe -> {A.ref}")
    sys.exit(0)

ref = json.load(open(A.ref))
out = A.out or f"/tmp/divergence-{A.label}.json"
rows = []
for r in ref:
    ids = r["prompt_ids"] + r["cont_ids"]
    n_p = len(r["prompt_ids"])
    d = post("/v1/completions", {"model": A.model, "prompt": ids, "max_tokens": 1,
                                 "temperature": 0, "prompt_logprobs": 1})
    pl = d["choices"][0]["prompt_logprobs"]

    def collect(lo, hi):
        lps, ranks = [], []
        for pos in range(lo, hi):
            e = pl[pos] or {}
            k = str(ids[pos])
            if k not in e:                 # the forced token is always included; guard anyway
                continue
            lps.append(e[k]["logprob"])
            ranks.append(e[k].get("rank", 99))
        return lps, ranks

    # The CONTEXT is real agent text (8k-30k tokens) and comes out of the same call --
    # a huge sample, but input text. The CONTINUATION is what the model itself
    # erzeugt, also verhaltensrelevant. Beide getrennt, weil sie verschiedene Fragen
    # beantworten.
    clp, crk = collect(1, n_p)
    glp, grk = collect(n_p, len(ids))
    if not glp:
        print(f"  {r['id'][:40]:<42} keine Logprobs — uebersprungen")
        continue
    rows.append({"id": r["id"], "n": len(glp), "n_ctx": len(clp),
                 "ctx_nll": -statistics.fmean(clp) if clp else None,
                 "ctx_top1": (sum(1 for x in crk if x == 1) / len(crk)) if crk else None,
                 "nll": -statistics.fmean(glp),
                 "top1": sum(1 for x in grk if x == 1) / len(grk),
                 "nll_q": quarters(glp, lambda s: -statistics.fmean(s)),
                 "top1_q": quarters(grk, lambda s: sum(1 for x in s if x == 1) / len(s))})
    q = rows[-1]
    print(f"  {r['id'][:40]:<42} Kontext NLL {q['ctx_nll']:.4f}/top1 {q['ctx_top1']*100:5.1f}%"
          f"   Fortsetzung NLL {q['nll']:.4f}/top1 {q['top1']*100:5.1f}%")

json.dump(rows, open(out, "w"), indent=1)
if rows:
    print(f"\n{A.label}: n={len(rows)}  NLL {statistics.fmean(x['nll'] for x in rows):.4f}"
          f"  top1 {statistics.fmean(x['top1'] for x in rows)*100:.2f} %")
print(f"  -> {out}")
