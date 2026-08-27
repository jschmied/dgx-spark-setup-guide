#!/usr/bin/env python3
# Prefill(TTFT) vs decode throughput as a function of CONTEXT LENGTH.
# Key from the environment, never inline. Tolerates usage-only SSE chunks
# (--enable-force-include-usage sends a final chunk with choices == []).
import json, os, sys, time, urllib.request, statistics as st
BASE = "http://127.0.0.1:8080/v1/chat/completions"
KEY  = os.environ["OPENAI_API_KEY"]
MODEL = sys.argv[1] if len(sys.argv) > 1 else "qwen38-27b"
CTXS = [1024, 8192, 32768]; REPS = 2; MAXTOK = 200

def prompt(n, nonce=""):
    # deterministic filler, ~4 chars/token
    line = "The quick brown fox jumps over the lazy dog while the system logs each event. "
    return nonce + (line * (n // 16 + 8))[: n * 4] + "\n\nSummarise the text above in one sentence."

def run(ctx, nonce=""):
    body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": prompt(ctx, nonce)}],
                       "max_tokens": MAXTOK, "stream": True, "temperature": 0.6,
                       "chat_template_kwargs": {"reasoning_effort": "medium"}}).encode()
    req = urllib.request.Request(BASE, data=body,
          headers={"Authorization": "Bearer " + KEY, "Content-Type": "application/json"})
    t0 = time.time(); ttft = None; n = 0; toks = 0
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            if not raw.startswith(b"data: "): continue
            chunk = raw[6:].strip()
            if chunk == b"[DONE]": break
            try: j = json.loads(chunk)
            except Exception: continue
            u = j.get("usage")
            if u and u.get("completion_tokens"): toks = u["completion_tokens"]
            ch = j.get("choices") or []
            if not ch: continue                      # usage-only chunk
            d = ch[0].get("delta") or {}
            if not (d.get("content") or d.get("reasoning_content")): continue
            if ttft is None: ttft = time.time() - t0
            n += 1
    total = time.time() - t0
    real = toks or n   # usage tokens when available -- one SSE delta can carry several
    dec = real / (total - ttft) if ttft and total > ttft else float('nan')
    return ttft, dec, real

print(f"=== {MODEL} ===")
print(f"{'ctx~tokens':>11} | {'TTFT(s)':>8} | {'decode(tok/s)':>13} | {'tokens':>6}")
for c in CTXS:
    rs = [run(c, f"[run {c}/{i}/{os.getpid()}] ") for i in range(REPS)]
    print(f"{c:>11} | {st.median(x[0] for x in rs):8.2f} | {st.median(x[1] for x in rs):13.1f} | {rs[-1][2]:>6}")
