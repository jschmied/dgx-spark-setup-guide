#!/usr/bin/env python3
"""Async concurrency bench client for the qwen36 vLLM endpoint.

Fires REQUESTS chat-completions at fixed CONCURRENCY, streaming, each with a
unique ~INPUT_LEN-token prompt (busts prefix cache) and exactly OUTPUT_LEN
tokens (ignore_eos). Reports median single-request decode t/s, aggregate
decode t/s, and median TTFT. Sampling matches prod (temp 0.6/top_p .95/top_k 20).
"""
import argparse, asyncio, json, random, statistics, string, time
import urllib.request

def make_prompt(uid: int, approx_tokens: int) -> str:
    # ~0.75 tokens/word for english-ish filler; prepend unique id to defeat any caching
    words = approx_tokens  # 1 word ~ 1 token here (short random words push token count up a bit; fine)
    body = " ".join(
        "".join(random.choices(string.ascii_lowercase, k=random.randint(3, 8)))
        for _ in range(words)
    )
    return (f"[req {uid} {random.random()}] Summarize and then continue this text at length. {body}")

async def one_request(session_sem, url, key, model, uid, input_len, output_len, temp, results):
    async with session_sem:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": make_prompt(uid, input_len)}],
            "max_tokens": output_len,
            "temperature": temp, "top_p": 0.95, "top_k": 20, "min_p": 0,
            "stream": True,
            "stream_options": {"include_usage": True},
            "ignore_eos": True,
        }
        data = json.dumps(payload).encode()
        req = urllib.request.Request(url, data=data, method="POST",
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
        t0 = time.perf_counter()
        t_first = None
        t_last = None
        completion_tokens = None
        # blocking urllib in a thread to keep it dependency-free
        def do():
            nonlocal t_first, t_last, completion_tokens
            with urllib.request.urlopen(req, timeout=600) as resp:
                for raw in resp:
                    line = raw.decode("utf-8", "ignore").strip()
                    if not line.startswith("data:"):
                        continue
                    chunk = line[5:].strip()
                    if chunk == "[DONE]":
                        break
                    try:
                        obj = json.loads(chunk)
                    except Exception:
                        continue
                    now = time.perf_counter()
                    ch = obj.get("choices") or []
                    if ch and ch[0].get("delta", {}).get("content"):
                        if t_first is None:
                            t_first = now
                        t_last = now
                    if obj.get("usage"):
                        completion_tokens = obj["usage"].get("completion_tokens")
        await asyncio.to_thread(do)
        if t_first is None or completion_tokens is None or completion_tokens < 2:
            return
        ttft = t_first - t0
        window = max(t_last - t_first, 1e-6)
        decode_tps = (completion_tokens - 1) / window
        results.append({"ttft": ttft, "decode_tps": decode_tps,
                        "tokens": completion_tokens, "t0": t0, "t_last": t_last})

async def run(args):
    sem = asyncio.Semaphore(args.concurrency)
    results = []
    # warmup (discarded)
    await one_request(sem, args.url, args.key, args.model, 999999,
                      args.input_len, args.output_len, args.temp, [])
    wall0 = time.perf_counter()
    tasks = [one_request(sem, args.url, args.key, args.model, i,
                         args.input_len, args.output_len, args.temp, results)
             for i in range(args.requests)]
    await asyncio.gather(*tasks)
    wall = time.perf_counter() - wall0
    if not results:
        print(json.dumps({"error": "no results"})); return
    dtps = sorted(r["decode_tps"] for r in results)
    ttfts = sorted(r["ttft"] for r in results)
    total_out = sum(r["tokens"] for r in results)
    agg = total_out / wall
    print(json.dumps({
        "concurrency": args.concurrency, "n": len(results),
        "decode_tps_median": round(statistics.median(dtps), 1),
        "decode_tps_p10": round(dtps[max(0, len(dtps)//10)], 1),
        "aggregate_tps": round(agg, 1),
        "ttft_median_s": round(statistics.median(ttfts), 3),
        "wall_s": round(wall, 1), "total_out_tokens": total_out,
    }))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8080/v1/chat/completions")
    p.add_argument("--key", default="sk-bench")
    p.add_argument("--model", default="qwen36-35b-a3b")
    p.add_argument("--concurrency", type=int, required=True)
    p.add_argument("--requests", type=int, required=True)
    p.add_argument("--input-len", type=int, default=4000)
    p.add_argument("--output-len", type=int, default=512)
    p.add_argument("--temp", type=float, default=0.6)
    asyncio.run(run(p.parse_args()))
