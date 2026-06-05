# 8. Performance tuning

[← Public access](07-public-access-cloudflare.md) · [Index](README.md) · [Next: Monitoring →](09-monitoring.md)

This page replaces the minimal unit from page 6 with a **tuned** one. Every flag is justified by measurements taken on a live GB10 box; the benchmark tables at the end show what each change does.

## 8.1 The tuned systemd unit

```ini
[Unit]
Description=llama-server (MODEL_NAME)
After=network-online.target
Wants=network-online.target

[Service]
User=SERVICE_USER
Group=SERVICE_USER
EnvironmentFile=/etc/llama-server/MODEL_NAME.env

ExecStart=/opt/llm/runtime/llama.cpp/build/bin/llama-server \
  --model ${MODEL_PATH} \
  --host ${LLAMA_HOST} \
  --port ${LLAMA_PORT} \
  --ctx-size 131072 \
  --n-gpu-layers 999 \
  --cache-ram 65536 \
  --cache-reuse 256 \
  --ctx-checkpoints 48 \
  --flash-attn on \
  --mlock \
  --kv-unified \
  --batch-size 2048 \
  --ubatch-size 2048 \
  --parallel 4 \
  --threads 8 \
  --cont-batching \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --temp 0.6 \
  --top-p 0.95 \
  --top-k 20 \
  --min-p 0.0 \
  --repeat-penalty 1.05 \
  --api-key-file /etc/llama-server/api_keys.txt \
  --metrics

Restart=always
RestartSec=5
LimitNOFILE=1048576
LimitMEMLOCK=infinity

NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=full
ProtectHome=true
ReadWritePaths=/opt/llm /tmp

[Install]
WantedBy=multi-user.target
```

Apply it:

```bash
sudo cp -p /etc/systemd/system/MODEL_NAME.service /etc/systemd/system/MODEL_NAME.service.bak
sudo nano /etc/systemd/system/MODEL_NAME.service       # paste the new unit
sudo systemctl daemon-reload
sudo systemctl restart MODEL_NAME
sudo journalctl -u MODEL_NAME -f
```

## 8.2 What every new flag does, and why

| Flag | Reason |
|---|---|
| `--mlock` | Pins the ~50 GB model in RAM so the kernel can't evict pages under memory pressure. On unified-memory boxes (CPU and GPU share LPDDR5X) eviction directly slows GPU access, looking like a "service fell back to CPU" stall. |
| `LimitMEMLOCK=infinity` | Required for `--mlock` to actually work. The default rlimit is 8 MiB; without raising it, mlock silently fails on a ~667 MB buffer and the warning hides in the journal. |
| `--kv-unified` | Slots share one KV pool. Re-enables `--cache-idle-slots` (otherwise auto-disabled at startup). With 4 parallel slots, each can address the full 131 K context instead of getting 1/4 of it. |
| `--batch-size 2048` `--ubatch-size 2048` | Larger physical batches for prefill. Default ubatch is 512. On bandwidth-bound GB10 this typically improves long-prefill throughput. |
| `--parallel 4` | Serve 4 concurrent requests without queueing. With `--kv-unified`, no extra memory cost vs `--parallel 2`. |
| `--threads 8` | All compute is on the GPU. Cutting CPU threads from auto-20 reduces scheduler noise. |
| `--cache-ram 65536` | 64 GiB prompt-cache pool, up from 8 GiB default. Bigger pool = more conversations / sessions cached for reuse. |
| `--cache-reuse 256` | **Off by default — the underrated big lever.** Enables KV-shift reuse of cached chunks ≥ 256 tokens, so a shared system prompt across many users skips re-prefill on every call. |
| `--ctx-checkpoints 48` | More per-slot rollback points (was 32). Helps with conversational branching / edit. |
| `--metrics` | Exposes the Prometheus `/metrics` endpoint used by the monitoring stack in [page 9](09-monitoring.md). |

## 8.3 Measured baselines (single-stream, 1197 → 2048 tokens)

| Run | Wall | Prefill | Generation | GPU util |
|---|---|---|---|---|
| Baseline (default flags + `--parallel 2`) | 40.5 s | 1012 t/s | **50.82 t/s** | 88–94 % |
| After `--mlock` + `--kv-unified` + `--ubatch-size 2048` | 41.4 s | 1020 t/s | 50.93 t/s | 93 % steady |
| After `--parallel 4` | 42.0 s | (n/a) | ~49 t/s | (n/a) |

**Single-stream is flat across all configs** because we're already at the GPU's natural limit (~50 t/s generation, ~1000 t/s prefill). The optimizations don't help in this scenario — and don't hurt. They pay off under concurrent load.

## 8.4 Measured concurrent throughput (4 users × 1168 → 1024 tokens)

| Config | Wall | Per-user (active) | Combined | Queueing |
|---|---|---|---|---|
| `--parallel 2` + opts | 56.8 s | 35 t/s | 72 t/s | 2 users wait 29 s |
| `--parallel 4` + opts | **44.6 s** | 23 t/s steady | **91.9 t/s** | none |

`--parallel 4` is a strict win for multi-user: 21 % less wall time, 28 % more combined throughput, no queueing.

## 8.5 Measured cache-reuse (4 concurrent, shared ~1000-token system prompt)

| Round | Cache state | Tokens server actually prefilled per user | Server prefill time | Wall |
|---|---|---|---|---|
| 1 (cold) | empty | 1053–1063 | 1.8–3.2 s | 33.8 s |
| 2 (warm) | hot | **30–34** ← skipped 1017 cached tokens | **0.30 s** | 33.3 s |

The server prefilled **33× fewer tokens** the second time, dropping prefill time **10×**. Total wall barely moved (2 %) only because generation (~33 s) dominated that particular workload. Cache-reuse gains scale with `prefill_tokens / total_tokens` — wins are largest for short completions and long shared contexts.

## 8.6 Things investigated and NOT adopted

| Option | Why not |
|---|---|
| **External draft model / speculative decoding** | llama.cpp returns `"speculative decoding not supported by this context"` for the `qwen3next` arch. Even on standard `qwen3moe` siblings, benchmarks find net slowdown at batch=1 due to MoE expert-routing union overhead. |
| **MTP (Multi-Token Prediction)** | Unsloth's GGUF doesn't include the MTP head tensors, and llama.cpp's MTP code path is wired for `qwen3moe`, not `qwen3next`. Watch for an MTP-variant GGUF or upstream support. |
| **`--cache-type-k/v q4_0`** | Breaks processing on Qwen3-Coder-Next (model gives degenerate output). Stays at `q8_0`. |
| **Lowering `--ctx-size` from 131072** | Use-case decision. If most prompts stay under 32 K, lowering this frees memory for more prompt-cache pool. |
| **NVFP4 quant** | Would leverage `BLACKWELL_NATIVE_FP4=1`. Requires a Qwen3-Coder-Next NVFP4 GGUF (not yet published as of writing). |

## 8.7 If you change the model

Most flags transfer cleanly to other GGUFs. Two caveats:

- **`--cache-reuse 256`** assumes prompts share a non-trivial prefix. Harmless for other workloads but the gain depends on usage patterns.
- **`--cache-type-k/v q8_0`** is conservative. Some models tolerate `q4_0` KV without quality loss (saves memory + bandwidth); test before committing.

---

[← Public access](07-public-access-cloudflare.md) · [Index](README.md) · [Next: Monitoring →](09-monitoring.md)
