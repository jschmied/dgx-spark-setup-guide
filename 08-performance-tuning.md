# 8. Performance tuning

[← Public access](07-public-access-cloudflare.md) · [Index](README.md) · [Next: Monitoring →](09-monitoring.md)

This page replaces the minimal `[*]` section of the preset from page 6 with a **tuned** one, and makes one change to the unit. Every flag is justified by measurements taken on a live GB10 box; the benchmark tables at the end show what each change does.

The router (the unit) doesn't change shape — the tuning lives in `/etc/llama-server/models.ini`, so every model the router spawns inherits it.

## 8.1 The tuned preset

Replace the `[*]` section in `/etc/llama-server/models.ini` with the full set below. Leave the per-model sections (`[qwen3-coder-next]`, `[qwen36-35b-a3b]`, …) as they are.

```ini
; Tuned shared defaults — applied to every model the router spawns.
[*]
ctx-size       = 131072
n-gpu-layers   = 999
cache-ram      = 65536
cache-reuse    = 256
ctx-checkpoints = 48
flash-attn     = on
mlock          = true
kv-unified     = true
batch-size     = 2048
ubatch-size    = 2048
parallel       = 4
threads        = 8
cont-batching  = true
cache-type-k   = q8_0
cache-type-v   = q8_0
temp           = 0.6
top-p          = 0.95
top-k          = 20
min-p          = 0.0
repeat-penalty = 1.05
metrics        = true
```

## 8.2 The one unit change: `LimitMEMLOCK`

`--mlock` only works if the unit raises the memlock rlimit. Add this line to the `[Service]` block of `/etc/systemd/system/llama-router.service` (the default rlimit is 8 MiB; without raising it `mlock` silently fails):

```ini
LimitMEMLOCK=infinity
```

Apply both changes:

```bash
sudo cp -p /etc/llama-server/models.ini /etc/llama-server/models.ini.bak
sudo cp -p /etc/systemd/system/llama-router.service /etc/systemd/system/llama-router.service.bak
sudo nano /etc/llama-server/models.ini                 # paste the tuned [*] section
sudo nano /etc/systemd/system/llama-router.service     # add LimitMEMLOCK=infinity
sudo systemctl daemon-reload
sudo systemctl restart llama-router
sudo journalctl -u llama-router -f
```

> With `--models-max 1` only one model is resident at a time, so `--mlock` pins just the active model (~50 GB for the Q4 coder) — well within 128 GB. On a swap the previous model is unmapped and its lock released before the next one loads.

## 8.3 What every new flag does, and why

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
| `metrics = true` | Each model instance exposes the Prometheus `/metrics` endpoint, which the router proxies per-model (`/metrics?model=<id>`). Used by the monitoring stack in [page 9](09-monitoring.md). Set in the preset, not the unit. |

## 8.4 Measured baselines (single-stream, 1197 → 2048 tokens)

| Run | Wall | Prefill | Generation | GPU util |
|---|---|---|---|---|
| Baseline (default flags + `--parallel 2`) | 40.5 s | 1012 t/s | **50.82 t/s** | 88–94 % |
| After `--mlock` + `--kv-unified` + `--ubatch-size 2048` | 41.4 s | 1020 t/s | 50.93 t/s | 93 % steady |
| After `--parallel 4` | 42.0 s | (n/a) | ~49 t/s | (n/a) |

**Single-stream is flat across all configs** because we're already at the GPU's natural limit (~50 t/s generation, ~1000 t/s prefill). The optimizations don't help in this scenario — and don't hurt. They pay off under concurrent load.

## 8.5 Measured concurrent throughput (4 users × 1168 → 1024 tokens)

| Config | Wall | Per-user (active) | Combined | Queueing |
|---|---|---|---|---|
| `--parallel 2` + opts | 56.8 s | 35 t/s | 72 t/s | 2 users wait 29 s |
| `--parallel 4` + opts | **44.6 s** | 23 t/s steady | **91.9 t/s** | none |

`--parallel 4` is a strict win for multi-user: 21 % less wall time, 28 % more combined throughput, no queueing.

## 8.6 Measured cache-reuse (4 concurrent, shared ~1000-token system prompt)

| Round | Cache state | Tokens server actually prefilled per user | Server prefill time | Wall |
|---|---|---|---|---|
| 1 (cold) | empty | 1053–1063 | 1.8–3.2 s | 33.8 s |
| 2 (warm) | hot | **30–34** ← skipped 1017 cached tokens | **0.30 s** | 33.3 s |

The server prefilled **33× fewer tokens** the second time, dropping prefill time **10×**. Total wall barely moved (2 %) only because generation (~33 s) dominated that particular workload. Cache-reuse gains scale with `prefill_tokens / total_tokens` — wins are largest for short completions and long shared contexts.

## 8.7 Things investigated and NOT adopted

| Option | Why not |
|---|---|
| **External draft model / speculative decoding** | llama.cpp returns `"speculative decoding not supported by this context"` for the `qwen3next` arch. Even on standard `qwen3moe` siblings, benchmarks find net slowdown at batch=1 due to MoE expert-routing union overhead. |
| **MTP (Multi-Token Prediction)** | Neither served GGUF ships the MTP head, so it's off on both. **Qwen3-Coder-Next** (`qwen3next`): no MTP tensors, and llama.cpp's MTP path is wired for `qwen3moe`. **Qwen3.6-35B-A3B** (`qwen35moe`): the GGUF's tensor table is only `blk.0`–`blk.39` with no `nextn.*`/`mtp.*` tensors and no `num_nextn_predict_layers` key, and the running instance has no `spec-type`/`model-draft` flags — confirmed plain single-token decode (~54.5 t/s, see 8.9). Enabling MTP would need both an MTP-head GGUF *and* `spec-type = draft-mtp` in that model's preset section (plus `qwen35moe` support in the build). Watch for an MTP-variant GGUF. |
| **`--cache-type-k/v q4_0`** | Breaks processing on Qwen3-Coder-Next (model gives degenerate output). Stays at `q8_0`. |
| **Lowering `--ctx-size` from 131072** | Use-case decision. If most prompts stay under 32 K, lowering this frees memory for more prompt-cache pool. |
| **NVFP4 quant** | Would leverage `BLACKWELL_NATIVE_FP4=1`. Requires a Qwen3-Coder-Next NVFP4 GGUF (not yet published as of writing). |

## 8.8 Per-model overrides

The tuned values live in `[*]`, so they apply to **every** model the router serves. When one model needs different settings, override the key in that model's own section — it wins over `[*]` (precedence: command line > model section > `[*]`). For example, the Qwen3.6 35B model wants its own sampling per the model card, a corrected chat template, and `preserve_thinking` pinned off:

```ini
[qwen36-35b-a3b]
model               = /opt/llm/models/qwen36-35b-a3b/Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf
chat-template-file  = /opt/llm/models/qwen36-35b-a3b/chat_template.jinja
chat-template-kwargs = {"preserve_thinking": false}
temp                = 0.6
top-p               = 0.95
top-k               = 20
min-p               = 0.0
presence-penalty    = 0.0
repeat-penalty      = 1.0
```

These sampler keys are server-side **defaults** for that model; a client can still override per-request. The only one that differs from the tuned `[*]` here is `repeat-penalty` (1.0 vs the global 1.05) — the rest are pinned for clarity so the model's behaviour doesn't drift if you later retune `[*]`.

Most flags transfer cleanly to other GGUFs. Four things worth pinning per-model rather than globally:

- **`cache-reuse = 256`** assumes prompts share a non-trivial prefix. Harmless for other workloads but the gain depends on usage patterns.
- **`cache-type-k/v = q8_0`** is conservative. Some models tolerate `q4_0` KV without quality loss (saves memory + bandwidth) — but on Qwen3-Coder-Next `q4_0` gives degenerate output (see 8.7), which is exactly why per-model overrides matter once you serve a mix.
- **`chat-template-file`** points at a Jinja template that overrides the one embedded in the GGUF. Some GGUFs ship a buggy template (wrong tool-call or thinking handling); a community-fixed template fixes it without re-quantizing. Download the `.jinja` next to the model and reference it with an absolute path. Jinja is on by default, so the file is accepted as-is; the override applies the next time that model loads. (Verify after a restart with `journalctl -u llama-router | grep chat-template-file` and a sample completion.)
- **`chat-template-kwargs`** passes a JSON object straight into the Jinja template's variables — pin it here rather than trusting the client to send it. The fixed Qwen3.6 template above exposes `preserve_thinking` (whether *previous* turns' `reasoning_content` is re-fed into the prompt; default `false`). Setting it `false` keeps the context lean across multi-turn chats and stops a client from flipping it on. The value is a single line of valid JSON — `{"preserve_thinking": false}` — and is passed to the child as one argument intact, spaces and all. (Don't confuse it with `enable_thinking`, which controls whether the model thinks *at all*; that's still `true`.)

> The benchmark tables in 8.4–8.6 were measured on **Qwen3-Coder-Next** (the 80B-A3B Q4 coder). The numbers for the second model live in 8.9.

## 8.9 Measured: Qwen3.6-35B-A3B (Q8)

Same box, same router, measured through the deployed per-model config from 8.8 (Q8_K_XL, `cache-type q8_0`, `parallel 4`, fixed chat template). Throughput was driven against the native `/completion` endpoint with `ignore_eos:true` so each run generates exactly `n_predict` tokens — reasoning vs. final-answer content doesn't affect decode speed, only token count, and `ignore_eos` removes that variability. Prompt = 1291 tokens. First (warm-up) run discarded.

### Single-stream (1291-token prompt)

| Metric | Value |
|---|---|
| Prefill | **~2080 t/s** (1291 tokens ≈ 0.62 s) |
| Generation | **~54.5 t/s**, flat from 256 → 1200 tokens |
| GPU utilization | 94–95 % steady |
| GPU power | ~32 W |

Both numbers are **higher than the 80B coder** (~1000 t/s prefill, ~50 t/s gen): same 3 B active parameters, but far fewer total weights to stream means roughly 2× prefill, and Q8 decode still clears the coder by a few t/s. GPU sits pinned at ~95 %, so this is the GB10's natural single-stream ceiling for this model — the tuning flags don't move it (same conclusion as 8.4), they pay off under load and on cache reuse below.

### Concurrent (4 streams × 1291 → 256 tokens)

| Metric | Value |
|---|---|
| Per-stream generation | ~24 t/s steady |
| Combined generation | **~96 t/s** (1.76× single-stream) |
| Wall-clock (4 × 256 = 1024 tokens incl. prefill) | 13.1 s → ~78 t/s |
| Queueing | none (4 slots, `--parallel 4`) |

`--parallel 4` is the same win here as for the coder: ~1.8× aggregate decode throughput with no queueing, at the cost of per-stream rate. Confirms the `[*]` tuning carries to this model unchanged.

### Cache reuse (repeated 1291-token prefix)

| Prefill path | Tokens actually prefilled | Prefill time |
|---|---|---|
| No cache (`cache_prompt:false`) | 1291 | ~0.62 s |
| Warm (`cache-reuse`, prefix already seen) | 4 (1287 reused) | **~0.036 s** |

A repeated prefix collapses prefill **~17×** (1287 of 1291 tokens served from cache). As in 8.6, the wall-clock win scales with `prefill_tokens / total_tokens` — largest for short completions over a long shared context.

**Bottom line:** Qwen3.6-35B-A3B runs comfortably on the same tuned `[*]` config; no model-specific performance flags were needed beyond the sampling/template overrides in 8.8. It is the faster of the two models on this box for both prefill and single-stream generation.

---

[← Public access](07-public-access-cloudflare.md) · [Index](README.md) · [Next: Monitoring →](09-monitoring.md)
