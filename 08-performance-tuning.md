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
| **MTP (Multi-Token Prediction)** | **Adopted on three of five models**: the two Ornstein merges (embedded head — see 8.8/8.10/8.11) and Gemma 4 (separate assistant draft — see 8.8/8.12). The other two (qwen3-coder-next, qwen36-35b-a3b) have no MTP path. **Qwen3-Coder-Next** (`qwen3next`): no MTP tensors, and llama.cpp's MTP path isn't wired for the arch. **Qwen3.6-35B-A3B** (`qwen35moe`, the *non*-Ornstein one): tensor table is only `blk.0`–`blk.39` with no `nextn.*`/`mtp.*` tensors — confirmed plain single-token decode (~54.5 t/s, see 8.9). **Ornstein 27B** (`qwen35`, dense) and **Ornstein 35B-A3B** (`qwen35moe`, MoE): both ship an embedded head (`nextn_predict_layers = 1`, `blk.*.nextn.*`), supported since build **b9502 (6ddc9430b)** — enabled with `spec-type = draft-mtp` (lossless). **Gemma 4 26B-A4B** (`gemma4`, MoE): no embedded head, but ships a small separate **assistant** GGUF used as the draft — enabled with `spec-type = draft-mtp` + `spec-draft-model = …assistant-F16.gguf`, supported since build **b9571** (PRs #23398/#24282). Gain scales with weight-streaming per token: ~2× on the dense 27B (9 → ~19 t/s, 8.10), ~+30 % on the 3 B-active Ornstein MoE (~57 → ~75 t/s, 8.11), and only ~+12 % avg / up to +32 % on code (−5 % on prose) for the already-cheap 4 B-active Gemma (8.12). Must be set explicitly: without it the server logs `common_speculative_init: no implementations specified for speculative decoding` and the draft sits unused. (The Ornstein archs also log `fused Gated Delta Net (chunked) not supported`, which caps *prefill*, not decode.) |
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

A third model, the Qwen3.6-27B "Ornstein" merge (`ornstein36-27B`), shows two more per-model patterns: the **other** template pattern, and **MTP self-speculation**. Its embedded chat template is correct, so instead of an external `chat-template-file` you just turn Jinja on for it. It is also a *dense* model (slow to decode — see 8.10), so it's the one dense model that benefits from its built-in MTP head as a draft. Note its sampling: this merge **must run at `temp = 1.0`** (its embedded recommendation) — at `temp 0.6` (and 0.2) it falls into reasoning-repetition loops and emits nothing (see page 13). Add the two `spec-*` keys for MTP:

```ini
[ornstein36-27B]
model            = /opt/llm/models/ornstein36-27B/Qwen3.6-27B-MTP-NSC-ACE-SABER-Ornstein-Q6_K.gguf
jinja            = true
spec-type        = draft-mtp   ; use the model's own MTP head as its draft
spec-draft-n-max = 3           ; draft depth; 3 is the sweet spot here (see 8.10)
temp             = 1.0         ; embedded rec; lower temps make this merge loop
top-p            = 0.95
top-k            = 20
min-p            = 0.0
presence-penalty = 0.0
repeat-penalty   = 1.0
```

`spec-type = draft-mtp` tells the child to build a speculative draft context from the model's **own** embedded MTP head (`nextn_predict_layers`) — no separate draft model, ~160 MiB extra. The head proposes up to `spec-draft-n-max` tokens per step and the full model verifies them in one pass; accepted tokens are free. It is **lossless** (the verifier preserves the model's output distribution). On this box it roughly **doubles** decode (9 → ~19 t/s); details and the depth sweep are in 8.10. Confirm it's active after a restart with `journalctl -u llama-router | grep draft-mtp` (look for `adding speculative implementation 'draft-mtp'`).

This Q6_K GGUF ships an MTP head (`nextn_predict_layers = 1`, tensors `blk.*.nextn.*`) — unlike the two models in 8.7 — and the `spec-type = draft-mtp` keys above turn it on, roughly doubling decode. (Without those keys the server logs `common_speculative_init: no implementations specified for speculative decoding` and the head sits unused — it must be enabled explicitly.) The model is also dense, so even with MTP it's slower than the MoE models; see 8.10 for the measurements, the MTP depth sweep, and the bandwidth analysis, plus the MTP row in 8.7.

A fifth model, **Gemma 4 26B-A4B** (`gemma-4-26B-A4B`), shows a different-vendor pattern: embedded template (`jinja = true`), its **own** sampling rather than the Qwen `[*]` defaults, and a **third MTP variant** — a *separate* assistant draft model rather than an embedded head (see below). Gemma needs `temp ≈ 1.0` — at low temperature it loops (8.12) — so pin its sampling per-model:

```ini
[gemma-4-26B-A4B]
model            = /opt/llm/models/gemma-4-26B-A4B/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf
spec-type        = draft-mtp
spec-draft-model = /opt/llm/models/gemma-4-26B-A4B/gemma-4-26B-A4B-it-assistant-F16.gguf
spec-draft-n-max = 3
jinja            = true
temp             = 1.0
top-k            = 64
top-p            = 0.95
min-p            = 0.0
repeat-penalty   = 1.0
```

Gemma 4's MTP is **not** an embedded head like the Ornstein merges — it ships a small separate **assistant** GGUF (`gemma-4-26B-A4B-it-assistant-F16.gguf`, ~816 MB) that serves as the draft, so you point `spec-draft-model` at it rather than relying on `nextn.*` tensors inside the main GGUF. This needs build **b9571+** (Gemma 4 MTP landed after b9502 — PRs #23398/#24282). The win is real but smaller and content-dependent than the dense Ornstein's: ~+12 % average decode on code (up to +32 % when drafts accept well), ~−5 % on free-form prose — see 8.12. It is the fastest model on the box either way (~74–85 t/s baseline).

**Don't let a model silently inherit the wrong sampling.** The `[*]` defaults (`temp 0.6`, etc.) are tuned for the Qwen3.6 *reasoning* models. `qwen3-coder-next` is a non-thinking coder and wants the Qwen3-Coder card values instead — pin them rather than inheriting `[*]`:

```ini
[qwen3-coder-next]
model           = /opt/llm/models/qwen3-coder-next/Qwen3-Coder-Next-UD-Q4_K_XL.gguf
temp            = 0.7
top-p           = 0.8          ; tighter than the reasoning models' 0.95
top-k           = 20
repeat-penalty  = 1.05
```

A quick way to find the author-intended values for any GGUF is the embedded `general.sampling.*` metadata (e.g. `temp`, `top_k`, `top_p`) — read it before deciding what to override. That audit is what set Gemma to `temp 1.0` and the dense Ornstein to `temp 1.0` (it loops lower), and gave the coder model the values above.

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

**Bottom line:** Qwen3.6-35B-A3B runs comfortably on the same tuned `[*]` config; no model-specific performance flags were needed beyond the sampling/template overrides in 8.8. It is the faster of the two MoE/coder models on this box for both prefill and single-stream generation.

## 8.10 Measured: Qwen3.6-27B Ornstein (Q6, **dense**) — MTP doubles a bandwidth-bound model

This is a **dense** model: out of the box it decodes at **~9 t/s**, ~6× slower than the 35B MoE. Turning on its built-in MTP head (`spec-type = draft-mtp`, 8.8) **roughly doubles that to ~19–21 t/s**. Same box, same router, measured via `/completion` with `ignore_eos:true`, warm-up discarded.

| Metric | Ornstein 27B — no MTP | Ornstein 27B — **MTP `n-max=3`** | Qwen3.6-35B-A3B (Q8, MoE) |
|---|---|---|---|
| Generation | ~9.1 t/s | **~19–21 t/s** (2.0–2.3×) | ~54.5 t/s |
| Prefill (1801-token prompt) | ~690 t/s | ~690 t/s (unchanged) | ~2080 t/s |
| GPU during decode | 96 % util, ~38 W | 96 % util | 95 % util, ~32 W |

**Why the *baseline* is ~9 t/s — memory bandwidth, not compute.** The GGUF's FFN tensors are `blk.*.ffn_{gate,up,down}` with **no `_exps`** — it is **dense**, so every token streams the whole ~22.4 GB of weights from memory. The 35B is MoE (`*.ffn_*_exps`) and activates only ~3 B params (~3–4 GB) per token. On the GB10's ~250–273 GB/s unified memory the hard ceiling for a 22.4 GB dense model is `BW / size` ≈ **11–12 t/s**; we measure 9.1 (~80 % of ceiling, normal). The 96 % util at only ~38 W is the giveaway: the GPU is **stalled on memory reads, not doing FLOPs** (a compute-bound run would pull 80–140 W). No `[*]` flag moves this wall — `parallel`, `flash-attn`, `mlock`, `ubatch` etc. don't reduce bytes-streamed-per-token.

**Why MTP breaks the wall.** MTP self-speculation drafts several tokens from the model's own next-token head, then verifies them in **one** forward pass — so one weight-stream from memory can commit multiple tokens instead of one. That directly attacks the bandwidth limit. On build **b9502 (6ddc9430b)** it is supported and enabled with `spec-type = draft-mtp` (no separate draft model, ~160 MiB extra context). At load you'll see `adding speculative implementation 'draft-mtp'` and `speculative decoding context initialized`; during generation the child logs `draft-mtp` acceptance statistics (~57 % of drafted tokens accepted on code-like prompts here). It is **lossless** — the verifier preserves the model's output distribution, so quality is identical to plain decoding.

**Draft-depth sweep (`spec-draft-n-max`), single stream:**

| `n-max` | Decode | Note |
|---|---|---|
| off | 9.1 t/s | dense baseline |
| 2 | 17.2 t/s | |
| **3** | **21.3 t/s** | **best — the default** |
| 4 | 18.7 t/s | acceptance drops, wasted draft compute |
| 5 | 19.9 t/s | |

`n-max=3` wins: deeper drafts propose more tokens but get rejected more often, and the rejected ones are pure overhead. (Through the live router with the full `ctx-size 262144` config the same setting lands at ~18.6 t/s vs the 21.3 measured on a lean scratch context — same ~2× win.)

**Prefill stays ~690 t/s** (MTP only helps decode). It's below the MoE's ~2080 partly from dense compute and partly the hybrid attention: arch `qwen35` is a Gated Delta Net / SSM hybrid (`qwen35.ssm.*`, full attention only every 4th layer — `full_attention_interval = 4`), and on b9502 the server logs `fused Gated Delta Net (chunked) not supported, set to disabled`, so prefill takes a slower scan. A future build with that fused kernel should lift prefill; watch for it.

**Other lever — smaller quant.** Decode scales with bytes/token, so a Q4_K_M (~15 GB) would raise even the *baseline* ceiling to ~16–18 t/s, and MTP would stack on top — at some quality cost and a re-download. Q6 + MTP is the better default unless you specifically need it faster.

**Bottom line:** Ornstein is dense and memory-bound, but `spec-type = draft-mtp` (n-max 3) gives a free, lossless ~2× to ~19–21 t/s — very usable. The MoE models are still faster by construction; reach for Ornstein when you want *its* behaviour.

## 8.11 Measured: Qwen3.6-35B-A3B Ornstein (Q8, **MoE**) — MTP on a model that's already fast

The MoE sibling (`ornstein36-35b-a3b`, arch `qwen35moe`, 256 experts / 8 active ≈ 3 B/token, Q8_0, with an MTP head) is a different story from the dense 27B: it's **already the fastest model on the box** before any speculation, and MTP adds a smaller — but real — gain. Same harness as 8.9/8.10.

| Config | Decode | Prefill (1801 tok) |
|---|---|---|
| baseline (no MTP) | ~57 t/s (very stable) | ~2100 t/s |
| **MTP `n-max=3`** | **~75 t/s** (range 64–90, content-dependent) | ~1900 t/s |
| MTP `n-max=2` | ~72 t/s (range 70–77) | ~1970 t/s |

**Why the MTP gain is smaller here (~+30 %) than on the dense 27B (~2×).** MTP pays off in proportion to how much weight-streaming each token costs. The dense model streams ~22 GB/token, so committing several tokens per stream is a huge win. This MoE activates only ~3 B params (~3–4 GB) per token, so each token is already cheap — there's much less to amortise, and the draft+verify overhead eats into the gain. Net is still positive (~57 → ~75 t/s) and lossless, so it's worth enabling, but don't expect the dense model's doubling.

**Sweet spot / variance.** `n-max=3` (the default, and what's configured) has the highest average and ceiling; `n-max=2` is slightly more consistent. The run-to-run spread (64–90 t/s) is acceptance-rate variance — how often the drafted tokens match what the model would have produced depends on the content being generated. Both clearly beat baseline; the choice between 2 and 3 is within noise.

This model's config mirrors `qwen36-35b-a3b`'s sampling (8.8) plus `jinja = true` (its embedded template is correct, so no external `chat-template-file`) and the two `spec-*` MTP keys. Like the dense Ornstein it logs `fused Gated Delta Net (chunked) not supported` (hybrid attention, prefill only) — but at ~2100 t/s prefill that's not a practical concern here.

**Bottom line:** `ornstein36-35b-a3b` is the fastest of the *Qwen-family* models on this box (~75 t/s with MTP, vs qwen36's ~54.5 and the dense Ornstein's ~19) — though the smaller Gemma 4 (8.12) edges it overall (and now also gains a little from MTP, via a separate assistant draft). MTP here is a modest, free top-up rather than the transformative win it is on the dense model. (For *coding quality* it's a mixed bag — strong on Go, self-inconsistent tests on the Spring task — see [page 13](13-model-evaluation.md).)

## 8.12 Measured: Gemma 4 26B-A4B (Q4 QAT, **MoE**) — fastest on the box, MTP via a separate assistant

A different vendor and architecture (`gemma4`, supported as of build **b9502**): MoE with 128 experts / 8 active (~4 B/token), unsloth QAT Q4_K_XL (~14 GB). It is the **throughput leader** even before speculation. Unlike the Ornstein merges it has **no embedded MTP head**, but Gemma 4 ships a small separate **assistant** draft model — and as of build **b9571** llama.cpp can use it for MTP self-speculation. Same harness as 8.9–8.11.

| Metric | Baseline (no MTP) | **MTP (assistant draft, n-max=3)** |
|---|---|---|
| Generation — code (short ctx) | ~85 t/s | **~96 t/s avg, up to 114** (+12 % avg, +32 % peak) |
| Generation — prose (short ctx) | ~85 t/s | ~81 t/s (−5 %) |
| Generation (1800-token ctx) | ~74 t/s | — |
| Prefill (1801 tok) | ~2800 t/s | ~300–420 t/s¹ |
| GPU during decode | 94 % util, ~36 W | 94 % util, ~36 W |

¹ Prefill in the MTP column is from the short-prompt benchmark harness, not the 1801-token sweep — not directly comparable; MTP affects decode, not prefill throughput.

**Why it's the fastest** despite being the most numerous-expert MoE: only ~4 B params active per token *and* a 4-bit quant, so each token streams the fewest bytes of any model here (~14 GB total, far less active). It's still memory-bandwidth bound (94 % util at 36 W — the usual signature), just with the least to move. Beats the Q8 MoEs (qwen36 54.5, ornstein-35B 57 baseline / ~75 with MTP) and trounces the dense 27B (~9–19).

**Why the MTP gain is small and content-dependent.** Gemma's draft is a *separate* 816 MB assistant model, not an embedded head, and the main model already activates only ~4 B params per token — so each token is cheap and there's little weight-streaming to amortise (same reason the Ornstein MoE gains less than the dense 27B, 8.11). When the draft accepts well (predictable code) you get up to +32 %; on less predictable prose the draft+verify overhead isn't hidden and it costs ~5 %. It is **lossless** either way. Net positive for a coding-focused box, so it's enabled — at the cost of +816 MB resident for the draft.

**Config differences from the Qwen models** (see its preset section in 8.8/6.2):
- **`jinja = true`** — embedded template is correct; no external `chat-template-file`.
- **MTP via a separate assistant** — `spec-type = draft-mtp` **plus** `spec-draft-model = …assistant-F16.gguf` (the Ornstein merges need only `spec-type`, drafting from their embedded head). Needs build b9571+. Confirm after a restart with `journalctl -u llama-router | grep draft-mtp` (look for `loading draft model '…assistant…'` and `adding speculative implementation 'draft-mtp'`).
- **Gemma's own sampling**, not the Qwen `[*]` defaults: `temp = 1.0`, `top-k = 64`, `top-p = 0.95`. This matters: at low temperature Gemma 4 falls into **degenerate repetition loops** — a known Gemma trait. Run it at ~1.0. The per-model section pins these so a client can't accidentally drive it cold.
- Loads with an automatic `tokenizer.ggml.add_bos_token → true` override (logged at startup) and uses Gemma's sliding-window attention; `flash-attn on` and `cache-type q8_0` from `[*]` work unchanged.

**Bottom line:** Gemma 4 26B-A4B is the **fastest model on the box** (~74–85 t/s baseline, ~96 t/s on code with MTP, ~2800 t/s prefill) and the lightest (~14 GB). MTP via its separate assistant draft is a modest, lossless top-up for code (worth the 816 MB on a coding box; skip it if you mostly generate prose). Its one operational gotcha is sampling — keep `temp` near 1.0. (Coding quality: strong, with sampling and self-consistency caveats — see [page 13](13-model-evaluation.md).)

---

[← Public access](07-public-access-cloudflare.md) · [Index](README.md) · [Next: Monitoring →](09-monitoring.md)
