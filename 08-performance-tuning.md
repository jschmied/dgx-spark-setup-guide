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
| **MTP (Multi-Token Prediction)** | **Adopted on three of five models**: the dense Qwen3.6-27B base (`qwen36-27b`, embedded head — see 8.8) and both Gemma 4 models (`gemma-4-26B-A4B` / `gemma-4-31B`, separate draft — see 8.8/8.10). The other two (qwen3-coder-next, qwen36-35b-a3b) have no MTP path. **Qwen3-Coder-Next** (`qwen3next`): no MTP tensors, and llama.cpp's MTP path isn't wired for the arch. **Qwen3.6-35B-A3B** (`qwen35moe`): tensor table is only `blk.0`–`blk.39` with no `nextn.*`/`mtp.*` tensors — confirmed plain single-token decode (~54.5 t/s, see 8.9). **Qwen3.6-27B** (`qwen35`, dense, `qwen36-27b`): ships an embedded head (`nextn_predict_layers = 1`, `blk.*.nextn.*`), supported since build **b9502 (6ddc9430b)** — enabled with `spec-type = draft-mtp` (lossless). **Gemma 4** (`gemma4`, MoE — both 26B-A4B and 31B): no embedded head, but ships a small separate draft GGUF — enabled with `spec-type = draft-mtp` + `spec-draft-model = …it-Q8_0-MTP.gguf`, supported since build **b9571** (PRs #23398/#24282). Gain scales with weight-streaming per token: ~2× on the dense 27B (`qwen36-27b`, ~9 → ~20–22 t/s with MTP), and ~+25 % on the 1800-token synthetic (75 → 94 t/s, higher on well-accepted code) for the already-cheap 4 B-active Gemma 26B (8.10). Must be set explicitly: without it the server logs `common_speculative_init: no implementations specified for speculative decoding` and the draft sits unused. |
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

A third model, **`qwen36-27b`** — the plain **Qwen3.6-27B** (unsloth `Qwen3.6-27B-MTP-GGUF`, `UD-Q5_K_XL`, ~19 GB) — shows two more per-model patterns: the **other** template pattern, and **MTP self-speculation**. Its embedded chat template is correct, so instead of an external `chat-template-file` you just turn Jinja on for it. It is a *dense* `qwen35` model (slow to decode — every token streams the full ~19 GB of weights), so it's the one model here that benefits from its built-in MTP head as its own draft. Note its sampling: it **must run at `temp = 1.0`** (vendor recommendation) — at `temp 0.6`/`0.2` this arch falls into reasoning-repetition loops and emits nothing. It runs at the full native 256 K, so it also gets `ctx-size`. Add the two `spec-*` keys for MTP:

```ini
[qwen36-27b]
model            = /opt/llm/models/qwen36-27b/Qwen3.6-27B-UD-Q5_K_XL.gguf
ctx-size         = 262144
jinja            = true
spec-type        = draft-mtp   ; use the model's own embedded MTP head as its draft
spec-draft-n-max = 3           ; draft depth; 3 is the sweet spot on this box
temp             = 1.0         ; vendor sampling; lower temps loop
top-p            = 0.95
top-k            = 20
min-p            = 0.0
presence-penalty = 0.0
repeat-penalty   = 1.0
```

`spec-type = draft-mtp` tells the child to build a speculative draft context from the model's **own** embedded MTP head (`nextn_predict_layers`) — no separate draft model, ~160 MiB extra. The head proposes up to `spec-draft-n-max` tokens per step and the full model verifies them in one pass; accepted tokens are free and **lossless** (the verifier preserves the model's output distribution). This Q5 GGUF ships the head (`nextn_predict_layers = 1`, tensors `blk.*.nextn.*`) — unlike the two models in 8.7 — and the keys above turn it on. (Without them the server logs `common_speculative_init: no implementations specified for speculative decoding` and the head sits unused — it must be enabled explicitly.) Confirm it's active after a restart with `journalctl -u llama-router | grep draft-mtp` (look for `adding speculative implementation 'draft-mtp'`).

Measured on this box (b9641, FP4 build): MTP loads cleanly (`adding speculative implementation 'draft-mtp'`, `n_embd = 5120`, ~2.3 GB MTP context) with **65–85 % draft acceptance**, decoding **~20–22 t/s** — roughly **2× the MTP-off baseline** estimated at ~9 t/s (a 19 GB dense model against this box's ~250–273 GB/s unified memory; MTP commits several tokens per weight-stream, which is what breaks that bandwidth wall). MTP only helps decode; prefill is unaffected. On the page-13 coding scorecard it is the weakest local coder — Go **1/4**, Java **3/4** — and slow with it.

A fourth model, **Gemma 4 26B-A4B** (`gemma-4-26B-A4B`), shows a different-vendor pattern: embedded template (`jinja = true`), its **own** sampling rather than the Qwen `[*]` defaults (and notably **min-p instead of top-p** — see below), and the **other MTP variant** — a *separate* draft model rather than an embedded head. Gemma needs `temp ≈ 1.0` — at low temperature it loops (8.10) — so pin its sampling per-model:

```ini
[gemma-4-26B-A4B]
model            = /opt/llm/models/gemma-4-26B-A4B/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf
spec-type        = draft-mtp
spec-draft-model = /opt/llm/models/gemma-4-26B-A4B/gemma-4-26B-A4B-it-Q8_0-MTP.gguf
spec-draft-n-max = 3
jinja            = true
temp             = 1.0          ; Google's recommended Gemma 4 defaults
top-k            = 64
top-p            = 0.95
repeat-penalty   = 1.0
```

**Why min-p, not top-p, for Gemma.** At `temp 1.0 / top-p 0.95` Gemma's Go code was production-correct only 1 in 4 samples (a different rare-token bug each time); at `temp 1.0 / min-p 0.1` (top-p off) it is **4/4**, with Java still perfect and no loops. `min_p` keeps only tokens with probability ≥ 10 % of the most-likely token, adaptively cutting the tail that caused the failures — without the repetition loops a blanket low temperature triggers. Full experiment on [page 14](14-sampling-and-variance.md).

Gemma 4's MTP is **not** an embedded head like `qwen36-27b`'s — it ships a small separate draft GGUF (`gemma-4-26B-A4B-it-Q8_0-MTP.gguf`, ~440 MB — a Q8_0 of Google's F16 assistant draft, swapped in 2026-06-16) that serves as the draft, so you point `spec-draft-model` at it rather than relying on `nextn.*` tensors inside the main GGUF. This needs build **b9571+** (Gemma 4 MTP landed after b9502 — PRs #23398/#24282). The win is real but content-dependent: **+25 % decode on the 1800-token synthetic** (75 → 94 t/s, 61 % accept), higher on well-accepted code (~86 % accept), roughly flat on free-form prose — see 8.10. It is the fastest model on the box either way (~75 t/s baseline).

**Don't let a model silently inherit the wrong sampling.** The `[*]` defaults (`temp 0.6`, etc.) are tuned for the Qwen3.6 *reasoning* models. `qwen3-coder-next` is a non-thinking coder and wants the Qwen3-Coder card values instead — pin them rather than inheriting `[*]`:

```ini
[qwen3-coder-next]
model           = /opt/llm/models/qwen3-coder-next/Qwen3-Coder-Next-UD-Q4_K_XL.gguf
temp            = 0.7
top-p           = 0.8          ; tighter than the reasoning models' 0.95
top-k           = 20
repeat-penalty  = 1.05
```

A quick way to find the author-intended values for any GGUF is the embedded `general.sampling.*` metadata (e.g. `temp`, `top_k`, `top_p`) — read it before deciding what to override. That audit is what set Gemma to `temp 1.0` and the dense `qwen36-27b` to `temp 1.0` (it loops lower), and gave the coder model the values above.

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

## 8.10 Measured: Gemma 4 (26B-A4B and 31B, Q4 QAT, **MoE**) — fastest on the box, MTP via a separate draft

A different vendor and architecture (`gemma4`, supported as of build **b9502**): MoE with 128 experts / 8 active (~4 B/token), unsloth QAT Q4_K_XL (~14 GB). It is the **throughput leader** even before speculation. It has **no embedded MTP head**, but Gemma 4 ships a small separate draft model — and as of build **b9571** llama.cpp can use it for MTP self-speculation. Same harness as 8.9.

| Metric | Baseline (no MTP) | **MTP (Q8_0-MTP draft, n-max=3)** |
|---|---|---|
| Generation (1800-token synthetic ctx) | ~75 t/s | **~94 t/s** (+25 %, 61 % draft accept) |
| Prefill (1801 tok) | ~2950 t/s | ~2770 t/s (MTP-neutral; small dip is cross-restart noise) |
| GPU during decode | 94 % util, ~36 W | 94 % util, ~36 W |

The 1800-token row is a clean baseline-vs-MTP toggle on the same `bench/throughput.sh` harness (build b9641, **Q8_0-MTP draft**, 2026-06-16). On the more predictable page-13 coding prompts the draft accepts **~86 %** (vs 61 % on this synthetic), so the code-path gain runs higher than the +25 % shown here; free-form prose accepts worst and nets roughly flat-to-slightly-negative as before. Lossless either way.

**Why it's the fastest** despite being the most numerous-expert MoE: only ~4 B params active per token *and* a 4-bit quant, so each token streams the fewest bytes of any model here (~14 GB total, far less active). It's still memory-bandwidth bound (94 % util at 36 W — the usual signature), just with the least to move. Beats the Q8 MoE (qwen36-35b-a3b ~54.5) and trounces the dense 27B (`qwen36-27b`, ~9 baseline / ~20–22 with MTP).

**Why the MTP gain is content-dependent.** Gemma's draft is a *separate* model (the `it-Q8_0-MTP` build, ~440 MB resident — a Q8_0 of the original F16 assistant draft, swapped in 2026-06-16 at half the footprint), not an embedded head, and the main model already activates only ~4 B params per token — so each token is cheap and there's less weight-streaming to amortise than on the dense 27B. When the draft accepts well (predictable code, ~86 %) the gain is largest; on less predictable prose the draft+verify overhead isn't fully hidden and it nets roughly flat. It is **lossless** either way. Net positive for a coding-focused box, so it's enabled — at the cost of +440 MB resident for the draft.

**Config differences from the Qwen models** (see its preset section in 8.8/6.2):
- **`jinja = true`** — embedded template is correct; no external `chat-template-file`.
- **MTP via a separate draft model** — `spec-type = draft-mtp` **plus** `spec-draft-model = …it-Q8_0-MTP.gguf` (`qwen36-27b` needs only `spec-type`, drafting from its embedded head). Needs build b9571+. Confirm after a restart with `journalctl -u llama-router | grep draft-mtp` (look for `loading draft model '…Q8_0-MTP…'` and `adding speculative implementation 'draft-mtp'`). *(Draft swapped from the original `assistant-F16.gguf` to the smaller `Q8_0-MTP.gguf` build on 2026-06-16 — ~440 MB vs ~815 MB resident, re-benched on this draft: +25 % decode on the 1800-token synthetic at 61 % accept, ~86 % accept on coding prompts.)*
- **Gemma's own sampling**, not the Qwen `[*]` defaults: `temp = 1.0`, `top-k = 64`, and **`min-p = 0.1` with top-p disabled**. Two things matter here: at low temperature Gemma 4 falls into **degenerate repetition loops** (a known Gemma trait — run it at ~1.0), and at `temp 1.0` plain top-p leaves a long tail that wrecked its Go code (1/4 correct); **min-p 0.1 fixed that to 4/4** (page 14). The per-model section pins these so a client can't accidentally drive it cold.
- Loads with an automatic `tokenizer.ggml.add_bos_token → true` override (logged at startup) and uses Gemma's sliding-window attention; `flash-attn on` and `cache-type q8_0` from `[*]` work unchanged.

**Bottom line:** Gemma 4 26B-A4B is the **fastest model on the box** (~75 t/s baseline, ~94 t/s with MTP on the 1800-token synthetic / higher on well-accepted code, ~2800 t/s prefill) and the lightest (~14 GB). MTP via its separate Q8_0-MTP draft is a lossless top-up for code (worth the ~440 MB on a coding box; skip it if you mostly generate prose). Its one operational gotcha is sampling — `temp ≈ 1.0` with **min-p 0.1 (top-p off)**. (Coding quality with that sampling: production-correct on both tasks — Go **4/4**, Java 4/4 — the best local default; see [page 13](13-model-evaluation.md) and the sampling story on [page 14](14-sampling-and-variance.md).)

### Sibling: Gemma 4 31B (QAT Q4_K_XL, ~17 GB) — a *reasoning* variant

Gemma 4 also ships a larger **31B** (`gemma-4-31B`), set up identically to the 26B: `gemma4` arch, embedded template (`jinja = true`), and the same separate-draft MTP (its own `gemma-4-31B-it-Q8_0-MTP.gguf`, ~515 MB, ~440 MB resident; `spec-type = draft-mtp` + `spec-draft-model`). Two differences matter operationally:

- **It is a reasoning model.** Unlike the 26B-A4B, the 31B emits a thinking block into `reasoning_content` before its final `content`. Clients need a generous `max_tokens` (a small budget returns empty `content` because generation was cut off mid-thought), and the chat layer must read `reasoning_content`/`content` accordingly.
- **Sampling** is Google's defaults *without* min-p: `temp = 1.0`, `top-p = 0.95`, `top-k = 64`.

Throughput on this box isn't separately benchmarked yet, but the MTP draft loads cleanly (`adding speculative implementation 'draft-mtp'`) with high acceptance (1.0 on a trivial prompt, ~0.86 on a short coding prompt). On the page-13 coding bench (4 samples/task at its sampling) it scores **Go 3/4, Java 4/4** on the neutral suite — the one Go miss was a trivial unused-variable compile error, and on Java its *production* code is 4/4 though its *own delivered tests* are often self-inconsistent (they under-count a legitimate second `save()`); see [page 13](13-model-evaluation.md).

---

[← Public access](07-public-access-cloudflare.md) · [Index](README.md) · [Next: Monitoring →](09-monitoring.md)
