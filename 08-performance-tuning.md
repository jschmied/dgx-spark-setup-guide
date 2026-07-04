# 8. Performance tuning

[← Public access](07-public-access-cloudflare.md) · [Index](README.md) · [Next: Monitoring →](09-monitoring.md)

This page replaces the minimal `[*]` section of the preset from page 6 with a **tuned** one, and makes one change to the unit. Every flag is justified by measurements taken on a live GB10 box; the benchmark tables at the end show what each change does.

The router (the unit) doesn't change shape — the tuning lives in `/etc/llama-server/models.ini`, so every model the router spawns inherits it.

### At a glance: speculative-decode speedups per model

Single-stream **decode** throughput (t/s) on this GB10 box. *Base* = no speculation. *MTP* / *DFlash* are each model's adopted spec path at its tuned `spec-draft-n-max`. Figures are indicative — base/MTP/DFlash were taken on different workloads (see the linked section for exact conditions); DFlash numbers are at the **coding-prompt optimum** where that's the deployed use.

| Model | Quant | Base | MTP | DFlash | Deployed | § |
|---|---|---|---|---|---|---|
| `qwen3-coder-next` (80B-A3B) | UD-Q4_K_XL | ~51.7 | — *(no MTP path)* | **~58** (n=3) | **DFlash** | 8.8 · 8.11 |
| `qwen36-35b-a3b` (MoE) | UD-Q5_K_XL | ~61 | — *(no MTP path)* | **~112** (n=6) | **DFlash** | 8.9 · 8.11 |
| `qwen36-27b` (dense) | UD-Q5_K_XL | ~9 | ~20 | **~34** (n=6) | **DFlash** | 8.8 · 8.11 |
| `qwen36-27b-uncensored` (dense) | Q5_K_M | ~9 | — | **~38** (n=6) | **DFlash** | 8.8 · 8.11 |
| `step-37` (Step-3.7-Flash, ~196B) | UD-Q3_K_XL | n/b | **~34** (n=4) | — *(Qwen-only)* | **MTP** | 8.8 |
| `gemma-4-26B-A4B` (MoE) | QAT Q4_K_XL | ~75 | **~94** (n=3) | — *(no DFlash drafter)* | **MTP** | 8.10 |
| `gemma-4-31B` (MoE) | QAT Q4_K_XL | n/b | *loads, n/b* | — *(no DFlash drafter)* | **MTP** | 8.10 |
| `glm-4.5-air` (MoE) | UD-Q4_K_XL | n/b | *needs nextn-bearing GGUF* | — *(Qwen-only)* | none yet | 8.7 |

Takeaways: DFlash beats MTP head-to-head on the 27B (~34 vs ~20, same build/prompt — 8.11) and gives the two previously-spec-less models (`qwen3-coder-next`, `qwen36-35b-a3b`) a path they never had. Gemma stays on MTP (no DFlash drafter published). `— ` = not applicable; `n/b` = not benchmarked.

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
| **External draft model / speculative decoding** | **Superseded — now ADOPTED via DFlash (see 8.11).** Retained for context: a *generic* external draft on `qwen3moe` siblings net-slowed at batch=1 (MoE expert-routing union overhead), and llama.cpp once returned `"speculative decoding not supported by this context"` for `qwen3next`. **DFlash** — a block-diffusion drafter, upstream since **b9831** (PRs #22105 + #25110) — sidesteps both and is the spec path now used on the coder and both qwen36 models. |
| **MTP (Multi-Token Prediction)** | **Now adopted on the two Gemma 4 models and Step-3.7-Flash** (`gemma-4-26B-A4B` / `gemma-4-31B` / `step-37`, each via a separate draft GGUF — see 8.8/8.10). The dense Qwen3.6-27B (`qwen36-27b`) *had* MTP from its embedded head but was **switched to DFlash** (faster in a same-build A/B — see 8.11). The other two (qwen3-coder-next, qwen36-35b-a3b) have no MTP path and are now served by **DFlash** (8.11). **Qwen3-Coder-Next** (`qwen3next`): no MTP tensors, and llama.cpp's MTP path isn't wired for the arch. **Qwen3.6-35B-A3B** (`qwen35moe`): tensor table is only `blk.0`–`blk.39` with no `nextn.*`/`mtp.*` tensors — confirmed plain single-token decode (~54.5 t/s, see 8.9). **Qwen3.6-27B** (`qwen35`, dense, `qwen36-27b`): ships an embedded head (`nextn_predict_layers = 1`, `blk.*.nextn.*`), supported since build **b9502 (6ddc9430b)** — enabled with `spec-type = draft-mtp` (lossless). **Gemma 4** (`gemma4`, MoE — both 26B-A4B and 31B): no embedded head, but ships a small separate draft GGUF — enabled with `spec-type = draft-mtp` + `spec-draft-model = …it-Q8_0-MTP.gguf`, supported since build **b9571** (PRs #23398/#24282). Gain scales with weight-streaming per token: ~2× on the dense 27B (`qwen36-27b`, ~9 → ~20–22 t/s with MTP), and ~+25 % on the 1800-token synthetic (75 → 94 t/s, higher on well-accepted code) for the already-cheap 4 B-active Gemma 26B (8.10). Must be set explicitly: without it the server logs `common_speculative_init: no implementations specified for speculative decoding` and the draft sits unused. |
| **EAGLE3 (`spec-type = draft-eagle3`)** | New as of build **b9777** (PR #24593, "spec: support eagle3 for qwen3.5 & 3.6"). Like `draft-mtp` it needs a **separate, target-specific draft GGUF** (`-md` / `spec-draft-model`), but the draft is an EAGLE3 head trained on the target's hidden states rather than an embedded `nextn.*` head. Published drafts so far are for the **dense** `qwen36-27b` (`Ex0bit/Qwen3.6-27B-PRISM-EAGLE3`) and `Qwen3.5-9B` (`BLR2/Qwen3.5-9B-Eagle3-ShareGPT`); the PR demos dense models only and reports ~1.78× (12.6 → 22.3 t/s) at ~50 % accept. No EAGLE3 draft is published for the MoE `qwen36-35b-a3b`. The two blockers noted at b9777 have since cleared: (1) the PRISM repo still ships **safetensors only**, needing `convert_hf_to_gguf.py --target-model-dir …`; and (2) bug [#24541](https://github.com/ggml-org/llama.cpp/issues/24541) — the missing `t_layer_inp` tensor in `qwen35.cpp` — is **fixed as of b9843** (`qwen35.cpp` and `qwen35moe.cpp` both now register it), so EAGLE3 would work without a source patch. **Not adopted anyway: DFlash (8.11) was chosen instead** — it covers all three models (including the previously-spec-less 35B-A3B), and the same `t_layer_inp` mechanism EAGLE3 needs is what DFlash uses. EAGLE3 remains an untested alternative. |
| **`--cache-type-k/v q4_0`** | Breaks processing on Qwen3-Coder-Next (model gives degenerate output). Stays at `q8_0`. |
| **Lowering `--ctx-size` from 131072** | Use-case decision. If most prompts stay under 32 K, lowering this frees memory for more prompt-cache pool. |
| **NVFP4 quant** | Would leverage `BLACKWELL_NATIVE_FP4=1`. Requires a Qwen3-Coder-Next NVFP4 GGUF (not yet published as of writing). |

## 8.8 Per-model overrides

The tuned values live in `[*]`, so they apply to **every** model the router serves. When one model needs different settings, override the key in that model's own section — it wins over `[*]` (precedence: command line > model section > `[*]`). For example, the Qwen3.6 35B model wants its own sampling per the model card, a corrected chat template, and `preserve_thinking` pinned off:

```ini
[qwen36-35b-a3b]
model               = /opt/llm/models/qwen36-35b-a3b/Qwen3.6-35B-A3B-UD-Q5_K_XL.gguf
jinja               = true
chat-template-file  = /opt/llm/models/qwen36-35b-a3b/chat_template.jinja
chat-template-kwargs = {"preserve_thinking": false}
spec-type           = draft-dflash   ; block-diffusion drafter (8.11) — this model has no MTP head
spec-draft-model    = /opt/llm/models/qwen36-35b-a3b/Qwen3.6-35B-A3B-DFlash-Q8_0.gguf
spec-draft-n-max    = 6               ; coding-prompt sweep optimum (8.11)
temp                = 0.6
top-p               = 0.95
top-k               = 20
min-p               = 0.0
presence-penalty    = 0.0
repeat-penalty      = 1.0
```

These sampler keys are server-side **defaults** for that model; a client can still override per-request. The only one that differs from the tuned `[*]` here is `repeat-penalty` (1.0 vs the global 1.05) — the rest are pinned for clarity so the model's behaviour doesn't drift if you later retune `[*]`. The three `spec-*` keys are new: this MoE has **no** embedded MTP head (8.7), so it gets its speculation from a separate **DFlash** drafter — see 8.11 (~61 → ~112 t/s on code). The target is **Q5_K_XL**, not Q8: on this box Q5 is ~12 % faster with DFlash and ~10 GB smaller at near-lossless quality (8.9).

A third model, **`qwen36-27b`** — the plain **Qwen3.6-27B** (unsloth `Qwen3.6-27B-MTP-GGUF`, `UD-Q5_K_XL`, ~19 GB) — has **two** speculation options and now runs the faster one. Being the same Qwen3.6 family as the 35B, it uses the **same** external `chat_template.jinja` (the `qwen3.6-froggeric-v20` template; keep a copy in its own model dir so it's self-contained) with `jinja = true` — standardising the family on one template. It is a *dense* `qwen35` model (slow to decode — every token streams the full ~19 GB of weights), so speculation pays off hard here. It ships an embedded **MTP** head (`nextn.*`) *and* there's a separate **DFlash** drafter for it — in a same-build A/B (8.11) DFlash won (**~34 vs ~20 t/s**), so it's deployed on DFlash; the old MTP line is kept commented for easy revert. Note its sampling: it **must run at `temp = 1.0`** (vendor recommendation) — at `temp 0.6`/`0.2` this arch falls into reasoning-repetition loops and emits nothing. It runs at the full native 256 K, so it also gets `ctx-size`. Its `spec-*` keys:

```ini
[qwen36-27b]
model            = /opt/llm/models/qwen36-27b/Qwen3.6-27B-UD-Q5_K_XL.gguf
ctx-size         = 262144
jinja            = true
chat-template-file   = /opt/llm/models/qwen36-27b/chat_template.jinja   ; same froggeric template as the 35B
chat-template-kwargs = {"preserve_thinking": false}
; spec-type      = draft-mtp   ; A/B baseline (embedded head, ~20 t/s) — kept for revert
spec-type        = draft-dflash   ; DFlash drafter won the A/B (~34 t/s) — see 8.11
spec-draft-model = /opt/llm/models/qwen36-27b/Qwen3.6-27B-DFlash-Q8_0.gguf
spec-draft-n-max = 6           ; coding-prompt sweep optimum (8.11)
temp             = 1.0         ; vendor sampling; lower temps loop
top-p            = 0.95
top-k            = 20
min-p            = 0.0
presence-penalty = 0.0
repeat-penalty   = 1.0
```

This model is the one case on the box where both speculation styles are available, so it's worth understanding the contrast. **`draft-mtp`** builds the draft from the model's **own** embedded MTP head (`nextn_predict_layers`, tensors `blk.*.nextn.*`) — no separate draft model, ~160 MiB extra; this Q5 GGUF ships that head. **`draft-dflash`** uses a small separate block-diffusion drafter GGUF (`-md` / `spec-draft-model`, ~1.8 GB here). Both are distribution-preserving; on this model DFlash decodes faster and accepts more (8.11), so it's deployed. Either way the keys must be set explicitly — without a `spec-type` the server logs `common_speculative_init: no implementations specified for speculative decoding` and decode runs single-token. Confirm after a restart with `journalctl -u llama-router | grep -E 'draft-mtp|draft-dflash'` (look for `adding speculative implementation '…'`).

Measured on this box: the earlier MTP run (b9641) loaded cleanly with **65–85 % draft acceptance**, decoding **~20–22 t/s** — roughly **2× the spec-off baseline** estimated at ~9 t/s (a 19 GB dense model against this box's ~250–273 GB/s unified memory; speculation commits several tokens per weight-stream, which is what breaks that bandwidth wall). The current DFlash config (b9843, n-max 6) lifts that to **~34 t/s** on a coding prompt (8.11). Speculation only helps decode; prefill is unaffected. On the page-13 coding scorecard it is the weakest local coder — Go **1/4**, Java **3/4** — and slow with it.

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

**Don't let a model silently inherit the wrong sampling — and use the *right* card.** The `[*]` defaults (`temp 0.6`, etc.) are tuned for the Qwen3.6 *reasoning* models. `qwen3-coder-next` is a non-thinking coder; pin its own values rather than inheriting `[*]`. Note the card matters: the older **Qwen3-Coder** recommends `0.7 / 0.8 / 20`, but **Qwen3-Coder-Next** (this model) recommends **`temp 1.0 / top-p 0.95 / top-k 40`** ([unsloth/Qwen3-Coder-Next-GGUF](https://huggingface.co/unsloth/Qwen3-Coder-Next-GGUF), "Best Practices") — don't conflate the two. It also runs **DFlash** speculation (8.11):

```ini
[qwen3-coder-next]
model            = /opt/llm/models/qwen3-coder-next/Qwen3-Coder-Next-UD-Q4_K_XL.gguf
jinja            = true         ; its OWN embedded (XML) template — required for tool calls; do NOT use the froggeric file
spec-type        = draft-dflash ; DFlash drafter — qwen3next has no MTP head (8.11)
spec-draft-model = /opt/llm/models/qwen3-coder-next/Qwen3-Coder-Next-DFlash-Q8_0.gguf
spec-draft-n-max = 3            ; this drafter's accept decays with depth → stays at 3 (unlike the qwen36 pair at 6)
temp             = 1.0          ; Qwen3-Coder-Next card (NOT Qwen3-Coder's 0.7)
top-p            = 0.95
top-k            = 40
repeat-penalty   = 1.05
```

A quick way to find the author-intended values for any GGUF is the embedded `general.sampling.*` metadata (e.g. `temp`, `top_k`, `top_p`) — read it before deciding what to override. That audit is what set Gemma to `temp 1.0` and the dense `qwen36-27b` to `temp 1.0` (it loops lower), and confirmed the coder-next card values above.

Most flags transfer cleanly to other GGUFs. Four things worth pinning per-model rather than globally:

- **`cache-reuse = 256`** assumes prompts share a non-trivial prefix. Harmless for other workloads but the gain depends on usage patterns.
- **`cache-type-k/v = q8_0`** is conservative. Some models tolerate `q4_0` KV without quality loss (saves memory + bandwidth) — but on Qwen3-Coder-Next `q4_0` gives degenerate output (see 8.7), which is exactly why per-model overrides matter once you serve a mix.
- **`chat-template-file`** points at a Jinja template that overrides the one embedded in the GGUF. Some GGUFs ship a buggy template (wrong tool-call or thinking handling); a community-fixed template fixes it without re-quantizing. Download the `.jinja` next to the model and reference it with an absolute path. **It requires `jinja = true` in the same section** — Jinja is *not* on by default, and `chat-template-file`/`chat-template-kwargs` are silently ignored without it (the model then loads, but tool calls break — see "Jinja and tool calling" below). The override applies the next time that model loads; verify after a restart with `journalctl -u llama-router | grep chat-template-file` and a sample completion.
- **`chat-template-kwargs`** passes a JSON object straight into the Jinja template's variables — pin it here rather than trusting the client to send it. The fixed Qwen3.6 template above exposes `preserve_thinking` (whether *previous* turns' `reasoning_content` is re-fed into the prompt; default `false`). Setting it `false` keeps the context lean across multi-turn chats and stops a client from flipping it on. The value is a single line of valid JSON — `{"preserve_thinking": false}` — and is passed to the child as one argument intact, spaces and all. (Don't confuse it with `enable_thinking`, which controls whether the model thinks *at all*; that's still `true`.)

**Jinja and tool calling.** `jinja = true` does two things: it selects the model-specific (minja) template path — so `chat-template-file`/`chat-template-kwargs` take effect — *and* it builds the structured tool-call grammar from that template. Leave it off and the server uses a generic parser that doesn't know the model's tool-call syntax: the model still emits its `<tool_call>…</tool_call>`, but the server can't turn that into a structured `tool_calls` field. It logs `common_chat_peg_parse: unparsed peg-native output: <tool_call>` and returns malformed output that makes agent/tool clients (e.g. a VS Code LLM gateway) fail mid-stream with errors like *"output does not match the expected peg-native format."* Plain chat is unaffected, which is why the gap hides until something actually requests tools. **Every model that should support tool calling needs `jinja = true`** — the embedded-template models (Gemma, Step, GLM) already carry it; the two Qwen3.6 models needed it added alongside their external template, and `qwen3-coder-next` needed it added on its own.

**Which template each model uses.**

| Model | Template | Dialect | Note |
|---|---|---|---|
| `qwen3-coder-next` | own embedded | XML `<function=…><parameter=…>`, non-thinking | tools advertised as nested `<function><name>` XML |
| `qwen36-35b-a3b` | external froggeric | XML, thinking | embedded template is broken |
| `qwen36-27b` | external froggeric (own copy) | XML, thinking | same Qwen3.6 family as the 35B |
| `gemma-4-26B-A4B` / `-31B`, `step-37`, `glm-4.5-air` | own embedded | (vendor) | embedded template is correct |

The `qwen3.6-froggeric-v20` template is **Qwen3.6-family only**. Do **not** point `qwen3-coder-next` at it: although both emit the same `<function=…><parameter=…>` call dialect, the froggeric template *advertises* tools as JSON schema and injects `<think>` scaffolding the coder was never trained on (it is a non-thinking model that advertises tools as nested XML). The coder's own embedded template is correct — it just needs `jinja = true`.

**Backporting froggeric's extras (optional).** Beyond the broken-template fix, the froggeric template carries a few *dialect-independent* robustness features that could be grafted onto the coder's own template if wanted: **tool-response truncation** (`max_tool_response_chars` — caps huge tool outputs with a `[TRUNCATED …]` marker; the clear win for an agent-mode coder), **tool-arg truncation** (`max_tool_arg_chars`), and **consecutive-failure warnings** (injects an escalating `⚠️ SYSTEM WARNING` when tool results look like errors). Two caveats: they are **opt-in and currently inert** — the `max_*_chars` knobs default to `0` (off) and only activate via `chat-template-kwargs`, which here only pins `preserve_thinking` — and adopting them means **maintaining a forked coder template**. Everything else in froggeric (thinking extraction, `<|think_on/off|>` toggles, vision handling, JSON tool advertisement) is Qwen3.6-specific and must not be ported.

> The benchmark tables in 8.4–8.6 were measured on **Qwen3-Coder-Next** (the 80B-A3B Q4 coder). The numbers for the second model live in 8.9.

## 8.9 Measured: Qwen3.6-35B-A3B

**Deployed quant: `UD-Q5_K_XL`.** Isolated A/B vs `UD-Q8_K_XL` (single-stream, base / with DFlash n6): Q5 **~61 / ~112 t/s** vs Q8 **~55 / ~100** — ~10–12 % faster and ~10 GB smaller, at a ~4-pt lower DFlash accept (68 vs 72 %) that the faster compute more than offsets. So Q5 is the better pick here; the usual "lower quant = less bandwidth = faster" holds (the Q5 dynamic mix is near-lossless on a 3 B-active MoE). **Benchmarking caveat:** measure with the router **stopped** — on GB10's *unified* memory a second resident model (or a large download) starves bandwidth and skews t/s badly.

The tables below were measured on the older Q8 config but remain representative of the architecture's prefill/concurrency/cache *behaviour*. Driven against `/completion` with `ignore_eos:true` (exactly `n_predict` tokens). Prompt = 1291 tokens; warm-up discarded.

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

**Bottom line:** Qwen3.6-35B-A3B runs on the tuned `[*]` config at **Q5 + DFlash**. With no MTP head it was the slow path until **DFlash** (8.11) lifted decode to **~112 t/s on code** (n-max 6). It's the faster of the two large MoE/coder models on this box.

## 8.10 Measured: Gemma 4 (26B-A4B and 31B, Q4 QAT, **MoE**) — fastest on the box, MTP via a separate draft

A different vendor and architecture (`gemma4`, supported as of build **b9502**): MoE with 128 experts / 8 active (~4 B/token), unsloth QAT Q4_K_XL (~14 GB). It is the **throughput leader** even before speculation. It has **no embedded MTP head**, but Gemma 4 ships a small separate draft model — and as of build **b9571** llama.cpp can use it for MTP self-speculation. Same harness as 8.9.

| Metric | Baseline (no MTP) | **MTP (Q8_0-MTP draft, n-max=3)** |
|---|---|---|
| Generation (1800-token synthetic ctx) | ~75 t/s | **~94 t/s** (+25 %, 61 % draft accept) |
| Prefill (1801 tok) | ~2950 t/s | ~2770 t/s (MTP-neutral; small dip is cross-restart noise) |
| GPU during decode | 94 % util, ~36 W | 94 % util, ~36 W |

The 1800-token row is a clean baseline-vs-MTP toggle on the same `bench/throughput.sh` harness (build b9641, **Q8_0-MTP draft**, 2026-06-16). On the more predictable page-13 coding prompts the draft accepts **~86 %** (vs 61 % on this synthetic), so the code-path gain runs higher than the +25 % shown here; free-form prose accepts worst and nets roughly flat-to-slightly-negative as before. Lossless either way.

**Why it's the fastest** despite being the most numerous-expert MoE: only ~4 B params active per token *and* a 4-bit quant, so each token streams the fewest bytes of any model here (~14 GB total, far less active). It's still memory-bandwidth bound (94 % util at 36 W — the usual signature), just with the least to move. Beats the 35B MoE (qwen36-35b-a3b ~61 at Q5) and trounces the dense 27B (`qwen36-27b`, ~9 baseline / ~20–22 with MTP).

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

## 8.11 DFlash speculative decoding (the coder + both qwen36 models)

**DFlash** is a *block-diffusion* drafter: a tiny separate model that proposes a whole block of tokens in parallel (here `block_size = 16`), which the target then verifies in one pass — distribution-preserving like MTP, but it works on architectures that have no embedded MTP head. It landed upstream in **b9831** (PRs #22105 + #25110); this box runs it on **b9843**. It's the spec path for the three models that either had none or had a slower one:

| Target | Arch | Drafter (z-lab, converted to Q8_0) | Was |
|---|---|---|---|
| `qwen3-coder-next` | `qwen3next` | `Qwen3-Coder-Next-DFlash` (487 MB) | no spec path |
| `qwen36-35b-a3b` | `qwen35moe` | `Qwen3.6-35B-A3B-DFlash` (410 MB) | no spec path |
| `qwen36-27b` | `qwen35` (dense) | `Qwen3.6-27B-DFlash` (1.8 GB) | MTP (slower) |

**Arch requirement.** The target must expose its hidden states at the drafter's extract layers — in llama.cpp that's `res->t_layer_inp[il]` in the model graph. `qwen35.cpp` and `qwen35moe.cpp` both register it as of **b9843** (this closed bug [#24541](https://github.com/ggml-org/llama.cpp/issues/24541), the same gap that had blocked EAGLE3 — 8.7). The `qwen3next` graph did **not**, so `qwen3-coder-next` carries a **one-line local patch** (`res->t_layer_inp[il] = inpL;` in `qwen3next.cpp`) — it asserts at load without it, and the patch must be re-applied after a clean `git pull` until it lands upstream.

**Building a drafter GGUF.** The z-lab repos ship safetensors only, so convert with the build's own converter, pointing `--target-model-dir` at the *target's* HF tokenizer (drafter embeds the target vocab):

```bash
/opt/llm/runtime/convert-venv/bin/python \
  /opt/llm/runtime/llama.cpp/convert_hf_to_gguf.py \
  /opt/llm/models/_dflash-src/qwen36-35b-a3b-dflash \
  --target-model-dir /opt/llm/models/_dflash-src/qwen36-35b-a3b-tok \
  --outtype q8_0 \
  --outfile /opt/llm/models/qwen36-35b-a3b/Qwen3.6-35B-A3B-DFlash-Q8_0.gguf
```

*Quirk:* the converter shifts `target_layers` by **+1** vs the drafter's `config.json` (e.g. 35B `[1,6,…,37]` → GGUF `[2,7,…,38]`) — intentional layer-input mapping, not a bug; just confirm the max id is `< target block_count` (35B = 40 layers, 27B = 65).

**Tuning `spec-draft-n-max`.** `block_size = 16` allows up to 15. Swept on a small coding prompt (`write is_prime`), idle box, median of 5 — code accepts well, so the optimum sits higher than the n=3 that suits general prompts:

| n-max | 35B-A3B t/s | 27B t/s |
|---|---|---|
| 3 | 88.3 | 27.5 |
| 4 | 94.3 | 30.6 |
| 5 | 98.7 | 34.0 |
| **6** | **108.8** | **34.5** |
| 8 | 97.1 | 34.7 |
| 10 | 92.9 | 34.0 |

Both deployed at **n-max 6** (35B a clear peak; 27B a plateau from n=5 where 6 holds max t/s at a healthier ~67 % accept vs ~57 % at n=8). `qwen3-coder-next` is the exception: its drafter's acceptance decays faster with depth (84 %→63 % from n3→n5), so it stays at **n-max 3** (~58 t/s @ ~81 % on the same prompt at its `temp 1.0` sampling).

**27B: DFlash vs MTP (A/B).** Same `b9843` build, same prompt and sampling (`temp 1.0`):

| 27B drafter | t/s | accept |
|---|---|---|
| MTP (embedded head) | 19.9 | 57 % |
| **DFlash** | **23.3** | **66 %** |

DFlash wins, so the 27B was switched (8.8). Verify any of these after a restart with `journalctl -u llama-router | grep draft-dflash` (look for `adding speculative implementation 'draft-dflash'` and the `block_size=16, … n_extract=N` line). The startup line `dflash requires ctx_other to be set` is a benign memory-fitting warning, not an error.

**Reusing a drafter across a finetune.** A DFlash drafter pairs with a *base* model's hidden states, but it works on a **finetune of the same arch + tokenizer** without re-conversion. `qwen36-27b-uncensored` (an AEON merge of Qwen3.6-27B, same `qwen35` arch, identical 248320-token vocab, 64 vs 65 layers so the drafter's max extract layer 62 is still in range) just points `spec-draft-model` at the **base 27B's** `Qwen3.6-27B-DFlash-Q8_0.gguf` — no download. Acceptance barely dropped (65.6 % vs the base's 66 %), decoding ~38 t/s on the coding prompt. The same trick won't help `step-37` (Step-3.7-Flash) or `glm-4.5-air` — DFlash only implements the Qwen3 arch — so **Step** uses **MTP** instead via a separate community draft (`notSnix/Step-3.7-Flash-MTP-Draft-GGUF`, `spec-type = draft-mtp` + `spec-draft-p-min = 0.60`; throughput plateaus from n-max 3 — n2 ~29 t/s @ 88 %, n3 ~34 @ 85 %, n4 ~34 @ 88 % — deployed at **n-max 4** for the higher accept at equal speed), and **GLM-4.5-Air** has no spec path yet (its GGUF ships with the `nextn` head stripped — 8.7).

> n-max is tuned for **coding** prompts (high acceptance). On free-form prose, acceptance is lower and n=6 slightly over-drafts — drop to ~3 if a model is mostly used for prose.

---

[← Public access](07-public-access-cloudflare.md) · [Index](README.md) · [Next: Monitoring →](09-monitoring.md)
