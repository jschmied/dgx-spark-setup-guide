# Runbook: Qwen3.8-27B-NVFP4 + MTP on DGX Spark, vLLM 0.27.1

One GB10 (sm_121a, 128 GB unified), CUDA 13, **stock `pip install vllm==0.27.1`** — no fork, no
custom image. Result: full **262 k** context, image input, tool-calling, and a **~2× decode
speed-up** from the checkpoint's built-in MTP head (11.4 → 25.5 tok/s on short prompts at n=3;
see §4 and §5 — the honest figure on *real* long-reasoning work is closer to 1.5×).

⚠️ **Read §9 before touching memory settings.** Raising `--gpu-memory-utilization` and
`--kv-cache-memory` together froze this box hard enough to need a powercycle, and cgroup limits
cannot prevent it.

Measured on release day+1 (2026-08-15) against `unsloth/Qwen3.8-27B-NVFP4`.

---

## 0. The headline: a benchmark trap that inverts the answer

**Under speculative decoding, vLLM packs several accepted tokens into ONE streaming chunk.**
A probe that counts SSE deltas (`n += 1` per chunk) therefore measures *deltas/s*, which **falls**
as real throughput **rises**.

Our first sweep "showed" MTP making the model monotonically slower — 11.4 → 7.6 tok/s as depth
went 0 → 4. Entirely an artifact. The same prompt emitted **283 deltas at n=0 but 105 at n=3**,
a 2.7× drop that exactly tracks the mean acceptance length (3.29).

```python
# WRONG — measures deltas/s; makes spec-decode look like a regression
if delta.get("content"): n += 1

# RIGHT — usage.completion_tokens is authoritative
# needs stream_options.include_usage + --enable-force-include-usage
if payload.get("usage"): n = payload["usage"]["completion_tokens"]
```

**Self-check: print a `tok/delta` column.** It must read **~1.00 with spec off** and rise toward
the acceptance length as depth grows. If it stays at 1.00 with spec on, your server isn't
speculating; if you never look, you will conclude speculation is a pessimization.

> This is worth checking in *any* spec-decode harness, including the community ones. We did not
> verify whether other engines batch tokens per chunk the way vLLM does — if yours emits one
> token per chunk, delta-counting is fine. Check rather than assume, in both directions.

---

## 1. Install — the torch pin is the whole difficulty

0.27.1 hard-pins `torch==2.13.0`. On GB10 you are on a `+cu130` local-version build that is **not
on PyPI**, so a naive upgrade silently swaps in the wrong wheel. Clone the venv, install the torch
trio from the cu130 index *first*, then vLLM:

```bash
cp -a vllm-venv-026 vllm-venv-027          # clone; never `python -m venv` fresh
# ... rewrite absolute paths in bin/*, pyvenv.cfg, lib/**/*.pth ...

pip install --index-url https://download.pytorch.org/whl/cu130 \
    torch==2.13.0+cu130 torchvision==0.28.0+cu130 torchaudio==2.11.0+cu130
python -c "import torch; assert '+cu130' in torch.__version__"   # gate BEFORE continuing

pip install -U vllm==0.27.1
python -c "import torch; assert '+cu130' in torch.__version__"   # gate AGAIN — pip can swap it
```

`torch==2.13.0+cu130` satisfies vLLM's `torch==2.13.0` pin (PEP 440 local version), so the second
install leaves it alone. **This bump also moves triton 3.6.0 → 3.7.1**, unlike 0.25→0.26 which
touched neither.

**`FLASHINFER_DISABLE_VERSION_CHECK=1` is now mandatory.** `flashinfer-cubin` is stuck at 0.6.13
(newest that exists) while 0.27.1 pins `flashinfer-python` 0.6.16.post3, and **0.6.16 raises on the
mismatch at import** — 0.26 did not. Not fixable by upgrading.

---

## 2. Serve command

```bash
export CUTE_DSL_ARCH=sm_121a FLASHINFER_DISABLE_VERSION_CHECK=1
export MAX_JOBS=2 FLASHINFER_NVCC_THREADS=1        # bound JIT fan-out; unbounded ninja + high
                                                   # util = global OOM that kills the box
vllm serve /models/qwen38-27b-nvfp4 \
  --served-model-name qwen38-27b \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --kv-cache-dtype fp8 --attention-backend flashinfer \
  --reasoning-parser qwen3 --tool-call-parser qwen3_xml --enable-auto-tool-choice \
  --enable-force-include-usage \
  --limit-mm-per-prompt '{"image":4,"video":0}' \
  --mm-processor-kwargs '{"max_pixels":1048576}' \
  --max-model-len 262144 --max-num-seqs 16 --max-num-batched-tokens 8192 \
  --enable-chunked-prefill --enable-prefix-caching \
  --default-chat-template-kwargs '{"reasoning_effort":"medium"}' \
  --compilation-config '{"cudagraph_mode":"PIECEWISE","cudagraph_capture_sizes":[1,2,4,8,16]}' \
  --gpu-memory-utilization 0.76 --load-format fastsafetensors --trust-remote-code
```

`cudagraph_capture_sizes` **must cover `max-num-seqs`** — batches larger than the biggest captured
size silently fall back to eager. Raising one without the other is a quiet regression.

PIECEWISE (not FULL) is also the correct mode under speculation: FULL capture is reported to
corrupt output when combined with spec-decode. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
is required for the full 262 k window.

**No `--override-generation-config`** — the checkpoint's `generation_config.json`
(temp 1.0 / top_p 0.95 / top_k 20) *is* the vendor thinking-mode set. Non-thinking clients should
send temp 0.7 / top_p 0.80 / presence_penalty 1.5 per request.

### Sizing: two ceilings, and which one binds

The hybrid attention makes KV unusually cheap — only **16 of 64 layers** carry full-attention KV,
the rest are linear attention. At `fp8` KV that works out to **37.2 KiB/token**. Measured on a
121 GB box:

| `gpu-memory-utilization` | KV | tokens | sessions @262 k | free RAM left |
|---|---|---|---|---|
| 0.50 | 35.45 GiB | 1,114,865 | 4.3 | ~55 GB |
| 0.60 | 46.98 GiB | 1,325,364 | 5.1 | ~40 GB |
| **0.76** | **66.74 GiB** | **1,883,336** | **7.2** | **~20 GB** |

Two limits apply at once and the crossover moves with utilisation:

```
context per session > KV_tokens / max_num_seqs   ->  KV memory binds
context per session < KV_tokens / max_num_seqs   ->  max-num-seqs binds
```

At 0.76 with `max-num-seqs 16` the crossover is **~118 k tokens/session**. Below that you get all
16 sessions; above it you get `KV_tokens / context`. vLLM prints the answer itself at startup:
`Maximum concurrency for 262,144 tokens per request: 7.18x`.

⚠️ **Do not chase the last few GB.** `0.9` plus a 75 GiB KV pin killed this box outright (global
OOM, power-cycle required). 0.76 leaves ~20 GB, which is enough headroom for a FlashInfer JIT
recompile *provided* `MAX_JOBS`/`FLASHINFER_NVCC_THREADS` are bounded as above. Verify after any
change that `free -g` still shows a double-digit "available" figure.

### Verify the native FP4 path is live
```
INFO [__init__.py:1077] Using FlashInferCutlassNvFp4LinearKernel for NVFP4 GEMM
INFO [__init__.py:665]  Selected CutlassFP8ScaledMMLinearKernel for CompressedTensorsW8A8Fp8
```
Both lines appear on **stock PyPI vLLM**. Claims that GB10 requires a special nightly or a
re-hosted container image are false for this checkpoint — and pulling a third-party mirrored image
to run `--gpus all` is a supply-chain risk you do not need to take.

---

## 3. Large-prefill warm is mandatory after every (re)start

A `"hello"` warmup only compiles the **small-shape** kernels. The first *big* prompt then pays the
full FP4 JIT cost and can stall or return garbled output — which is what an agent client sending a
20 k-token system preamble will do the moment it connects.

```bash
# fire ~26k tokens of REAL text after health, before clients connect (~21 s)
curl -s "$URL/v1/chat/completions" -H "Authorization: Bearer $KEY" \
  -d "{\"model\":\"qwen38-27b\",\"max_tokens\":32,\"messages\":[{\"role\":\"user\",
       \"content\":$(python3 -c 'import json;print(json.dumps(open("corpus.txt").read()[:95000]))')}]}"
```

The linear-attention kernels JIT on first use too — watch for
`fused_recurrent_gated_delta_rule_packed_decode_kernel` and `_causal_conv1d_update_kernel` in the
log. Cached afterwards, so this is a per-cache-wipe cost, not per-restart.

---

## 4. MTP depth — the optimum is deeper than every published source says

Paired sweep, 3 runs × 3 prompts (2 code / 1 prose), large-prefill warmed, util 0.5, kv fp8,
tokens counted from `usage.completion_tokens`:

| n | overall median | vs base | P1 code | P2 prose | P3 code | TTFT |
|---|---|---|---|---|---|---|
| 0 | 11.4 | 1.00× | 11.4 | 11.4 | 11.3 | 193 ms |
| 1 | 18.4 | 1.61× | 19.0 | 15.9 | 18.4 | 218 ms |
| 2 | 22.8 | 2.00× | 24.9 | 18.5 | 22.8 | 237 ms |
| 3 | 25.5 | 2.24× | 29.4 | 17.6 | 25.5 | 257 ms |
| **4** | **28.3** | **2.48×** | 33.4 | 17.5 | 28.3 | 277 ms |
| 5 | 26.2 | 2.30× | **36.7** | 16.3 | 26.2 | 295 ms |
| 6 | 27.0 | 2.37× | 34.9 | 17.3 | 27.0 | 316 ms |

⚠️ **These are SHORT-PROMPT numbers, and the apparent n=4 optimum does NOT generalise.** Repeat the
measurement at a realistic **4000-token input** (§5) and n=3 wins or ties everywhere. Published
recommendations are n=2 (Unsloth) and n=3 (official vLLM recipe + both community repos); after
correcting for context length, **n=3 is right and we ship it**.

The lesson is that **the MTP optimum shifts down as context grows.** Mean acceptance length at n=4
fell from **3.76 on short prompts to 2.84 at 4 k** — richer context is harder for the single reused
head, so the deep draft positions stop paying. Benchmarking speculative decoding on short synthetic
prompts will systematically over-recommend depth.

The shape *does* hold at every context length: code prompts keep climbing with depth, prose
flatlines from n=1 onward.

**The shape matters more than the peak:** code prompts keep climbing to n=5 (36.7 tok/s, 3.2×)
while **prose flatlines at ~16–18 from n=1 onward.** If your workload is code, go deeper; if it's
prose, almost all the win is already at n=1.

Per-position acceptance at n=4: **0.85 / 0.71 / 0.62 / 0.48** (avg ~64 %). Position 4 still lands
nearly half the time — far above break-even.

### ⚠️ Real-work acceptance is about HALF the synthetic number

Measured on an actual coding task (long reasoning trace) versus the short-prompt probe:

| | short prompts | real coding task |
|---|---|---|
| per-position | 0.91 / 0.76 / 0.63 | **0.49 / 0.30 / 0.19** |
| avg draft acceptance | 66–76 % | **33–39 %** |
| mean acceptance length | 3.29 | **1.99–2.16** |

Long chain-of-thought is far less predictable for the single reused head than a short generic code
request. **Expect ~1.5× on real work, not the ~2.2× the probe reports.** MTP is still a clear win;
the magnitude just does not survive contact with a real workload. vLLM says as much at startup:
*"Enabling num_speculative_tokens > 1 will run multiple times of forward on same MTP layer, which
may result in lower acceptance rate."*

### "depth ≥ 4 crashes / emits invalid tokens" is false here
n=4, 5 and 6 all ran clean, with no invalid-token or spec-failure signatures in the logs. There is
also **no cap in vLLM**: the only check is `num_speculative_tokens % n_predict == 0`
(`config/speculative.py:1012`), and `n_predict == 1` (`mtp_num_hidden_layers`), so every integer
passes. Depth is limited by *drift*, not by a limit — the single MTP head is reused for every draft
position (`qwen3_5_mtp.py:162`, `spec_step_idx % num_mtp_layers`), and it was trained to predict one
step ahead from the **main model's** hidden state. Contrast a DeepSeek-style `n_predict=3`
checkpoint: three separately trained heads, no drift at depth 3.

---

## 5. Concurrency — single-stream is not the whole story

Single-stream tuning can mislead: Unsloth justify their n=2 recommendation with *"faster decode but
somewhat less throughput"* — i.e. aggregate under load, where rejected drafts burn compute a
batched request could have used. So measure both axes.

Real-text prompts, 4000-token input, 256-token output, `ignore_eos`, tokens from
`usage.completion_tokens`:

| n | c=1 decode | c=4 agg | c=8 agg | c=16 agg | mean accept len |
|---|---|---|---|---|---|
| 0 | 11.2 | 31.0 | 30.8 | 30.8 | — |
| 2 | 19.8 | **47.8** | 47.8 | 49.6 | 2.32 |
| **3** | **21.6** | 47.3 | **49.0** | **51.7** | 2.62 |
| 4 | 21.4 | 42.7 | 48.4 | 51.0 | 2.84 |

**MTP wins at every concurrency level — +55–68 % aggregate.** The "speculation wastes compute when
batched" concern does not materialise here, so there is no throughput reason to run shallow.

**n=3 is best or tied-best everywhere**, and n=4 is distinctly worse at c=4 (42.7 vs 47.3). Combined
with §4 this is the whole argument for shipping n=3.

⚠️ **The c=8 and c=16 rows are queueing, not scaling.** They were run at `--max-num-seqs 4`, so
aggregate saturates at c=4 while TTFT grows linearly — 7.8 s → 40 s → **105 s** at n=0. If you raise
`max-num-seqs`, re-measure before drawing any conclusion from those columns. And read §9 first: KV
and concurrency changes are exactly what killed this machine.

---

## 6. Thinking, reasoning effort, and tool calling

**Thinking is ON by default and the default effort is `xhigh`** — the most expensive level, applied
to any client that says nothing. The template also injects *"think carefully through the task,
validate key assumptions, consider plausible alternatives"* into the system message at that level.

### ⚠️ `xhigh` does not converge on real coding work — use `medium`

Running the Go coding task at pure vendor defaults, the model spent its **entire 26,000-token
budget inside `<think>` and emitted zero content**:

```
finish: "length"   completion_tokens: 26000
has_reasoning: true   reasoning_len: 97708   content_len: 0
```

A benchmark harness scores that as a failed task. It is not — it is a **budget artifact**, and it
will silently corrupt any scorecard that does not check `finish_reason` and content length. At
`reasoning_effort=medium` the same task converged in **10,544 tokens** (20 k chars reasoning,
14.8 k chars content) and produced correct code.

**Ship `medium` for agentic/coding work.** It is also the only level that injects no instruction
text, so your own system prompt stands alone. Either send it per request, or set it server-side:

```bash
--default-chat-template-kwargs '{"reasoning_effort":"medium"}'
```

| effort | accepted? | note |
|---|---|---|
| `xhigh` | ✅ | default; injects the "think carefully" preamble |
| `medium` | ✅ | **injects nothing** — the no-instruction escape hatch |
| `low` | ✅ | injects "keep your thinking brief" |
| `high` | ✅ | silently **aliased to `xhigh`** |
| `none` | ✅ | vLLM converts it to `enable_thinking=false` |
| **`minimal`** | ❌ **HTTP 400** | vLLM's API advertises it; the template raises |
| **`max`** | ❌ **HTTP 400** | same |

`minimal` and `max` are in vLLM's `Literal["none","minimal","low","medium","high","xhigh","max"]`
and are forwarded straight into the template, which only accepts `xhigh|medium|low`. A
standards-following client can therefore 400 against a perfectly healthy server.

**To suppress the injected text without forking the template**, use `medium` — or set it
server-side with `--default-chat-template-kwargs '{"reasoning_effort":"medium"}'`. To invent your
*own* levels you need both a forked template and to pass them via `chat_template_kwargs`; the
top-level field's pydantic `Literal` rejects novel names before rendering.

**Precedence is backwards from intuition:** `merge_kwargs(chat_template_kwargs, extra_kwargs)`
returns `defaults | overrides`, so the **top-level field beats `chat_template_kwargs`**. `None` is
filtered, so the escape hatch works only when the standard field is absent.

### Two silent client-breakers

- **`reasoning_content` → `reasoning`.** 0.27.1 renamed the response field. Clients written against
  the old key see **no reasoning at all, with no error**. (This cost us a full sweep: our harness
  read the old key, reported 0 reasoning tokens and 17–40 s TTFT, and we briefly believed tokens
  were being generated and dropped.)
- **`qwen3_xml` and `qwen3_coder` are aliases** — both resolve to `Qwen3EngineToolParser`. The
  official recipe says `qwen3_coder`, other guides say `qwen3_xml`; they are the same class. Use
  either. (`hermes` genuinely does not work — the payload is XML, not JSON.)

### Tool calling at the checkpoint's own sampling
**6/6 with thinking on, 6/6 with thinking off, at temp 1.0, under MTP-4.** Prior Qwen3.6 NVFP4
guidance that temp 1.0 breaks tool-calling (1/8 vs 7/8 at temp 0.2) does **not** carry to this
model — do not blindly port that override.

### `preserve_thinking` defaults to TRUE — think before disabling it
The obvious move is to turn it off and save context. It is probably wrong: with `false`,
`ns.last_query_index` advances each turn, so concluded turns get their `<think>` blocks stripped
**retroactively**, the rendered prefix changes underneath you, and **prefix caching is invalidated
every turn.** Prefill is the expensive half on this box. The real tradeoff is fewer tokens vs.
cached prefill, and the vendor default is the safer side of it.

---

## 7. Gotcha → fix

| symptom | cause | fix |
|---|---|---|
| `RuntimeError: flashinfer-cubin (0.6.13) does not match flashinfer (0.6.16.post3)` at import | 0.6.16 enforces a check 0.6.14 didn't; no newer cubin exists | `export FLASHINFER_DISABLE_VERSION_CHECK=1` |
| torch silently becomes a non-`+cu130` build | vLLM pins bare `torch==2.13.0`; PyPI wheel wins | install the trio from the cu130 index first; assert `+cu130` before *and* after |
| MTP appears to make decode slower | probe counts SSE deltas, not tokens | use `usage.completion_tokens`; assert `tok/delta ≈ 1.00` with spec off |
| first large prompt stalls or returns garbage | only small-shape FP4 kernels were compiled | fire a ~26 k-token warm request after health |
| `reasoning_content` always empty | renamed to `reasoning` in 0.27.1 | read `reasoning` (accept both for portability) |
| HTTP 400 `Unexpected reasoning effort minimal` | vLLM's Literal is wider than the template's whitelist | send `low`/`medium`/`high`/`xhigh`/`none`, or fork the template to map them |
| `ValueError: moe_backend='flashinfer_b12x' is not supported for FP8 MoE` | mixed-precision **MoE** checkpoint + a forced global `--moe-backend` | not applicable to this dense model; for MoE add `"flashinfer_b12x": Fp8MoeBackend.MARLIN` to the `mapping` dict **inside** `map_fp8_backend()` — it moved there in 0.27 |
| `Torchcodec` load error | video decoding | `apt install ffmpeg` (or keep `"video":0`) |
| box dies during first boot | unbounded FlashInfer JIT fan-out at high util | `MAX_JOBS=2`, `FLASHINFER_NVCC_THREADS=1`, warm at low util |

---

## 7b. What NVFP4 actually costs, measured against BF16

The checkpoint is **not** uniformly NVFP4. `quantization_config` declares
`format: mixed-precision` with two groups:

| | group_1 — NVFP4 W4A4 | group_0 — FP8 W8A8 |
|---|---|---|
| targets | `mlp.(gate\|up\|down)_proj` | attention `q/k/v/o`, GDN `in_proj/out_proj`, **`lm_head`**, **MLP of layers 56–63** |
| weights | 4-bit, `tensor_group`, group 16, scale in FP8-E4M3 | 8-bit, per-channel, `memoryless_minmax` |
| activations | 4-bit, group 16, `dynamic: local` | 8-bit, per-token, `dynamic: True` |

168 MLP tensors are packed to 4-bit; 27 stay at 8-bit. **The last eight layers and the
`lm_head` are deliberately left at FP8** — the standard guard against damage nearest the output,
and the reason the card notes it will not load in SGLang.

### Weight error against the BF16 original

Dequantized both checkpoints and compared them to `Qwen/Qwen3.8-27B` BF16, 12 MLP tensors
(layers 0–3). Only **one 4 GB shard** of the 55.6 GB original is needed — shard 1 covers layers 0–4.

| comparison | relative L2 |
|---|---|
| **NVFP4 ↔ BF16** | **0.1210** |
| **FP8 ↔ BF16** | **0.0266** |
| NVFP4 ↔ FP8 | 0.1238 |

**NVFP4 sits 4.5× further from the original than FP8**, and remarkably evenly so (0.110–0.154
across tensors). Note `N↔F ≈ N↔B`: the distance to FP8 *is* the distance to the truth, because FP8
is so close — which is what makes FP8 a usable stand-in when the BF16 weights are not on disk.

### Why this closes the recalibration question

Of the 1,968 tensors, exactly **200 are calibrated**: 168 `input_global_scale` and 32 KV
`k_scale`/`v_scale` (`observer: static_minmax`, `dynamic: False`). Everything else — including
**all weight scales** — is derived from the weights themselves (`memoryless_minmax`), so **no
calibration set touched them and none can change them**.

That means 12.1 % of weight error is fixed before a single activation is computed. Recalibrating on
our own agent traces would only move the activation and KV scales. **The dominant term is out of
reach**, so a recalibration effort is aimed at the smaller lever.

Two further measurements support leaving it alone:

- **Teacher-forced logprob divergence NVFP4 vs FP8** on 17 real agent contexts (129,892 positions):
  mean 1.11 nats, median 0.056, 30.4 % of positions above 0.5 — roughly **80× the fp8-vs-bf16
  reference** of 0.014. But it is **flat across context depth** (0.90 → 1.08 from <512 to >8192
  tokens), so the error does **not compound** as the KV cache fills. That is the reassuring half for
  long agent runs.
- The comparison cannot be settled on SWE-bench either: a paired McNemar test over 100 instances
  resolves roughly **12 pp**; detecting the 2–3 pp a quantization change would plausibly produce
  needs **1,700–3,900 instances**, six to thirteen times what SWE-bench Multilingual contains.

### The honest framing

The same checkpoint carrying 12.1 % weight error and 1.11 nats of distributional distance from FP8
**solves 70 of 100 real repository bugs**. Quantization error does not translate linearly into task
failure, which is exactly why neither number should be read as an alarm.

**If you want more quality, change the format, not the calibration.** FP8 is 4.5× closer to the
original — and costs 30.9 GB instead of 22.6, and ran measurably slower here (225 s vs 136 s for the
same eight agent contexts). That is the real trade, now quantified rather than guessed.

> Reproduce: `/tmp/weight_compare.py` (CPU only, one tensor at a time). One trap — NVFP4's
> `weight_global_scale` is a **reciprocal** (amax-derived, value 6400). Multiplying instead of
> dividing yields weights of ~1e7 and a relative error of 4e7; the absurd magnitude is the tell.

## 8. Notes, and what we did not test

- **NVFP4 over FP8 for speed.** Two independent reports have FP8 costing ~30 % decode without
  improving acceptance — decode here is memory-bound (~273 GB/s), so fewer weight bytes win. FP8
  remains a *quality* lever, not a speed one. `MXFP4` does not load on NVIDIA devices at all.
- **1M context**: the vLLM recipe uses `--max-model-len 1010000` plus
  `--hf-overrides '{"text_config":{"max_position_embeddings":1010000}}'` — **not** YaRN rope-scaling
  (which is how SGLang does it). Qwen's own guidance is to **tune the YaRN factor to your actual
  context** (2.0 for ~512 k), not to blanket-apply 4.0. We stay native 262 144 with YaRN off.
- **Prefix caching + this hybrid arch is flagged experimental** by vLLM
  (`Mamba cache 'align' mode`), and we run with it enabled.
- **Not tested:** output quality at MTP depth (speculative decoding is distribution-preserving in
  theory; we did not verify empirically), and other NVFP4 quants (`RadixArk`, `Inferact`) against
  Unsloth's. Checkpoint provenance is a plausible source of the ~50 % throughput disagreement
  between published reports.

### DSpark drafting — works, but it is not a drop-in win

**Provenance first, because this is third-party weights running in your inference process.**

| | |
|---|---|
| Checkpoint | `RadixArk/Qwen3.8-27B-DSpark` — **not** published by Qwen/Alibaba |
| Method | DSpark, extending **DFlash** (`github.com/z-lab/dflash`, arXiv 2602.06036) — an academic block-diffusion drafting method, not a vendor feature |
| Trained with | **SpecForge** (`github.com/sgl-project/SpecForge`), served upstream via SGLang |
| Checkpoint tag | `epoch_2_step_4166` — i.e. 2 epochs |
| Size / dtype | 1.36 B params, BF16, 5 full-attention layers, GQA 40 Q / 8 KV heads |
| Wiring into the target | auxiliary features tapped from target layers **`[4, 16, 28, 40, 52]`**; vanilla Markov confidence head, rank 256; block size 7 |
| **Training data** | **not disclosed.** The card names the method and the checkpoint step but never the corpus. |
| License | `other`, no named author |

**⚠️ It was trained against a different quantization of the target.** The card states the target is
`Qwen/Qwen3.8-27B-FP8`; we serve `unsloth/Qwen3.8-27B-NVFP4`. The drafter consumes hidden states
from those five target layers, so it is being fed activations from weights it was not trained on.
That is a plausible cause of the underwhelming result below, and it is testable: the FP8 checkpoint
would isolate it.

Their own reported acceptance length is **3.39 mean over 1,164 requests** (11 workloads, FP8 target
+ BF16 draft, SGLang) — 3.47 on HumanEval down to 2.71 on Arena-Hard-v2. Ours is a different target
quantization and a different engine, so those numbers do not transfer.

**Risk framing:** speculative decoding verifies every drafted token against the target, so the
distribution is preserved and a poor or even hostile drafter costs *throughput*, not correctness.
That bounds the exposure — but it is still unaudited third-party code (`dflash.py`, `dspark.py` ship
with the checkpoint and require `--trust-remote-code`) executing in the server process.

Related drafters on this box, for the same provenance question:
- `z-lab/Qwen3.6-27B-DFlash` — academic, and its own card says *"still under training, and inference
  engine support may not be fully available yet"*. Compute credited to Modal, InnoMatrix, Yotta Labs.
  Training data not disclosed.
- `poolside/Laguna-S-2.1-DFlash-NVFP4` — **the only first-party one**: poolside publishes both the
  base and the speculator, and states `e0630_rhiemann_baseline` SFT, DFlash Stage-2, 15 k steps.
  Corpus still not disclosed.

The DSpark draft model *does* run against this target on vLLM 0.27.1, after two non-obvious steps:

1. The checkpoint ships `architectures: ["DSparkDraftModel"]`, which vLLM's registry maps to
   **DeepSeek-V4** (`registry.py:617`). A Qwen3 target needs `["Qwen3DSparkModel"]`
   (`registry.py:618` → `qwen3_dspark.py`). Patch the drafter's `config.json`.
2. Pass the drafter path **explicitly**. With `model` unset, the dspark branch
   (`config/speculative.py:717`) assumes DeepSeek-style in-checkpoint weights.

```bash
--speculative-config '{"method":"dspark","model":"/models/qwen38-27b-dspark","num_speculative_tokens":7}'
```

Paired against MTP n=3 (same util, `--max-model-len 32768`, `usage` tokens, 3 runs):

| prompt | MTP n=3 | DSpark k=7 | DSpark k=14 |
|---|---|---|---|
| code (py) | 30.3 | **43.8 (+45 %)** | 42.8 |
| prose | 18.0 | 16.4 (−9 %) | 13.8 (−23 %) |
| code (bash) | 25.3 | 23.4 (−8 %) | 19.9 (−21 %) |
| mean | 24.9 | **28.0 (+12 %)** | 26.0 |

**`k=14` is worse than `k=7` here**, contradicting the published recommendation, and the reported
72–75 tok/s single-stream did not reproduce — our best single figure is 43.8 on one code prompt.

#### Measured on real agent traffic — DSpark loses by 27 %

The synthetic result above invited a workload explanation: published DSpark numbers come from an
*edit-heavy* harness, and agent traffic echoes file contents constantly, so the real-work gain
should be larger. **It is not.** Replaying 24 genuine agent contexts from our SWE-bench
trajectories (2 k–43 k tokens, median 10.7 k), paired, `usage.completion_tokens`:

| | MTP n=3 | DSpark k=7 |
|---|---|---|
| median | **26.4 tok/s** | 18.9 tok/s |
| mean | 26.3 | 20.1 |
| wins | — | **3 / 24** |
| median TTFT | 7.74 s | 7.39 s |

> **Do not use tokens-per-SSE-delta as an acceptance measure.** A chunk can span several
> verification steps, so the ratio exceeds the theoretical maximum — MTP `n=3` caps at 4 accepted
> tokens per step, yet it measured 5.81. vLLM exposes the real counters:
> `vllm:spec_decode_num_{drafts,draft_tokens,accepted_tokens}_total` and, most usefully,
> `..._accepted_tokens_per_pos_total{position=N}`.

**−27 % median, and the deficit is flat across context size** — −27.4 % below 8 k, −24.3 % at
8–20 k, −27.4 % above 20 k. If the workload theory held, the gap would close as context grows; it
does not move. Acceptance settles it: DSpark drafts 7 tokens and lands *fewer* than MTP's 3. TTFT
is unchanged, so this is decode, not a prefill trade.

There is also a capacity cost that the throughput number hides:

| | KV | tokens | sessions @262 k |
|---|---|---|---|
| MTP n=3 | 66.74 GiB | 1,883,336 | **7.18×** |
| DSpark k=7 | 27.06 GiB | 506,909 | **1.93×** |

~40 GB leaves the KV pool for a 2.7 GB drafter, because `k=7` reserves buffers for seven
speculative positions and scales `max_num_scheduled_tokens` with them.

**We stay on MTP n=3.** It ships *inside* the target checkpoint, so it costs no extra memory, no
capacity, and carries no separate provenance question. The remaining open question is whether the
FP8-vs-NVFP4 training mismatch explains the deficit — testable by serving the FP8 target, with no
training involved.

Constraint: DSpark does **not** fit alongside the 262 k window at util 0.6 (needs ~14 GiB KV,
6.5 available). Either shorten the context or raise utilisation.

### What a drafter would actually have to deliver here

Measured acceptance settles the DSpark question and, more usefully, sets the bar for any future
drafter. vLLM's own counters (`vllm:spec_decode_num_accepted_tokens_per_pos_total`), 8 real agent
contexts:

| position | MTP n=3 | DSpark k=7 |
|---|---|---|
| 0 | **87.2 %** | 62.4 % |
| 1 | 76.7 % | 41.2 % |
| 2 | 67.2 % | 26.0 % |
| 3–6 | — | 18.6 / 11.3 / 7.3 / 5.6 % |
| **acceptance rate** | **77.0 %** | 24.7 % |
| **tokens per step (incl. bonus)** | **3.31** | 2.73 |

DSpark loses at *position 0* — it mispredicts the immediate next token 38 % of the time where MTP
misses 13 %. Everything downstream follows from that.

**This is why quantizing the drafter cannot rescue it.** MTP yields 3.31 tokens per step; DSpark
yields 2.73. Shrink the drafter to *zero bytes* — physically impossible, but the limit — and DSpark
still moves the same bytes as MTP while producing **17.5 % fewer tokens per step**. The deficit is
prediction quality, not bandwidth. Bandwidth makes it worse (a bf16 drafter at `k=7` is 45 % of all
bytes moved), but it is not the cause.

Break-even, then. Assuming constant acceptance *p* per position, a drafter beats MTP only above:

| drafter | k=2 | k=3 | k=4 | k=7 |
|---|---|---|---|---|
| 1.36 B bf16 (2.7 GB) | impossible | impossible | 98.5 % | 91.7 % |
| 1.36 B fp8 (1.4 GB) | impossible | 97.9 % | 89.7 % | 83.7 % |
| 1.36 B nvfp4 (0.7 GB) | impossible | 92.8 % | 84.7 % | 78.6 % |
| 0.5 B nvfp4 (0.25 GB) | impossible | 89.1 % | 81.0 % | 74.4 % |
| MTP-sized head (~0.05 GB) | impossible | 87.3 % | 79.2 % | 72.2 % |

Three things fall out, and they are the design brief:

1. **`k=2` is unwinnable at any acceptance.** MTP n=3 already emits 3.31 tokens/step; a 2-deep
   drafter caps at 3 even at 100 %. Depth ≥ 3 is a hard floor.
2. **A bf16 drafter against a 4-bit target is structurally uncompetitive** — it would need 91.7 %
   sustained across seven positions. The better the target is quantized, the more a fat drafter
   costs *relatively*. This is the reason recommendations tuned on FP8 targets do not transfer.
3. **Sustained acceptance beats peak acceptance.** MTP holds 87/77/67 over three positions. A
   drafter must hold ~75–80 % over *seven* — much harder than beating 87 % once.

### Measured: nothing available beats the MTP head that ships with the target

Four configurations, same 8 agent contexts, acceptance from vLLM's per-position counters:

| config | tokens/step | bytes/step | acceptance | sessions @262 k |
|---|---|---|---|---|
| **MTP n=3** | **3.31** | 22.8 GB | 77.0 % | 7.18× |
| DSpark k=7 | 2.73 | 41.5 GB | 24.7 % | 1.93× |
| n-gram k=8 | 2.62 | 22.6 GB | 20.3 % | 7.07× |
| n-gram k=4 | 2.26 | 22.6 GB | 31.5 % | 7.61× |

Per-position acceptance, which is where the differences come from:

| position | MTP n=3 | DSpark k=7 | n-gram k=8 |
|---|---|---|---|
| 0 | **87.2 %** | 62.4 % | 42.3 % |
| 1 | 76.7 % | 41.2 % | 27.4 % |
| 2 | 67.2 % | 26.0 % | 22.1 % |
| 3–7 | — | 18.6 → 5.6 % | 17.4 → 12.0 % |

**The copy-based proposal below was too optimistic, and the number that killed it is position 0.**
The reasoning was that agent traffic is saturated with verbatim repetition — true of the *prompts*
(1.52 M prompt vs 1.34 M assistant tokens in our corpus, and the prompts are largely echoed source)
but not of what the model *generates*. Explanations, diff hunks and commands with varying arguments
are new text. Only 42–45 % of immediate next tokens are copyable. Inferring predictability from a
corpus statistic instead of measuring it was the error; the measurement cost one hour.

n-gram does have the flattest tail of the three (12 % still accepted at position 7 vs DSpark's
5.6 %) — when a copy match exists it tends to continue. Deeper `k` therefore helps it: `k=8` beat
`k=4` by 16 %. Extrapolating the tail, matching MTP would need roughly `k=14`, at rising scheduler
cost. Not obviously worth it, but the trend is real and cheap to re-test.

**Why MTP is hard to beat:** it ships *inside* the target checkpoint (15 tensors — one transformer
layer plus a projection), so it costs no extra weight bytes per step, needs no second precision
class, and was fitted to exactly these activations at exactly this quantization. Every external
drafter must pay its own bandwidth *and* out-predict it.

#### The one proposal whose economics work: more MTP heads

MTP's ceiling is drift, not capability. There is **one** head, reused at every draft position
(`qwen3_5_mtp.py:162` — `spec_step_idx % num_mtp_layers` is always 0), so position 2 consumes hidden
states derived from its own guesses. The measured decay 87 → 77 → 67 is exactly that, and it is why
`nspec=3` is the optimum while `nspec=4` collapses.

DeepSeek-style **separate heads per position** remove the drift by construction. If acceptance held
near 87 % across four positions, that is ~4.2 tokens/step against today's 3.31 — **+27 % with zero
additional bytes per step**, because the heads live in the target checkpoint like the existing one.

Training such heads is real work, but note what it avoids: no second checkpoint to keep in step, no
provenance question, no quantization mismatch, and no capacity loss. The MTP head is also a natural
**initialization** for them — it is already fitted to this target at this precision.

> **Not** a distillation target, though: acceptance is defined against what the *target* samples, so
> the target is the teacher. Training a drafter against the MTP head would inherit its 12.8 % miss
> at position 0 as a floor, and the head has no signal at all beyond position 2 — silent exactly
> where a deeper drafter needs to learn.

### Rejected after measurement: n-gram / suffix drafting

The table's bottom row is the interesting one: at near-zero drafter cost the bar drops to ~79 % at
`k=4`. vLLM already ships two drafters that cost **no model bytes at all** —
`--speculative-config '{"method":"ngram",...}'` and `{"method":"suffix",...}` — which predict by
copying from the existing context rather than from a network.

That matches this workload's dominant characteristic. Agent traffic is saturated with verbatim
repetition: file contents echoed into tool results and then quoted back, diffs restated, paths and
identifiers repeated across turns. Our own corpus makes the point — 100 SWE-bench trajectories hold
1.34 M assistant tokens against 1.52 M prompt tokens, and the prompts are largely echoed source.
Copy-based drafting is *exactly* the mechanism that regime rewards, and it has no training, no
provenance question, no extra weights, and no separate checkpoint to keep in step with the target.

It is not free of cost: `k` speculative positions still enlarge scheduler buffers, which is what
took KV from 66.74 to 27.06 GiB at `k=7` — budget for that, or run a smaller `k`.

**Result: it fell short** — see the measurement above. `k=4` gave 2.26 tokens/step and `k=8` gave
2.62, against MTP's 3.31, for the same bytes moved. The idea was still worth the hour: it cost no
download and no training, and it converted a plausible-sounding argument into a number. `suffix`
was not run separately — it searches the same generated text for copies, so the position-0 ceiling
of ~42 % applies to it too.

#### Was it the quantization mismatch? Measured on both targets — no

DSpark was trained against `Qwen/Qwen3.8-27B-FP8`, so a natural objection to the result above is
that we fed it activations from an NVFP4 target it was never fitted to. That is testable without
any training: serve the FP8 target and repeat. Same drafter, same 8 agent contexts, same seed.

**Acceptance at draft position 0:**

| | MTP n=3 | DSpark k=7 | gap |
|---|---|---|---|
| **NVFP4 target** | **87.2 %** | 62.4 % | **−24.8 pp** |
| **FP8 target** | 83.4 % | 65.0 % | **−18.4 pp** |
| Δ from target | −3.8 pp | **+2.6 pp** | |

**Effective tokens per step:**

| | MTP n=3 | DSpark k=7 |
|---|---|---|
| NVFP4 | **3.31** | 2.73 |
| FP8 | 3.20 | 2.78 |

Overall acceptance: MTP 77.0 % / 73.4 %, DSpark 24.7 % / 25.4 %.

**The mismatch is real but small.** DSpark gains **+2.6 pp** on its native target — the right
direction, and about a tenth of the gap to MTP. Crucially the comparison now holds *within one
target*: 83.4 % against 65.0 % on FP8. The confound is quantified rather than assumed away.

The other diagonal is the surprise: **MTP loses 3.8 pp on FP8**, i.e. the built-in head does better
on the *more* aggressively quantized target. Counter-intuitive, not robust at N=8 prompts, but
enough to retire the assumption that a higher-precision target automatically helps a drafter.

FP8 also costs throughput outright — 225 s versus 136 s for the same eight prompts — so even a
favourable result here would have been a diagnosis, not a configuration worth adopting.

> **Also ruled out: the upstream DSpark routing bug.** vllm-project/vllm#50851 reports that
> `use_dflash()` excludes `"dspark"`, so the model runner never sets `use_aux_hidden_state_outputs`
> and a drafter designed to read target layers `[4,16,28,40,52]` silently gets none of them. Our
> 0.27.1 does contain the excluding code, so this looked like the explanation. Patching all four
> sites and re-measuring produced a **bit-identical** result and an identical KV footprint — on
> this version and this checkpoint the path is already equivalent. The patch was reverted rather
> than carried as unverified local drift. Related: #51581 (quantized drafters, filed against this
> exact GB10 setup), #49646 (arch routing), #49614.

### Third-party claims we could not reproduce

- **"CUTLASS FP4 kernels emit silent garbage on SM121."** We run exactly that path
  (`CutlassNvFp4LinearKernel` + `CutlassFp4GemmRunner`) and scored **70/100 on SWE-bench
  Multilingual**, plus 25/28 on a Java/JS slice reproduced twice with zero variance. Garbage output
  does not solve real repository bugs. Treat as version-specific or wrong.
- **"`W4A4` buys nothing."** That is a *performance* argument about activation precision; it is
  presented alongside the correctness claim above and should not be read as one.
- **Confirmed and already in the serve command:** `VLLM_MARLIN_USE_ATOMIC_ADD=1`, PIECEWISE
  cudagraphs, `expandable_segments:True`.

### ⚠️ `thinking_token_budget` may not apply

vLLM 0.27.1 logs `Model Runner V2 does not yet support the thinking_token_budget request
parameter`. This is the *only* sampler-enforced brake on runaway thinking, so where it is
unavailable the effort level is a prompt nudge and nothing more. Observed consequence on real
agent work: one SWE-bench instance burned `max_tokens` inside `<think>` five times at
`reasoning_effort=medium` and never emitted a tool call, scoring as an empty patch. **Verify the
parameter takes effect on your runner before relying on it.**
- **Effort-level cost is unmeasured.** Three *semantically identical* requests produced 74 / 362 /
  489 reasoning tokens at temp 1.0. Any `medium`-vs-`xhigh` cost claim needs N ≥ 10 per cell.

## 9. ⚠️ Memory: cgroup limits CANNOT protect this box

**This config froze the machine hard enough to require a physical powercycle.** Raising
`--gpu-memory-utilization` 0.5→0.9, pinning 75 GiB KV, `max-num-seqs` 4→32 and extending the
cudagraph capture list — all in one step — hung it during startup. There was no kernel `oom-kill`
entry: it thrashed the unified pool until nothing was schedulable. Worse, the unit was `enabled`,
so the reboot auto-started it straight back into the same config.

### Why `MemoryMax=` does not save you

Measured on one running vLLM at util 0.5:

| source | value |
|---|---|
| cgroup `memory.current` | 13.96 GiB |
| Σ process `VmRSS` | ~7.2 GiB |
| `nvidia-smi` used_gpu_memory | **57.9 GiB** ← the real consumer |

GB10 is unified memory, so "GPU" memory *is* system RAM — but it is allocated through the NVIDIA
driver and charged to **neither** process RSS **nor** the cgroup. So `MemoryMax=`/`MemoryHigh=`
would bind on ~14 GiB of host-side memory while the ~58 GiB that actually fills the box goes
unaccounted. **`systemd-oomd` is equally blind to it** — it acts on cgroup pressure, and vLLM's
cgroup looks small and calm.

**The only effective limits are vLLM's own `--gpu-memory-utilization` and `--kv-cache-memory`.**
Treat them as the safety system, and change **one at a time**, verifying between each.

### What does work: protect the login path

When the driver consumes RAM invisibly, the kernel reclaims from cgroups that *do* have accounted
memory — sshd and your shell. `memory.min` is hard protection in cgroup v2 and prevents exactly
that eviction.

```ini
# /etc/systemd/system/ssh.service.d/oom-protect.conf
[Service]
MemoryMin=192M
MemoryLow=256M
OOMScoreAdjust=-900

# /etc/systemd/system/user.slice.d/oom-protect.conf   <-- the one that actually matters
[Slice]
MemoryMin=4G
MemoryLow=6G
```

**Protecting `ssh.service` alone is a trap.** logind moves an SSH login into
`user.slice/user-<uid>.slice/session-N.scope` — a different cgroup — so you would connect fine and
then watch the shell hang. Size the guarantee *above* observed usage (`user.slice` idles ~1.5 GiB;
a 1 GiB floor leaves the difference reclaimable and achieves nothing).

Finally, `vm.swappiness` 60 → **10**. With 121 GiB of RAM and a 16 GiB swapfile the default made
the kernel swap anonymous memory under pressure, converting an over-allocation into a multi-minute
freeze instead of a clean, survivable kill.

## 10. Coding-task results

`reasoning_effort=medium`, vendor sampling (temp 1.0), scored by a **neutral** suite we wrote as
well as the model's own tests:

| task | own suite | neutral suite | verdict |
|---|---|---|---|
| **Go** — concurrent generic LRU cache | FAIL | **PASS** | implementation correct; its *own test* is broken |
| **Java** — Spring Boot + JPA transfer, optimistic locking | ok (2+8 tests) | **PASS** (5 tests) | clean pass |

The Go defect is specific: the implementation declares `Get(key K) (V, bool)` but its own test
calls `_ = c.Get("a")` — a one-value assignment from a two-value function, so the *test* fails to
compile. Our neutral suite compiled against the same `cache.go` and passed under `-race`, including
eviction ordering. The model writes correct code and an inconsistent test for it.

> Harness note: `bench/run-go.sh`'s extractor asked for `m.group(2)` against a single-group regex,
> raising `IndexError` on any response that actually contained a fenced code block — it only ever
> appeared to work on empty responses. Group the language tag (`^```(\w*)\n(.*?)^```) to fix it.

## 11. SWE-bench Multilingual — 70/100

100-instance slice (seed 20260815), mini-swe-agent, `reasoning_effort=medium`, temp 0.6,
`max_tokens 24000`, 6 workers.

| outcome | n | meaning |
|---|---|---|
| resolved | **70** | patch applied, tests pass |
| unresolved | 28 | real patch, tests fail — model miss |
| patch did not apply | 1 | edited a *generated* file (`src/parser.c` from `parser.y`) |
| empty patch | 1 | burned `max_tokens` inside `<think>`, never emitted a tool call |

**70.0 %** of the full slice. Every non-result is accounted for, which is the point — see the
harness bug below, which turned 18 solved instances into apparent model failures.

Head-to-head against **qwen3.6-27B** on the same 28 Java/JS instances:

| | resolved | Δ |
|---|---|---|
| qwen3.6-27B | 21/28 = 75.0 % | — |
| **qwen3.8-27B** | **25/28 = 89.3 %** | +14.3 pp (wins 6, loses 2) |

3.6's two non-results are genuine, not harness artifacts (checked: zero discarded submissions,
zero shell noise in those images). One is a `ContextWindowExceededError` that 3.8 solves — the
262 k native window is a plausible cause. 3.8 reproduced 25/28 across two independent runs with
**zero instance-level variance** at temp 0.6. N=28 still means roughly ±10 pp, so treat the gap as
a strong signal rather than a precise figure.

### ⚠️ The harness bug that fakes model failures

mini-swe-agent's `_check_finished` (`environments/docker.py:140`) accepts the submit marker **only
on line 0**. Many `swebench/sweb.eval.x86_64.*` images ship a `/root/.bashrc` that sources a
nonexistent conda path, and `config/benchmarks/swebench.yaml` sets `BASH_ENV=/root/.bashrc`, so
every command prints an error *before* its real output. The submission is discarded, the agent
idles to `step_limit`, and a solved task is recorded as an **empty patch**.

Note the trigger is `BASH_ENV`, **not** the login shell — `swebench.yaml` already sets
`interpreter: ["bash","-c"]`, so overriding that in a config overlay is a **no-op**.

Measured on 18 affected instances, identical config otherwise:

| | before | after |
|---|---|---|
| exited `Submitted` | 0/18 | **18/18** |
| trajectory length | ~500 messages (step limit) | 25–103 messages |
| of those, resolved | — | **13** |

Thirteen solved instances — 13 pp on a 100-slice — were being scored as failures. A second
symptom: when a submission *does* register with noise behind the marker, the noise is stored
inside `model_patch` (12 of 28 predictions), which mostly still scores but adds an unexplained
patch-application risk.

Fix: skip shell-startup noise before the line-0 check, and again inside the payload.

```python
_STARTUP_NOISE = re.compile(r"^\S*(?:bashrc|bash_profile|profile|BASH_ENV):\s*line\s+\d+:")
```

**Any empty-patch count from an unpatched harness is not a model result.** Check before quoting
one — including your own historical numbers.

## References

- [unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4) — the checkpoint
- [vLLM recipe: Qwen3.8-27B](https://recipes.vllm.ai/Qwen/Qwen3.8-27B) — official flags
- [SGLang cookbook: Qwen3.8-27B](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-27B)
- [Qwen3.8-27B model card](https://huggingface.co/Qwen/Qwen3.8-27B) — sampling + YaRN guidance
