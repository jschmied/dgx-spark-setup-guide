# Runbook: Qwen3.8-27B-NVFP4 + MTP on DGX Spark, vLLM 0.27.1

One GB10 (sm_121a, 128 GB unified), CUDA 13, **stock `pip install vllm==0.27.1`** — no fork, no
custom image. Result: full **262 k** context, image input, tool-calling, and a **~2× decode
speed-up** from the checkpoint's built-in MTP head (11.4 → 25.5 tok/s on short prompts at n=3;
see §4 and §5 — the honest figure on *real* long-reasoning work is closer to 1.5×).

⚠️ **Read §9 before touching memory settings.** Raising `--gpu-memory-utilization` and
`--kv-cache-memory` together froze this box hard enough to need a powercycle, and cgroup limits
cannot prevent it.

**Which stack?** vLLM with MTP n=3, *unless* you are willing to run SGLang — there DSpark reaches
30.1 tok/s at 4.6 s TTFT, **+12 % throughput and −28 % TTFT against the vLLM path** (§7e). vLLM
cannot deliver DSpark: it skips the drafter's confidence head by design.

**Which checkpoint?** `unsloth/Qwen3.8-27B-NVFP4`, and §7c says why: it is the only one that
keeps `lm_head` and the last eight MLP layers out of 4-bit *and* ships calibrated KV scales,
while staying on the fast CUTLASS path. Eight alternatives were compared
structurally; three checkpoints were measured against BF16.

Measured on release day+1 (2026-08-15) against `unsloth/Qwen3.8-27B-NVFP4`; checkpoint
comparison and divergence measurements added 2026-08-18; the `lm_head` precision
experiment and the `humming-kernels` 0.1.12 upgrade added 2026-08-19 (§7f).

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

### ⚠️ Do not read weight-L2 as a quality number

**12 % is what the format costs, and it is not 1/16.** NVFP4 is E2M1 — *one* mantissa bit, so only
two values per octave (`1, 1.5` then `2, 3` then `4, 6`). Relative grid steps are 29–67 %, averaging
~40 %, and round-to-nearest over a ~40 % step gives an RMS relative error near `0.4/√12 ≈ 11.5 %`.
The correct intuition is "one mantissa bit", not "16 levels"; 1/16 would apply to *linear*
quantization. Simulating ideal E2M1 with group-16 `amax` scaling on the same tensors yields
**9.43 %**, so 12 % is the expected order of magnitude.

**But the shipped checkpoint is 12.59 % where ideal round-to-nearest is 9.43 %, and that gap is
informative in the opposite direction from what it looks like.** Ruled out: nibble order (the
alternative gives 141.69 %, i.e. garbage), scale derivation (stored scales are the ideal `amax/6`,
median ratio 0.9998), and FP8 storage of the scale (worth 0.05 pp). The cause is the rounding
itself — only **74.1 %** of weights carry the nearest E2M1 code; 13.1 % sit one grid step high,
12.0 % one step low, mean offset +0.012.

That symmetric, non-systematic pattern is the signature of a quantizer optimising **layer output
error** rather than weight error — the GPTQ/AWQ family deliberately accepts larger weight deviation
to compensate downstream. So a *better* quantizer scores *worse* on this metric.

**Consequence: weight-L2 is a valid format comparison (NVFP4 vs FP8, same method both sides) but
not a quality measure.** It says how far the numbers moved, not how much capability was lost — and
this checkpoint carries 12.59 % weight deviation while solving 70 of 100 real repository bugs.

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

The same checkpoint carrying 12.6 % weight deviation and 1.11 nats of distributional distance from
FP8 **solves 70 of 100 real repository bugs**. Neither number is a quality measure: the first is
inflated by a quantizer that optimises output error over weight error (see the warning above), and
the second measures distance from FP8, which is not ground truth for anything we care about.

**If you want more quality, change the format, not the calibration.** FP8 is 4.5× closer to the
original — and costs 30.9 GB instead of 22.6, and ran measurably slower here (225 s vs 136 s for the
same eight agent contexts). That is the real trade, now quantified rather than guessed.

> Reproduce: `/tmp/weight_compare.py` (CPU only, one tensor at a time). One trap — NVFP4's
> `weight_global_scale` is a **reciprocal** (amax-derived, value 6400). Multiplying instead of
> dividing yields weights of ~1e7 and a relative error of 4e7; the absurd magnitude is the tell.

## 7c. Which NVFP4 checkpoint — the field, measured

Fifty-odd NVFP4 conversions of this model exist. Almost all differ in ways that do not matter,
and three ways that do. Everything below was read from the live HuggingFace API (config +
tensor index) and, where a number is given, measured on this box.

### The field

A module counts as 4-bit if it carries `weight_scale_2` (modelopt) or `weight_packed`
(compressed-tensors), 8-bit if it has `weight_scale` and neither, BF16 if it has no scale.
That classification is format-agnostic; the tensor *name* is not — in modelopt format the packed
4-bit tensor is also just called `weight`.

| checkpoint | scheme | format | size | MLP 4/8/bf | attn | GDN | lm_head | KV scales | calibration |
|---|---|---|---|---|---|---|---|---|---|
| **unsloth/Qwen3.8-27B-NVFP4** | W4A4 | comp.-tensors | 23.4 GB | 168/**24**/0 | FP8 | FP8 | **FP8** | **32** ✓ | undocumented |
| RadixArk/Qwen3.8-27B-NVFP4 | W4A4 | modelopt | 21.9 GB | 192/0/0 | FP8 | FP8 | NVFP4 | 0 | `cnn_dailymail` 1024 × **512** |
| Inferact/Qwen3.8-27B-NVFP4 | W4A4 | modelopt | 26.4 GB | 192/0/0 | **4-bit** | 48 4-bit / 96 bf16 | BF16 | 0 | undocumented |
| sakamakismile/…-MTP-NVFP4 | W4A4 | comp.-tensors | 20.6 GB | 192/0/0 | **4-bit** | **4-bit** | BF16 | 0 | `neuralmagic` 32 × 8192 |
| gittensor/…-NVFP4-RTX5090 | W4A4 | modelopt | 20.6 GB | 192/0/0 | **4-bit** | **4-bit** | BF16 | 0 | 128 image-text samples |
| huginnfork/…-NVFP4A16 | W4A16 | comp.-tensors | **31.0 GB** | 192/0/0 | **BF16** | **BF16** | BF16 | 0 | n/a (weights only) |
| r0b0tlab/…-NVFP4-MTP-sm121 | W4A16 | modelopt | 21.9 GB | 192/0/0 | FP8 | FP8 | NVFP4 | 0 | `cnn_dailymail` + Nemotron (code/math) |
| *nvidia/Qwen3.6-27B-NVFP4* (recipe origin) | W4A16 | modelopt | 21.9 GB | 192/0/0 | FP8 | FP8 | NVFP4 | 0 | `cnn_dailymail` + Nemotron |
| ours (rebuild, not published) | W4A16 | comp.-tensors | 23.8 GB | 192/0/0 | FP8 | FP8 | FP8→BF16 | 0 | none (dynamic) |

Excluded: abliterated/uncensored derivatives, GGUF and MLX conversions.

### What actually decides it

**1. The scheme decides the kernel, and the kernel decides latency.** W4A4 runs CUTLASS with
native FP4 tensor cores; W4A16 falls back to Marlin, which dequantizes to BF16 before the GEMM.
Measured paired over 24 real agent contexts, both with MTP n=3 at util 0.60, nothing else on the box:

| | tok/s median | TTFT | MTP acceptance |
|---|---|---|---|
| W4A16 (Marlin) | 22.2 | 10.0 s | 75.4 % |
| W4A4 (CUTLASS) | **26.6** | **6.4 s** | 72.7 % |

−14.8 % median throughput, **W4A16 faster in 0 of 20 prompts**. The prefill penalty scales with
context: **+3.0 s TTFT below 12k, +8.7 s at ≥12k**. Note the acceptance goes the *other* way, so
the deficit is per-step compute, not worse speculation. SGLang's cookbook greys NVFP4 out on H200
for exactly this reason — no FP4 tensor cores means the Marlin fallback, and they do not offer it
as an operating point.

**2. `lm_head` precision is the largest single quality lever found.** Teacher-forced against a
BF16 reference over 20 agent contexts (see §7d), top-1 agreement:

| | lm_head distance to BF16 | Δtop1 vs BF16 |
|---|---|---|
| ours / Unsloth (FP8 head) | 3.05 % | −1.11 / −1.53 pp |
| r0b0tlab (NVFP4 head) | 10.29 % | −1.69 pp |

r0b0tlab's MLP weights are within **1.1–1.3 %** of ours — both are data-free round-to-nearest from
the same BF16 source, and both land at 9.46 % from it. It nevertheless loses the most, and the
ranking follows `lm_head` precision exactly. One tensor, ~0.6 GB, measurable at the model output
because that tensor *is* the output.

**Unsloth protects it at FP8; NVIDIA's recipe and its derivatives quantize it to NVFP4.** The data
supports Unsloth.

**3. The FP8 KV cache needs calibrated scales, and almost nobody ships them.** We serve
`--kv-cache-dtype fp8`. Unsloth ships 32 `k_scale`/`v_scale`; every other checkpoint here ships
none and silently falls back to a scaling factor of 1.0. RadixArk states the consequence in its own
qualification file: *"no scaling factors provided … may lead to less accurate results"*.

### Calibration is not the axis it looks like

NVFP4 **weight** scales are `amax` per group of 16 — data-free by construction. No calibration set
can change them, which is why our rebuild (no calibration at all) and r0b0tlab's (cnn_dailymail +
Nemotron) land on identical weights to two decimals. Calibration only touches **activation** scales,
and only where they are static: 168 `input_global_scale` in Unsloth, 401 `input_scale` in RadixArk,
**zero** in a W4A16 build, which keeps activations in BF16 and has nothing to calibrate.

Where it is published at all, it is news prose: RadixArk uses 1024 CNN/DailyMail articles at
**512 tokens** for a model people run at 8–30k on code. NVIDIA's own documented path pairs that
corpus with Nemotron-Post-Training-v2 including code and math splits — RadixArk kept the news half
and dropped the rest. Unsloth publishes nothing.

### Verdict

**Use `unsloth/Qwen3.8-27B-NVFP4`.** It is not the smallest, not the newest, and not the one the
SGLang cookbook nominates — it is the one whose four deviations from the reference recipe all point
the same way:

- **W4A4** → the CUTLASS path, 15 % more decode and 3.6 s less TTFT than the Marlin alternative
- **`lm_head` at FP8** → 3.05 % from BF16 instead of 10.29 %, the largest lever we found
- **MLP layers 56–63 at FP8** → the standard guard nearest the output, which no other checkpoint here applies
- **32 calibrated KV scales** → the only checkpoint that does not fall back to 1.0 under `--kv-cache-dtype fp8`

That is a coherent set of conservative choices by someone who understood which parts of the network
tolerate four bits, not a difference in quantizer craft — the weights themselves are the same
round-to-nearest everyone else does.

**Runner-up, and when to pick it:** our W4A16 rebuild preserves 0.42 pp more top-1 fidelity to BF16
(paired t=2.90, better in 15/20). If output fidelity mattered more than latency — offline scoring,
batch evaluation, a distillation reference — that is the trade. For interactive agent work it is
not: 0.42 pp does not buy back +8.7 s of TTFT per turn.

**Suspect, but NOT measured here:** the checkpoints that push attention and GDN to 4 bit as well
(sakamakismile, gittensor, Inferact). The reasoning is that with 48 of 64 layers being gated
DeltaNet, that is the recurrent state path, where an error persists across the sequence instead of
being re-derived each token — so it is the last place to spend bits. **That is an inference from
architecture, not a measurement**, and nobody publishing those checkpoints has measured it either.
Treat it as a reason to test before adopting, not as a verdict. Same status for
`huginnfork/…-NVFP4A16`: it pays the Marlin latency penalty *and* carries 9 GB of unquantized
attention at 31 GB, which looks like the worst of both — untested.

**What would settle it:** three of the nine were measured against BF16 (Unsloth, our W4A16 rebuild,
r0b0tlab). The two worth adding are `RadixArk` — W4A4 *with* an NVFP4 `lm_head`, the missing cell
that would turn the `lm_head` finding from a correlation into a 2×2 — and one all-4-bit build to
test the GDN claim above. Roughly 42 GB and two hours of download.

### What this verdict does not say

Everything above ranks **fidelity to BF16**, not capability. The reference continuation was
generated *by* BF16, so a model that mimics BF16 wins by construction — including where BF16 is
wrong. Capability would need task benchmarks, and those cannot resolve it: a paired McNemar test
over 100 SWE-bench instances detects ~12 pp, while the differences here are 0.4 pp. The gap is a
factor of thirty, and no amount of running the existing harness closes it.

## 7d. Measuring quantization damage without a task benchmark

SWE-bench cannot settle a 0.4 pp question (§7c). What can, cheaply, is teacher-forced logprob
divergence against the BF16 original — but only if three design traps are avoided, each of which
cost a run here before it produced a number.

**Let BF16 generate once, score everyone on that.** If each candidate generates freely, the texts
diverge after a handful of tokens and every later difference measures drift, not quantization. BF16
produces one long greedy continuation per prompt; every candidate is then forced through that exact
token sequence.

**Pass token IDs, not text.** The same string can retokenize differently. Tokenize the prompt once
and the continuation once, then hand the identical ID list to all candidates. Otherwise you compare
tokenizer boundaries.

**Read both numbers out of one pass.** vLLM's `prompt_logprobs` returns, for every position, the
forced token's logprob **and its rank**. That gives divergence (ΔNLL, i.e. the log perplexity ratio)
and top-1 agreement — how often the candidate would itself have picked BF16's token — without a
second request. For agentic work the rank is the more meaningful of the two: one wrong token
derails a tool call.

```bash
# reference, once
python3 divergence.py --mode ref   --label bf16 --gen-tokens 768
# each candidate, against that same reference
python3 divergence.py --mode score --label ours-w4a16
```

### The self-check, and why it is not 100 %

BF16 scoring its own generation is the control: if the harness is sound, the model must agree with
itself. It came out at **96.86 %**, not 100 %, and the reason is worth knowing before you conclude
your pipeline is broken.

`ignore_eos` is needed to get long continuations — greedy generation on agent prompts stops after a
median of **144 tokens**, far too short to see whether error accumulates with depth; with it, the
median is 747. But vLLM **masks EOS while generating and not while scoring**, so at every position
where the model wanted to stop, EOS is the argmax and the forced token drops to rank 2. The
per-quarter profile shows exactly that signature: **98.2 → 97.2 → 97.4 → 94.8 %**.

So 100 % is unreachable by construction once `ignore_eos` is set. Gate the run against the measured
self-score, not against a number you assumed.

### Score the continuation, not the context

`prompt_logprobs` covers the whole prompt, so the 8–30k tokens of real agent context come along for
free — file contents, diffs, tool output. It is tempting to treat that as the better metric because
the token count is 250,000 against 14,000.

**It is not.** The independent unit is the prompt, not the token: n = 20 either way, and the spread
between prompts is large. The context metric duly produced an impossible result — it ranked a 4-bit
model **better than its own BF16 source** (16/20 prompts, t = −2.72). Report it if you like, but do
not decide on it.

BF16 context NLL 5.53 at 52.7 % top-1, against 0.32 and 96.9 % on its own continuation, is itself
worth noting: the model predicts what it writes almost perfectly and real input text only half the
time. That is a property of agent traffic, not of quantization.

### Cost and the memory trap

Roughly 4 minutes per candidate at 20 prompts once the reference exists (~35 min at BF16 speed).
The whole comparison fits in an afternoon.

One hazard specific to this box: `prompt_logprobs` materializes logprobs for every position, and at
31,681 positions × 248,320 vocabulary that is a **31.5 GB** tensor. It is affordable at util 0.40
(38.6 s for the largest case, full ranks available) and fatal at util 0.80 with 55.6 GB of BF16
weights resident — the engine dies with `RPC call to sample_tokens timed out`. Serve the BF16
reference at **util 0.62** and raise `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS`; see §9 for why sequencing
two large models is the actual OOM trigger.

## 7e. DSpark works — on SGLang. vLLM's implementation is the reason it looks bad here.

Same checkpoint, same drafter, same box, same 24 agent contexts. Only the stack changes:

| | tok/s | TTFT | vs its own MTP |
|---|---|---|---|
| vLLM MTP n=3 | 26.6 | 6.4 s | — |
| vLLM DSpark k=7 | 20.9 | 6.4 s | **−21.4 %** |
| SGLang MTP (EAGLE 3/1/4) | 22.5 | 4.9 s | — |
| **SGLang DSpark block-7** | **30.1** | **4.6 s** | **+33.7 %**, paired, 18/20 |

Against the production path here (vLLM + MTP n=3): **+12.3 % throughput and TTFT 6.4 → 4.6 s**,
paired, faster in 14 of 20 — the fastest figure measured for this model on this box.

**A stack switch on its own loses.** SGLang's MTP is 22.5 against vLLM's 26.6; vLLM decodes 18 %
faster. The entire gain comes from DSpark, which vLLM cannot deliver. The choice is *vLLM with MTP*
or *SGLang with DSpark* — not "which server is better".

### Why vLLM cannot deliver it

vLLM ships a real DSpark speculator — block drafting in one parallel pass, Markov head, all 62
drafter tensors matching by name. Two pieces are missing, and the first is stated in its own source:

```python
# vllm/model_executor/models/qwen3_dspark.py:185
# confidence_head is not wired into inference yet; skip its weights.
skip_substrs = ["mask_embedding", "confidence_head"]
```

The drafter ships `enable_confidence_head: true`, `confidence_head_alpha: 1.0` and the weights;
vLLM skips loading them. SGLang drives exactly this path through
`--speculative-accept-threshold-acc`. The second gap is `--enable-linear-replayssm-spec`, which
SGLang enables on precisely rtx5090 / rtx6000 / dgx-spark to hold GDN state across speculation;
vLLM has no equivalent, and with 48 of 64 layers being gated DeltaNet a rejected block has to roll
that state back.

The symptom is acceptance that is bad **from position 0** — 61–63 % against MTP's 87 % — which no
downstream filter can repair, and which is flat across block size:

```
DSpark n=6   26.5 %  P0=63 P1=41 P2=25 P3=14 P4=9  P5=6            20.6 tok/s
DSpark n=7   23.1 %  P0=61 P1=38 P2=25 P3=16 P4=11 P5=7  P6=4      20.9 tok/s
DSpark n=8   22.4 %  P0=63 P1=43 P2=27 P3=17 P4=12 P5=8  P6=6 P7=3 20.8 tok/s
MTP n=3      72.7 %  P0=87 P1=77 P2=67                             26.6 tok/s
```

Three explanations died in the process, and they are worth recording because each looked solid:
matching MiaAI-Lab's `--speculative-num-draft-tokens 8` (block 7 + anchor) changed nothing; block-5
changed nothing; and the target-layer resolution is **fine** — `eagle3_utils.py` has a five-step
fallback whose second step reads `dflash_config.target_layer_ids` and applies the +1 conversion.
Reading only `speculative.py`'s Gemma4-only branch made it look broken.

### Reproducing on GB10 — two traps

**GPU access in docker is CDI-only.** There is no `nvidia` runtime on this box:

```bash
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
docker run --device nvidia.com/gpu=all ... lmsysorg/sglang:qwen38-27b
```

**The drafter needs its original `config.json`.** Ours is patched to `Qwen3DSparkModel` for vLLM's
registry; SGLang wants the vendor's `DSparkDraftModel`. And **copy** the drafter rather than
symlinking it — absolute symlinks into `/opt/llm/models` dangle inside the container, where the
mount is `/models`.

Script: `/opt/llm/sglang-bench.sh`. Note also that `unsloth/Qwen3.8-27B-NVFP4` **does** load under
SGLang, contrary to the note on its model card.

**Not measured: quality.** The divergence harness in §7d relies on vLLM's `prompt_logprobs`.
Speculation verifies against the same target model, but `--num-continuous-decode-steps 2` and the
different prefill handling are unchecked.

## 7f. Quantizing `lm_head` — four defects deep, and the verdict flipped

§7c leaves `lm_head` in high precision because every checkpoint that quantizes it scores worse.
That result does not say *why*. Two candidates: the head's **weights** lose too much, or its
**activations** do. They are separable — FP8 weights can be paired with 16-bit activations
(`W8A16`) instead of 8-bit ones (`W8A8`) — so we built the third variant and measured all three.

Getting there took working around three independent defects. None of them is documented anywhere
we could find, and the path is blocked without all three.

### Defect 1+2 — vLLM: `ParallelLMHead` is not a `LinearBase`

Routing `lm_head` through a compressed-tensors FP8 target group hits `CompressedTensorsW8A16Fp8`,
which prepares a Humming layer. `prepare_humming_layer` reads `LinearBase`-only attributes:

```
AttributeError: 'ParallelLMHead' object has no attribute 'output_partition_sizes'
… then, once that is guarded …
AttributeError: 'ParallelLMHead' object has no attribute 'has_bias'
```

`ParallelLMHead` extends `VocabParallelEmbedding`, not `LinearBase`. The same class of bug was
already fixed once for `input_size_per_partition` (vllm#46420), and **that fix's comment names
`ParallelLMHead` explicitly** — two sibling attributes were missed.

Already reported: **[vllm#48856](https://github.com/vllm-project/vllm/pull/48856)**, open since
2026-07-16, conflict-blocked since 2026-08-08. Do not open another — vLLM's `AGENTS.md` says so
directly ("If an open PR already addresses the same fix, do not open another"), and a second,
independently-authored PR (vllm#52543) was self-closed for exactly that reason plus the
AI-contribution bar. Backport the PR's hunks locally instead; they are ~26 lines in one file.

### Defect 3 — humming-kernels on unified memory: NVML has no memory clock

With the layer prepared, the worker dies inside the kernel heuristics:

```
humming/utils/device.py:33  nvmlDeviceGetMaxClockInfo(handle, NVML_CLOCK_MEM)
→ pynvml.NVMLError_NotSupported
```

On GB10 both inputs to Humming's bandwidth formula are unavailable — the memory clock raises, and
`nvmlDeviceGetMemoryBusWidth` returns **0**, so `(clock × 2 × width) / 8 / 1000` could not be
evaluated even if the clock worked. The `except` above it catches only `FunctionNotFound`.

**Fixed by the vendor in `humming-kernels` 0.1.12** — same 273.0 GB/s constant, keyed on compute
capability `(12, 1)`. vLLM 0.27.1 pins `==0.1.10`; vLLM main already pins 0.1.12.

```bash
# the pin is too tight, not binding — verified on this box
pip install --no-deps --force-reinstall "humming-kernels==0.1.12"
```

Upgrading is safe here and worth doing, but **not for speed**: it also introduces
`Sm120Heuristics` for sm_120/121 (0.1.10 falls back to `Sm89Heuristics`, i.e. Ada), and that
changed nothing measurable — **−0.3 %**. Production NVFP4 + MTP n=3 measured **26.4 tok/s under
0.1.12 against 26.6 under 0.1.10 (−0.8 %, noise)**. Budget ~7 minutes for the first start after
the upgrade: it invalidates the JIT cache. Keep `MAX_JOBS=2` / `FLASHINFER_NVCC_THREADS=1` while
it recompiles — unbounded ninja fan-out is what killed this box once before (§9).

### Defect 4 — the humming fix lives per-venv, and a side branch will not have it

A trap for anyone running more than one vLLM tree. The `ParallelLMHead` fix and the humming
upgrade are edits to *site-packages*, so they exist only in the venv they were applied to. Our
0.27.1 had humming **0.1.12** and the patch; the #52816 branch venv still had **0.1.10**, whose
package layout has no `humming.transform` at all:

```
ModuleNotFoundError: No module named 'humming.transform'
```

Both FP8-A16-head cells died on that during the DFlash2 sweep, and the error names neither humming's
version nor the venv — it looks like a broken checkpoint. Copying the package across is safe here
(both venvs on torch 2.13.0+cu130, and humming ships only pre-built launchers under
`_native/aarch64`, no torch-ABI extensions); `pip install -U` is not, because it moves vLLM's pins.

The `ParallelLMHead` patch has **three** sites, not two — an earlier two-site version is still
floating around in our scripts and fails at the third:

```
AttributeError: 'ParallelLMHead' object has no attribute 'has_bias'
```

Take the fix from a venv where it works (`diff` against `*.orig-vor-humming-fix`), not from a
script that documents an earlier attempt.

### The one head quantized further than ours — weights *and* activations in 4-bit

`RadixArk/Qwen3.8-27B-NVFP4` goes a step past every head discussed above. From its
`hf_quant_config.json`, `lm_head` sits in the NVFP4 group with **`W=float4, A=float4`** — weights
*and* activations in 4-bit, where our NVFP4 heads leave activations at 16-bit. The tensor set gives
it away: `weight` + `weight_scale` + `weight_scale_2` **+ `input_scale`**; that last one is what our
builds do not have.

It is also the only checkpoint here whose MLP activations are **static** (`dynamic: false`), against
Unsloth's `dynamic: local`.

Measured 2026-08-20, 20 real agent contexts:

| | decode | TTFT | accept length | s/turn @130 tok |
| :--- | ---: | ---: | ---: | ---: |
| MTP n=3 | 30.1 tok/s | 4.6 s | 3.29 | 8.9 s |
| **DFlash2 n=7** | **40.3 tok/s** | **4.5 s** | **4.32** | **7.7 s** |

Both are records on this page. The TTFT is the lowest measured anywhere here — 1.1 s below the next
body — and 7.7 s per agent turn beats the previous best of 9.0 s by 14 %. It is the only cell that
leads on prefill *and* decode at once.

**And it is the least faithful cell measured**, at **−2.19 pp** against BF16, worse than the
previous floor of −2.06. That is the same property from the other side: fewest bytes moved, least
of BF16 retained. Its head is where the map's speed axis and its quality axis are most directly the
same knob.

Two consequences worth carrying:

- **It needs the quantized-head guard removed to draft at all.** On stock vLLM it refuses DFlash2 —
  it is precisely the case §7g describes.
- **The "dominated" column was not dominated.** Static W4A4 activations were treated here as an
  uninteresting corner; they save the per-token scale computation, and that is the most plausible
  source of the prefill lead. It went unmeasured because the label said not to bother.

### The measurement

Same NVFP4 body, three heads. Divergence is teacher-forced top-1 agreement against the BF16
original (§7d), throughput is MTP n=3 over 20 real agent contexts.

| `lm_head` | Δtop-1 vs BF16 | decode | TTFT |
| :--- | ---: | ---: | ---: |
| BF16 (as shipped) | **−1.11 pp** | 22.2 tok/s | — |
| FP8 `W8A8` | −1.33 pp | 26.4 tok/s | — |
| FP8 `W8A16` | −1.29 pp | **27.0 tok/s** | 9.9 s |

**The activation precision is not the loss source.** `W8A16` and `W8A8` differ by **0.04 pp** —
noise. Both lose ~0.2 pp against a BF16 head, and that loss sits in the head's **weights**. The
16-bit activation path buys nothing measurable in quality, and its speed edge over `W8A8`
(+0.6 tok/s) is inside the same noise band while costing a longer prefill.

**Verdict under MTP: do not do this.** `W8A16` needs a patched vLLM to load at all; `W8A8` needs
nothing and lands in the same place. If you want the 0.2 pp back, keep the head in BF16 and pay
4.8 tok/s.

> **⚠️ This verdict does not survive DFlash2.** It was reached with MTP n=3, where the head runs
> three times per step. Under DFlash2 n=7 it runs seven times, and the ordering inverts: the five
> fastest of eleven measured cells all have quantized heads, and the four unquantized 2.54 GB rows
> are the slowest (§7g). The most aggressively quantized head on this page — RadixArk's NVFP4 with
> 4-bit activations — is the fastest cell measured anywhere here. The quality cost is real and gets
> worse (−2.19 pp), but "not worth it" was a statement about a drafter, not about heads.

### ⚠️ Never compare across speculation settings

Our first reading of the `W8A16` cell had it at **11.6 tok/s** — "half the speed of everything
else". It was measured with `VLLM_QWEN38_SPEC=off` while the comparison grid ran MTP n=3. The
serve script defaults to `mtp`, so the grid scripts never set it and the difference is invisible
in the script source; only the ratio gives it away (27.0 / 11.6 ≈ 2.3, the MTP speed-up). The
unspeculated baselines are **11.0 tok/s** (production NVFP4) and **11.6** (`W8A16`) — consistent
with each other, and not comparable to any number elsewhere in this document.

## 7g. DFlash2 — the fastest thing measured here, and why it is not the recommendation yet

`z-lab/Qwen3.8-27B-DFlash2` is a block-diffusion drafter: it predicts a whole block in one pass
and traces a path through per-position candidates. Both engines have it in flight, neither has
shipped it. Measured 2026-08-19 on the branch of
[vllm#52816](https://github.com/vllm-project/vllm/pull/52816).

### The unquantized-lm_head requirement is not real — it was a guard

First attempt against the NVFP4 production checkpoint dies at startup:

```
ValueError: DFlash2 requires an unquantized target LM head for candidate TopK.
```

Two things are wrong with that, one small and one large.

**The small one:** the guard was over-strict even on its own terms. `compute_candidates` tested
`isinstance(quant_method, UnquantizedEmbeddingMethod)`, and a checkpoint that *leaves* `lm_head`
unquantized while carrying a `quantization_config` gets `UnquantizedLinearMethod` instead — so the
check only passed on models with no quant config at all. Fixed in
[vllm#52883](https://github.com/vllm-project/vllm/pull/52883); with it our W4A16 build (`lm_head`
in the ignore list) loads.

**The large one:** the requirement itself is gratuitous. The line the guard protects is

```python
logits = self.lm_head.quant_method.apply(self.lm_head, hidden_states, bias=None)
```

`quant_method.apply()` is the generic interface every scheme implements, and the padding and TP
handling underneath it are scheme-agnostic. Nothing about candidate TopK needs raw weights.
Deleting the eight-line guard is enough — no dequantization, the head stays at its packed size.
A [public DGX-Spark recipe](https://github.com/Weschera/Qwen3.8-27B-NVFP4-DFlash2-DGX-Spark)
reached the same conclusion on SGLang, where the plumbing had to be added; in vLLM it was already
correct and only the refusal stood in the way.

Measured consequence, 2026-08-19: **seven cells with engine-quantized heads draft fine, and the five
fastest of eleven are all among them** — see the sweep below. The guard was not describing the method.

Take the NaN fix too — [`31840cf3`](https://github.com/vllm-project/vllm/commit/31840cf3ead3632f3c99db4a24e4aba39ad54ef6),
"patches for probabilistic drafting safety". Two testers reported `!!!!!` garbage without it. With
it we saw none (0.0 % exclamation marks across 20 completions).

### Throughput: +74.5 % over MTP on the same target

Target `qwen38-27b-w4a16`, 20 real agent contexts, same branch for both arms — the MTP figures
elsewhere in this document are from 0.27.1 and are *not* comparable.

| | decode | TTFT | acceptance |
| :--- | ---: | ---: | :--- |
| MTP n=3 | 20.8 tok/s | 10.0 s | — |
| **DFlash2 n=7** | **36.3 tok/s** | 9.5 s | 43.6 %, length 4.05 |

Per position: `P0 79 % · P1 62 % · P2 47 % · P3 39 % · P4 33 % · P5 26 % · P6 20 %`.

Length 4.05 sits just under z-lab's H200/BF16 range (4.10–5.46) — expected against a 4-bit target.
It also **exceeds what MTP n=3 can reach by construction** (3 drafts + 1 bonus = 4.00 ceiling), so
the win comes from depth, not from a better hit rate per position.

### ⚠️ But it deviates from unspeculated output more than MTP does

Speculative decoding is advertised as lossless. It is not, here — for *either* method. Six prompts
at temperature 0, compared against the same target served with speculation off:

| | identical to reference | mean similarity |
| :--- | ---: | ---: |
| MTP n=3 | 0/4 | 0.722 |
| DFlash2 n=7 | 0/4 | **0.517** |

(Two of the six prompts return 500 tokens of thinking and no content under *every* configuration —
a `max_tokens` artifact, excluded. Including them inverts the ranking, which is how we first
misread this.)

The box itself is deterministic: two consecutive unspeculated runs were **6/6 byte-identical**, so
this is not measurement noise. Divergence starts early — at character 52 and 109 in two cases, on
ordinary words. One prompt is decisive: MTP scores **0.957** against the reference where DFlash2
scores **0.162**. MTP demonstrates that near-fidelity is attainable on that prompt, so DFlash2's
departure there cannot be dismissed as floating-point tie-flipping.

**Then the same test on the production server dissolved most of that signal.** Send one prompt at
temperature 0 three times alone: byte-identical every time (1719 chars). Send *the same prompt*
concurrently with three unrelated filler requests: **1723 chars, diverging at character 421**,
similarity 0.873 — `"Remove and return the item at the top of the stack."` becomes
`"Remove and return the top item from the stack."` Same model, same method, same seed, same
prompt. Only the batch shape changed.

That is the mechanism: the target's forward pass runs over a different number of positions, so
GEMM shapes change, so reduction order changes, so the last bits of the logits change, so a
near-tie argmax flips — and one flipped token rewrites everything after it. Speculation triggers
it because verifying K+1 positions is a different shape than decoding one; concurrency triggers it
for exactly the same reason, with no speculation involved at all.

That reframes the numbers but does not dismiss them, and the depth-matched rerun settles it. Held
at **identical verification width**:

| | mean similarity to unspeculated reference |
| :--- | ---: |
| MTP n=3 | **0.722** |
| DFlash2 n=3 | 0.469 |
| DFlash2 n=7 | 0.517 |

Depth does *not* explain the gap — DFlash2 at n=3 is if anything slightly further out than at n=7,
and both sit well below MTP at the same depth. Batch shape sets the floor for how close anything
can be expected to land (0.873 from concurrency alone), and MTP at 0.722 is much nearer that floor
than DFlash2 at 0.469. Under otherwise identical conditions, **DFlash2 departs from unspeculated
output more than MTP does**, and that is a property of the drafter, not of the harness.

n is 4 prompts. What this does not yet separate: whether DFlash2 is *nondeterministic* (its
candidate selector walks probabilistically, so two DFlash2 runs may differ from each other) or
*systematically biased* (consistently deviating in one direction). Running DFlash2 twice on the
same prompts would tell them apart, and is the obvious next step.

The reviewer's own reservation on the PR still stands as an open question — *"I'm still not entirely convinced of the
correctness of the probabilistic code here … we'll probably need to do some heavy full-scale evals
to make sure it's not biasing or corrupting more subtly."* Our n is 4 prompts: a signal, not a
verdict.

**What does survive, and matters more:** at temperature 0 this server is reproducible only at
batch size 1. Under concurrent load the same request returns different text — in production, today,
with no speculation flag involved. If you have been treating `temperature: 0` as a reproducibility
guarantee for evals or regression tests, it is not one here unless you also pin concurrency to 1.

### ⚠️ n=7 is not the optimum — that was an artefact of how the sweep was aggregated

An earlier version of this section concluded that DFlash2's published block size of 7 was right
here. It rested on comparing **unpaired medians on paired data**: 36.5 tok/s at n=7 against 36.4 at
n=11, a 0.2 difference. Re-read the same four runs correctly and the answer flips.

| n | block | median | mean | **throughput** Σtok/Σt | corpus wall time |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 4 | 25.3 | 24.9 | 23.8 | 135 s |
| 7 | 8 | **36.5** | 36.0 | 31.4 | 103 s |
| **11** | 12 | 36.4 | **38.5** | **33.8** | **91 s** |
| 15 | 16 | 35.5 | 37.3 | 32.1 | 97 s |

Three readings, two answers. The median picks n=7 by 0.2; the mean picks n=11 by 1.1; total tokens
over total decode time — the thing a single-stream user actually experiences — picks n=11 by 1.7,
and finishes the same twenty contexts in **12 % less time**.

Paired, which is the comparison the data supports: **n=11 beats n=7 in 11 of 20 contexts**, median
difference **+1.7 tok/s**, range −6.6 to +15.7. Note that the median of the per-context differences
(+1.7, favouring n=11) has the opposite sign to the difference of the medians (−0.1, favouring
n=7). The gain lives in a handful of high-acceptance contexts, and the median is precisely the
statistic that discards them.

Depth costs nothing in prefill: TTFT is 9.6–9.8 s across all four settings.

### The optimum is a property of the workload, not of the drafter

Splitting the same twenty contexts by how predictable their continuations are — classified by MTP's
per-context rate on the same target, so the split does not come from the numbers being compared:

| | n=3 | n=7 | n=11 | n=15 |
| :--- | ---: | ---: | ---: | ---: |
| less predictable (MTP 19.9 tok/s) | 24.5 | 36.4 | **38.7** | 36.2 |
| more predictable (MTP 23.4 tok/s) | 25.2 | 35.6 | 38.3 | **38.5** |

The optimum moves outward as acceptance rises: 11 on the harder half, 15 on the easier one. Deeper
drafts only pay when the drafter keeps hitting, so the right depth depends on what the model is
being asked to do — editing existing code accepts far more readily than writing new code.

This also explains a third-party report of `k=14` working well on this hardware for edit-heavy
work while our sweep said the tail collapses past 11. Both are right about their own corpus.

**Consequence for everything else in this document:** a single median over a mixed corpus hides a
factor of two. Across twenty contexts one target ranges from 26.4 to 52.5 tok/s, and that spread
correlates with neither context length (−0.10) nor completion length (−0.16) nor TTFT (−0.10) — it
is a property of the content. Any tuning done against the median is tuning against the middle of a
distribution whose ends behave differently.

### With the guard gone: eleven cells, and the five fastest all have quantized heads

Same 20 agent contexts, same branch, DFlash2 n=7 throughout, FP32 top-k ranking active on all of
them. Head sizes: BF16 / FP8-wide 2.54 GB, FP8-narrow 1.27 GB, NVFP4 0.64 GB. Sorted by decode.

| head | body | decode | TTFT | accept length |
| :--- | :--- | ---: | ---: | ---: |
| FP8 · A16 · 1.27 GB | ours W4A16 | **41.7** | 9.6 s | 4.23 |
| FP8 · A8 · 1.27 GB | ours W4A16 | 41.2 | 9.6 s | **4.54** |
| NVFP4 · 0.64 GB | r0b0tlab W4A16 | 40.6 | 8.5 s | 4.23 |
| NVFP4 · 0.64 GB | ours W4A16 | 39.1 | 9.6 s | 4.06 |
| NVFP4 · 0.64 GB | unsloth W4A4 | 38.6 | 5.6 s | 4.26 |
| FP8 · A16 *wide* 2.54 GB | r0b0tlab W4A16 | 37.8 | 8.5 s | — |
| BF16 · 2.54 GB | ours W4A16 | 37.5 | 9.6 s | — |
| FP8 · A16 · 1.27 GB | unsloth W4A4 | 36.9 | 5.6 s | 4.46 |
| FP8 · A8 · 1.27 GB | unsloth W4A4 | 36.5 | 5.6 s | 4.38 |
| FP8 · A16 *wide* 2.54 GB | unsloth W4A4 | 35.2 | 5.6 s | — |
| BF16 · 2.54 GB | unsloth W4A4 | 34.7 | 5.6 s | — |

Seven cells carry an **engine-quantized** head (1.27 / 0.64 GB); the four 2.54 GB rows are stored
at BF16 width, so the engine sees them as unquantized — they are exactly the ones stock vLLM would
run at all.

**The five fastest all have quantized heads**, and both slowest are unquantized. The ordering is
not clean beyond that: a wide head on the fast r0b0tlab body (37.8) beats a narrow head on the
slower Unsloth body (36.9). Head class and body are not separable by simply reading down this
column — see the decomposition below.

TTFT is a property of the body alone and does not move with the head: 5.6 s on the W4A4 body,
9.6 s on ours, 8.5 s on r0b0tlab's, whichever head sits on it.

### What head bytes actually buy: steps/s, not tok/s

Under speculation the decode rate factors as **steps/s × accept length**. Head bytes govern the
first term, not the product — and reading them as if they governed the product inverts conclusions.
Four cells measured under identical conditions:

| body | head | tok/s | accept length | steps/s |
| :--- | :--- | ---: | ---: | ---: |
| ours W4A16 | FP8 A8 · 1.27 GB | **41.2** | 4.54 | 9.08 |
| ours W4A16 | NVFP4 · 0.64 GB | 39.1 | 4.06 | **9.63** |
| unsloth W4A4 | FP8 A8 · 1.27 GB | 36.5 | 4.38 | 8.33 |
| unsloth W4A4 | NVFP4 · 0.64 GB | **38.6** | 4.26 | 9.06 |

Halving the head buys steps on **both** bodies, +0.56 and +0.73 steps/s for the same 0.63 GB — the
byte model, holding exactly where it should. What differs is the accept length it costs: −0.48 on
our body against −0.12 on Unsloth's. So the same head swap is worth −2.1 tok/s on one body and
+2.1 on the other.

Two caveats that survive every pairing:

- **Accept length is not a property of the head.** The *same* 0.64 GB NVFP4 head accepts 4.26 on
  one body and 4.06 on the other; the same FP8 A16 head, 4.46 and 4.23. The drafter's hit rate
  depends on the target's weights. The two-axis reading — head owns decode, body owns prefill —
  does not cover this.
- **Head activation precision is a kernel choice, not an axis.** A8 → A16 at *identical* head
  size moves steps/s by −0.05 on one body and +0.78 on the other. W8A8 and Humming's W8A16 are
  different kernels whose relative cost depends on what else competes for bandwidth.

### ⚠️ FP32 top-k ranking does not reproduce here

The DGX-Spark recipe credits ranking the candidate top-k in FP32 rather than the head's own
precision with lifting accept length 3.01 → 3.24, about 8 %. Tested both ways on the same NVFP4
head, same 20 contexts:

| | accept length | decode |
| :--- | ---: | ---: |
| without FP32 ranking | 4.26 | 38.6 tok/s |
| with FP32 ranking | 4.26 | 38.6 tok/s |

**No effect.** On the FP8 production head, +0.1 %. For scale: repeat runs of the same cell
reproduce accept length to ±0.01, so a real 8 % would have been unmissable. The patch is kept
because it costs nothing, not because it earns anything.

### Porting DFlash2 onto 0.27.1: it runs, and it is slower

DFlash2 numbers come from the #52816 branch (0.26.1rc1) while every MTP number here comes from
0.27.1 — so within-cell comparisons cross an engine boundary and **understate** DFlash2, since the
branch's own MTP is slower (20.8 against 22.2 tok/s on the same target). Attempted 2026-08-20 to
remove that by forward-porting DFlash2 onto 0.27.1. **Abandoned.**

It runs — right engine banner, `method: dflash`, sane output — but underperforms the branch by a
margin that is identical on both bodies:

| cell | branch | ported 0.27.1 |
| :--- | ---: | ---: |
| NVFP4 head · unsloth W4A4 | 38.6 / len 4.26 | 36.4 / len 4.03 |
| NVFP4 head · ours W4A16 | 39.1 / len 4.06 | 38.0 / len 3.81 |

TTFT identical; the deficit is −0.23/−0.25 accept length, body-independent, and repeat runs
reproduce to ±0.01 so it is not noise.

**0.27.1 already ships DFlash.** What it lacks is DFlash2: two modules plus four hooks. Three of
the four were found only by diffing `qwen3_dflash.py` in full — **none of them contains the string
"dflash2"**, because they are changes to *DFlash1* that DFlash2 depends on:

1. `model_cls` class attribute on `DFlashQwen3ForCausalLM`. Without it the parent hard-codes its
   own model and the subclass's is ignored → loud error, `candidate_selector` weights have nowhere
   to go.
2. `decoder_layer_cls` on `DFlashQwen3Model`. Without it DFlash1 decoder layers get built
   **silently** — no error, only a worse draft, and no way to tell from the outside.
3. **RoPE-layout propagation** (`dflash_target_rope_is_neox_style` plus a `load_model` hook that
   stamps it onto the draft config). The branch's own docstring calls the failure mode out:
   *"a mismatch is silent — acceptance collapses but nothing errors and the output stays correct."*
   Restoring it recovered **+0.08** of the 0.24. This one is a defect in 0.27.1's DFlash1 in its
   own right, for anyone using it.
4. Two speculator fixes. `sample_idx_mapping` initialised to −1 instead of 0 — **tested, zero
   effect**. And a valid-context Triton kernel guard, entangled with context-parallel variables and
   not cleanly extractable; that is the remaining suspect for the −0.24.

Wholesale copy is blocked: the branch's `dflash/speculator.py` imports `cp_local_slot`, absent from
0.27.1.

**Practical upshot:** measure DFlash2 on the branch venv and keep the engine caveat. A known
footnote beats an unexplained 5.7 % deviation. Items 1–3 are worth carrying into 0.27.1 regardless
of DFlash2, because item 3 costs plain DFlash1 acceptance there today.

### SGLang: blocked on two merges, not one

SGLang merged DFlash2 support on 2026-08-19 ([sgl#35371](https://github.com/sgl-project/sglang/pull/35371),
pure Python — five source files, Triton kernels, nothing compiled). No arm64 image carries it yet;
the newest `nightly-dev-cu13` predates the merge. An older image fails explicitly:

```
ValueError: Cannot find model module. 'DFlash2DraftModel' is not a registered model
```

(it registers `DFlashDraftModel` — DFlash **1** — plus the Laguna variants). And a fresh image
alone would not be enough: SGLang's selector has the *same* quantized-`lm_head` restriction, with
two PRs still open (sgl#35462, sgl#35496). Without them the NVFP4 production checkpoint is
rejected there too, and our unquantized-head W4A16 build fails to load in SGLang for an unrelated
reason (`AttributeError: 'NoneType' object has no attribute 'num_bits'`).

## 8. Notes, and what we did not test

### ⚠️ Every speed number above ranks agent work wrong

The figures in this document are decode rates. For agent and tool-loop work that is the wrong
quantity to rank by, and not merely imprecise — **the ranking inverts**.

An agent turn is a long prompt and a short completion. Across the 20 real agent contexts replayed
here the median completion is **130 tokens** (range 60–256) against a median context of **9,138**
(range 2,008–31,228). At ~40 tok/s those 130 tokens take about 3 s while the prefill takes 5.6 to
9.6. What you wait for is the prompt.

Normalising to a fixed 130-token completion — `TTFT + 130/rate`, so verbosity cannot leak into a
speed comparison:

| cell | TTFT | tok/s | s / turn @130 tok | of which TTFT |
| :--- | ---: | ---: | ---: | ---: |
| NVFP4 **A4** head · RadixArk W4A4 | 4.5 s | 40.3 | **7.7 s** | 58 % |
| NVFP4 head · unsloth W4A4 | 5.6 s | 38.6 | 9.0 s | 62 % |
| FP8 A16 head · unsloth W4A4 | 5.6 s | 36.9 | 9.1 s | 61 % |
| NVFP4 head · r0b0tlab W4A16 | 8.5 s | 40.6 | 11.7 s | 73 % |
| FP8 A16 head · ours W4A16 | 9.6 s | **41.7** | 12.7 s | 75 % |
| FP8 A8 head · ours W4A16 | 9.6 s | 41.2 | 12.8 s | 75 % |
| NVFP4 head · ours W4A16 | 9.6 s | 39.1 | 12.9 s | 74 % |

The fastest cell by decode rate is **fifth of seven** per turn. First place goes to the row with
the *lowest* decode rate of the group at the time it was added — RadixArk led on turn time at
30.1 tok/s before its DFlash2 figure existed, purely on a 5-second prefill advantage. TTFT is 61–75 % of the turn, so a 13 % decode advantage cannot outrun a
4-second prefill deficit.

**Where the crossover sits.** A faster body wins only once the completion is long enough to repay
its prefill:

```
N > ΔTTFT / (1/r_slow − 1/r_fast)
```

For these two bodies: (9.6 − 5.6) / (1/38.6 − 1/41.7) = **2,077 output tokens**. Real agent turns
emit 130 — short by a factor of sixteen. At 1,000 tokens the slow-decode body is still ahead
(31.5 s against 33.6); they only draw level at 2,000.

**And it worsens through a session.** TTFT grows with context, unevenly: below 8k it is 2.0 s
against 4.1 s, at ≥15k it is 15.3 s against 23.8 s. The penalty grows from 2.1 s to 8.5 s per turn,
and agent context only accumulates.

**Why normalised rather than measured.** Wall time as it occurred is not comparable between runs —
and not because one drafter says more. Paired over the same 20 prompts the median difference in
completion length between an MTP and a DFlash2 run is **+1 token**, and six of twenty hit the
256-token cap in each. What moves is the median itself: bootstrapped at n = 20 these completions
give a 95 % range of **104–252** tokens. Two runs of the same model can differ by 35 tokens in
median output for no reason at all, which in one cell turned a real 10 % gain into an apparent 3 %.
A measured turn runs ~13 % longer than the normalised index, because long contexts carry both a
high TTFT *and* a long completion — so the index orders cells correctly but must not be quoted as a
duration.

**Practical rule:** for tool loops, rank by TTFT at your working context length. Decode rate only
becomes the criterion for long continuous generation.


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

### ⚠️ "temp 1.0 breaks tool-calling" does not generalize across models

Recorded here because it was carried from one model family to another without a re-test, and it was
wrong. On `qwen36-35b-a3b-nvfp4` (A4Q stack) a temperature of 1.0 — the `generation_config.json`
default — collapses tool-calling to **1/8** against 7/8 at 0.2. That result is real and is why the
fleet serves at 0.6.

It does **not** transfer. Same shape of test on `RadixArk/Qwen3.8-27B-NVFP4`, one prompt requiring a
tool call, 8 samples per temperature:

| temp | 0.2 | 0.6 | 1.0 | server default |
| :--- | ---: | ---: | ---: | ---: |
| clean `tool_calls` | 8/8 | 8/8 | **8/8** | 8/8 |

No degradation at 1.0 at all. Whatever breaks at high temperature on the older family is not a
property of NVFP4 checkpoints, or of temperature 1.0, or of this tool-call parser. Re-test per
model before treating it as a constraint.

Setting a server-side default, if wanted, needs no change to the shared serve script — it already
carries a `VLLM_QWEN38_EXTRA` hook at the end of the `vllm serve` line:

```
-E VLLM_QWEN38_EXTRA='--override-generation-config {"temperature":0.6}'
```

No space inside the JSON: the variable is expanded unquoted, so a space splits it into three
arguments. `override_generation_config` does a `config.update()`, so only the named key changes and
the vendor's `top_p`/`top_k` survive.

## 9. ⚠️ Memory: cgroup limits CANNOT protect this box

**This config froze the machine hard enough to require a physical powercycle.** Raising
`--gpu-memory-utilization` 0.5→0.9, pinning 75 GiB KV, `max-num-seqs` 4→32 and extending the
cudagraph capture list — all in one step — hung it during startup. There was no kernel `oom-kill`
entry: it thrashed the unified pool until nothing was schedulable. Worse, the unit was `enabled`,
so the reboot auto-started it straight back into the same config.

### The other trigger: sequencing two large models

`systemctl stop` returns long before vLLM has released its memory. A benchmark harness that waits
only for **port 8080 to close** therefore starts the next server while the previous one still holds
55 GB. Two BF16 models at once took this box to 121/121 GB, load 128, `NVRM: Out of memory` in the
kernel log, and `systemctl` itself stopped answering.

Gate the next start on **memory, not on the port**:

```bash
alive=$(pgrep -cf "bin/vll[m]" || true)
avail=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
[ "$alive" = 0 ] && [ "$avail" -ge 90 ] || wait
```

Recovering from it: kill the **orchestrator first**, then the servers — otherwise the script keeps
spawning replacements and memory never comes back (the PIDs kept changing between kill attempts).
`sudo` is unusable at that load, PAM times out; processes started via `systemd-run --uid=1000` are
yours, so plain `kill -9` works. Use bracketed patterns (`bin/vll[m]`) so `pgrep`/`pkill` cannot
match their own command line.

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
