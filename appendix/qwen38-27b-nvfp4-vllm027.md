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
  --max-model-len 262144 --max-num-seqs 4 --max-num-batched-tokens 8192 \
  --enable-chunked-prefill --enable-prefix-caching \
  --compilation-config '{"cudagraph_mode":"PIECEWISE","cudagraph_capture_sizes":[1,2,4,8]}' \
  --gpu-memory-utilization 0.5 --load-format fastsafetensors --trust-remote-code
```

**No `--override-generation-config`** — the checkpoint's `generation_config.json`
(temp 1.0 / top_p 0.95 / top_k 20) *is* the vendor thinking-mode set. Non-thinking clients should
send temp 0.7 / top_p 0.80 / presence_penalty 1.5 per request.

**`--gpu-memory-utilization 0.5` is enough**, and that is not a typo: it yields **35.45 GiB KV =
1,114,865 tokens**, against the ~1.05 M needed for 4 sequences at the full 262 k window. The hybrid
attention is why — only **16 of 64 layers** carry full-attention KV, the rest are linear attention.
Leaving 0.4 of the machine unused costs nothing here and keeps you clear of the JIT-OOM failure mode.

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
  theory; we did not verify empirically), the `RadixArk/Qwen3.8-27B-DSpark` draft model as an
  MTP alternative, and other NVFP4 quants (`RadixArk`, `Inferact`) against Unsloth's. Checkpoint
  provenance is a plausible source of the ~50 % throughput disagreement between published reports.
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

## References

- [unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4) — the checkpoint
- [vLLM recipe: Qwen3.8-27B](https://recipes.vllm.ai/Qwen/Qwen3.8-27B) — official flags
- [SGLang cookbook: Qwen3.8-27B](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-27B)
- [Qwen3.8-27B model card](https://huggingface.co/Qwen/Qwen3.8-27B) — sampling + YaRN guidance
