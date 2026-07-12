# Appendix: GB10 NVFP4 benchmarks — MoE kernel (marlin vs b12x) and KV-cache dtype

Measured findings for serving **Qwen3.6-35B-A3B NVFP4** on a single **DGX Spark / GB10** (sm_121,
128 GB unified memory) with **vLLM 0.24.0**. Two independent questions:

1. **MoE kernel** — the default `marlin` (weight-only dequant → 16-bit compute) vs the native-FP4
   `flashinfer_b12x` (FP4 tensor cores, opt-in: `--moe-backend flashinfer_b12x` + `CUTE_DSL_ARCH=sm_121a`).
2. **KV-cache dtype** — `fp8` (prod) vs `bf16` (full precision) vs `nvfp4`.

All runs share identical serving params (fp8 KV, MTP `nspec=3`, `max-num-seqs 24`, `gpu-mem-util 0.75`,
flashinfer attention) so only the named variable changes. Two weight sets: **NVIDIA** (uniform
W4A16_NVFP4) and **Unsloth** (W4A4 experts + FP8 on attention/lm_head/last-8-layers).

The 2×2 configs: **A** = NVIDIA·marlin, **B** = Unsloth·marlin, **C** = Unsloth·b12x, **D** = NVIDIA·b12x.

> **Superseded in part by [Appendix E](nvfp4-agentic-coding.md).** This page's agentic study (§2) is
> **SWE-bench Lite, Python-only, temp 0, counting *submitted* patches**. A larger **multi-language,
> resolution-scored** re-run (Appendix E) revises two conclusions: (a) the *"NVIDIA wins agentic"* verdict
> is a **Python-only artifact** — on Java/JS, NVIDIA·b12x gives up 39 % of the time and **Unsloth wins
> overall**; and (b) *"b12x is worse/slower"* is a **misattribution** — marlin ≈ b12x in speed; the
> difference is give-up/persistence. The KV-cache ladder (§3) and the throughput sweep (§1) still stand.

---

## 1. Decode throughput — synthetic sweep (aggregate tok/s)

> Appendix B covers the b12x-on-mainline decode sweep in the A4Q narrative; it's summarised here
> so this page stands alone alongside the agentic (§2) and KV-quality (§3) results, which are new.

| c | A NV·marlin | B Uns·marlin | C Uns·b12x | D NV·b12x |
|---|---:|---:|---:|---:|
| 1  | 78.4  | 60.0  | 64.5  | 71.9 |
| 4  | 116.3 | 111.4 | 114.7 | **121.2** |
| 8  | 136.9 | 130.5 | 136.8 | **142.9** |
| 12 | 148.5 | 142.2 | 147.6 | **154.4** |
| **TTFT c1 (s)** | 2.06 | 2.11 | 1.95 | **1.89** |

- **Single-stream** decode (c=1, MTP): ~**117–129 tok/s** across configs (MTP-acceptance-dependent).
- **b12x ≥ marlin** at concurrency ≥4 on aggregate throughput *and* lower TTFT, for both weight sets —
  but the margin is small (~4–6%). **D (NVIDIA·b12x) is the throughput-optimal config** and needs no
  patch (NVIDIA's uniform FP4); Unsloth's mixed precision needs a one-line FP8-oracle shim for b12x.
- **Peak aggregate is far higher at large batch.** This sweep caps `max-num-seqs 24` with long context +
  MTP. A 3B-active MoE amortizes weight reads across a big batch, so at high concurrency (64–256 streams,
  short outputs) aggregate leaves the memory-bound regime and becomes **compute-bound**, where the FP4
  tensor cores drive it much higher — peak-aggregate numbers in the many-hundreds tok/s are plausible
  there. That is a *batch-throughput* figure, not per-user (single-stream stays ~115).

> **Retraction.** An earlier draft showed b12x "collapsing at c=12" (TTFT ~25 s). That was a benchmark
> artifact — the b12x services ran `max-num-seqs 6` while the marlin baseline ran `12` (plus
> async-scheduling + atomic-add), so b12x was queue-starved, not kernel-limited. With every serving
> param matched, the collapse vanishes. **Never compare backends across unmatched `--max-num-seqs`.**

Caveat: decode t/s carries **MTP-acceptance variance** (single-stream cells swing with the draft
accept rate) — trust *aggregate*, not single-cell single-stream.

---

## 2. Real agentic workload — SWE-bench (mini-SWE-agent), 16 tasks × 4 configs

The synthetic sweep favours b12x. **The real agentic workload does not.** 16 arm64 SWE-bench Lite
instances per config, temp 0, `timeout 1200s`/task, 64 trajectories total.

| axis | done | timeouts | mean calls/task | sec/turn | submits |
|---|---|---|---|---|---|
| **marlin** (A+B) | 32/32 | **0** | 99 | 4.1 | 26 |
| **b12x** (C+D) | 26/32 | **6** | 102 | 3.9 | 21 |
| **NVIDIA** (A+D) | 29/32 | 3 | **78** | **3.8** | **26** |
| **Unsloth** (B+C) | 29/32 | 3 | **124** | 4.3 | 21 |

**Kernel axis — marlin is the safer agentic choice, but *not* because b12x is slower.** On the 15 tasks
both kernels completed, b12x is **0.93× sec/turn — marginally *faster* per turn**. The 6 b12x timeouts
(both b12x configs, same weights as their marlin twins) come from **longer / looping trajectories** on
hard tasks: b12x's W4A4 activation quant nudges the logits → slightly worse decisions → more turns →
hits the wall-clock. It also generates less reasoning per turn (187 vs 269 tokens). So b12x's synthetic
throughput edge is a *decision-quality* liability in multi-turn agentic use.

**Weights axis — NVIDIA is the clear agentic win.** NVIDIA weights solve in **~40% fewer turns**
(78 vs 124), with faster turns (3.8 vs 4.3 s) and more submits. Unsloth also loops more (config B hit
the 240-call cap 4×). On the real workload, **weights matter more than the kernel.**

Caveats: **`submit` ≠ `resolved`** — this counts patches produced, not test-passing patches (no eval
harness scoring). n=16 with large trajectory divergence — the NVIDIA turn-efficiency gap is the robust
signal; the 6-timeout / submit deltas are suggestive, not conclusive.

---

## 3. KV-cache dtype quality — logprob divergence ladder

Greedy teacher-forced Δlogprob (`echo`+`logprobs`), same build, only KV dtype varies. **bf16 = anchor.**

| | bf16 (A) | fp8 (B, vs bf16) | nvfp4 (C, vs fp8) |
|---|---|---|---|
| mean \|Δ logprob\| | 0 | **0.0143** | **0.127** |
| perplexity cost | — | **+0.13%** | **+0.6%** |
| correlation w/ ref | — | 0.9988 | — |
| tokens > 0.5 nats | — | **0%** | ~4% argmax flips |
| depth trend | — | decays (no compounding) | decays (no compounding) |

- **bf16 ≈ fp8 — near-lossless.** fp8 KV: +0.13% perplexity, correlation 0.999, **zero** tokens above
  0.5 nats, divergence *decreases* with depth (no autoregressive compounding). **Prod's fp8 KV is
  validated** — no quality reason to pay bf16's 2× KV memory.
- **nvfp4 is ~9× the divergence of fp8** (0.127 vs 0.0143) and ~5× the perplexity cost — and greedy
  teacher-forced *understates* it (real work samples at temp>0 and drifts through its own 4-bit KV;
  extended real-task use found fp8 "considerably better").
- **nvfp4 KV is moot on GB10 anyway** — mainline vLLM gates it to `sm100f` (SM_100 datacenter, not
  sm_121): engine init fails with *"--kv-cache-dtype nvfp4 requires sm100f"*. The A4Q fork that provided
  it on sm_121 is decommissioned.

Ranking: **bf16 ≈ fp8 ≫ nvfp4.**

---

## Recommendations (GB10, agentic coding fleet)

- **Weights: NVIDIA** (W4A16_NVFP4) — clearly more turn-efficient than Unsloth on real agent tasks.
- **MoE kernel: marlin** for agentic/interactive use (0 timeouts, more submits; b12x's synthetic edge
  doesn't survive multi-turn work). Reserve **b12x** for compute-bound / high-batch *throughput* serving,
  where FP4 tensor cores actually pay off.
- **KV cache: fp8** — near-lossless vs bf16, half the memory; nvfp4 is worse *and* not runnable here.

Net: **NVIDIA·marlin·fp8-KV** (i.e. prod) is the agentic-optimal config on GB10; b12x is a
throughput-only lever, not a default.

> **Revised for coding agents ([Appendix E](nvfp4-agentic-coding.md)).** The scored multi-language re-run
> holds this verdict for **Python-only** work, but for a **polyglot coding agent** the pick is
> **Unsloth·b12x** (no Java/JS collapse, native-FP4 fast). And if you run NVIDIA weights, prefer **marlin
> over b12x** for quality reasons (fewer give-ups), not speed — the two kernels benchmark ~equal single-stream.
