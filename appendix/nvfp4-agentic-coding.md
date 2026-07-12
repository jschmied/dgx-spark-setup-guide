# Appendix: NVFP4 for agentic coding — weights (NVIDIA vs Unsloth) × kernel (marlin vs b12x)

> **Extends [Appendix D](gb10-nvfp4-benchmarks.md).** Appendix D benchmarked the same 2×2 on **SWE-bench
> Lite (Python-only), temp 0, counting patches *submitted*** and concluded *"NVIDIA·marlin·fp8-KV is
> agentic-optimal."* This appendix re-runs the comparison **bigger, multi-language, at prod sampling
> (temp 0.6), scoring real *resolution*** with the SWE-bench harness — and the picture changes: the
> "NVIDIA wins" verdict was a **Python-only artifact**, and the b12x "slowness" was a **misattribution**.

Serving **Qwen3.6-35B-A3B NVFP4** on a single **GB10** (sm_121, 128 GB unified, vLLM 0.24.0). The four
configs, identical serving params except the named variable (fp8 KV, MTP `nspec=3`, temp 0.6, flashinfer
attention):

| | marlin · **W4A16** (16-bit activations) | b12x · **W4A4** (native FP4 activations) |
|---|---|---|
| **NVIDIA** modelopt | **A** | **D** |
| **Unsloth** compressed-tensors | **B** | **C** |

**A one-page visual summary lives at [`nvfp4-agentic-guide.html`](../nvfp4-agentic-guide.html)** (open in a
browser). This page is the detail behind it.

---

## 1. Quality — agentic resolution, and where NVIDIA gives up

Model-appropriate slices (tractable, single-file): **SWE-bench Verified-30** (Python) + **Multilingual-28**
(Java/JS), temp 0.6, mini-SWE-agent, patches **scored by the official harness** (apply + run FAIL_TO_PASS)
on an x86 box.

| config | Python /30 | Java/JS /28 | Java/JS give-ups (empty patch) |
|---|---:|---:|---:|
| **C** Unsloth·b12x | 17 | **17** | 6 |
| **A** NVIDIA·marlin | — | 14 | **3** |
| **D** NVIDIA·b12x | **19** | 10 | **11** |

- **Python: NVIDIA slightly ahead** (19 vs 17; 16 vs 14 on the 18 both actually test-ran) — real but
  *marginal*, not the +21 pt of the Lite run.
- **Java/JS: it flips.** On the instances where **both** produce a patch, quality is a **tie** — the gap is
  *coverage*: **NVIDIA·b12x emits an empty patch on 11/28 (39 %)**, the agent flailing and producing nothing,
  vs Unsloth's 6.
- **Combined /58: Unsloth wins raw (34 vs 29)** — entirely from Java/JS coverage.

**The give-up is a b12x *serving* artifact, not the model** — proven by the marlin control (config **A**):
the *same NVIDIA weights* on marlin (16-bit activations) give up only **~3/26**, half b12x's rate, and
resolve *more* (14/28 vs 10/28). So the collapse is caused by **b12x's runtime W4A4 activation quantization**
(uniform FP4 on all 40 expert layers), **not** the checkpoint weights or the base model. (The base model is
fine; a BF16 ceiling run was unnecessary — and doesn't serve here anyway, see §4.)

**Mechanism — two independent reasons NVIDIA·b12x is fragile on non-Python code:**

1. **Activation precision.** Give-ups track how hard the *expert activations* are quantized. 16-bit (marlin)
   or **dynamic** W4A4 + an FP8 carve-out (Unsloth) hold up; **uniform FP4 activations on every layer**
   (NVIDIA·b12x) drift the logits enough to derail multi-turn agentic reasoning on unfamiliar token
   distributions.
2. **Calibration.** NVIDIA's modelopt (v0.44.0) uses **static** activation scales calibrated on
   **cnn_dailymail — English news, no code** (its eval set is MMLU-Pro/GPQA/AIME/SciCode/IFBench, no
   Java/JS). Unsloth's compressed-tensors quant is **data-free / dynamic** (per-token scales computed at
   runtime), so it **cannot be domain-skewed** toward Python.

> **Don't trust a Python-only eval.** SWE-bench **Lite is 100 % Python**; on it NVIDIA looked far ahead
> (Lite n=16: NVIDIA 59 % vs Unsloth 38 %). That verdict does not survive contact with Java/JS. Benchmark
> the languages you'll actually run.

---

## 2. Speed — marlin ≈ b12x (the "b12x is faster / marlin is too slow" folklore is wrong)

Prefill (time-to-first-token) and decode measured **separately** across context length — single-stream,
cache-busted (unique tokens force real prefill), isolated box, MTP on.

| context | marlin TTFT (s) | b12x TTFT (s) | Δ prefill | marlin decode (t/s) | b12x decode (t/s) |
|---|---:|---:|---:|---:|---:|
| 1k  | 0.96  | 0.95  | 1.01× | 34.1 | 33.8 |
| 8k  | 11.33 | 10.56 | 1.07× | 31.5 | 31.8 |
| 32k | 95.70 | 92.50 | 1.03× | 24.8 | 25.1 |

- **Prefill and decode are within ~3–7 %** — no widening gap. At long context the prefill is
  **attention-dominated** (same flashinfer kernel for both), so marlin's dequant-to-16-bit MoE barely shows.
- **"marlin is too slow for agents" is a misattribution.** marlin's longer *agentic* wall-time comes from
  **persistence** — it gives up less (3 vs 11), so it keeps working hard tasks (more turns) → hits the
  per-step timeout. **b12x's apparent agentic speed was partly it quitting early.** Give a persistent config
  a generous per-step request timeout.
- The **32k ≈ 95 s prefill** is a real cost, but it's a **model/GB10 property, kernel-independent** — big
  tool observations (large Java files) are the tax. b12x's genuine edge is **~5 % aggregate at concurrency
  ≥4** (Appendix D §1), a *throughput* lever, not latency.

This **revises Appendix D §2**, which read the 6 b12x timeouts as a kernel liability. The corrected reading:
the kernel is ~the same speed; the timeouts are the give-up/persistence dynamic surfacing under a fixed
wall-clock.

---

## 3. Choosing, in order

1. **Language mix.** Polyglot / any Java·JS·TS → weights that don't collapse (**Unsloth**, or NVIDIA on
   marlin). Pure Python → NVIDIA's slim edge is real but marginal.
2. **Weights.** **Unsloth** is the safe default (dynamic activations, nothing to mis-calibrate, no language
   hole). NVIDIA is competitive *only* on marlin.
3. **Kernel.** Unsloth → **b12x** (what its FP4 design is for; needs a one-line FP8-oracle shim). NVIDIA →
   **marlin** (b12x is the give-up trap). Speed is ~a wash either way.
4. **Sampling.** **temp 0.6** coding preset. The generation-config default of **1.0 breaks tool-calling**
   (7/8 vs 1/8 tool calls at low temp — see Appendix C).
5. **Serving stack.** b12x needs **CUDA 13 + `nvidia-cutlass-dsl 4.5.2`**. On 0.24 that's a manual pin;
   **vLLM 0.25 upstreams it** (`requirements/cuda.txt` pins `nvidia-cutlass-dsl[cu13]==4.5.2` +
   `flashinfer 0.6.13`). Then **opt in explicitly** and run the preflight (§4).
6. **Timeout.** A config that doesn't give up takes more turns on hard tasks — set a generous per-step
   request timeout so persistence isn't misread as a hang.

---

## 4. Bad-by-default traps

Each silently degrades the out-of-box experience — you get a worse or slower model and don't know it.

- **b12x is *not* the default on Spark.** vLLM auto-select picks **marlin** on sm_121; you must pass
  `--moe-backend flashinfer_b12x` + `CUTE_DSL_ARCH=sm_121a`. A naive head-to-head is secretly
  marlin-vs-marlin. Verify with the preflight:
  ```bash
  CUTE_DSL_ARCH=sm_121a python -c "
  import torch; from vllm.utils.flashinfer import has_flashinfer_b12x_gemm as g, has_flashinfer_b12x_moe as m
  cap=torch.cuda.get_device_capability(); print('cap',cap,'| gemm',g(),'| moe',m())
  assert cap[0]==12 and g() and m(), 'b12x unavailable -> would degrade to marlin W4A16'"
  ```
  and by the serve log line: `Using 'X' NvFp4 MoE backend`.
- **NVIDIA·b12x's Java/JS collapse is invisible on Python.** Static English-news calibration + uniform FP4
  activations = 39 % give-ups that a Python-only eval never surfaces.
- **The default temperature breaks agents.** generation_config ships temp 1.0; override to 0.6.
- **BF16 won't serve here.** The 65 GB hybrid (linear-attn/mamba) + multimodal model OOMs vLLM at
  KV/mamba profiling on 128 GB unified (weights load fine; it dies in the profiling forward, even with
  `--enforce-eager`, text-only, gpu-util 0.72). On this box the practical full-precision proxy is the
  **Q8 GGUF** (llama.cpp), not BF16.

---

## Recommendations (revising Appendix D for a coding agent)

- **Polyglot / general agentic coding → Unsloth·b12x (config C).** Fastest path that doesn't collapse on
  any language: 34/58 resolved, no calibration skew, native FP4 speed.
- **If you must run NVIDIA weights → marlin (A), not b12x (D).** Same speed, ~half the Java/JS give-ups.
- **Pure-Python agent → NVIDIA is fine** (slim edge), on marlin; Appendix D's "NVIDIA·marlin·fp8-KV" verdict
  holds for that case.
- **KV: fp8** (unchanged from Appendix D — near-lossless, half the memory; nvfp4 KV worse *and* sm100f-gated
  off GB10).

**Method / caveats.** One GB10 box, one agent scaffold (mini-SWE-agent), patches scored by the official
SWE-bench harness on x86. Slices chosen model-appropriate; **n is modest per language** — treat as
*direction, not decimals*. Configs matched on everything but the named variable; speed is single-stream on
an isolated box (marlin's 5 Java/JS eval errors are likely timeout/infra, not model). Full run logs and the
autonomous orchestration are recorded in the project's knowledge tree.
