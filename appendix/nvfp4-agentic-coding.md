# Appendix: NVFP4 for agentic coding — weights (NVIDIA vs Unsloth) × kernel (marlin vs b12x)

> **Extends [Appendix D](gb10-nvfp4-benchmarks.md).** Appendix D benchmarked the same 2×2 on **SWE-bench
> Lite (Python-only), temp 0, counting patches *submitted*** and concluded *"NVIDIA·marlin·fp8-KV is
> agentic-optimal."* This appendix re-runs the comparison **bigger, multi-language, at prod sampling
> (temp 0.6), scoring real *resolution*** with the SWE-bench harness — and the picture changes: **which
> weights win depends on the language**, and the b12x "slowness" is a **misattribution** (the kernels
> measure ~equal, §2).
>
> **⚠️ This page was corrected on 2026-07-14.** Its original headline — a Java/JS "give-up collapse" for
> NVIDIA·b12x — was a **benchmarking artifact** (uncached Docker eval images timing out, so instances were
> never attempted). See the correction box in §1. The *language-dependent* result survives; the give-up
> finding and its mechanism do not.

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

## 1. Quality — agentic resolution by language

> ### ⚠️ Correction (2026-07-14): the "give-up" finding was a benchmarking artifact
>
> An earlier version of this appendix reported that **NVIDIA·b12x emits an empty patch on 11/28 Java/JS
> instances (39 %)** and built a mechanism (activation precision + calibration skew) on top of that
> "coverage collapse." **That finding does not survive scrutiny, and the mechanism is withdrawn.**
>
> mini-SWE-agent starts each instance with `docker run … sleep 2h` and a **120 s `pull_timeout`**. Any
> instance whose eval image is **not already cached** dies in `docker run` with `TimeoutExpired` — the model
> is **never called**, no trajectory is produced, and the harness records an **empty patch**, which is
> indistinguishable in the results from the model giving up.
>
> The three configs were run **back-to-back against a warming Docker cache**, so the failures track *run
> order*, not the model:
>
> | run order | config | `docker run` timeouts | reported "give-ups" |
> |---|---|---:|---:|
> | 1st (Jul 11) | **D** NVIDIA·b12x | **8** | 11 |
> | 2nd (Jul 12) | **C** Unsloth·b12x | **2** | 6 |
> | 3rd (Jul 12) | **A** NVIDIA·marlin | **0** | 3 |
>
> D ran first, against a cold cache, and paid for it. The instances it lost are exactly the
> alphabetically-first ones (`lucene-*`, `axios-*`, `babel-*`) — the ones a cold cache hits first.
> **Corrected give-ups are ≈3 / ≈4 / 3 — no meaningful difference between configs.** The same bug hit the
> Python runs (D: 4 infra losses = all 4 of its "empty patches"; C: 1).
>
> **Fix for future runs:** pre-pull every eval image, or raise the limits in an overlay (this changes no
> sampling, so results stay comparable):
> ```yaml
> environment:
>   container_timeout: "8h"   # slow local models blow the 2 h default mid-trajectory
>   pull_timeout: 900         # 120 s default cannot pull a cold multi-GB eval image
> ```

Model-appropriate slices (tractable, single-file): **SWE-bench Verified-30** (Python) + **Multilingual-28**
(Java/JS), temp 0.6, mini-SWE-agent, patches **scored by the official harness** (apply + run FAIL_TO_PASS)
on an x86 box.

The 15 infra-lost instances were **re-run** (2026-07-15) on the identical stacks with images pre-cached
(0 infra errors), so these are scored over the **full slice**, not just the attempted subset:

| config | Python | Java/JS | real give-ups (Java/JS) |
|---|---:|---:|---:|
| **D** NVIDIA·b12x | **23/30 — 77 %** | 15/28 — 54 % | ~3 |
| **C** Unsloth·b12x | 17/29 — 59 % | **19/28 — 68 %** | ~4 |
| **A** NVIDIA·marlin | — | 14/28 — 50 % | 3 |

(In the re-run, D resolved 5 of its 8 previously-"given-up" Java/JS instances and 4/4 Python — confirming
those were Docker-timeout losses, not model give-ups. C-Python is /29: one instance was cut mid-run, immaterial.)

- **The language flip survives — it was the one real finding.** NVIDIA is **better on Python**, Unsloth is
  **better on Java/JS**. Correcting the artifact actually *widens* NVIDIA's Python lead (73 % vs 59 %, where
  the old numbers called it "marginal") and leaves Unsloth ahead on Java/JS by a smaller but consistent margin.
- **There is no coverage collapse.** All configs give up at roughly the same low rate (~3-4/28). The old
  claim that b12x's W4A4 activations "derail multi-turn reasoning" had nothing left to explain once the
  Docker failures were subtracted, and is withdrawn.
- **Calibration remains a plausible story for the *resolution* gap, not for give-ups.** NVIDIA's modelopt
  uses **static** activation scales calibrated on **cnn_dailymail — English news, no code**; Unsloth's
  compressed-tensors quant is **data-free / dynamic** (per-token scales at runtime), so it cannot be
  domain-skewed. That would predict exactly this shape: NVIDIA strongest where its calibration is least
  wrong, weaker on token distributions further from it. **Untested — treat as hypothesis.**
- **n is small and the denominators now differ per config.** A 3-instance swing moves these numbers by
  >10 points. Direction, not decimals.

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
- **"marlin is too slow for agents" is still a misattribution — but for a simpler reason than we thought.**
  These direct measurements show the kernels are ~the same speed, and that stands on its own. An earlier
  version explained marlin's longer *agentic* wall-time as **persistence** ("it gives up less — 3 vs 11 — so
  it keeps working hard tasks"). **That explanation is withdrawn:** the 3-vs-11 gap was the Docker-cache
  artifact of §1, not a behavioural difference. b12x did not "quit early"; 8 of its instances never started.
  The agentic wall-time difference is currently **unexplained** — and with give-up rates equal, there may be
  nothing left to explain.
- The **32k ≈ 95 s prefill** is a real cost, but it's a **model/GB10 property, kernel-independent** — big
  tool observations (large Java files) are the tax. b12x's genuine edge is **~5 % aggregate at concurrency
  ≥4** (Appendix D §1), a *throughput* lever, not latency.

This **revises Appendix D §2**, which read the 6 b12x timeouts as a kernel liability. The corrected reading:
**the kernel is ~the same speed** (measured directly, above) — so the timeouts were never evidence about the
kernel either way.

---

## 3. Choosing, in order

1. **Language mix — this is the live variable.** Polyglot / any Java·JS·TS → **Unsloth**. Predominantly
   Python → **NVIDIA** (its edge there is real: 73 % vs 59 % on attempted instances).
2. **Weights.** **Unsloth** is the safe default: it wins the polyglot case and loses Python by a margin
   smaller than the slice can resolve. Pick NVIDIA only if you know your workload is Python-heavy.
   *(Note: the old reason for this — "NVIDIA collapses on Java/JS" — was an artifact, §1. The
   recommendation survives on resolution rate; the scary story behind it does not.)*
3. **Kernel.** Unsloth → **b12x** (what its FP4 design is for; needs a one-line FP8-oracle shim). NVIDIA →
   either; **speed is a wash** (§2) and there is **no give-up trap** — that was the artifact. marlin remains
   the safer pick simply because it is the default and needs no opt-in.
4. **Sampling.** **temp 0.6** coding preset. The generation-config default of **1.0 breaks tool-calling**
   (7/8 vs 1/8 tool calls at low temp — see Appendix C).
5. **Serving stack.** b12x needs **CUDA 13 + `nvidia-cutlass-dsl 4.5.2`**. On 0.24 that's a manual pin;
   **vLLM 0.25 upstreams it** (`requirements/cuda.txt` pins `nvidia-cutlass-dsl[cu13]==4.5.2` +
   `flashinfer 0.6.13`). Then **opt in explicitly** and run the preflight (§4).
6. **Benchmark hygiene (learned the hard way).** **Pre-pull every SWE-bench eval image before a scored run**,
   or raise `pull_timeout`/`container_timeout` (§1). A cold Docker cache silently penalises whichever config
   runs *first*, and the damage looks exactly like the model giving up. Always subtract
   `grep -c "Error processing instance"` from any empty-patch count before believing it.

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
- **NVIDIA's Java/JS weakness is invisible on Python.** It resolves **73 %** of Python but only **50 %** of
  Java/JS — a Python-only eval (SWE-bench Lite) shows you its *best* face and hides the gap.
- **A cold Docker cache fabricates "give-ups".** An uncached eval image blows mini-SWE-agent's 120 s
  `pull_timeout`; the instance is **never attempted** but recorded as an **empty patch** and scored as a
  failure. This produced — and then destroyed — this page's original headline (§1). Pre-pull images, and
  subtract `grep -c "Error processing instance"` from any empty-patch count before believing it.
- **The default temperature breaks agents.** generation_config ships temp 1.0; override to 0.6.
- **BF16 won't serve here.** The 65 GB hybrid (linear-attn/mamba) + multimodal model OOMs vLLM at
  KV/mamba profiling on 128 GB unified (weights load fine; it dies in the profiling forward, even with
  `--enforce-eager`, text-only, gpu-util 0.72). On this box the practical full-precision proxy is the
  **Q8 GGUF** (llama.cpp), not BF16.

---

## Recommendations (revising Appendix D for a coding agent)

- **Polyglot / general agentic coding → Unsloth·b12x (config C).** Best Java/JS score (**65 %** vs NVIDIA's
  50 %), no calibration skew, native FP4 speed. It gives up Python (59 % vs 73 %) — a real trade, not a free lunch.
- **Python-heavy agent → NVIDIA.** Its Python edge is **larger than originally reported** (73 % vs 59 %):
  correcting the benchmark artifact *widened* it. Appendix D's "NVIDIA·marlin·fp8-KV" verdict holds for that case.
- **Kernel is not a quality lever.** b12x vs marlin is ~a speed wash and there is **no give-up trap** — that
  was the artifact. Use b12x with Unsloth (its FP4 design), marlin with NVIDIA (it's the default, no opt-in).
- **KV: fp8** (unchanged from Appendix D — near-lossless, half the memory; nvfp4 KV worse *and* sm100f-gated
  off GB10).

**Method / caveats.** One GB10 box, one agent scaffold (mini-SWE-agent), patches scored by the official
SWE-bench harness on x86. Slices chosen model-appropriate; **n is modest per language** — treat as
*direction, not decimals*. Configs matched on everything but the named variable; speed is single-stream on
an isolated box (marlin's 5 Java/JS eval errors are likely timeout/infra, not model). Full run logs and the
autonomous orchestration are recorded in the project's knowledge tree.
