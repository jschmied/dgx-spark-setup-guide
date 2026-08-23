---
license: apache-2.0
base_model:
  - z-lab/Qwen3.8-27B-DFlash2
tags:
  - dflash2
  - speculative-decoding
  - draft-model
  - fp8
  - compressed-tensors
  - dgx-spark
inference: false
---

# Qwen3.8-27B-DFlash2-FP8

An **FP8 quantization of the DFlash2 draft model** for `Qwen/Qwen3.8-27B`. It is not a
standalone language model: it drafts tokens inside a speculative-decoding server for a
target model to verify.

Quantizing a *drafter* is unusually low-risk: every token it proposes is verified against
the target, so a worse drafter costs speed, never correctness. The upside is real on a
bandwidth-bound machine — the drafter runs once per step, so its weight bytes are paid
every step.

## ⚠️ Requires a patched vLLM

**This will not load on stock vLLM.** Two separate defects sit in front of it:

1. The draft model's quantization config never reaches `packed_modules_mapping`, so fused
   projections are built unquantized and loading dies with a confusing
   `AttributeError: 'MergedColumnParallelLinear' object has no attribute 'data'` —
   [vllm#53116](https://github.com/vllm-project/vllm/issues/53116),
   [vllm#53107](https://github.com/vllm-project/vllm/issues/53107),
   PR [#53122](https://github.com/vllm-project/vllm/pull/53122).
2. DFlash's fused context-KV projection slices the raw `qkv_proj.weight` and applies it
   with a bare `F.linear`, bypassing `quant_method` — so a quantized drafter dies at
   startup profiling with `expected mat1 and mat2 to have the same dtype` —
   [vllm#51581](https://github.com/vllm-project/vllm/issues/51581),
   PRs [#51620](https://github.com/vllm-project/vllm/pull/51620),
   [#51684](https://github.com/vllm-project/vllm/pull/51684).

Until those land, this checkpoint is mainly useful as a **test artifact** for the people
fixing them: reproducing either defect needs a quantized DFlash drafter.

## Quantization scheme

W8A8, FP8 `e4m3` weights with **per-channel** scales and **dynamic per-token** activation
scales. There is no stored `input_scale`; the activation scale is computed at runtime.
vLLM selects `CompressedTensorsW8A8Fp8`.

Quantized: `self_attn.{q,k,v,o}_proj`, `mlp.{gate,up,down}_proj`, `fc` — 2-D weights only.

**Deliberately left in BF16:** the two-tap convolutions (`*_conv`), the
`candidate_selector` and its codebooks, and all norms. Together they are 0.39 GB — the
saving is small and they are the parts that make block-diffusion drafting work.

The exact script that produced this checkpoint is included as `quantize.py`.

Result: **3.85 GB → 2.25 GB.**

## Measurements

NVIDIA GB10 (DGX Spark, `sm_121a`), TP=1, target `RadixArk/Qwen3.8-27B-NVFP4` (NVFP4
W4A4 head), `num_speculative_tokens=7`, FlashInfer, fp8 KV cache. 24 replayed real agent
contexts, `max_tokens=256`. Both arms on the same engine and the same session — only the
drafter differs.

| drafter | decode | acceptance length | TTFT |
| --- | ---: | ---: | ---: |
| BF16 (`z-lab/Qwen3.8-27B-DFlash2`) | 41.3 tok/s | 4.24 | 5.6 s |
| **FP8 (this)** | **43.8 tok/s** | **4.24** | 5.6 s |

**+6.1 % decode at an unchanged acceptance length.** The quantization was predicted to
buy ~7.5 % from the byte budget alone (the drafter is 15.5 % of the bytes moved per step
on a machine measured at 231.8 of 273 GB/s); the measured 6.1 % is close, and the
unchanged acceptance length says the drafts themselves did not get worse.

**Scope of these numbers:** one target, one harness, n=24, single stream. They are not a
general claim about FP8 drafters.

## Related work

[`lued/Qwen3.8-27B-DFlash2-W8`](https://huggingface.co/lued/Qwen3.8-27B-DFlash2-W8)
(2026-08-19) quantizes the same base drafter, and predates this one. It is a **different
scheme**, not a competing build of the same thing:

| | `lued/…-W8` | this |
| --- | --- | --- |
| weights | INT8, group-128, `pack-quantized` | FP8 `e4m3`, per-channel, `float-quantized` |
| activations | none (W8A16, weight-only) | dynamic per-token (W8A8) |
| built with | llm-compressor `QuantizationModifier` | the included `quantize.py` |
| size | 2.02 GiB | 2.25 GB |

The two have since been measured against each other, on the same engine and in the same
session as the numbers above — only the drafter was swapped:

| drafter | decode | acceptance length | TTFT |
| --- | ---: | ---: | ---: |
| BF16 (`z-lab/Qwen3.8-27B-DFlash2`) | 41.3 tok/s | 4.24 | 5.6 s |
| INT8 W8A16 (`lued/…-W8`) | 43.2 tok/s | 4.16 | 5.6 s |
| FP8 W8A8 (this) | 43.8 tok/s | 4.24 | 5.6 s |

The expectation that FP8's native Blackwell support would beat Marlin-routed INT8 on GB10
was **not** confirmed. FP8 is 1.4 % ahead of INT8, which is the same order as the ~2.5 %
session-to-session drift we have measured elsewhere on this box: at this sample size the
two schemes are **not distinguishable**. Pick either.

What the run does support is the other half: **both quantized drafters land roughly 5 %
above BF16** — 4.6 % for INT8, 6.1 % for FP8 — **at a practically unchanged acceptance
length** (4.16 and 4.24 against BF16's 4.24). That is the result worth carrying over.

Same scope limit as above: n=24, single stream, one target, one harness. Their card
reports acceptance unchanged against BF16 as well, from a different harness.

## Credits

Base model: [`z-lab/Qwen3.8-27B-DFlash2`](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2),
itself a mirror of [`incoai/Qwen3.8-27B-DFlash2`](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2).
DFlash2: [blog](https://inco.ai/blog/dflash2/) · [code](https://github.com/z-lab/dflash).
Apache-2.0, inherited from the base model.
