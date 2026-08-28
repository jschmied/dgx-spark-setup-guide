# Flash-Next quantization comparison

The same idea as `bench/nvfp4-table/` for Qwen3.8-27B, but the axes are different because a
different thing dominates.

On the 27B the interesting axes were the **`lm_head`** and the **MLP**. On Qwen3.8-Flash-Next
they are:

- **the PLE / n-gram table** — 51.2 B parameters that no quantizer touches by default, and the
  single largest object in the checkpoint (95.4 GiB at BF16, 47.7 at FP8, 26.8 at NVFP4). It
  decides whether the model *fits*.
- **the dense projections** (attention q/k/v/o + GDN `in_proj`/`out_proj`) — 2.67 B parameters
  read on every token. They decide how *fast* it decodes. Most published checkpoints leave them
  in BF16.

`lm_head`, `shared_expert` and the hyper-connections are a third axis that, as of 2026-08-28,
**nobody in the field had quantized**. We started on it here.

## Files

| | |
|---|---|
| `data/cells.csv` | every measured / censused checkpoint, machine-readable |
| `flashnext-quants.html` | the published comparison page |

## Rules for this table, learned the hard way

**Read placement from the files, never from the model card or the repo name.** Repo names are
actively misleading here: `Inferact/…-NVFP4` has *identical* dense BF16 to RadixArk (the whole
44 GiB difference is PLE precision), and `…-NVFP4-FP8` says nothing about which layers got which.
Read `hf_quant_config.json` → `quantization.quantized_layers`, and cross-check tensor dtypes from
the safetensors headers.

**Distinguish "read per token" from "present in the checkpoint".** Three groups are *not* per-token
reads and must be excluded from any bandwidth claim: the vision tower (only with an image),
`embed_tokens` (one row, not the matrix), and the MTP drafter (only when speculating). We
over-counted our own roofline by including them.

**Do not trust published quality metrics to differentiate variants.** `gsm8k_metrics.json` and
`aime26_metrics.json` are byte-identical (`sha256 88766f7e…`, same `latency_seconds` to ten
decimals) across a plain NVFP4 build, an FP8 dense-quantized fork, *and* a 512→448 expert-pruned
variant. `sha256sum` them before letting them inform anything.

**Verify the scale convention empirically before writing a quantized tensor.** ModelOpt FP8_PB_WO
stores `weight_scale_inv`, which despite the name is the **scale, not the reciprocal**:
`w_fp8 * scale` reconstructs to 2.25% relative error; `w_fp8 / scale` to 5.7e8%. Getting it
backwards yields fluent garbage, not a crash.

**Two GB10 env vars are mandatory for blockwise-FP8 weights**, both undocumented upstream:
`VLLM_USE_DEEP_GEMM=0` (else a CUDA `unspecified launch failure` — vllm#54125) and
`VLLM_GDN_DECODE_KERNEL=triton` (else the engine hangs at concurrency ~32, silently).

**Log `clocks.sm`.** GB10 parks bandwidth-bound decode at ~82% of max SM clock (2411-2522 against
3003) and locking does not move it. Within-box comparisons are fine; cross-project absolute
figures are not.

## Cross-stack numbers are not comparable

Rows marked `ours` are one box, one method, same day. Field numbers from SGLang or llama.cpp
builds belong in prose, not in this table — they differ in engine, PLE handling, speculation and
clock policy simultaneously.
