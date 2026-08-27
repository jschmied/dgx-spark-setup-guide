This is already solved three times outside this thread, and none of them link back here. Collecting it, because #51620, #51684 and #53122 are converging on the same code without knowing about each other.

**A fix exists and is in production use.** [`syv-ai/qwen38-27b-rtx3090`](https://github.com/syv-ai/qwen38-27b-rtx3090) → `patches/dflash2-backport.patch` replaces the same line with `_dense_kv_rows(a)`, dequantizing pack-quantized W4A16/W8A16 K/V rows once at load. Its docstring states the timing that makes it safe:

> before the Marlin repack, so `weight_packed`/`weight_scale` are still in the plain checkpoint layout

That matters for #53122, whose guard message asserts the opposite (`process_weights_after_loading` has transposed the weight). It has not run at that point — `_build_fused_kv_buffers()` is called at the end of `load_weights()`. Two independent implementations rely on this.

The same patch also relaxes the LM-head restriction via `hasattr(qm, "apply")` + `qm.apply(...)`, which is the #52816 head fix arrived at separately.

[`noonghunna/club-3090`](https://github.com/noonghunna/club-3090) vendors that patch byte-near-identically and credits syv-ai.

**The defect is already shaping published checkpoints.** [`magiccodingman/Qwen3.8-27B-heretic-ara-DFlash2-fp8`](https://huggingface.co/magiccodingman/Qwen3.8-27B-heretic-ara-DFlash2-fp8) leaves Q/K/V in BF16 on purpose — only 15 MLP matrices are FP8 — and gives the reason as:

> the current DFlash fused context-KV path reads their raw weights and bypasses quantization dispatch

**Scope:** [`0xWhiteMage/…-Kearuga-DFlash2-FP8-E4M3`](https://huggingface.co/0xWhiteMage/Qwen3.8-27B-Kearuga-DFlash2-FP8-E4M3) quantizes Q/K/V and serves on SGLang, so this looks vLLM-specific.

**Neither published form covers both storage layouts:** syv-ai's returns a plain `weight` untouched, so an FP8 drafter still dies there; the patch I posted above handled FP8 but not packed. Combined branch, on the #52816 merge commit:
https://github.com/vllm-project/vllm/compare/b389ac29465b33f9e9c534df221ea3c129e9793f...jschmied:vllm:dflash-quantized-drafter

**Measured**, GB10 (`sm_121a`) TP=1, target `RadixArk/Qwen3.8-27B-NVFP4`, n=7, FlashInfer, fp8 KV, 24 replayed agent contexts, same engine and session, only the drafter swapped:

| drafter | decode | acceptance length |
|---|---:|---:|
| BF16 `z-lab/Qwen3.8-27B-DFlash2` | 41.3 tok/s | 4.24 |
| INT8 W8A16 `lued/Qwen3.8-27B-DFlash2-W8` | 43.2 tok/s | 4.16 |
| FP8 W8A8 `josch15366/Qwen3.8-27B-DFlash2-FP8` | 43.8 tok/s | 4.24 |

Both quantized drafters ~5 % over BF16 at unchanged acceptance; FP8 vs INT8 is 1.4 %, not distinguishable at n=24. Loading either still needs the `packed_modules_mapping` fix (#53116 / #53122) first.
