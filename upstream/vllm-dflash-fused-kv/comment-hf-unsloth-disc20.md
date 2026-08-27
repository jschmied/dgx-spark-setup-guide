Correction to my comment above ("head has wrong quant") — that no longer applies. **#52816 is merged**: the candidate selector now goes through `quant_method.apply()`, so a quantized head drafts on stock vLLM. Verified here on GB10 against an NVFP4 W4A4 head.

@MaMo7x — the patches you are carrying by hand likely have upstream numbers now:
- draft quant config never reaching `packed_modules_mapping` → [#53116](https://github.com/vllm-project/vllm/issues/53116), PR [#53122](https://github.com/vllm-project/vllm/pull/53122)
- DFlash's fused context-KV slicing `qkv_proj.weight` raw and applying it with `F.linear` → [#51581](https://github.com/vllm-project/vllm/issues/51581)

Drafter quantization measured on GB10 — 24 replayed agent contexts, same engine and session, only the drafter swapped:

| drafter | decode | acceptance length |
|---|---:|---:|
| BF16 `z-lab/Qwen3.8-27B-DFlash2` | 41.3 tok/s | 4.24 |
| INT8 W8A16 `lued/…-DFlash2-W8` | 43.2 tok/s | 4.16 |
| FP8 W8A8 `josch15366/…-DFlash2-FP8` | 43.8 tok/s | 4.24 |

Both quantized drafters land ~5 % over BF16 at unchanged acceptance; FP8 vs INT8 is not distinguishable at this sample size.
