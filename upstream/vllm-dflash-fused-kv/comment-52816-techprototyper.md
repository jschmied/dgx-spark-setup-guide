@TechPrototyper That failure has a number: [#53116](https://github.com/vllm-project/vllm/issues/53116) — the draft model's quant config never reaches `packed_modules_mapping`, so the fused layers are built without a scheme and the per-shard `weight_scale` resolves to the module. Fix in [PR #53122](https://github.com/vllm-project/vllm/pull/53122). The confusing `AttributeError` itself is [#53107](https://github.com/vllm-project/vllm/issues/53107).

Worth knowing before you get there: **a second wall sits right behind it.** Once the drafter loads, DFlash's fused context-KV precompute slices `qkv_proj.weight` raw and applies it with a bare `F.linear`, bypassing `quant_method` — so a drafter with quantized `q/k/v` dies at startup profiling with `expected mat1 and mat2 to have the same dtype`. That is [#51581](https://github.com/vllm-project/vllm/issues/51581); #53122 now carries a dequantize-at-buffer-build fix for it too. Your unfused-only result is consistent with exactly this: `o_proj`/`down_proj` never touch that path.

Same hardware here (GB10, sm_121a), and your acceptance observation reproduces — 24 replayed agent contexts against an NVFP4 W4A4 target, same engine and session, only the drafter swapped:

| drafter | decode | acceptance length |
|---|---:|---:|
| BF16 | 41.3 tok/s | 4.24 |
| INT8 W8A16 | 43.2 tok/s | 4.16 |
| FP8 W8A8 (per-channel weights, dynamic activations) | 43.8 tok/s | 4.24 |
