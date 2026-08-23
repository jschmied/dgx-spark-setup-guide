@kamb-code's one-block back-off explains our numbers on a second configuration, and the arithmetic
lands exactly — except in one case, which may be the third wrinkle.

**Setup.** GB10 / sm_121a, `RadixArk/Qwen3.8-27B-NVFP4` (hybrid GDN), vLLM `0.26.1rc1.dev912`,
**DFlash2** drafter with `num_speculative_tokens: 7`, `--kv-cache-dtype fp8`, chunked prefill,
`max_num_batched_tokens: 8192`. Block size is raised to **1648** so the attention page matches the
mamba page. `mamba_cache_mode` resolves to `align`. Each prompt sent three times, `max_tokens: 1`,
counters read from `/metrics` around every single request.

| prompt tokens | blocks | predicted cacheable | measured hits |
|---:|---:|---:|---:|
| 8 702 | 5.28 | ⌊8703/1648⌋−1 = 4 → 6 592 | **6 592** ✅ |
| 9 155 | 5.56 | ⌊9156/1648⌋−1 = 4 → 6 592 | **6 592** ✅ |
| 6 592 | **exactly 4** | ⌊6593/1648⌋−1 = 3 → 4 944 | **3 296** ❌ (one block short) |

So the `use_eagle` back-off predicts the unaligned cases exactly. `use_eagle()` returns True for
`dflash`, so this path is not Eagle/MTP-specific.

**A prompt ending exactly on a block boundary loses one more block** than the formula predicts. The
6 592-token prompt was calibrated through `/tokenize` + `/detokenize` and verified at exactly 6 592,
so this is not a rounding artefact. Candidate explanation from the code: that prefill is a single
chunk which is both the final chunk and a boundary, and the final-chunk exemption removes the state
that the boundary rule would have registered — but I have not confirmed that.

**Second observation, in every case:** the *first* repeat never hits. Request 1 → 0, request 2 → 0,
request 3 → the numbers above. An 8-second pause before request 2 changes nothing, so it is not a
release race, and once populated every later request hits immediately. Whatever writes the
cacheable state does not do so on the run that first computes the prefix.

Prefill time tracks it: 3.78 s / 3.79 s / 0.94 s for the three requests, against 3.80 s for an
unrelated prompt of the same length as a control.

For scale on this configuration: 1 648 tokens of an ~8 000-token agent prompt is about a fifth of
the reusable prefix, given up per request.
