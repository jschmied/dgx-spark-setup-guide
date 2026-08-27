@seanyourhighness — I argued for keeping the fused GEMM without ever measuring what it is worth, so
here is the number. It supports your recommendation.

Production configuration on GB10: RadixArk Qwen3.8-27B-NVFP4, DFlash2 `n=7` with a W4A16 drafter,
`--kv-cache-dtype fp8`. The fused projection runs once per prefill, so TTFT is where it can pay;
decode is untouched. Unique prompts so prefix caching cannot shorten the prefill, three repetitions,
median:

| context | fused | per-layer (#51620 approach) | delta |
|---|---:|---:|---:|
| ~1k | 0.470 s | 0.517 s | +10 % |
| ~8k | 3.396 s | 3.475 s | +2.3 % |
| ~30k | 14.638 s | 14.825 s | +1.3 % |

**1–2 % at working contexts, and the run-to-run spreads overlap at every length** — at ~1k the
ranges are 0.36–0.525 against 0.374–0.634, so that 10 % is one 50 ms gap between two three-sample
medians, not a result. The per-layer fallback costs approximately nothing here, and "it gives up the
cross-layer fusion" is not a reason to hold up a correctness fix.

Scope worth stating: this drafter has 5 layers and its context-KV projection is small next to the
target's prefill. A larger drafter, or a cheaper target, would shift the ratio.

Both arms log from inside `_project_context_kv` so the journal proves which path ran in each.
