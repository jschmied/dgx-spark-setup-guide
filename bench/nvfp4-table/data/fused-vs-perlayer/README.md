# What the fused context-KV GEMM is actually worth

Our patch for vllm#51581 dequantizes the K/V slices so the fused cross-layer GEMM survives.
The competing fix (#51620) routes each projection through its own `quant_method.apply()`
instead, giving the fusion up — and on #51581 @seanyourhighness recommended landing that
conservative path as the correctness baseline.

The argument for keeping the fusion had never been measured. It is now.

Prod configuration: RadixArk Qwen3.8-27B-NVFP4, **DFlash2 n=7** with the W4A16 drafter,
`--kv-cache-dtype fp8`. The fused GEMM runs **once per prefill**, so TTFT is where it can pay;
decode is untouched. Unique prompts per request so prefix caching cannot shorten the prefill,
three repetitions, median reported.

| context | fused | per-layer | delta | individual runs |
|---|---:|---:|---:|---|
| ~1k | 0.470 s | 0.517 s | +10.0 % | `0.36 0.47 0.525` vs `0.374 0.517 0.634` |
| ~8k | 3.396 s | 3.475 s | +2.3 % | `2.884 3.396 3.51` vs `2.919 3.475 3.539` |
| ~30k | 14.638 s | 14.825 s | +1.3 % | `13.263 14.638 15.421` vs `13.432 14.825 15.576` |

**The fusion buys 1–2 % of TTFT at working context lengths, and the run-to-run spreads overlap
at every length.** The 10 % at ~1k is one 50 ms difference between two three-sample medians whose
ranges overlap; it is not a result. So the conservative per-layer path costs approximately nothing
here, and the recommendation to land it first is well founded on this hardware.

Scope: the drafter has 5 layers and its context-KV projection is small next to the target's
prefill. A larger drafter, or a target whose prefill is cheaper, would shift the ratio.

Both arms log `KVPROBE[FUSED|PERLAYER]` from inside `_project_context_kv`, and the journal
timestamps place each marker in its own arm's window — after the venv-shebang episode
(`../prefix-cache-prs/README.md`) no arm here is trusted without that.
