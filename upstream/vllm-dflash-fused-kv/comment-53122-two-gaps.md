Heads-up that the `_dequant_kv_slice` here is the version from 21 Aug — two things have been added
to it since, and both are load-or-fail for real checkpoints rather than polish.

**1. Pack-quantized drafters have no path.** The helper reads `qkv.weight`, but a compressed-tensors
`pack-quantized` checkpoint (W4A16 / W8A16) has no plain `weight` at all — values sit in
`weight_packed` as int32 with group scales. That is not a corner case: the INT4 group-128 drafter is
the fastest one I have measured (44.6 tok/s against 41.3 for BF16, acceptance 4.13 vs 4.24) and it
is what I run in production.

**2. A per-tensor scheme on a fused `qkv_proj` stores one scalar per shard**, so `weight_scale` has
shape `(3,)` — neither `numel() == 1` nor one entry per row, so it falls through to the ValueError.
@TechPrototyper hit exactly this, and confirmed the fix against a real per-tensor fused fp8
checkpoint on sm121 (#52816): loads, 41.5 tok/s single-stream, acceptance 5.22.

Both are on the branch, on this PR's base:
https://github.com/jschmied/vllm/compare/b389ac29465b33f9e9c534df221ea3c129e9793f...jschmied:vllm:dflash-quantized-drafter

The shard scalars have to be expanded over the rows they own *before* the `[q_size:]` slice —
slicing a `(3,)` scale hands K the scale belonging to V, which does not crash and only shows up as
worse acceptance. The packed path slices rows before unpacking rather than after; both tensors are
row-major over output features, so it is the same result for a third of the transient allocation.

There is also a test covering all four storage forms plus both rejection paths
(`tests/v1/spec_decode/test_dflash_context_kv_dequant.py`, config-only, no GPU) — yours and mine
overlap on two cases, so it is worth merging rather than taking wholesale.
