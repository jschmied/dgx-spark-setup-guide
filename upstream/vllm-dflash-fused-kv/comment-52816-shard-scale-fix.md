@TechPrototyper Thanks — and you found a real gap, not a documentation gap. Fixed.

The dequant path handled a per-tensor scalar and a per-row vector, but not the third form your checkpoint has: a per-tensor scheme on a **fused** layer stores one scalar *per shard*, so `weight_scale` is `(3,)` for q/k/v — neither `numel()==1` nor `numel()==out_features`. It now expands each shard's scalar over the rows that shard owns, using `output_partition_sizes` (falling back to `output_sizes`), and only then slices `[q_size:]`, so K and V each get their own scale:

```python
sizes = getattr(qkv, "output_partition_sizes", None) or getattr(qkv, "output_sizes", None)
if sizes is not None and s.numel() == len(sizes) and sum(sizes) == w.shape[0]:
    s = torch.cat([s[i].expand(int(n)) for i, n in enumerate(sizes)])
```

The `sum(sizes) == w.shape[0]` guard keeps it from firing on a coincidental length match; anything it still cannot map keeps the explicit error rather than silently broadcasting one shard's scale across another's rows — which would not crash and would only show up as worse acceptance.

Branch, on the #52816 merge commit:
https://github.com/vllm-project/vllm/compare/b389ac29465b33f9e9c534df221ea3c129e9793f...jschmied:vllm:dflash-quantized-drafter

Verified against synthetic tensors for all four cases — per-shard `(3,)`, per-tensor scalar, per-channel full-length, and unquantized pass-through — plus the error path. Not verified against a real per-tensor fused checkpoint, since I do not have one; if your offer to test that variant still stands, that is the piece I am missing.

@Tejas-Raj01 — worth folding into #53122 alongside the part you already took.

Your numbers are a useful second data point: fp8-full 42.5 tok/s single-stream and 205.2 at c=8 against 38–42 / 179 for BF16, acceptance 5.05 vs 5.11. Same direction as ours on a different target and harness — quantizing the drafter buys throughput and costs no measurable acceptance.
