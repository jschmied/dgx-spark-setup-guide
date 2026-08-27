### Your current environment

vLLM `0.26.1rc1.dev912` (main @ `8d6b1832`), GB10 / sm_121a, `Qwen3.8-27B-NVFP4` (hybrid GDN),
`--enable-prefix-caching`, `--kv-cache-dtype fp8`, MTP `num_speculative_tokens: 3`, block size 1600.

### Describe the bug

`prefix_cache_retention_interval` defaults to **0**, which per its own docstring "retains only
semantic checkpoints, including the latest replay boundary and shared-prefix junctions". On a hybrid
model with prefix caching enabled, `MambaManager.reachable_block_mask` then skips the segment branch
entirely and masks `True` only for `reachable_boundaries`, so every other boundary state is filed and
never hashed — it can never serve a hit.

Nothing says so. **`retention_interval` is not logged anywhere in the tree**, `Prefix cache hit rate`
still reports non-zero, and the only related message is the deprecation warning that fires when you
*set* the env var — you are told once you have fixed it, never while you are affected.

The asymmetry is the part I would call a defect. vLLM already announces the neighbouring derived
default (`model_executor/models/config.py:606`):

```
Mamba cache mode is set to 'align' for Qwen3_5ForConditionalGeneration by default
when prefix caching is enabled
```

And `_validate_prefix_cache_retention_interval` (`v1/core/kv_cache_coordinator.py:32`) already
inspects exactly the model shape needed:

```python
if not any(isinstance(g.kv_cache_spec, (SlidingWindowSpec, MambaSpec))
           for g in kv_cache_config.kv_cache_groups):
    if retention_interval == 0:
        return
    raise ValueError("prefix_cache_retention_interval is set but this model has "
                     "no sliding-window or Mamba KV cache group, so retention has no effect. ...")
```

It raises a detailed error for the harmless case — value set where it has no effect — and returns
silently for the harmful one: the model *does* have Mamba groups, the interval is 0, and prefix
caching is on.

### Measured effect

Production configuration, nothing patched, only the interval changed:

| | default (0) | interval = block size |
|---|---|---|
| 7 292-token prompt, first hit on request | **3** | **2** |
| 6 592 (block-aligned), first hit | 3 | 3 |
| hit amounts | 3 296 / 4 944 | unchanged |

One cold pass per fresh prefix. It also caps what store-side work can achieve: while investigating
#53479 — which makes every prefill chunk stop on a block boundary so a state exists there — the
carve was provably correct and the hits did not move, because the retention mask discards those
states after they are filed.

### Suggested fix

Log once, at startup, when prefix caching is enabled **and** the model has Mamba or sliding-window
groups **and** `retention_interval == 0`: that only semantic checkpoints will be retained, and which
knob changes it. The validator already holds every value needed.
