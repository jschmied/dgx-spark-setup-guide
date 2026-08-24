## Purpose

`prefix_cache_retention_interval` defaults to **0**, which per its own docstring "retains only
semantic checkpoints, including the latest replay boundary and shared-prefix junctions". On a model
with sliding-window or Mamba groups, `reachable_block_mask` then masks `True` only for
`reachable_boundaries` — every other boundary state is filed and never hashed, so it can never serve
a hit. `Prefix cache hit rate` still reports non-zero, so nothing looks wrong from outside.

Nothing says so. `retention_interval` is not logged anywhere in the tree, and the only related
message is the deprecation warning that fires when the env var is **set** — you are told once you
have changed it, never while the default is costing you.

The asymmetry is what this fixes. vLLM already announces the neighbouring derived default
(`model_executor/models/config.py`):

```
Mamba cache mode is set to 'align' for Qwen3_5ForConditionalGeneration by default
when prefix caching is enabled
```

and `_validate_prefix_cache_retention_interval` already inspects exactly the model shape needed: it
raises a detailed `ValueError` for the harmless case — a value set on a model where retention has no
effect — and returns silently for the harmful one, where the model *does* have a sparse group and
the interval is 0.

## Change

One `logger.info` at startup, gated on prefix caching being enabled **and** the model actually
having a sliding-window or Mamba group **and** the interval being 0. It names the consequence and
the knob. `enable_caching` is threaded into the validator, which is the only new argument.

## Measured

GB10 / sm_121a, `Qwen3.8-27B-NVFP4` (hybrid GDN), MTP `n=3`, block size 1600. Production code,
nothing else changed:

| | default (0) | interval = block size |
|---|---|---|
| 7 292-token prompt, first hit on request | **3** | **2** |
| 6 592 (block-aligned), first hit | 3 | 3 |
| hit amounts | 3 296 / 4 944 | unchanged |

One cold pass per fresh prefix. It also caps what store-side work can achieve: while measuring
#53479 the chunk carve was provably correct and the hits did not move, because this default
discards the states that PR creates.

## Test

`test_zero_retention_is_announced_on_sparse_models` asserts the line appears for a hybrid config at
the default, and does **not** appear with a periodic interval or with caching disabled. Reverting
the source change fails it; the rest of `test_prefix_caching.py` is unaffected.

Reported as #53595.
