## Purpose

`prefix_cache_retention_interval` defaults to **0**, which per its own docstring "retains only
semantic checkpoints, including the latest replay boundary and shared-prefix junctions". On a model
with sliding-window or Mamba groups, `reachable_block_mask` then masks `True` only for
`reachable_boundaries`: every other boundary state is filed and never hashed, so it can never serve
a hit.

**This costs measurable performance, not tidiness.** On our rig it is one extra cold prefill per
fresh prefix, and it caps what store-side work can achieve — while measuring #53479 the chunk carve
was provably correct and the hits did not move, because this default discards the states that PR
creates. Nothing in the log says the trade was made, and `Prefix cache hit rate` still reports
non-zero, so from outside it looks like the prompts simply do not share a prefix.

The asymmetry is what this closes. vLLM already announces the neighbouring derived default
(`model_executor/models/config.py`):

```
Mamba cache mode is set to 'align' for Qwen3_5ForConditionalGeneration by default
when prefix caching is enabled
```

and `_validate_prefix_cache_retention_interval` already inspects exactly the model shape needed. It
raises a detailed `ValueError` for the harmless case — a value set on a model where retention has no
effect — and returns silently for the harmful one, where the model *does* have a sparse group and
the interval is 0.

## Why this is not duplicating an existing PR

Checks run: `gh pr list --state open --search "53595 in:body"`,
`--search "prefix_cache_retention_interval"`, `--search "retention log warning mamba"`.

- **#52527** (`[Metrics] Report shared-prefix tokens lost to a missing sparse-retention checkpoint`)
  describes the same invisibility and is the closer neighbour. It adds a **runtime metric**:
  per-request signal that reuse was found and then discarded, in `stats.py` / `loggers.py`. This PR
  adds a **startup statement**: the configuration will discard, before any request is served, where
  it cannot be missed. Complementary rather than competing — the metric answers "did I lose reuse
  just now", the log answers "will I". If maintainers prefer one, #52527 is the more substantial
  change and this can be dropped.
- **#51886** adds retention support to `OffloadingConnector`; different feature, no overlap.

## Change

One `logger.info` at startup, gated on prefix caching being enabled **and** the model actually
having a sliding-window or Mamba group **and** the interval being 0. It names the consequence and
the knob. `enable_caching` is threaded into the validator; that is the only new argument.

## Tests

```
python -m pytest tests/v1/core/test_prefix_caching.py -k zero_retention_is_announced
# 1 passed
python -m pytest tests/v1/core/test_prefix_caching.py
# 91 passed, 1 failed (test_hybrid_local_kv_retention_interval_survives_recycling,
#   pre-existing on this checkout: MLAAttentionSpec has no 'tokens_per_state' in the
#   installed build; fails identically without this patch)
ruff check / ruff format --check on both changed files: clean
```

Reverting the source change fails the new test, which asserts the line appears at the default and
does **not** appear with a periodic interval or with caching disabled.

**Model evaluation:** not applicable. The change emits one log line and alters no scheduling,
caching, or sampling behaviour; no output, accuracy or serving path is touched.

## Measured

GB10 / sm_121a, `Qwen3.8-27B-NVFP4` (hybrid GDN), MTP `n=3`, block size 1600. Production code,
nothing patched, only the interval changed:

| | default (0) | interval = block size |
|---|---|---|
| 7 292-token prompt, first hit on request | **3** | **2** |
| 6 592 (block-aligned), first hit | 3 | 3 |
| hit amounts | 3 296 / 4 944 | unchanged |

Reported as #53595.

*AI assistance was used for this change (Claude Code): the investigation, patch and tests were
produced with it and are reviewed by me.*
