Found what gates it here, and it is not a defect in this PR — it is a default.

`prefix_cache_retention_interval` defaults to **0** (`config/cache.py`,
`_get_prefix_cache_retention_interval` returns 0 when the env var is unset), and per its own
docstring 0 "retains only semantic checkpoints, including the latest replay boundary and
shared-prefix junctions". In `MambaManager.reachable_block_mask`, `retention_interval == 0` skips
the segment branch entirely, so only `reachable_boundaries` are masked `True` — every other boundary
state is filed and then never hashed.

That is why the carve is provably right here and the hits do not move: this PR creates a state at
every boundary, and the retention mask discards all but the semantic ones. Your harness presumably
runs dense (`None`), which is exactly the cell where our numbers diverged.

Measured, production code, no PR applied, only the interval changed:

| | default (0) | interval = block size |
|---|---|---|
| 7 292-token prompt, first hit on request | **3** | **2** |
| 6 592 (block-aligned), first hit | 3 | 3 |
| hit amounts | 3 296 / 4 944 | unchanged |

So the default costs one cold pass on unaligned prompts, and it caps what any store-side change can
show. Worth stating in the PR: without `prefix_cache_retention_interval` set, the benefit is
invisible on a default install — which is most of them.
