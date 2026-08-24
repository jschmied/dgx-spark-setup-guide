Your description matches what I traced on a GB10, and I can add the cause your text does not name.

The invisibility comes from `prefix_cache_retention_interval` defaulting to **0**
(`config/cache.py`, `_get_prefix_cache_retention_interval` returns 0 when the env var is unset).
With that, `MambaManager.reachable_block_mask` skips the segment branch entirely and masks `True`
only for `reachable_boundaries` — so a sparse group files a state at every boundary it crosses and
all but the semantic ones are never hashed. Nothing logs it; the value appears in no log line in the
tree, and the only related message is the deprecation warning that fires when you *set* the env var.

Live per-group trace, `Qwen3.8-27B-NVFP4` (hybrid GDN), MTP n=3, block 1600, one 6 400-token prompt:

```
send 1:  chunks 0/1600/3200/4800    FILE_B mgr=Mamba computed=1600..6400 reprefill=0
send 2:  FullAttention=3200   Mamba=0      -> intersected to 0
send 3:  FullAttention=3200   Mamba=3200   -> 3200
```

Exactly your "found and then thrown away", with the filings proving the states were written.

Cost, production code, only the interval changed:

| | default (0) | interval = block size |
|---|---|---|
| 7 292-token prompt, first hit on request | **3** | **2** |
| hit amounts | 3 296 / 4 944 | unchanged |

One extra cold prefill per fresh prefix. It also caps store-side work: on #53479, whose chunks stop
at every boundary so a state exists there, the carve was provably correct and the hits did not move,
because this default discards those states.

I opened #53596 (a startup log line saying the trade was made) before finding this PR. Yours is the
more substantial change and answers "did I just lose reuse"; mine only answers "will I". Happy to
close mine if you would rather have one mechanism.

*AI assistance was used for this analysis (Claude Code); the hardware runs and the numbers are mine.*
