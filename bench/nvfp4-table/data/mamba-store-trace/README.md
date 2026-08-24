# Where the state is lost: a live per-group trace

vllm#53479 makes every prefill chunk stop at a block boundary so a reusable state exists at each
one. On this rig the chunking does exactly that and the hits still do not move. The open question
was whether the states are never written, or written and not found.

The trace answers it: **written for full attention, not for Mamba, until the second send.**

Instrumentation on the running server — `HITPROBE` from the hybrid
`KVCacheCoordinator.find_longest_cache_hit` (per-group hit lengths and whether the Eagle drop
applied), `STOREPROBE` from `KVCacheManager.cache_blocks`, `SCHEDPROBE` from
`_mamba_block_aligned_split`. MTP n=3, block 1600, one 6 400-token prompt sent three times.

```
PR arm, send 1:  chunks 0 / 1600 / 3200 / 4800     last_cache=6400   STORE num_computed=6400
PR arm, send 2:  FullAttention+eagledrop=3200   Mamba+eagledrop=0      -> final 0
PR arm, send 3:  FullAttention+eagledrop=3200   Mamba+eagledrop=3200   -> final 3200
```

Send 1 carves at all four boundaries — the branch's intent, and not a port artefact. Yet on send 2
the Mamba group reports **zero**, while full attention already has 3 200. The coordinator intersects
the groups, so the request misses. Only after send 2 does the Mamba group carry 3 200.

So the extra chunk stops do not by themselves cause a Mamba state to be published on the first
pass; the store side needs a second pass regardless of where the chunks end. The base arm shows the
same shape with fewer stops (send 1 carves only at 4 800).

Both groups carry the Eagle drop, Mamba included.

## Round two: the filings do reach every boundary

@kamb-code named a candidate at the filing side — `num_reprefillable_tokens = max(0,
num_prefill_lookahead - 1)`, applied when caching and, on the hybrid path, rounded down to the
scheduler block size — and asked for `num_tokens_to_cache` at those two sites. Instrumented:

```
FILE_B mgr=Mamba computed=1600 reprefill=0 finalized=1600 aligned=1600 cached_computed=1600
FILE_B mgr=Mamba computed=3200 reprefill=0 finalized=3200 aligned=3200 cached_computed=3200
FILE_B mgr=Mamba computed=4800 reprefill=0 finalized=4800 aligned=4800 cached_computed=4800
FILE_B mgr=Mamba computed=6400 reprefill=0 finalized=6400 aligned=6400 cached_computed=6400
```

**`reprefill=0` on this runtime**, so the window subtracts nothing and the rounding never bites.
The filings land on **every** boundary, for both groups, and the Mamba manager caches progressively
(`cached 0->1`, `1->2`, `2->3`, `3->4`). His candidate is ruled out.

And request 2 still reports `Mamba=0`. So the state is filed at each boundary and the lookup does
not find it — "published but not findable", not "never published". His separation test (parent once,
then sibling B, no repeats) returns **0 on both arms**, which under the filing evidence above cannot
mean the states were never written.

One observation not yet a conclusion: `MSTORE` counts blocks in `req_to_blocks` whose `block_hash`
is `None`, and in the branch arm's first filing all four are (`blocks=4 null=0 nohash=4`). That set
includes the in-flight tail, so it is a lead rather than a result.

Two probes were lost to my own instrumentation before this one landed: a lookup-side probe that
called into the block pool inside a log line (the logging module swallowed the exception and printed
the raw format string, and every request 500'd), and a `logger` reference in
`single_type_kv_cache_manager.py`, which has no module-level logger. **Check that the module you are
instrumenting has a logger, and keep log arguments to scalars you already hold.**
