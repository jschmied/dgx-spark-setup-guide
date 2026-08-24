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
