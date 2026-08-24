Hypothesis 1 is ruled out — the stop does recur per chunk in the port. Journal timestamps from a
single send:

```
12:35:31 SCHEDPROBE[PR53479] start=0    num_new=6400 last_cache=6400
12:35:31 SCHEDPROBE[PR53479] start=1600 num_new=4800 last_cache=6400
12:35:32 SCHEDPROBE[PR53479] start=3200 num_new=3200 last_cache=6400
12:35:33 SCHEDPROBE[PR53479] start=4800 num_new=1600 last_cache=6400
```

Send 1 carves `[1600, 3200, 4800, 6400]`, exactly the branch's intent. Send 2 (12:35:35) then starts
at 0 → 1600 again and hits **0**.

So your discriminating number is already in the data: **hits on request 2 are 0 on both arms**, base
and branch. The divergence therefore is not chunking — the chunks end where they should — but that
nothing reachable is published at those boundaries on this runtime. Which leaves your second
hypothesis, the live MTP lookahead deferring hashing, as the standing candidate.

The verbatim run on a #52789-inclusive build stands; sibling A at 3 400 shared remains the cell to
watch.
