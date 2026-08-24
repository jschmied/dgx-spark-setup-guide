Your candidate is ruled out, and it flips the direction: **`reprefill=0` on this runtime**, so the
window subtracts nothing and the hybrid rounding never bites.

```
FILE_B mgr=Mamba computed=1600 reprefill=0 finalized=1600 aligned=1600 cached_computed=1600
FILE_B mgr=Mamba computed=3200 reprefill=0 finalized=3200 aligned=3200 cached_computed=3200
FILE_B mgr=Mamba computed=4800 reprefill=0 finalized=4800 aligned=4800 cached_computed=4800
FILE_B mgr=Mamba computed=6400 reprefill=0 finalized=6400 aligned=6400 cached_computed=6400
```

Branch arm, send 1. The filings land on **every** boundary for both groups, and the Mamba manager
caches progressively (`0->1`, `1->2`, `2->3`, `3->4`). Request 2 still reports `Mamba=0`.

So it is not "publication never happens" — it is **published and not findable**. Your separation
test agrees but cannot mean what it was meant to: parent once, then sibling B, no repeats → **0 on
both arms**, with the filings above proving the states were written.

One lead, not a result: my counter says all four blocks carry `block_hash is None` at that first
filing (`blocks=4 null=0 nohash=4`). That set includes the in-flight tail, so treat it as a
direction rather than a finding — but the lookup side is where I would look next.
