Reproduced on GB10 (Qwen3.8-27B NVFP4, fp8 KV, FlashInfer, single GPU). Agent-loop probe: ~7.3k
seed, 6 turns, each appending the reply + a new question. Cold cache per arm, hits from
`prefix_cache_hits_total` deltas.

| arm | first hit | plateau | hit rate | TTFT (turns 2+) |
|---|---|---|---|---|
| **no spec** | **turn 2** | **6272 = 4 x 1568** | **69.4 %** | **0.66 s** |
| mtp K=3 | turn 3 | 4800 = 3 x 1600 | 42.5 % | 2.04 s |
| dflash2 K=7 | turn 3 | 4944 = 3 x 1648 | 43.3 % | 1.93 s |

Speculation costs **one whole block of reachable prefix (4 -> 3) plus one extra cold turn**:
-27 points and 3x TTFT, well past the 6-10 points in the filing. MTP and DFlash2 are
indistinguishable, so it is the EAGLE-family drop, not drafter-specific.

Two notes:

- We run **K>0 always**, so `skip_draft_when_k0` would not help this workload. What we pay is
  what a K>0 consumer pays today — which argues #50897 is the path that matters for
  prefix-reusing agent traffic.
- **Not a reversal:** at a 130-token turn the decode gain still wins here (7.8 s vs 12.4 s
  modelled; break-even ~28 output tokens). It narrows the margin, not erases it.

Incidental, since the filing cites 1648: block size tracks speculative depth on this box —
1568 / 1600 / 1648 at K=0 / 3 / 7.

_Disclosure: AI-assisted analysis (Claude Code); I ran the benchmarks and reviewed the traces myself._
