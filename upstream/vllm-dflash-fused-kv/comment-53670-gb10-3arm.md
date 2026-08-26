Independent reproduction on GB10 (DGX Spark, sm_121), single GPU, Qwen3.8-27B NVFP4 (mixed,
FP8 `lm_head`), fp8 KV, FlashInfer, prefix caching on. Agent-loop probe: ~7.3k seed document,
six turns, each turn appends the reply plus a new question so the prefix grows the way an agent's
does. Cold cache per arm, hits read from `vllm:prefix_cache_hits_total` deltas.

| arm | first hit on turn | hit plateau | hit rate | mean TTFT (turns 2+) |
|---|---|---|---|---|
| **no spec** | **2** | **6272 = 4 x 1568** | **69.4 %** | **0.66 s** |
| mtp, K=3 | 3 | 4800 = 3 x 1600 | 42.5 % | 2.04 s |
| dflash (DFlash2), K=7 | 3 | 4944 = 3 x 1648 | 43.3 % | 1.93 s |

Two costs, both charged by enabling any EAGLE-family speculation:

1. **One whole block of reachable prefix** — four blocks become three. The plateau never grows as
   the prompt reaches ~7.9k; it is pinned one block below the replay boundary.
2. **One extra cold turn** — first hit slips from turn 2 to turn 3.

Together: **-27 points of hit rate and 3x TTFT**, considerably worse than the 6-10 points in the
filing. MTP and DFlash2 are indistinguishable here, which supports this being the EAGLE-family
last-block drop rather than anything drafter-specific.

**Relevant to your scoping:** we run K>0 at all times, so the `skip_draft_when_k0` variant you
propose would not help this workload — the drafter genuinely reads the tail. The cost above is
therefore what a K>0 consumer pays today, and it argues that the successor-aware path (#50897) is
the one that matters for prefix-reusing agent traffic, not only the tiered-K=0 case.

**Not a reversal, though**, and worth stating so the number is not over-read: at a realistic
130-token agent turn the decode gain still dominates on our box (dflash 7.8 s vs no-spec 12.4 s
modelled turn time; break-even ~28 output tokens). Speculation remains net positive here — the
last-block drop narrows the margin rather than erasing it. For very short turns it would flip.

Incidental, since the filing cites 1648: on this box the scheduler block size tracks speculative
depth — 1568 at K=0, 1600 at K=3, 1648 at K=7 — so that number is a property of the depth, not
only of the layout.

_Disclosure: AI-assisted analysis (Claude Code); I ran the benchmarks and reviewed the traces myself._
