Negative data point from CUDA, to help bound this: **does not reproduce on sm121 (GB10) with
FlashInfer.** Qwen3.8-27B NVFP4 target, DFlash2 drafter, 7 draft tokens, 24 requests per arm.

| concurrency | acceptance length | draft accept rate | aggregate |
|---:|---:|---:|---:|
| 1 | 3.96 | 42.3% | 34.0 tok/s |
| 4 | 3.97 | 42.4% | 74.0 tok/s |
| 8 | 3.89 | 41.3% | 101.2 tok/s |

The per-position acceptance is what rules it out here — evicted sliding-window K/V should cost the
*deep* positions disproportionately, and they are unchanged:

```
c=1:  0.80 0.63 0.50 0.39 0.29 0.21 0.15
c=8:  0.80 0.62 0.47 0.38 0.27 0.21 0.15
```

So the sliding-window layers in the DFlash2 drafter are not sufficient on their own — this looks
specific to the ROCm prefix-attention path rather than to the drafter architecture.
