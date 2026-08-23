A number for what speculative decoding costs the hybrid prefix cache, since this thread has the
mechanism but I did not find the price anywhere.

Same rig as my earlier comment (GB10 / sm_121a, `Qwen3.8-27B-NVFP4`, hybrid GDN, `align`), MTP
`num_speculative_tokens: 3`. Identical prompts, calibrated to exact token counts, each sent three
times; the only variable is whether a speculative config is present.

| | block size | 6 400-token prompt | 7 100-token prompt | first hit on request | warm prefill |
|---|---:|---:|---:|---:|---:|
| MTP n=3 | 1 600 | 3 200 (2 of 4 blocks) | 4 800 | 3 | 1.42 s |
| **no speculative config** | 1 568 | **6 272 (4 of 4)** | **6 272** | **2** | **0.15 s** |

So on this configuration speculative decoding costs half the reusable prefix on an aligned prompt,
one extra cold pass before anything is reusable at all, and roughly 9x on the warm prefill. Cold
prefill is ~3.1 s either way, so this is entirely the cache, not the drafter.

Worth noting there are now **two** gates keyed on the same predicate, not one. Besides the
back-off, `#52789` added `mamba_has_prefill_checkpoint_blocks` — 9–25 % TTFT — and it carries
`# TODO: support spec decoding` right above `and not self.use_eagle`. And `use_eagle()` returns
true for `eagle`, `eagle3`, `mtp`, `dflash` and `dspark`, so this is the whole speculative family
on hybrids, not an EAGLE corner.

For completeness: I tried both candidate fixes on this rig (#50897 and #52244, each against its own
merge-base, with and without `--prefix-match-unit`) and neither moved the numbers — details in
those threads. The no-drafter row above is what tells me the probe would have seen it.
