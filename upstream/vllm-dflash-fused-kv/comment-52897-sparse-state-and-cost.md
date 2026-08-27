@kamb-code — your sparse-state result explains a runtime observation I could not account for in my
previous comment, and it lets me put a price on the gate. Both from the GPU side, same rig
(GB10 / sm_121a, `Qwen3.8-27B-NVFP4`, hybrid GDN, `align`).

**"One snapshot per chunk end" is visible end-to-end.** In every arm I ran, the *first* repeat never
hits — the third request is the first that does. An 8-second pause before the second request changes
nothing, so it is not a release race. Your model accounts for it exactly: a 6 400-token prompt fits
in one chunk at a 8 192 budget, so the only chunk end is 6 400, which is above the `prompt_len - 1`
cap and above it again after the back-off — unreachable. Request 1 caches the full-attention blocks;
that shifts request 2's carve so a chunk now ends at 3 200, which finally publishes a reachable
state; request 3 hits it. The measured hit is exactly 3 200.

**The price of the gate**, identical prompts, the only variable being whether a speculative config
is present (MTP `n=3`):

| | block size | 6 400 tok | 7 100 tok | first hit on request | warm prefill |
|---|---:|---:|---:|---:|---:|
| MTP n=3 | 1 600 | 3 200 (2 of 4 blocks) | 4 800 | 3 | 1.42 s |
| **no speculative config** | 1 568 | **6 272 (4 of 4)** | **6 272** | **2** | **0.15 s** |

Half the reusable prefix on an aligned prompt, one extra cold pass before anything is reusable at
all, and ~9x on the warm prefill. Cold prefill is ~3.1 s either way, so this is the cache, not the
drafter.

Two gates now key on the same predicate, not one: besides the back-off, #52789 added
`mamba_has_prefill_checkpoint_blocks` (9–25 % TTFT) carrying `# TODO: support spec decoding` right
above `and not self.use_eagle`. And `use_eagle()` is true for `eagle`, `eagle3`, `mtp`, `dflash` and
`dspark` — the whole speculative family on hybrids, not an EAGLE corner.

For completeness: I ran both candidate fixes here (#50897 and #52244, each against its own
merge-base, with and without `--prefix-match-unit`) and neither moved these numbers; details in
those threads. The no-drafter row is what tells me the probe would have seen it.
