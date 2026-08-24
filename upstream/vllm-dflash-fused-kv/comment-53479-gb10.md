Ran this on the GB10 rig as asked. Both changes are provably active — a log line inside
`_mamba_block_aligned_split` shows `last_cache_position` going 4 800 → 6 400 and an extra chunk stop
appearing at 1 600 — but the hits do not move: 3 200 on a 6 400-token prompt, 4 800 on 7 100, same
as base.

**Caveat that may be the whole story:** our build predates #52789, so there is no
`use_internal_checkpoint`. I applied your intent for that case — no back-off, unconditional boundary
stop — rather than your branch verbatim. On the same rig #50897 does move these prompts
(4 800 / 6 400), so the probe resolves this kind of change.

Happy to re-run against your branch on a build that has #52789 if that is the more useful test.
