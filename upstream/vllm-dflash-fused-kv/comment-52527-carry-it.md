Yes to both — please link the trace.

On the duplication: with `df6372608` on this branch, #53596 and this PR now carry the same
commit and will conflict at merge. I would rather this one carried it, since the metric is the
substantial half and the log line is three lines of it. Say the word and I will close #53596
pointing here; if reviewers would rather land the startup line separately first, I will keep it
open and you can drop the cherry-pick instead.

One scoping note for the description if you link the trace: it was measured at
`prefix_cache_retention_interval` default 0 on a GB10 with MTP, block size 1,600 — the default
costs one extra cold pass on unaligned prompts there (first hit on request 3 rather than 2).
