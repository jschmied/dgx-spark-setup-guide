@TechPrototyper That closes it — thanks, and thanks for flagging the port; a verbatim hunk on your
line answers what I actually needed, which was whether the layout loads outside synthetic tensors.

I would not read anything into per-tensor's 5.22 being the highest: your own per-channel runs span
5.05–5.60, so all three are indistinguishable on acceptance. That is the useful result — scale
granularity is free, so pick it for the kernel, not for the drafting.

The branch now has a regression test for this case:
`tests/v1/spec_decode/test_dflash_context_kv_dequant.py` (config-only, no GPU). All four storage
forms plus both rejection paths; the per-shard case is mutation-checked.

Since you build from a pin: DFlash2 does not load on current main at all — #52560 reverted the
`decoder_layer_cls` indirection this PR added (#53428, fix in #53435).

@Tejas-Raj01 — real-checkpoint confirmation plus a test now, if that helps it into #53122.
