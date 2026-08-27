This is the most useful thing anyone has produced on this — and your section 3 identifies a test
that only your side can run.

**The control I would ask for: make Inferact look like RadixArk on that one axis.** Add
`*.self_attn.*` as a wildcard to Inferact's exclusion list (or otherwise force the QSA
`q/k/v/o_proj` to stay BF16) and re-run prompt A. If the output degrades, the bug is vLLM's QSA
path mishandling *unquantized* projections, and everything else in this thread is noise. If it
stays coherent, the exclusion gap is incidental and I will stop pointing at it.

That is a much cheaper experiment than anything left on my side, because you can produce both
arms; I can only produce the broken one.

You are right that it is the wrong direction on its face. What makes me take it seriously anyway
is that it is now the *only* isolated difference left: same architecture, same per-expert NVFP4
layout, same PLE format handling, same image version string, same TP=1 + `mp`, and I have
independently eliminated the PLE end to end (its FP8 rows match the official BF16 table at cosine
0.999635; I also dequantized in the worker and handed the GPU a BF16 buffer, making it
structurally identical to yours — byte-identical garbage). Your "a fused path that assumes its
inputs are quantized" is exactly the shape of thing that survives all of that.

**Your A/B/C reference is on my list to run**, and I will post the comparison. One caveat I want
to flag before I do, so the result is not over-read: my box may be the variable. I replicated
[blazux](https://github.com/blazux/qwen3.8-Flash-DGX)'s single-Spark RadixArk recipe down to the
image digest, their `vllm_ple_mmap` hook and their 12-entry `splitting_ops`, and still get salad
— on hardware that serves Qwen3.8-27B coherently every day. So a divergence pattern from my side
may be measuring my environment rather than the checkpoint. That is filed separately
([blazux#1](https://github.com/blazux/qwen3.8-Flash-DGX/issues/1)), driver question first.

Given that, the Inferact-with-self_attn-excluded control is worth more than my A/B/C numbers:
it is a single-variable experiment inside one known-good environment, with no box difference to
confound it.

Thanks also for killing `group_size` — I had not gotten to it, and the `:1066` default plus the
`config_groups` fallback is exactly the kind of thing I would have spent an hour on.
