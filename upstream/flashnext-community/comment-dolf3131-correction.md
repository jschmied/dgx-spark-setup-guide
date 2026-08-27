**Correction to section 4 of the above — the body is not the suspect, and I was wrong to point
there.**

[blazux/qwen3.8-Flash-DGX](https://github.com/blazux/qwen3.8-Flash-DGX) serves
`RadixArk/Qwen3.8-Flash-Next-NVFP4` **coherently on a single Spark**, on the same image digest I
used (`sha256:fc120ece0a38`), with a smoke test that asserts the output ("The capital of France
is Paris."). So the RadixArk body, the checkpoint and that image are all fine, and my "3.3 GiB
body difference" lead was a dead end. That gap is benign anyway: RadixArk simply keeps ~3.5 GiB
more in BF16 (MTP included) where Inferact quantizes it.

**What actually separates my run from both working RadixArk recipes: `VLLM_PLE_CPU_OFFLOAD=1`.**
blazux replaces the PLE layer with an mmap gather; OsakaTX shards it as a
`VocabParallelEmbedding` across TP=2. Neither uses the offload worker. Every known-good RadixArk
run avoids it — which reframes my report: the FP8-PLE **offload** path is the unexercised one,
not FP8-PLE as such.

Two further corrections to my own list, both from other people's write-ups:

- **`--enforce-eager` does not eliminate cudagraph capture here.** Per blazux, the mamba /
  short-conv path still captures, so my "cudagraph vs eager" elimination was invalid. Their
  tested configuration is `PIECEWISE` with an explicit `splitting_ops` list.
- **Prefix caching is implicated by two independent engines.** blazux ships
  `--no-enable-prefix-caching` because a GDN `in_proj` GEMM hits `CUBLAS_STATUS_INTERNAL_ERROR`
  on the cached-block path; tonyd2wild's SGLang recipe disables radix cache because it
  "triggers the spec-verify garbage".

And the lead I had dismissed, which OsakaTX documents against exactly my symptom: the PR adds a
**sigmoid** output-gate variant to `fused_gdn_decode_post_conv_mtp` (qwen4_exp uses sigmoid where
earlier Qwen GDN models used silu), and prebuilt binaries predate it — with
`VLLM_GDN_DECODE_KERNEL=triton` as a zero-build workaround. I had ruled that out on the grounds
that the op is gated behind `num_spec_decodes > 0`; that reasoning covered only one call site.

Testing the offload-free and Triton-GDN configurations now; I will report the outcome either way.
Apologies for sending you down the body path.
