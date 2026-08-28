**Correcting my own report, and adding two negative results that narrow it further.**

@jahnclawdmonet — your diagnosis is right and mine was not. I verified the attribute mismatch in
the same image: `ModelOptFp8PbWoLinearMethod.__init__` assigns `self.w8a8_block_fp8_linear`
(line 784), while `process_weights_after_loading` guards on `hasattr(self, "fp8_linear")`
(line 824) — the name every *other* method in that file uses (503/524, 593/608). The guard never
fires, so the kernel's weight post-processing is skipped. My "the `120` capability family is too
coarse" framing was wrong; please read the title as inaccurate.

**But fixing the guard is not sufficient**, at least against a checkpoint whose scales were written
elsewhere. Two runs on GB10 / sm_121, `lovedheart/Qwen3.8-Flash-Next-NVFP4-FP8` (ModelOpt
`MIXED_PRECISION`, 156 `FP8_PB_WO` tensors, plain **FP32** block scales):

| run | result |
|---|---|
| guard fixed, `VLLM_USE_DEEP_GEMM=1` | same crash — `CUDA_ERROR_LAUNCH_FAILED` in `fp8_gemm_nt` |
| guard fixed, `VLLM_USE_DEEP_GEMM=1`, `VLLM_USE_DEEP_GEMM_E8M0=0` | **different** crash — `illegal memory access` (Triton) |
| `VLLM_USE_DEEP_GEMM=0` | **works** — falls back to `CutlassFp8BlockScaledMMKernel`, serves correctly |

The guard fix restores the post-processing *call*, but it cannot retroactively produce UE8M0-format
scales for a checkpoint that never contained them — the failing frame is
`_fp8_gemm_nt_impl(..., disable_ue8m0_cast=not use_ue8m0, ...)`, and on sm_121
`is_deep_gemm_e8m0_used()` returns True through the same family-120 check. Disabling the E8M0 cast
then fails differently, which suggests the non-E8M0 path is not viable here either rather than that
the scales are merely mis-formatted.

So the accurate statement seems to be: **the attribute bug is real and worth fixing on its own**,
and separately, **DeepGEMM's E8M0 variant is selected on sm_121 for checkpoints that do not supply
E8M0 scales.** Whether sm_121 can run DeepGEMM at all with correctly post-processed weights is
something your dummy-weight micro-reproducers answer better than my end-to-end runs do — you
report clean runs with properly packed UE8M0 scales, which I cannot reproduce from this checkpoint
because the packing step is exactly what was skipped when the weights were made.

Practical note for anyone landing here from a search: `VLLM_USE_DEEP_GEMM=0` is the working
workaround on GB10 today, and the Cutlass fallback serves correctly — 26.1 tok/s single-stream on
this checkpoint, no quality regression against the unquantized-dense build.

Happy to run any specific A/B on GB10 with real weights if it would help pin the remaining half.
