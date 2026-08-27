Your checkpoint table lists `RadixArk/Qwen3.8-Flash-Next-NVFP4` as **loads? no**. It does load —
the blocker is one `isinstance` check — but it then generates garbage, so your practical
conclusion stands. Posting the details because you have the working Inferact setup and are the
person best placed to diff against it. Everything below is on a single DGX Spark (GB10, sm_121),
image `vllm/vllm-openai:qwen38-flash-next` @ `sha256:fc120ece0a38`.

## 1. Why it is rejected, and how to get past it

`models/qwen3_8_flash_next/nvidia/ple_layer.py`:

```python
def _get_ple_embedding_quant_method(quant_config, prefix):
    """Select global-scale FP8 only for quantized PLE checkpoint shards."""
    if not isinstance(quant_config, Fp8Config):
        return None          # RadixArk is modelopt/NVFP4 -> rejected here
```

The gate keys on the **body's** quant config, not the PLE's format. Accepting
`modelopt`/`modelopt_fp4` as well loads the model: 73.77 GiB resident, 294,638-token KV,
`Application startup complete`. Note that under `VLLM_PLE_CPU_OFFLOAD` the method must be handed
out **only in the offload process** (`envs.VLLM_PLE_CPU_OFFLOAD and not is_offload_process()`
returns `None`), otherwise `create_weights()` registers a GPU-side `weight_scale` that
`load_weights()` never fills, and it shadows `_offload_weight_scale`.

## 2. RadixArk's PLE is a faithful quantization

I compared its FP8 rows against `Qwen/Qwen3.8-Flash-Next`'s BF16 rows (HTTP range reads, a few
KB, no download):

    official row0[:6] : [-0.009216, 0.014282, -0.016479, 0.013306, -0.00708,  0.00705]
    dequantized [:6]  : [-0.009567, 0.014351, -0.015945, 0.012756, -0.007175, 0.007175]
    cosine 0.999635   rel err 2.4%   absmax 0.040039 vs 0.041458

Format is 128 x `F8_E4M3 [2500012, 160]` plus one global `BF16 [1]` scale — exactly what
`Qwen3_8FlashNextPLEFp8EmbeddingMethod` implements. So "44 GiB less to download and half the
swap" is real in principle; something else is wrong.

## 3. It still emits garbage, and here is what it is *not*

Nineteen hypotheses eliminated. The ones likely to save you time:

- **Not the PLE data path.** Verified on served requests (not warmup — the first call is a dummy
  forward that zeroes the buffer and signals, which will mislead you): fp8 branch taken, scale
  `0.00019931793212890625`, post-dequant absmax 0.026-0.035, matching the official table's 0.040.
- **Not the MoE backend.** `FLASHINFER_CUTLASS` and `MARLIN` both corrupt identically; `triton`
  is invalid for NVFP4 and `flashinfer_trtllm` refuses the config.
- **Not the exclusion list.** RadixArk's 13 wildcards vs Inferact's 1267 explicit entries looked
  like the answer; matching every module gives **73,728 quantized correctly kept, 1,319
  unquantized correctly excluded, 0 wrong**. vLLM's modelopt parser does handle wildcards.
- **Not the expert layout.** I checked yours: `nvfp4_experts-*` carries
  `experts.N.{gate,up,down}_proj.weight` as `U8` with `F8_E4M3` scales — byte-for-byte the same
  per-expert split layout RadixArk uses.
- Also not: `--no-enable-flashinfer-autotune`, async scheduling (off in both), MRV1 vs MRV2,
  cudagraph vs eager, shard ordering (parsed numerically and shape-validated), or
  vllm#40252's `linear_attn` naming (identical to the official checkpoint, BF16, unquantized).

- **Not the PLE dtype either.** I patched the offload worker to dequantize the fp8 rows itself
  and hand the GPU a **BF16** buffer, making the path structurally identical to Inferact's. The
  output is **byte-identical** to every other run. Combined with the above, the corruption is
  invariant to everything I have varied, so it sits upstream of all of it.

**Where that points: the body.** Your Inferact body is ~74.9 GiB (170.2 total - 95.4 PLE);
RadixArk's is ~78.2 GiB (125.9 - 47.7). The two bodies are **3.3 GiB apart**, so despite the
identical per-expert layout something outside the experts is quantized differently. That is the
next thing I would bisect, and where your working setup gives you an advantage I do not have.

## 4. What would settle it, if you have a minute

A first-token logit diff on your Inferact setup against the same prompt at temperature 0 would
say immediately whether the divergence starts at layer 1 (PLE) or later (body). I cannot run
that control here: a single Spark cannot hold this model with the PLE resident, because with
offload disabled the table is a CUDA allocation and does not swap.

Unrelated but worth having: the TP=1 offload hang you documented was fixed upstream this morning
— commit `95dc96d1d012` in PR #53899 adds `spawn_ple_offload()` / `wait_ple_offload_ready()` to
`uniproc_executor.py`. Your `--distributed-executor-backend mp` workaround remains correct for
builds predating it, and I confirmed it independently on GB10 before that landed.

Full write-up, including the failed attempts: https://github.com/jschmied/qwen38-flash-next-gb10

_Disclosure: AI-assisted analysis (Claude Code); I ran the measurements and reviewed them myself._
