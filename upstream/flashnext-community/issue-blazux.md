Your recipe is the only documented case of `RadixArk/Qwen3.8-Flash-Next-NVFP4` producing coherent
output on a **single** GB10, so I tried to reproduce it here and cannot. Everything I can match,
I matched — including your `src/vllm_ple_mmap.py` and your `splitting_ops` list. Output is token
salad. I suspect an environment difference rather than a config one, and you would spot it faster
than I can guess it.

**Mine**

    NVIDIA GB10, driver 580.173.02, compute_cap 12.1, aarch64, Ubuntu 24.04.4
    DGX Spark (not GX10), 121 GiB unified
    vllm/vllm-openai@sha256:fc120ece0a388cc0aa1caad4a9f1cd92113484ab7ec2fd0efadd62585be05bf8
    checkpoint RadixArk/Qwen3.8-Flash-Next-NVFP4, 418/418 files, every size matching the HF API

Launch (your hook appended to a pristine `ple_layer.py`, `VLLM_PLE_MMAP=1`, no PLE offload):

```
-e VLLM_PLE_MMAP=1 -e VLLM_PLE_MMAP_WORKERS=32 -e VLLM_PLE_MMAP_PREWARM=0
-e VLLM_USE_FLASHINFER_SAMPLER=1
--tensor-parallel-size 1 --max-model-len 4096 --max-num-seqs 1
--max-num-batched-tokens 8192 --gpu-memory-utilization 0.80
--load-format safetensors --no-enable-flashinfer-autotune
--no-enable-prefix-caching --enable-chunked-prefill
-cc.cudagraph_mode=PIECEWISE -cc.splitting_ops=<your 12-entry list verbatim>
```

It loads cleanly and your hook reports what it should:

    PLE mmap: layer 1, 128 shards, 320001536 rows x 160 B (47.7 GiB on disk), dtype F8_E4M3
    Model loading took 74.34 GiB memory
    GPU KV cache size: 364,916 tokens

Then:

    "The capital of France is"  ->  " andufteth,,allwaysas2.logasasas1.myas2 kkl2 IIl1inkl l ul l lllK"

**What I have ruled out** (details:
https://github.com/jschmied/qwen38-flash-next-gb10) — the checkpoint (its FP8 PLE rows match the
official BF16 table at cosine 0.999635, so the quantization and the global scale are right), the
MoE backend (CUTLASS and MARLIN corrupt identically), flashinfer autotune, prefix caching, MRV1
vs MRV2, async scheduling, `VLLM_GDN_DECODE_KERNEL=triton`, and the PLE path in every form
including your mmap gather. The output is byte-identical across most of those, so the fault is
upstream of all of them. This same box serves Qwen3.8-27B coherently every day, so the GPU is not
generally miscomputing.

**Questions, if you have a minute**

1. Driver and firmware version on your GX10? Board-vendor or driver skew is my leading suspect
   now, since I have matched everything above the driver.
2. Do you run any container flags beyond `--gpus all --ipc=host --shm-size 16g`?
3. Does your smoke test still pass on a *fresh* pull of that digest today?

Not asking you to debug my box — a one-line answer to (1) would probably settle it.

_Disclosure: AI-assisted analysis (Claude Code); I ran the tests and reviewed the traces myself._
