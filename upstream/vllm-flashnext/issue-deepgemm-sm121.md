### Your current environment

<details>
<summary>Environment</summary>

```
GPU      : NVIDIA GB10 (DGX Spark), compute capability 12.1 (sm_121), 128 GB unified
vLLM     : 0.1.dev20073+g8e685d198 (Qwen3.8-Flash-Next preview build, PR #53896)
torch    : 2.13.0+cu130   CUDA 13.0   aarch64 / Ubuntu 24.04
model    : lovedheart/Qwen3.8-Flash-Next-NVFP4-FP8 (ModelOpt MIXED_PRECISION,
           FP8_PB_WO blockwise-FP8 attention + GDN projections, NVFP4 experts)
```

</details>

### 🐛 Describe the bug

`support_deep_gemm()` returns **True on sm_121**, but DeepGEMM's blockwise FP8 GEMM then dies with
a hard CUDA fault. The gate accepts the whole `120` capability *family*, and GB10 is in that family
without being a device DeepGEMM actually runs on.

```python
# vllm/platforms/cuda.py:669
def support_deep_gemm(cls) -> bool:
    """Currently, only Hopper and Blackwell GPUs are supported."""
    return (
        cls.is_device_capability(90)
        or cls.is_device_capability_family(100)
        or cls.is_device_capability_family(120)   # <- sm_121 (GB10) lands here
    )
```

Measured on the machine:

```
capability                : DeviceCapability(major=12, minor=1)
is_device_capability_family(120) : True
is_deep_gemm_supported()         : True
```

#### What happens

Selection picks DeepGEMM for the blockwise-FP8 linear method:

```
INFO [__init__.py:689] Selected DeepGemmFp8BlockScaledMMKernel for ModelOptFp8PbWoLinearMethod
```

and the first real launch — inside the startup profile run, so before serving a single request —
faults:

```
torch.AcceleratorError: CUDA error: unspecified launch failure
  File "vllm/utils/deep_gemm.py", line 464, in fp8_gemm_nt
  File "vllm/v1/worker/gpu/model_runner.py", line 854, in profile_run
  File "vllm/v1/worker/gpu_worker.py", line 634, in determine_available_memory
```

The engine then reports only `Engine core initialization failed. Failed core proc(s): {}`, so the
DeepGEMM frame is easy to miss.

#### Workaround

`VLLM_USE_DEEP_GEMM=0` falls back cleanly:

```
INFO [__init__.py:689] Selected CutlassFp8BlockScaledMMKernel for ModelOptFp8PbWoLinearMethod
```

and the model serves correctly — 23.7 tok/s single-stream, 156 tok/s aggregate at 16 concurrent
streams, NLL/token 0.7610 against 0.7748 for the unquantized-dense build of the same model.

#### Suggested fix

Either exclude sm_121 from `support_deep_gemm()`, or narrow the family checks to the capabilities
DeepGEMM actually ships kernels for. If sm_121 *is* meant to be supported, then this is a DeepGEMM
kernel bug rather than a gating one — but either way the current behaviour is a hard crash on a
supported-by-declaration device.

#### Why this is worth fixing rather than documenting

The path is reachable **only** through blockwise-FP8 weights. GB10 users running the common NVFP4
checkpoints never touch it, so it stays invisible — until they adopt a checkpoint that quantizes
the dense projections, which on this hardware is a large win (we measure +39% single-stream,
because a "6B-active" model was moving 11 GB/token with its attention and GDN projections left in
BF16). That makes this a fault that greets people exactly when they try the thing that helps most.

There is precedent for capability-family checks being too coarse for GB10: flashinfer's
trtllm-gen decode kernels are SM100-only and *silently emit garbage* on sm_121, reported
independently by another DGX Spark project. This one at least crashes.

### Before submitting a new issue...

- [x] Searched existing issues; #30938 (which added FP8_PB_WO) and the DeepGEMM issues do not
  cover sm_121 gating.
