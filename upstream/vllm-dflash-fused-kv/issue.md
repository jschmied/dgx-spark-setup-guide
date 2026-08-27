# [Bug]: DFlash's fused context-KV projection bypasses `quant_method`, so a quantized drafter dies with a dtype mismatch

## Your current environment

NVIDIA GB10 (DGX Spark, `sm_121a`, aarch64), driver 580.173.02, CUDA 13.0, torch
2.13.0+cu130, vLLM `0.26.1rc1.dev1049+gf8e060271` — a nightly wheel containing the merge
of #52816. Full `vllm collect-env` in `env.md`.

## Describe the bug

`precompute_and_store_context_kv` writes all context K/V up front as **one GEMM across
every draft layer**. `_build_context_kv_buffers` concatenates the raw K/V weight rows and
`_project_context_kv` applies the result with a bare `F.linear`, never going through
`quant_method.apply()`:

```python
# vllm/model_executor/models/qwen3_dflash.py
kv_weights = [a.qkv_proj.weight[a.q_size :] for a in layers_attn]
self._fused_kv_weight = torch.cat(kv_weights, dim=0)
...
all_kv_flat = F.linear(normed_context_states, self._fused_kv_weight, self._fused_kv_bias)
```

With a **quantized** drafter that buffer is still in the storage dtype, so this is a BF16
activation against an FP8 weight and the engine dies during startup profiling:

```
RuntimeError: Worker failed with error 'expected mat1 and mat2 to have the same dtype,
but got: c10::BFloat16 != c10::Float8_e4m3fn'

vllm/v1/worker/gpu_worker.py:519                        determine_available_memory
vllm/v1/worker/gpu/spec_decode/dflash/speculator.py:368 propose
vllm/model_executor/models/qwen3_dflash.py:621          precompute_and_store_context_kv
vllm/model_executor/models/qwen3_dflash.py:563          _project_context_kv
```

The quantization plumbing itself is fine — the log shows `Selected
CutlassFP8ScaledMMLinearKernel for CompressedTensorsW8A8Fp8` for the drafter and the
ordinary decoder path works. Only this hand-fused GEMM is uncovered. It is the same
defect class #52816 just fixed for the LM head, at a second site.

### Reproduction

Target `RadixArk/Qwen3.8-27B-NVFP4`, drafter `z-lab/Qwen3.8-27B-DFlash2` quantized to
FP8 (`compressed-tensors`, weight-only, targets `re:.*self_attn\.(q|k|v|o)_proj$` and
`re:.*mlp\.(gate|up|down)_proj$`; norms, `*_conv`, `candidate_selector` in `ignore`):

```bash
vllm serve /models/qwen38-27b-radixark \
  --kv-cache-dtype fp8 --attention-backend flashinfer \
  --speculative-config '{"method":"dflash","model":"/models/qwen38-27b-dflash2-fp8","num_speculative_tokens":7}'
```

**Prerequisite:** on stock vLLM this fails earlier, while loading the drafter, because the
draft model's quantization config never reaches `packed_modules_mapping` — that is #53107.
Work around that first or you never reach this crash.

### Suggested fix

Dequantize the K/V slices where the fused buffer is built, before `torch.cat`. Keeps the
cross-layer fusion; per-layer `apply()` would give it up. Costs the K/V half of the
drafter's attention weights, ~52 MB for a 5-layer Qwen3.8 drafter. The slice is returned
untouched when the dtype already matches, so the unquantized path is unchanged.

```diff
--- a/vllm/model_executor/models/qwen3_dflash.py
+++ b/vllm/model_executor/models/qwen3_dflash.py
@@ -479,6 +479,42 @@
             embeds = torch.where(is_mask, self.mask_embedding.to(embeds.dtype), embeds)
         return embeds
 
+    @staticmethod
+    def _dequant_kv_slice(attn: nn.Module, act_dtype: torch.dtype) -> torch.Tensor:
+        """The layer's K/V rows in `act_dtype`, dequantizing if stored quantized.
+
+        `_project_context_kv` runs ONE fused `F.linear` over every layer's K/V
+        weights, bypassing `quant_method.apply()`. Dequantizing here keeps that
+        cross-layer fusion; going through `apply()` per layer would give it up.
+        """
+        qkv = attn.qkv_proj
+        kv = qkv.weight[attn.q_size :]
+        if kv.dtype == act_dtype:
+            return kv
+
+        scale = getattr(qkv, "weight_scale", None)
+        if scale is None:
+            raise ValueError(
+                f"DFlash context-KV precompute needs to dequantize {kv.dtype} "
+                f"weights, but {type(qkv).__name__} exposes no weight_scale. "
+                f"Serve this drafter unquantized, or route the fused KV GEMM "
+                f"through quant_method.apply()."
+            )
+        s = scale.data if hasattr(scale, "data") else scale
+        out = kv.to(act_dtype)
+        if s.numel() == 1:
+            return out * s.to(act_dtype).reshape(())
+        # Per-channel scales: take the K/V rows, matching the weight slice.
+        s = s.reshape(-1)
+        if s.numel() == qkv.weight.shape[0]:
+            s = s[attn.q_size :]
+        if s.numel() != out.shape[0]:
+            raise ValueError(
+                f"DFlash context-KV precompute cannot map a weight_scale of "
+                f"{tuple(scale.shape)} onto a K/V slice of {tuple(out.shape)}."
+            )
+        return out * s.to(act_dtype).reshape(-1, 1)
+
     def _build_context_kv_buffers(
         self,
         layers_attn: list[nn.Module],
@@ -487,7 +523,10 @@
         self._hidden_norm_weight = self.hidden_norm.weight.data
 
         # KV projection weights: [num_layers * 2 * kv_size, hidden_size]
-        kv_weights = [a.qkv_proj.weight[a.q_size :] for a in layers_attn]
+        kv_weights = [
+            self._dequant_kv_slice(a, self._hidden_norm_weight.dtype)
+            for a in layers_attn
+        ]
         self._fused_kv_weight = torch.cat(kv_weights, dim=0)
         if has_bias:
             kv_biases = [a.qkv_proj.bias[a.q_size :] for a in layers_attn]
```

Happy to open this as a PR. One open question: is the per-tensor / per-channel
`weight_scale` split the right level of generality, or should this go through a
scheme-provided dequantization helper?

### Measurements after the fix

GB10, TP=1, `num_speculative_tokens=7`, single stream, 6 requests, 4000 prompt / 512
output, temperature 0.6. Acceptance length is `1 + accepted/drafts` from `/metrics`.

| drafter | tok/s (median decode) | acceptance length | TTFT |
|---|---|---|---|
| BF16 (unquantized) | 32.8 | 3.56 | 2.02 s |
| FP8 (quantized) | 35.1 | 3.75 | 1.50 s |

The quantized drafter serves correctly and costs nothing — plausibly faster because it is
2.25 GB instead of 3.85 GB. I would not read anything into the acceptance-length
difference at this sample size.

### Before submitting a new issue...

- [x] Searched existing issues and read the documentation.
