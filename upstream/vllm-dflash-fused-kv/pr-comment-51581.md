Still present on merged `main` post-#52816 — vLLM `0.26.1rc1.dev1049+gf8e060271`, GB10 (`sm_121a`), driver 580.173.02, torch 2.13.0+cu130. Same trace, `qwen3_dflash.py:563`.

Target `RadixArk/Qwen3.8-27B-NVFP4`, drafter `z-lab/Qwen3.8-27B-DFlash2` quantized to FP8 (compressed-tensors, weight-only, per-tensor `weight_scale`).

**Prerequisite for reproducing:** the drafter fails to load before reaching this, because the draft quant config never gets `packed_modules_mapping` — #53116, PR #53122.

**Relevant for #51620 / #51684:** `_build_fused_kv_buffers()` runs at the end of `load_weights()` (`qwen3_dflash.py:896`), i.e. **before** `process_weights_after_loading` (`model_loader/base_loader.py:63-80`). At that point the weight is still `[out, in]` and still carries `weight_scale` — no transposition to undo, no per-kernel layout to reverse. A dequantize-in-place at the buffer build is enough:

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

**Measured** with that, GB10 TP=1, `num_speculative_tokens=7`, FlashInfer, fp8 KV, single stream, 6 requests, 4000/512, temp 0.6, AL = `1 + accepted/drafts`:

| drafter | tok/s | AL | TTFT |
|---|---|---|---|
| BF16 | 32.8 | 3.56 | 2.02 s |
| FP8 | 35.1 | 3.75 | 1.50 s |

Output coherent, no acceptance collapse — so the silent-`weight_scale` hazard you describe does not bite here once the scale is applied. AL delta not significant at n=6.

Happy to run #51620 or #51684 on this hardware; the repro needs a quantized DFlash drafter.
