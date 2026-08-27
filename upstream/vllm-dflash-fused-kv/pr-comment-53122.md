Filed #53116. Ran both halves on GB10 (`sm_121a`) against a quantized drafter. Part 1 matches what I run locally. Part 2:

**The guard's rationale does not hold at that call site.** It says *"process_weights_after_loading has transposed the weight to [in, out]"* — that hook has not run yet. `model_loader/base_loader.py`:

```python
self.load_weights(model, model_config)      # _build_fused_kv_buffers() runs at the END of this
...
process_weights_after_loading(model, model_config, target_device)
```

`_build_fused_kv_buffers()` is called at `qwen3_dflash.py:896`, end of `load_weights()`. Weight is still `[out, in]` and still carries `weight_scale`. Cutlass does transpose (`kernels/linear/scaled_mm/cutlass.py:50-55`) — afterwards.

Second check: if it were `[in, out]` there, `F.linear(x[N, H], W[L*(H-Q), Q+2K])` fails on **shape**. The reported symptom is **dtype**.

Consequence: the guard rejects a working case, and a quantized drafter still cannot serve after this PR — only the message improves.

**Working alternative** — dequantize the K/V slices where the buffer is built. Keeps the cross-layer fusion (per-layer `apply()` gives it up), costs ~52 MB for a 5-layer Qwen3.8 drafter, returns the slice untouched when dtypes match (unquantized path byte-identical).

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

**Measured**, GB10 TP=1, target `RadixArk/Qwen3.8-27B-NVFP4`, drafter `z-lab/Qwen3.8-27B-DFlash2` quantized locally to FP8 (compressed-tensors, weight-only), `num_speculative_tokens=7`, FlashInfer, fp8 KV, single stream, 6 requests, 4000/512, temp 0.6, AL = `1 + accepted/drafts`:

| drafter | tok/s | AL | TTFT |
|---|---|---|---|
| BF16 | 32.8 | 3.56 | 2.02 s |
| FP8 | 35.1 | 3.75 | 1.50 s |

Output coherent. FP8 plausibly faster at 2.25 GB vs 3.85 GB; AL delta not significant at n=6.

- `get_draft_quant_config` calls `get_model_architecture()` unguarded — mine wraps it so an unresolvable draft arch falls back instead of becoming a load failure.
- Fused-KV half overlaps #51581 (open PRs #51620, #51684). Happy to test any of them here — the repro needs a quantized DFlash drafter.
