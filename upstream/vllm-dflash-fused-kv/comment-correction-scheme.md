Correction to my drafter description above — two details were wrong, and both matter for anyone reproducing.

I wrote **weight-only** with a **per-tensor** `weight_scale`. It is neither. The checkpoint is **W8A8 with dynamic activation quantization**, and the weight scales are **per-channel**:

```json
"weights":           {"num_bits": 8, "type": "float", "strategy": "channel", "symmetric": true, "dynamic": false}
"input_activations": {"num_bits": 8, "type": "float", "strategy": "token",   "symmetric": true, "dynamic": true}
```

```
layers.0.self_attn.q_proj.weight        F8_E4M3  [4096, 5120]
layers.0.self_attn.q_proj.weight_scale  F32      [4096, 1]
```

There is no stored `input_scale` because the activation scale is computed per token at runtime — which is why I misread it as weight-only. The scheme vLLM selects is `CompressedTensorsW8A8Fp8`, consistent with that.

`ignore` covers norms, the two-tap convolutions, the candidate selector and the mask embedding, so only `q/k/v/o_proj` and `mlp.(gate|up|down)_proj` are quantized.

The measurements are unaffected — they were taken on this artifact, only my description of it was wrong.
