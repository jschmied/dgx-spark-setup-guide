`config.json` declares `dtype: bfloat16` and carries no `quantization_config`, but `model.safetensors` holds 15 `F8_E4M3` tensors — `layers.{0..4}.mlp.{gate,up,down}_proj` — plus 15 `F32` `weight_scale` entries alongside 66 BF16 tensors. Read from the safetensors header via a range request; nothing downloaded.

As published, the weights and the config describe different models, so this cannot load as declared.

Your `Qwen3.8-27B-Kearuga-DFlash2-FP8-E4M3` repo carries the matching `quantization_config` block, if this one was meant to be the same artifact.
