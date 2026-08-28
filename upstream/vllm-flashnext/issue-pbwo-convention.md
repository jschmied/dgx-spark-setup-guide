### 🐛 Describe the bug

`ModelOptFp8PbWoLinearMethod` supports exactly one of the two `FP8_PB_WO` export conventions that
ModelOpt produces, and a checkpoint using the other fails in two confusing ways rather than one
clear one.

**What vLLM expects** (`modelopt.py`, `create_weights`):

```python
# Match ModelOpt's exported shape so weight loading works without a
# custom loader: [out_blk, 1, in_blk, 1]
weight_scale = BlockQuantScaleParameter(
    data=torch.empty((out_blks, 1, in_blks, 1), dtype=torch.float32), ...)
layer.register_parameter("weight_scale", weight_scale)
```

So: name `weight_scale`, rank 4. #30938 describes the format the same way — "block-scaled FP8
weight-only with **4D block scales**".

**What some ModelOpt exports actually ship.** `lovedheart/Qwen3.8-Flash-Next-NVFP4-FP8`
(`conversion_environment.json`: nvidia-modelopt **0.46.0**, `producer.name = modelopt+postproc`)
declares `quant_algo: "FP8_PB_WO"` for 156 tensors and writes them as:

```
model.language_model.layers.0.linear_attn.in_proj_qkv.weight            F8_E4M3 [10240, 2560]
model.language_model.layers.0.linear_attn.in_proj_qkv.weight_scale_inv  F32     [80, 20]
```

Name `weight_scale_inv`, rank **2** — i.e. the DeepSeek block convention that `fp8.py` already
implements (`weight_scale_name = "weight_scale_inv" if self.block_quant else "weight_scale"`, with
the comment *"the weight_scale_inv name is intentional for deepseekv3"*).

Both spellings denote the same quantity and feed the same `W8A8BlockFp8LinearOp` — `_inv` is a
naming legacy here, not a reciprocal.

#### Two failure modes, neither of which names the problem

**1. Wrong name → a misleading `AttributeError`.** No parameter matches `weight_scale_inv`, so
`LinearBase.load_weights` passes the *module* to the weight loader:

```
AttributeError: 'MergedColumnParallelLinear' object has no attribute 'data'
```

That is #53107 — the module-substitution masking whatever the real cause was. Nothing points at a
scale-name mismatch.

**2. Wrong rank → a bare assert.** After bridging the name, loading a fused layer hits:

```
parameter.py:175  assert param_data.shape == loaded_weight.shape
  param=(16, 1, 20, 1)  loaded=(16, 20)   # out-blocks 16, in-blocks 20
```

No message, no tensor name.

#### Suggested fix

`process_weights_after_loading` in this same method **already accepts both ranks**:

```python
scale = layer.weight_scale
if scale.dim() == 4:
    scale = scale.squeeze(1).squeeze(-1)   # [out_blk,1,in_blk,1] -> [out_blk,in_blk]
elif scale.dim() != 2:
    raise ValueError(...)
```

so rank-2 is already a supported *internal* shape — only `create_weights` hard-codes rank 4.
Accepting both conventions at load time would close this: register the scale under both spellings
as **distinct** parameters and coalesce whichever was filled.

One subtlety worth writing down, because it cost me a debugging cycle: registering the *same*
`Parameter` object under both names does not work. `named_parameters()` deduplicates shared
tensors, so the alias never becomes visible to the weight loader and failure mode 1 persists
unchanged. The second registration needs its own storage.

Locally I patched exactly that (distinct `weight_scale_inv` parameter, sentinel-initialised, and
`process_weights_after_loading` picks whichever is no longer all-sentinel), and the checkpoint then
**loads and serves correctly** — verified on output, not just on absence of errors: 23.7 tok/s
single-stream, 156 tok/s aggregate at 16 streams, and NLL/token 0.7610 versus 0.7748 for the
unquantized-dense build of the same model, so the bridged scales are demonstrably being applied.

One further wrinkle for whoever implements this: with a rank-2 scale,
`Parameter(scale.contiguous(), ...)` raises

```
RuntimeError: Creating a Parameter from an instance of type BlockQuantScaleParameter requires
that detach() returns an instance of the same type
```

The existing rank-4 path escapes it only incidentally, because `squeeze()` returns a plain Tensor
and drops the subclass. `scale.data.contiguous()` handles both.

Happy to send it as a PR if the approach looks right — or, if the intent is that ModelOpt only ever emits rank-4
`weight_scale`, then this checkpoint class should at least be **rejected with a clear message**
rather than producing an `AttributeError` about a module having no `.data`.

#### Related

- #30938 — added FP8_PB_WO, specified as 4-D
- #53107 — `LinearBase.load_weights` substituting the module for a missing parameter, which is
  what turns this into an unreadable error
