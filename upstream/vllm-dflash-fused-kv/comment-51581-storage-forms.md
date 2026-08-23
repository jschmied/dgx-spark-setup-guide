Scope note for whoever implements this, from reading the safetensors headers of the DFlash2
drafters published so far. The fused context-KV path has to cope with four storage forms, not one,
and they differ in ways the config alone does not reveal:

| checkpoint | scheme | what the layer actually carries |
|---|---|---|
| `syvai/…-DFlash2-W4A16`, `gratex/…-g128-sym-GPTQ` | INT4 g128 | `weight_packed` **int32** + 2-D `weight_scale` |
| `YourHighnessLA/…-DFlash2-NVFP4` | NVFP4 W4A16 g16 | `weight_packed` **uint8** + fp8 `weight_scale` + `weight_global_scale` |
| `tcclaviger/…-DFlash2-FP8` | block-scaled fp8 | `weight_scale_inv`, 2-D, no `weight_scale` |
| per-tensor fp8 on a fused qkv | fp8 W8A8 | `weight_scale` of shape `(3,)` — one scalar per shard |

The int32 and uint8 packings are the pair worth care: the bit width is normally inferred from the
column count, which silently yields `bits=16` for the uint8 layout instead of failing, so the
container dtype has to be checked rather than derived. The `(3,)` case is the other quiet one —
slicing that scale to the K/V rows hands K the scale belonging to V, which does not crash and only
shows up as degraded acceptance.

All four are handled or explicitly rejected on the branch linked in #53122.
