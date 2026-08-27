Verified on GB10 (DGX Spark, sm_121) against a **quantized W4A16 DFlash2 drafter**, on today's
nightly `0.26.1rc1.dev1183+g06ecec7a8` — i.e. after #53435 landed. Both files were byte-identical
to this PR's base there, so it applied verbatim.

**Without the PR**, the drafter fails to load — this is #51581 surfacing through #53107's
confusing-AttributeError path:

```
qwen3_dflash.py:492, in _build_context_kv_buffers
    kv_weights = [a.qkv_proj.weight[a.q_size :] for a in layers_attn]
AttributeError: 'QKVParallelLinear' object has no attribute 'weight'
```

**With the PR applied**, the same config boots past it (`Capturing model for DFlash2`),
serves, and benchmarks normally. Target `Qwen3.8-27B-NVFP4` (mixed NVFP4/FP8, FP8 `lm_head`),
drafter W4A16, `num_speculative_tokens 7`, FlashInfer, fp8 KV.

So #53435 fixed loading, but a quantized drafter still needs this. Happy to re-run anything
specific on this hardware.

One unrelated observation, **not** attributable to this PR: that nightly is ~10% slower on a
long-prose decode probe than our pinned pre-#52560 tree at identical config. It also carries
flashinfer 0.6.17 (from 0.6.16.post3), so I have not isolated the cause and mention it only so
nobody reads the boot-success as a throughput claim.

_Disclosure: AI-assisted analysis (Claude Code); I ran the benchmarks and reviewed the traces myself._
