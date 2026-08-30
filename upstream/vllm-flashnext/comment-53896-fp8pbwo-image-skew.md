**The prebuilt image has drifted behind this branch in a way that fails silently — worth a line in
the PR body.**

`vllm/vllm-openai:qwen38-flash-next` ships `0.1.dev20073+g8e685d198`, and its
`ModelOptMixedPrecisionConfig.get_quant_method()` dispatches:

```
FP8, MXFP8, NVFP4, W4A16_NVFP4
```

This branch (`2a4cd640`) and `main` both dispatch:

```
FP8, FP8_PB_WO, MXFP8, NVFP4, W4A16_NVFP4
```

So the branch is fine. But the image is what the PR body currently points people at, and on the
image a checkpoint declaring `quant_algo: "FP8_PB_WO"` falls through to:

```python
# Layer not in quantized_layers — leave unquantized
return UnquantizedLinearMethod()
```

which loads the **packed FP8 bytes into BF16 parameters**. No error, no warning, server starts
clean, output is fluent garbage. `ModelOptFp8PbWoLinearMethod` is already present in the image's
`modelopt.py` — only the dispatch branch is missing, so it is a one-line fix locally:

```python
if quant_algo == "FP8_PB_WO":
    return ModelOptFp8PbWoLinearMethod(self.fp8_config)
```

This is not hypothetical for this model. `lovedheart/Qwen3.8-Flash-Next-NVFP4-FP8` uses
`FP8_PB_WO` (blockwise 128×128, weight-only) for its 156 attention and GDN projection tensors —
i.e. every layer that matters — and it is one of the more attractive Flash-Next checkpoints right
now because it is the only servable one that quantizes the dense weights. Stock SGLang has the
same gap and its author documents it.

A ten-second offline check that would save people the hunt:

```bash
python - <<'PY'
import re, vllm.model_executor.layers.quantization.modelopt as m
s = open(m.__file__).read()
i = s.index("class ModelOptMixedPrecisionConfig"); j = s.index("def get_quant_method", i)
print(sorted(set(re.findall(r'quant_algo == "([A-Z0-9_]+)"', s[j:j+2600]))))
PY
jq -r '.quantization.quantized_layers[].quant_algo' hf_quant_config.json | sort -u
```

The second list must be a subset of the first. Suggestion: either refresh the tag, or add a line
to the PR body noting the image predates `FP8_PB_WO` — a format the runtime does not *recognise*
is much worse than one it rejects, because rejection is an error message and non-recognition is a
silent reinterpretation of the bytes.

---

**Two things from profiling this model on GB10 that may be useful to the thread.**

**1. Single-stream decode is bandwidth-bound on the *unquantized dense* weights, not on the
experts.** Torch profiler over a steady-state decode, TP=1, no speculation, 58.8 ms/token:

| | ms/token | % of wall |
|---|---:|---:|
| cuBLAS GEMV (BF16 mat-vec) | 40.77 | 69.4% |
| cutlass GEMM (the NVFP4 path) | 15.80 | 26.9% |
| MoE experts | 0.47 | 0.8% |

GPU is 95.5% busy (union of 253,529 kernel intervals across 52 streams), so this is not a
scheduling or graph-break effect. `RadixArk/Qwen3.8-Flash-Next-NVFP4` leaves ~4.84 B **dense**
params in BF16 — attention q/k/v/o, GDN projections, dense mlp, lm_head, shared_expert — read in
full every token: 9.68 GB of a 10.98 GB per-token budget. At 273 GB/s that is a 35.5 ms floor, and
an independent bandwidth model lands within 15% of the profiler. The properly-NVFP4 experts cost
0.47 ms/token by comparison. An "A6B" model moving 11 GB/token is the headline.

[hashd1ve](https://github.com/hashd1ve/qwen38-flash-next-one-dgx-spark) reached the same
conclusion independently and earlier, from SGLang.

**2. `VLLM_GDN_DECODE_KERNEL=triton` appears to be required once the GDN projections are FP8.**
Reported by `primitive-ai`, who bisected it module class by module class: with the default cuda
kernel, FP8 GDN projections *hang the engine deterministically* at concurrency ~32 — no error,
requests simply stall. Relevant to anyone serving a dense-quantized checkpoint of this model.

**3. Do not use the published `gsm8k_metrics.json` / `aime26_metrics.json` on these checkpoint
repos to compare variants.** They are byte-identical (sha256 `88766f7e…` / `eb4acd8c…`, same
`latency_seconds` to ten decimals) across a plain NVFP4 build, an FP8 dense-quantized fork of it,
and a 512→448 expert-**pruned** variant, and the `model` field inside them names the original
build. I repeated those figures myself before checking. `sha256sum` them before letting them
inform a decision.
