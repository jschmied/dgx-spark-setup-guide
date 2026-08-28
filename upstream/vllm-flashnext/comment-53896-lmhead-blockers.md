**`lm_head` cannot be quantized on this model, and one of the three reasons is in this PR's own
model file — worth a one-line change before merge.**

`lm_head` is BF16 `[248320, 2560]` in every published GPU checkpoint of Flash-Next (verified across
ten). That is not a checkpoint choice. Three independent things block it, and any one alone is
sufficient:

**1. This PR's model file never passes `quant_config` to `ParallelLMHead`.**

`vllm/models/qwen3_8_flash_next/nvidia/model.py`:

```python
self.lm_head = ParallelLMHead(
    config.vocab_size,
    config.hidden_size,
    prefix=maybe_prefix(prefix, "lm_head"),   # <- no quant_config
)
```

`VocabParallelEmbedding.__init__` then sees `quant_config is None` and installs
`UnquantizedEmbeddingMethod` unconditionally — `get_quant_method` is never called, so a checkpoint
declaring `lm_head` in `quantized_layers` is silently ignored. `self.quant_config` is already in
scope two lines up, and `ParallelLMHead` already accepts the kwarg:

```python
    quant_config=self.quant_config,
```

Same in `mtp.py`, which constructs its own `ParallelLMHead` the same way.

**2 and 3 are not this PR's** — recording them so the picture is complete. `config.json`'s
`quantization_config.ignore` contains `lm_head` in the checkpoints we tested (so `is_layer_excluded`
returns early), and `VocabParallelEmbedding`'s vocab-aware loader asserts
`loaded_weight.shape[output_dim] == self.org_vocab_size`, which is true of `weight` and false of a
block-scale companion shaped `[1940, 20]` — the embedding loader has no notion of a scale tensor.
We worked around the third locally by attaching a plain copy loader to the scale parameters, with
an explicit `NotImplementedError` for TP>1 since the scale would need sharding in *block* space.

**Why it is worth supporting: measured on one GB10, TP=1.**

| | `lm_head` BF16 | `lm_head` FP8 blockwise |
|---|---:|---:|
| code | 23.2 tok/s | **26.1** |
| factual | 23.7 tok/s | **26.2** |
| NLL/token | 0.9687 | **0.9628** |
| tasks | 10/10 | 10/10 |

Paired NLL over 14 chunks / 646 tokens of held-out prose, code, German, French and technical text;
9 chunks better and 5 worse, so we read the −0.6% as noise and report it as **no measurable
quality cost**, not as an improvement. **+11% decode** matches a bandwidth model: at `[248320,
2560]` the head is 1.27 GB read on every token, of a ~7.25 GB per-token budget.

Format matters more than the layer: on the sibling Qwen3.8-27B we previously measured an **NVFP4**
head at 2.4% worse NLL and declined it, while an **FP8** head is loss-neutral.

One prefix note for anyone implementing: vLLM asks about **`language_model.lm_head`**, not
`lm_head`, so a config keyed on the bare name still resolves (the candidate list handles it) but is
worth testing explicitly:

```python
c = ModelOptMixedPrecisionConfig.from_config(json.load(open("config.json"))["quantization_config"])
c.is_layer_excluded("language_model.lm_head")    # must be False
c._resolve_quant_algo("language_model.lm_head")  # must be the algo you wrote
```

That ten-second offline check would have saved us three ten-minute boot cycles.

Happy to send the one-line `quant_config` change as a PR against this branch if useful — the other
two are separate and belong upstream of the model.
