**GB10 / TP=1 data point for the NVFP4 PLE gate, plus MTP numbers on the same box.**

Confirming @OsakaTX's diagnosis (#issuecomment-5427847488) on hardware this thread has not covered
— every existing report of it is TP=2 or multi-GPU. Single GB10 (DGX Spark, sm_121), TP=1,
`RadixArk/Qwen3.8-Flash-Next-NVFP4`, `VLLM_PLE_CPU_OFFLOAD=1`, image `g8e685d198`.

Same failure, same function:

```
ValueError: There is no module or parameter named 'ngram_embedding.weight_scale'
            in Qwen3_8FlashNextNGramEmbedding
```

Accepting `modelopt` / `modelopt_fp4` in `_get_ple_embedding_quant_method()` fixes it, and the
model then serves correctly — verified on content, not just on absence of errors:

```
"The capital of France is"  -> Paris.
"17 sheep, all but 9 die"   -> 9
iterative Fibonacci         -> correct
Rayleigh scattering (DE)    -> correct
```

**One caveat that cost us a while, worth folding into whatever lands.** Under PLE **CPU offload**
the GPU-side process must *not* register `weight` / `weight_scale`. `load_weights()` retains only
`_offload_weight_scale`; a registered-but-never-loaded `weight_scale` then shadows it in
`_get_embedding_weight_scale()`, and the lookup dequantizes against an uninitialised value. That
fails **silently** — no error, correct-magnitude activations, fluent but wrong output. So the gate
needs to stay closed in the GPU process when offload is active:

```python
if envs.VLLM_PLE_CPU_OFFLOAD and not is_offload_process():
    return None
```

Resident cost on this box: **76.61 GiB** weights + 30.99 GiB KV, of 121.63 GiB, with a 64 GiB
swapfile carrying the cold part of the FP8 PLE table. So NVFP4 + PLE offload at TP=1 fits on one
Spark with headroom — noting the PR's validation matrix currently lists PLE offload as BF16/FP8
only.

**MTP works on this configuration**, which may be useful for the recipe page since the numbers
differ from the single-stream-only ones circulating:

| concurrency | no spec | MTP k=2 | MTP k=3 |
|---:|---:|---:|---:|
| 1 | 17.1 | **28.5** | 27.4 |
| 4 | 44.1 | 50.7 | **60.6** |
| 8 | 87.5 | 89.0 | **93.4** |

Acceptance k=2: 77.1% / 58.5% at positions 0-1, mean accepted length 2.36. k=3: 70.3% / 48.2% /
33.0%, mean 2.52.

**The optimal `num_speculative_tokens` shifts with load** — k=2 wins single-stream, k=3 wins from
c=4 up. k=3 drafts longer but costs more per step, and batching absorbs that cost. Since the
recipe page recommends k=3 and several reports recommend k=2, both are right for the concurrency
each was measured at; it may be worth stating the load assumption alongside the value.

Separately: `--max-num-seqs` matters far more than either. At c=8 the difference between the
value we first shipped (2) and 16 is **4x aggregate throughput**, and a request cap is
indistinguishable from saturation unless you also record
`vllm:request_queue_time_seconds_sum`. We nearly published a wrong ceiling because of it.

One more from this box, filed separately as #54097: an installed-but-unloadable `torchcodec`
kills startup because the guards catch `(ImportError, RuntimeError)` and `torch.ops.load_library`
raises `OSError`. Hits any host without system ffmpeg.
