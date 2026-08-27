**Not a bug report — a cross-stack datapoint from your own `bench/ndec.py`.**

I ran your `bench/ndec.py` unmodified in method against vLLM on a DGX Spark, so the numbers
line up with your table. Adapted only: endpoint/model/auth, and `reasoning_effort: medium`
pinned alongside `enable_thinking: false` so the server default was not silently replaced.
Warmup confirms thinking is off (`reasoning_content` empty), and the code probe stops at
`c2=510` against your MTP's 508 — same natural stop, so the probes are comparable.

| probe (n=5, median) | vLLM MTP n=3 | vLLM DFlash2 n=7 | your MTP | your DSpark | your DFlash2 |
|---|---|---|---|---|---|
| code — LRUCache | 30.13 (30.02–30.50) | **68.25 (67.26–69.09)** | 34.5 | 51.5 | 50.9 |
| essay — Babbage→GPUs | 21.79 (21.76–21.87) | **33.08 (32.40–33.40)** | 24.1 | 18.3 | 25.4 |

Both vLLM arms are on the same box and the same day, so the MTP→DFlash2 delta is attributable
to the drafter. I am including our MTP row deliberately: on your own probe our MTP is *below*
your MTP, so this is not a case of picking our best number against your average.

**What I am not claiming.** This is our box's vLLM against your box's SGLang. I did not run
SGLang here. Different physical unit, different day, and your README's power-cap drift warning
applies. Treat it as "same probe, same GPU model, different stack" and compare against your own
re-run, not as a stack race.

**The part that may be directly useful to you — and it cuts the other way from how I first
read it.** Per #8, a quantized head makes `_maybe_build_draft_sampler()` fall back to eager
because `is_dense_head_weight` fails, which is presumably why `start-dflash.sh` defaults to the
BF16-LMHead target. So your 50.9 / 25.4 are the *good* case: dense head, selector decode folded
into the draft CUDA graph.

The vLLM numbers above are on a **quantized** head — `RadixArk/Qwen3.8-27B-NVFP4`, with
`quantized_layers[lm_head] = {NVFP4, group_size 16}` — with no dequant patch and no derived
image. The boot logs `Capturing model for DFlash2 speculator...` and no quantized-head
fallback of any kind; whether vLLM implements an equivalent draft-sampler fold at all, I
cannot tell from here:

```
target : RadixArk/Qwen3.8-27B-NVFP4          (NVFP4 W4A4, quantized lm_head)
draft  : Qwen3.8-27B-DFlash2 W4A16, num_speculative_tokens 7
serve  : --kv-cache-dtype fp8 --attention-backend flashinfer
         --enable-chunked-prefill --enable-prefix-caching
         --max-num-batched-tokens 8192 --max-num-seqs 16
         --compilation-config '{"cudagraph_mode":"PIECEWISE",
                                "cudagraph_capture_sizes":[1,2,4,8,16]}'
         --gpu-memory-utilization 0.76
```

If the eager draft sampler on quantized heads is worth closing on your side, this is at least an
existence proof that the quantized-head path need not be the degraded one. It also means your
BF16-head default is paying ~1.2 GB of extra head weight per decode step on a bandwidth-bound
box to avoid that fallback — possibly a worthwhile trade on your stack, but a trade.

Two smaller notes:

- Your `mem-fraction-static` 0.95 reboot finding matches ours independently: this box freezes
  rather than OOM-killing, and the kernel OOM killer never fires. We run 0.76 with the KV pool
  pinned in bytes for that reason.
- Thanks for documenting the SSE event-counting artifact and scoping the 66.6 chat cell as
  counted differently from the MTP/DSpark cells in that row. We hit the same trap from the
  opposite direction once; the note saved me from misreading the table.

_Disclosure: AI-assisted analysis (Claude Code); I ran the benchmarks and reviewed the numbers
myself._
