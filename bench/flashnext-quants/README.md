# Flash-Next quantization comparison

The same idea as `bench/nvfp4-table/` for Qwen3.8-27B, but the axes are different because a
different thing dominates.

On the 27B the interesting axes were the **`lm_head`** and the **MLP**. On Qwen3.8-Flash-Next
they are:

- **the PLE / n-gram table** — 51.2 B parameters that no quantizer touches by default, and the
  single largest object in the checkpoint (95.4 GiB at BF16, 47.7 at FP8, 26.8 at NVFP4). It
  decides whether the model *fits*.
- **the dense projections** (attention q/k/v/o + GDN `in_proj`/`out_proj`) — 2.67 B parameters
  read on every token. They decide how *fast* it decodes. Most published checkpoints leave them
  in BF16.

`lm_head`, `shared_expert` and the hyper-connections are a third axis that, as of 2026-08-28,
**nobody in the field had quantized**. We started on it here.

## Files

| | |
|---|---|
| `data/cells.csv` | every measured / censused checkpoint, machine-readable |
| `flashnext-quants.html` | the published comparison page |

## Rules for this table, learned the hard way

**Read placement from the files, never from the model card or the repo name.** Repo names are
actively misleading here: `Inferact/…-NVFP4` has *identical* dense BF16 to RadixArk (the whole
44 GiB difference is PLE precision), and `…-NVFP4-FP8` says nothing about which layers got which.
Read `hf_quant_config.json` → `quantization.quantized_layers`, and cross-check tensor dtypes from
the safetensors headers.

**Distinguish "read per token" from "present in the checkpoint".** Three groups are *not* per-token
reads and must be excluded from any bandwidth claim: the vision tower (only with an image),
`embed_tokens` (one row, not the matrix), and the MTP drafter (only when speculating). We
over-counted our own roofline by including them.

**Do not trust published quality metrics to differentiate variants.** `gsm8k_metrics.json` and
`aime26_metrics.json` are byte-identical (`sha256 88766f7e…`, same `latency_seconds` to ten
decimals) across a plain NVFP4 build, an FP8 dense-quantized fork, *and* a 512→448 expert-pruned
variant. `sha256sum` them before letting them inform anything.

**Verify the scale convention empirically before writing a quantized tensor.** ModelOpt FP8_PB_WO
stores `weight_scale_inv`, which despite the name is the **scale, not the reciprocal**:
`w_fp8 * scale` reconstructs to 2.25% relative error; `w_fp8 / scale` to 5.7e8%. Getting it
backwards yields fluent garbage, not a crash.

**Two GB10 env vars are mandatory for blockwise-FP8 weights**, both undocumented upstream:
`VLLM_USE_DEEP_GEMM=0` (else a CUDA `unspecified launch failure` — vllm#54125) and
`VLLM_GDN_DECODE_KERNEL=triton` (else the engine hangs at concurrency ~32, silently).

**Log `clocks.sm`.** GB10 parks bandwidth-bound decode at ~82% of max SM clock (2411-2522 against
3003) and locking does not move it. Within-box comparisons are fine; cross-project absolute
figures are not.

## Record the conditions with every number

An aggregate t/s figure without a stated **prompt length** carries almost no information. Measured
2026-08-28 on `fp8head`, MTP k=2, c=16, output 256, everything else held fixed:

| input length | aggregate t/s | TTFT |
| --- | --- | --- |
| 256 | 192.7 | 2.72 s |
| 1000 | 127.5 | 2.44 s |
| 4000 | 74.9 | 10.39 s |

A **2.6x swing** from prompt length alone. This is how a recorded `167.8` and a re-measured `96.6`
are both correct — the first was a short-prompt run, the second used 4000-token inputs. The
`c=1` column is likewise **speculation-off**: re-measuring gave 26.4 against a recorded 26.1,
and MTP k=2 takes the same server to 38.0.

Every row must carry:

| field | why |
| --- | --- |
| `--input-len` | prefill is charged to wall clock; it moves aggregate t/s by 2.6x |
| MTP on/off + k | +44% at c=1, +2.6% at c=16 — the gain is regime-dependent |
| `--max-num-seqs` | a low cap silently ceilings concurrency (cost us a false 33 t/s ceiling once) |
| decode median *and* aggregate | they diverge ~10% at c=1 and much more under load |

## Cross-stack numbers are not comparable

Rows marked `ours` are one box, one method, same day. Field numbers from SGLang or llama.cpp
builds belong in prose, not in this table — they differ in engine, PLE handling, speculation and
clock policy simultaneously.

## External reference: 0xBakeer's inference-atlas

[inference-atlas](https://0xbakeer.github.io/inference-atlas/) publishes schema'd, hardware-
fingerprinted runs with full provenance — the only external source so far that is directly
comparable to ours, because it pins the **same engine build** (`0.1.dev20073+g8e685d198`) on the
**same hardware**.

Their `serve-single-i256-o256-v1` on `RadixArk` (concurrency 1, temperature 0, 50 requests):

```
decode_tok_s_per_request  mean 33.606  p50 33.434
ttft_ms                   mean 541.5
gpu_util_avg 89.48%   ram_peak 110.4 GB   KV 18.13 GiB
```

**33.6 tok/s on the checkpoint we measured at 28.5.** Their configuration differs in five ways;
the one we think matters is the **PLE mechanism**:

| | atlas | ours |
|---|---|---|
| PLE | `VLLM_PLE_MMAP=1` — in-process, page cache | `VLLM_PLE_CPU_OFFLOAD=1` — separate worker, CUDA IPC |
| swap | **off** | 79 GiB, actively swapping |
| MTP | k=3 | k=2 |
| max-num-seqs | 2 | 16 |
| gpu-memory-utilization | 0.85 | 0.90 |

Our own trace found the offload worker idle at 5–6% CPU, but that measures the **gather**, not the
round-trip around it — ZMQ request, CPU gather, pinned DMA, semaphore wait, every token. The mmap
hook deletes all of that. This is the mechanism 0xBakeer and paragontasx argued for and our
worker-CPU% metric structurally could not see; we flagged that limitation at the time and this is
evidence on their side of it.

**Also normalise the metric before comparing**: their `decode_tok_s_per_request` excludes TTFT,
ours divides output tokens by total wall. At c=1 that is only ~3% for us (TTFT 0.19 s of 8.4 s), so
it does not account for the gap — but it is not zero either.

**Open experiment:** mmap PLE **+** the FP8-dense checkpoint. They address unrelated costs (round-trip
latency vs bytes-per-token) and should compose.
