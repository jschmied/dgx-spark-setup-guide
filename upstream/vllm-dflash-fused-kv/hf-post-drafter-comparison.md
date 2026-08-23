## Five quantizations of the DFlash2 drafter, measured on a DGX Spark (GB10)

All of these draft for the same target, so the drafter is the only thing that changes. Posting the
numbers because the choice turns out to be simpler than it looks, and one result is not what the
file sizes predict.

**Setup.** Qwen3.8-27B NVFP4 target (`RadixArk/Qwen3.8-27B-NVFP4` @`554ebba`), vLLM on the #52816
branch, `num_speculative_tokens: 7` — the checkpoints declare `block_size: 8`, and vLLM derives the
convolution block as `1 + n`, so 7 is the only value that matches the trained boundary. Single
stream, temperature 0.6, real code-and-prose prompts with a nonce (random-word prompts collapse a
small drafter, repeated-paragraph prompts flatter it; both fake the result). Sizes are decimal GB
of the safetensors payload. AL is mean acceptance length including the bonus token.

| drafter | scheme | size | decode | AL |
|---|---|---:|---:|---:|
| `z-lab/…-DFlash2` | BF16 | 3.85 GB | 41.3 tok/s | 4.24 |
| `lued/…-DFlash2-W8` | INT8 W8A16 g128 | 2.17 GB | 43.2 tok/s | 4.16 |
| `josch15366/…-DFlash2-FP8` | FP8 W8A8, per-channel | 2.25 GB | 43.8 tok/s | 4.24 |
| `syvai/…-DFlash2-W4A16` | INT4 g128 | **1.28 GB** | **44.6 tok/s** | 4.13 |

**Throughput tracks drafter bytes, and only that.** The drafter runs once per verification step, so
its weights are read every step; halving them is a straight bandwidth win on a machine measured at
231.8 of 273 GB/s. The middle two rows are 1.4 % apart, which is session drift here — FP8 and INT8
are not distinguishable. Quantizing the drafter is structurally low-risk: the target verifies every
token, so only speed and acceptance can move, never correctness.

**But the returns are nearly gone.** BF16 → FP8 halves the bytes for +6 %. FP8 → INT4 halves them
again for **+1.8 %**. And 32 % of that 1.28 GB is still BF16 — the candidate-selector codebooks
(257 MB) and the convolution kernels (132 MB), which every quantizer leaves alone. Quantizing those
too would remove another fifth of the bytes and land inside the noise. The drafter is no longer the
bottleneck.

### Block-scaled FP8 costs more than its size

Measured separately, paired against the W4A16 drafter on the same harness and prompts (different
run conditions from the table above — only compare the two rows to each other):

| drafter | scheme | size | decode | AL | draft accept rate |
|---|---|---:|---:|---:|---:|
| `syvai/…-DFlash2-W4A16` | INT4 g128 | 1.28 GB | 39.1 tok/s | 3.96 | 42.3 % |
| `tcclaviger/…-DFlash2-FP8` | block-scaled fp8, 128×128 | 2.12 GB | 36.6 tok/s | 3.73 | 39.0 % |

Slower, which the size predicts — but it also **loses 0.23 acceptance length**, which the size does
not. A 128×128 tile is coarser than a per-channel scale, and the per-channel FP8 drafter at a
*larger* 2.25 GB holds AL 4.24. So block scaling gives up draft quality on top of bandwidth.

### Two things worth knowing before you pick one

**They are all the same model.** Every drafter above is a requantization of `z-lab/…-DFlash2`, not
an independent training. The tensors each quantizer preserves (`hidden_norm`, the conv kernels, the
selector projection) are bit-identical to the original in all five — checked by hashing them, and
for the remote repos by range-requesting just those few KB rather than downloading. So differences
in acceptance between these repos are quantization error, not training quality, and a genuinely
better drafter cannot come out of this set.

**A quantized drafter needs two vLLM fixes**, neither in #52816: the draft model's quant config
never reaches `packed_modules_mapping` ([#53116](https://github.com/vllm-project/vllm/issues/53116),
PR [#53122](https://github.com/vllm-project/vllm/pull/53122)), and DFlash's fused context-KV
precompute slices `qkv_proj.weight` raw and applies it with a bare `F.linear`, bypassing
`quant_method` ([#51581](https://github.com/vllm-project/vllm/issues/51581)). All rows above were
measured with both patched.

If you are choosing today: **INT4 g128**. It is the smallest, the fastest, and its acceptance cost
is 0.11 against BF16.
