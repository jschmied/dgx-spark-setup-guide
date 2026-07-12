# Runbook: Qwen3.6-35B-A3B NVFP4 (Unsloth) + b12x on DGX Spark, vLLM 0.25

One GB10 (sm_121, 128 GB unified), CUDA 13. Every line below is a bug we hit — this is the minimal set.
Result: full **262 k** context, image input, cudagraphs, MTP, **~90 tok/s** decode, tool-calling.

## 1. Install (no fork)
```bash
pip install vllm==0.25.0        # on a CUDA-13 host: pulls torch 2.11+cu130, nvidia-cutlass-dsl 4.5.2,
                                # flashinfer 0.6.13 — the b12x stack is stock in 0.25 (was manual on 0.24)
sudo apt install -y ffmpeg      # else 0.25's torchcodec fails to load (libavutil.so.*)
```

## 2. One-line source patch — Unsloth checkpoint only
Its mixed-precision FP8 tail rejects a global `flashinfer_b12x`. In
`vllm/model_executor/layers/fused_moe/oracle/fp8.py`, add to the `mapping` dict inside `map_fp8_backend`:
```python
"flashinfer_b12x": Fp8MoeBackend.MARLIN,   # route the FP8 tail to marlin
```
(NVIDIA's uniform-FP4 checkpoint needs no patch.)

## 3. Warm the kernel cache ONCE (else first boot OOMs)
0.25 eagerly compiles the whole `cute_dsl` FP4-kernel matrix in parallel while the model is resident → OOM
on a cold cache. Warm it first, then boot normally:
```bash
# boot A: warmup off — serves (slow, ~12 tok/s), send ~5 requests to populate the JIT cache
vllm serve ...  --kernel-config '{"enable_cutedsl_warmup":false}'
# boot B: remove that flag — loads cached kernels → fast (~90 tok/s), no OOM
```
Cache is persistent (`XDG_CACHE_HOME`); later restarts are fast. Re-warm only if the cache is wiped.

## 4. Serve command
```bash
export CUTE_DSL_ARCH=sm_121a                # REQUIRED; b12x is NOT auto-selected on sm_121
vllm serve <unsloth-nvfp4-dir> \
  --served-model-name qwen36-35b-a3b \
  --moe-backend flashinfer_b12x \                      # opt-in native FP4 MoE
  --kv-cache-dtype fp8 --attention-backend flashinfer \
  --reasoning-parser qwen3 --tool-call-parser qwen3_xml --enable-auto-tool-choice \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3,"moe_backend":"triton"}' \
  --override-generation-config '{"temperature":0.6,"top_p":0.95,"top_k":20}' \
  --limit-mm-per-prompt '{"image":4,"video":0}' \      # video profiling OOMs; keep image
  --mm-processor-kwargs '{"max_pixels":1048576}' \     # cap image so its profiling fits
  --max-model-len 262144 --max-num-seqs 4 --max-num-batched-tokens 8192 \
  --kv-cache-memory-bytes 34359738368 \                # pin KV (32 GB) — bug #44209
  --compilation-config '{"cudagraph_mode":"PIECEWISE","cudagraph_capture_sizes":[1,2,4,8]}' \
  --enable-prefix-caching --enable-chunked-prefill --trust-remote-code
```

## 5. Verify b12x is actually live (or you silently run slow marlin)
```bash
CUTE_DSL_ARCH=sm_121a python -c "import torch; from vllm.utils.flashinfer import \
 has_flashinfer_b12x_gemm as g, has_flashinfer_b12x_moe as m; \
 print(torch.cuda.get_device_capability(), g(), m())"     # -> (12,1) True True
```
And the serve log must say: `Using 'FLASHINFER_B12X' NvFp4 MoE backend`.

## Gotcha → fix
| symptom | cause | fix |
|---|---|---|
| `libtorchcodec…libavutil.so.* not found` at start | 0.25 added `torchcodec` (video) | `apt install ffmpeg` |
| server errors on FP8 MoE (Unsloth) | mixed-precision FP8 tail vs global b12x | 1-line `map_fp8_backend` patch (§2) |
| log says `Using 'MARLIN'` not b12x | b12x excluded from auto-select on sm_121 | `--moe-backend flashinfer_b12x` + `CUTE_DSL_ARCH=sm_121a` |
| OOM at `profiled with N image items…` | multimodal (video) profiling | `--limit-mm-per-prompt` video 0 + `--mm-processor-kwargs max_pixels` |
| OOM at CUDA-graph capture, non-deterministic | GDN KV over-allocation, vLLM **#44209** | `--kv-cache-memory-bytes <N>` + PIECEWISE cudagraphs |
| OOM before serving, `cicc` killed | cold-cache `cute_dsl` warmup wall | warm cache once (§3); keep `--max-num-seqs` small (4) |
| decode ~12 tok/s | `cute_dsl` warmup off → slow kernel path | warm cache + warmup ON → ~90 tok/s |
| tool calls fail | `generation_config` default temp 1.0 | send/override **temp 0.6** |

Root cause of the OOM saga: **vLLM 0.25 startup regressions** for this hybrid GDN+attention + multimodal model
(#44209 KV over-alloc, the new eager cute_dsl warmup, heavier mm profiling). None existed on 0.24. The flags
above neutralize them; steady-state memory is only ~50 GB (the crashes were the startup spike).
