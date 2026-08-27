Confirmed: with the two corrupt shards replaced, the same configuration serves correctly.

```
"The capital of France is"  ->  Paris.
"17 sheep, all but 9 die"   ->  9
iterative Fibonacci         ->  correct
Rayleigh scattering (DE)    ->  correct
```

76.61 GiB resident with the PLE offloaded to host, 30.99 GiB KV, **17.3 tok/s** single-stream and
**32.4 tok/s** across two concurrent streams (6% per-stream loss), no speculative decoding.

Two things were needed, and the second one is worth flagging because it is undocumented as far as
I can tell:

**1. One-line gate.** `_get_ple_embedding_quant_method()` in `ple_layer.py` rejects anything that
is not `Fp8Config`. RadixArk ships the PLE in exactly the format that method implements (F8_E4M3
shards + one global BF16 `ngram_embedding.weight_scale`), but the body is NVFP4, so
`quant_config` is `modelopt_fp4` and it is rejected — the embedding is built unquantized and
loading dies on `no module or parameter named 'ngram_embedding.weight_scale'`. Accepting
`modelopt`/`modelopt_fp4` fixes it. **So RadixArk NVFP4 does load on vLLM**, which contradicts the
checkpoint tables going around.

Caveat, because it fails *silently*: under PLE CPU offload the GPU-side process must not register
`weight`/`weight_scale`. `load_weights()` keeps only `_offload_weight_scale`, so a
registered-but-unloaded `weight_scale` shadows it in `_get_embedding_weight_scale()` and the
lookup dequantizes against an uninitialised value — fluent garbage, no error.

**2. `--cap-add=SYS_PTRACE` for PLE CPU offload in Docker.**

```
RuntimeError: pidfd_getfd: Operation not permitted
  torch/multiprocessing/reductions.py:179 in rebuild_cuda_tensor
  vllm/v1/ple_offload/worker.py:482 in accept_registrations
```

`PleOffloadWorker` hands CUDA tensors to the GPU worker over IPC; `rebuild_cuda_tensor` needs
`pidfd_getfd` and the default Docker seccomp/capability set denies it. Both workers load all 206
shards first, so the failure surfaces ~10 minutes in as an unhelpful `Engine core initialization
failed. Failed core proc(s): {}`. This does not affect the mmap approach (single process, no IPC
handoff) — which is presumably why it has not come up.

Full write-up: https://github.com/jschmied/qwen38-flash-next-gb10/blob/main/notes/results-radixark-vllm.md
