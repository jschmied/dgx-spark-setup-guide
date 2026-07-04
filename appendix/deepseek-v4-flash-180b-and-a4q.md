# Appendix B — Bare-metal vLLM forks on GB10: DeepSeek-V4-Flash-180B and A4Q

This appendix documents two experimental bare-metal vLLM efforts on the DGX Spark
(GB10, sm121/CC 12.1, 128 GB unified memory), warts and all. Both run **alongside**
the router / NVFP4 backends via the same `llm-switch` + systemd `Conflicts=` mechanism
from [Appendix A](vllm-switch.md).

> **Status:** experimental. DeepSeek-V4-Flash-180B is live via `llm-switch dsv4` but
> underperforms the daily-driver coding models. A4Q is validated on the qwen36 NVFP4
> fleet. Neither is a recommended production default — this is a field report.

---

## B.1 Does DeepSeek-V4-Flash fit on a single GB10?

| Checkpoint | Weights | Fits 128 GB? |
|---|---|---|
| `nvidia/DeepSeek-V4-Flash-NVFP4` (284B total / 13B active) | **168 GB** | ❌ ~40 GB over, before KV |
| `0xSero/DeepSeek-V4-Flash-180B` (REAP-pruned, K160 = 160 experts) | **103 GB** | ✅ tight — needs the whole box |
| `0xSero/DeepSeek-V4-Flash-180B-GGUF` (Q2 only) | 57 GB | fits, but Q2-of-pruned quality is poor — skipped |

The NVIDIA original is a datacenter model (deploy examples use TP across 4–8 B200/GB200).
Only the **REAP-pruned 180B** fits a single GB10, and only just — serving needs the box
to itself (unload the qwen fleet first).

Download with `aria2c` (single-connection, pinned `out=` names) through the HF Xet CDN;
`hf download` is unreliable for large pulls here. The `main` revision was byte-identical
to the author's pinned commit, so no revision pin was needed.

---

## B.2 It needs a *dedicated* vLLM — not the NVFP4 one

DeepSeek-V4-Flash is not upstream in vLLM. It requires a fork with the `deepseek_v4`
architecture, sparse-MLA (Compressed Sparse Attention), `deepseek_v4` tokenizer/tool/
reasoning parsers, and `deepseek_mtp` speculative decoding — **none of which exist in the
NVFP4 qwen vLLM** from Appendix A. The community recipe (`0xSero/deepseek-spark`) ships a
prebuilt container; we built it **from source** instead, to avoid trusting a pseudonymous
prebuilt image:

- Base: `jasl/vllm` @ `codex/ds4-sm120-min-enable`, compiled for `--gpu-arch 12.1a` via the
  `eugr/spark-vllm-docker` PR #219 harness.
- Plus `nvidia-cutlass-dsl[cu13]==4.5.2` (see the version drift note below).
- Plus a launch-time patcher (`patch_vllm_reap_gb10.py`): router fallback for the
  non-standard 160-expert count, MXFP4/Marlin per-layer memory hygiene, a FlashInfer
  CUDA-IPC fix.

### ⚠️ The GB10 build-OOM lesson

GB10 shares **one 128 GB pool across CPU and GPU**. The default `BUILD_JOBS=16` CUDA compile,
run while the NVFP4 35B (~30 GB) was resident, exhausted memory → **global OOM → the box hung
for ~3 h and needed a manual power cycle**. Rule for any heavy compile here:

```bash
llm-switch stop            # free the GPU pool (took available RAM 43 GiB → 116 GiB)
export BUILD_JOBS=4        # never the default 16; nvcc/cicc is multi-GB per job
# verify `free -h` shows ~110 GiB available before starting
```

Also: tmux does **not** survive a reboot, so a multi-hour build interrupted by a hang must be
relaunched — the BuildKit cache (~13 GB) resumes it.

---

## B.3 Five bare-metal-extraction fixes (source build → host venv)

We extracted the built environment from the image into a host venv under
`/opt/llm/models/deepseek-v4-flash-180b/venv` (copy-out; the sm120a wheels can't be rebuilt
from PyPI). Every one of these bit us:

1. **cutlass 4.5.2, not 4.5.1.** The `jasl/vllm` branch drifted forward; the built vLLM
   (`20260706.dev1+g616a5723a`) requires cutlass-dsl **4.5.2**, but the 0xSero cutlass layer
   pinned 4.5.1 → dependency conflict. Fix: install 4.5.2 (the base image already had it).
2. **Supplementary 160-expert router patch.** On the drifted vLLM the fused `sqrtsoftplus`
   CUDA router kernel rejects 160 experts (`Unsupported expert number: 160`) and the 0xSero
   patcher's fallback no longer catches the restructured path. One-line fix forces the existing
   pure-Torch fallback for non-templated counts:
   ```
   sed -i 's|    if current_platform.is_xpu():|    if current_platform.is_xpu() or gating_output.shape[-1] not in (16, 32, 64, 128, 192, 256, 320, 384, 512):|' .../fused_topk_bias_router.py
   ```
3. **Dangling `libnccl.so.2`.** The container's `nvidia-nccl` pip package symlinks
   `libnccl.so.2` to a *container* system path absent on the host → `docker cp` grabbed a
   dangling symlink → torch import fails. Fix: copy the real lib from the working NVFP4 venv
   (`/opt/llm/runtime/vllm-venv`, identical `nvidia-nccl-cu13` version).
4. **`chmod -R u+w` the venv.** `docker cp` preserves 444 read-only modes on some vLLM files;
   the patcher can't rewrite them.
5. **`CUDA_HOME=/usr/local/cuda`.** TileLang JIT reads `CUDA_HOME`/`CUDA_PATH`. Unset, it picks
   the venv's bundled nvcc **13.3** against the venv's **13.0** headers → CCCL
   *"compiler and toolkit headers are incompatible"* crash during sparse-MLA/cudagraph warmup.
   Forcing the host CUDA 13.0 (self-consistent nvcc+headers) fixes it — exactly what the
   container did.

Full serve config (200K ctx, `FULL_AND_PIECEWISE` cudagraphs, `deepseek_mtp` spec) validated:
coherent generation, reasoning-content splits correctly, **~22.7 tok/s**.

---

## B.4 Self-contained deployment + `llm-switch dsv4`

Everything to run lives in **one directory** the fleet can see under the service's
`ProtectHome=true` sandbox:

```
/opt/llm/models/deepseek-v4-flash-180b/     (llm:llm)
├── weights/              # 46 safetensors + config/tokenizer
├── venv/                 # extracted bare-metal vLLM
├── patch_vllm_host.py    # patcher, repointed at ./venv
└── launch.sh             # applies patches (idempotent) → vllm serve
```

A `vllm-dsv4.service` unit (mirroring Appendix A's sandbox + `Conflicts=`) and a `llm-switch dsv4`
case make it a fourth mutually-exclusive backend on `:8080`. Not boot-enabled — the default
stays the NVFP4 35B. First cold-start is slow (JIT + cudagraph capture); subsequent starts reuse
the persistent JIT caches under `/opt/llm/.cache`.

### Coding-bench result: underperforms

Run through the [standardized harness](../12-model-coding-test.md), both tasks, no truncation
(13.2k / 12.4k tokens generated):

- **Go: FAIL** — hallucinated `import "std"` (not a real package) breaks compilation.
- **Java: own tests FAIL / neutral PASS** — production code is correct against an independent
  suite, but the model's own tests hallucinate a class (`MockitoJUnitExtension` vs the real
  `MockitoExtension`) and a nonexistent helper.

Signature of REAP pruning: understands the task, makes small **fatal** precision errors.
**Below the qwen daily-drivers — not a coding upgrade.**

---

## B.5 A4Q: native fp4 attention for sm120/121

[A4Q](https://x.com/i/status/2073322454198649215) (by `jethac`) is a native fp4 attention
kernel for consumer Blackwell (5090 / RTX PRO 6000 / **DGX Spark**), headed to vLLM. It fixes
the gap that TensorRT-LLM ships no attention kernels for sm120/121, so today the 4-bit KV
unpacks through an fp16 convert chain and fp4 prefill runs ~1.8× **slower** than bf16. A4Q runs
QKᵀ on the native nvf4 block-scaled MMA, consuming K directly from the NVFP4 KV cache.

It ships prebuilt aarch64 wheels for GB10 (`jethac/vllm` sm121a + `jethac/flashinfer` +
`a4q-jit-modules-12.0f.tar.gz`), designed to layer on the "r10" serving stack (torch 2.11.0+cu130
— which the NVFP4 venv already is).

### A4Q on DeepSeek-V4: blocked on DeepGEMM

The A4Q vLLM has `DeepseekV4ForCausalLM`, so it *looks* like a drop-in for the 180B — no source
build. But it **hard-blocks**: the ds4 `SparseAttnIndexer` requires **DeepGEMM**
(`fp8_fp4_paged_mqa_logits`), which the A4Q release doesn't ship and which isn't in any of our
venvs. The source-build vLLM only served because *its* indexer needs DeepGEMM **only** for the
fp4-KV path (fp8 KV skips it); the A4Q fork requires it unconditionally. So A4Q-on-ds4 needs a
separate DeepGEMM build — not worth it. (Also a real gotcha: a copied venv's `bin/pip` keeps the
**source venv's shebang**, so a naive `pip install` clobbers the *original* venv — nearly took out
the production NVFP4 venv; restore from the pristine copy.)

### A4Q on qwen36 (the real target): ✅ works

The DeepGEMM wall was **sparse-attention-specific**. The qwen36 NVFP4 fleet uses **dense**
attention — no sparse indexer — so it clears it. Serving `qwen36-35b-a3b-nvfp4` under the A4Q
wheel with `--kv-cache-dtype nvfp4 --attention-backend flashinfer` + `VLLM_NVFP4_A4Q=1`:

```
Using FlashInfer FA2 backend for NVFP4 KV cache on SM12x (swizzled, in-kernel deswizzle)
A4Q: nvf4 block-scaled QK MMA enabled for FA2 NVFP4 prefill.
```

The kernel **engages**, output is **correct** (17+25→42, clean stop), prefill ~3174 tok/s.
Note: the A4Q FlashInfer backend on CC 12.x **rejects fp8 KV** by default (wants bf16/auto/nvfp4,
or set `VLLM_FLASHINFER_MM_PREFIX=1`); the wheel also needs the `gguf` dep (`--no-deps` skips it).

**Expected gains:** nvf4 KV **~halves** KV memory vs fp8 → more context/concurrency at qwen36's
262K; native nvf4 prefill. **Decode is ~unchanged** — A4Q is an attention/prefill kernel, and
qwen36's hot path is MoE + MTP decode.

### Quality: the open question

The real risk with 4-bit KV is **multi-turn agentic degradation** — precision loss compounds over
a growing trajectory. A controlled A/B (A4Q-nvf4 vs fp8 KV, same instances) via
[mini-SWE-agent](../15-swebench.md) is the faithful test. *[Results pending — see the SWE-bench
run; this section will be updated.]*

---

## B.6 Takeaways

- The **REAP-pruned 180B fits** a single GB10; the NVIDIA 284B original does not.
- Building the ds4 vLLM from source is a multi-hour, version-brittle affair with **five distinct
  extraction gotchas**; the prebuilt-image path trades that for trusting a pseudonymous binary.
- **Always `llm-switch stop` + `BUILD_JOBS=4`** before any heavy compile — the unified-memory OOM
  hangs the box.
- DeepSeek-V4-Flash-180B **underperforms** the qwen daily-drivers on coding (REAP precision loss).
- **A4Q works on the dense qwen36 fleet** (not ds4, which needs DeepGEMM) — nvf4 KV halves KV
  memory with the native fp4 prefill kernel; the multi-turn quality A/B is the deciding test.
