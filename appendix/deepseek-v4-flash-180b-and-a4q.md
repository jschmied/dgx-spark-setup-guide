# Appendix B — Bare-metal vLLM forks on GB10: DeepSeek-V4-Flash-180B and A4Q

This appendix documents two experimental bare-metal vLLM efforts on the DGX Spark
(GB10, sm121/CC 12.1, 128 GB unified memory), warts and all. Both run **alongside**
the router / NVFP4 backends via the same `llm-switch` + systemd `Conflicts=` mechanism
from [Appendix A](vllm-switch.md).

> **Status (2026-07-06):** DeepSeek-V4-Flash-180B was built, benchmarked, and
> **decommissioned** — it underperformed the daily-drivers (§B.4) and its 107 GB was
> better spent, so §B.1–B.4 are now a *historical* build report. **A4Q was kept and
> productionized** for the qwen36 NVFP4 fleet as `llm-switch a4q`: with CUDA graphs it
> matches vanilla-vLLM decode *and* roughly doubles prefill (§B.6) — a genuine
> recommendation for big-prompt agentic use, not just a field experiment.

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

### Quality A/B: does 4-bit KV degrade agentic work?

The real risk with 4-bit KV is **multi-turn agentic degradation** — the worry is that precision
loss *compounds* over a growing trajectory. Two instruments, one flawed and one clean:

**1. SWE-bench (mini-SWE-agent) — saturated, so inconclusive.** On the arm64-available Lite
subset, qwen36+A4Q-nvf4 resolved **12/12** cleanly-evaluated instances (0 test-failing patches;
another ~11 hit *scoring-harness* flakiness, not model failures). But 100% is a **ceiling** — that
subset is too easy to discriminate nvf4 from fp8. It rules out *catastrophic* breakage; it can't
measure *subtle* degradation. (Lesson: a saturated benchmark is a blind instrument.)

**2. Per-token logprob divergence — ceiling-free, and decisive.** Feed an identical fixed
~38.6k-token code context to the served model under **nvf4** then **fp8** KV (can't co-serve on one
`:8080`), capture per-position logprobs (`echo`+`logprobs`, greedy), and diff them:

```
mean |Δlogprob|          : 0.127      (small)
greedy-argmax divergence : 4.27 %     (top-1 token differs at ~1 in 23 positions)
perplexity               : nvf4 only ~0.6 % higher than fp8

By context position (the compounding test):
  pos     0– 3.9k :  6.53 %    ← most disagreement is EARLY
  ...
  pos  34.8k–38.6k:  1.50 %    ← deep context: nvf4 ≈ fp8 ~98.5 %
```

**The compounding fear is refuted.** Divergence is *highest at the start* and *decays* with depth —
the opposite of accumulating error. The disagreements concentrate where the model is **least
confident** (early, high-entropy, near-tie argmax); deep in a long context, predictions are
confidence-locked and nvf4 barely moves them. That also explains the SWE-bench 12/12: the confident,
context-constrained predictions that drive task completion are nearly untouched by 4-bit KV.

**Verdict:** nvf4 KV costs ~0.6 % perplexity and some low-confidence argmax flips, and **does not
compound over long context** — for long-context / multi-turn agentic use the KV-memory halving is
close to free. Caveat: this is teacher-forced next-token divergence (not full autoregressive drift,
where nvf4's own choices feed back into its own KV) on a single code context; treat as strong but
not the last word.

---

## B.6 Productionizing A4Q for the qwen fleet (and decommissioning ds4)

The ds4 build (107 GB: 97 GB weights + 11 GB venv), its `vllm-dsv4.service`, and the
`llm-switch dsv4` verb were **removed** — it underperformed the qwen daily-drivers (§B.4) and
the space was better spent. The A4Q venv was **kept and promoted to a first-class backend**:

- venv renamed `dsv4-a4q` → **`/opt/llm/models/a4q-vllm`** (65 hardcoded paths fixed — bin/
  shebangs, `pyvenv.cfg`, activate scripts, `.pth`)
- new `vllm-a4q.service` + `a4q-conflict.conf` drop-ins on the peers + a `llm-switch a4q` case
  → fourth mutually-exclusive backend on `:8080`.

### The MoE-backend map on GB10 (sm121)

A4Q fixes *attention* prefill, but the fleet still ran `--moe-backend marlin`, which logs *"no
native FP4 → weight-only compression, may degrade performance."* The fork exposes several NVFP4
MoE backends; only some are built for consumer Blackwell:

| Backend | Device gate | GB10 (sm121)? |
|---|---|---|
| `flashinfer_cutedsl` | capability family **100** (datacenter B200/GB200) | ❌ crashes: *"kernel does not support current device"* |
| `flashinfer_cutlass` | sm90 (Hopper) or family 100 | ❌ |
| **`flashinfer_b12x`** | **sm12x consumer Blackwell** | ⚠️ works, but excluded from auto-select pending an upstream *"CUTLASS SM121 MMA op guard"* |
| `marlin` | any (weight-only FP4) | ✅ the auto default |

### CUDA graphs are the real win (decode 27 → 73 tok/s)

The initial A4Q config ran `--enforce-eager` (cudagraphs were unverified on the fork). That
throttled decode badly — a 3B-active MoE is dominated by per-step kernel-launch overhead in eager
mode. **Dropping `enforce-eager` enables torch.compile (inductor) + cudagraphs, which capture
cleanly on the A4Q wheel** (`FULL_AND_PIECEWISE`, 0.63 GiB, 4 s — correctly piecewise-splitting the
model's *hybrid* mamba+attention layers):

| 35B-A3B, nvf4 KV (single-stream unless noted) | eager | **+ CUDA graphs** |
|---|---:|---:|
| decode, single-stream | 27 | **73** |
| decode, 4-concurrent (aggregate) | — | **157** |
| prefill, 64k ×4 | 3,663 | **4,608** (+26 %) |

73 tok/s **matches vanilla vLLM 0.24.0** on the same box (an external single-stream benchmark:
35B-A3B **76** decode / 6k prefill, vs **27B *dense* 13 / 1.1k** — the MoE is 5–6× faster, so stay
on 35B-A3B). And the A4Q backend's prefill (~12k single-stream at 6.5k) is ~**2× vanilla's 6k** —
the nvf4-QK kernel earning its keep.

### marlin vs b12x: the single-stream cliff

With graphs on, is the native `flashinfer_b12x` MoE worth it? It captures fine, but:

| Both + graphs | marlin | b12x |
|---|---:|---:|
| decode, single-stream | **73** | **17** ← 4× cliff |
| decode, 4-concurrent (agg) | 157 | 174 (+11 %) |
| prefill, 64k ×4 | 4,608 | 4,955 (+8 %) |

b12x wins the *compute-bound* regimes (prefill, saturated batch) but **collapses at batch-1
decode** — a compute-tuned cutlass kernel is terrible when memory-bound at low batch. Agentic
traffic is **bursty** (often 1–2 active streams), so the 17-tok/s cliff disqualifies b12x; marlin is
robust across the whole range for an 8–11 % best-case loss. `flashinfer_b12x` is a one-line swap
*only if* a permanently-saturated batch workload (bulk/offline) appears.

### MTP spec decode stacks on top (+39% decode)

Adding the model's MTP head (`--speculative-config
'{"method":"mtp","num_speculative_tokens":3,"moe_backend":"triton"}'`, matching prod)
**coexists with cudagraphs** on the A4Q wheel (still `FULL_AND_PIECEWISE`; the capture sizes just
expand to cover the spec batch multiplier):

| decode (35B-A3B) | no spec | + MTP |
|---|---:|---:|
| single-stream | 73 | **102** (+39%) |
| 4-concurrent (agg) | 157 | **171** (+9%) |

MTP helps at *both* batch 1 and batch 4 — so, contrary to the usual "spec hurts under concurrency,"
**no `disable_by_batch_size` cutoff is needed** for the 4–6-session range (it only bites at
saturation well above this workload). Cost is a ~6% prefill dip. 102 tok/s is at/above vanilla
vLLM 0.24.0's 76.

### Recommended config — big-prompt, 4–6-session agentic

```
A4Q wheel + CUDA graphs + MTP + nvf4 KV + 256k context + marlin MoE
  --kv-cache-dtype nvfp4 --attention-backend flashinfer --moe-backend marlin
  --speculative-config '{"method":"mtp","num_speculative_tokens":3,"moe_backend":"triton"}'
  --max-model-len 262144 --max-num-seqs 6 --max-num-batched-tokens 16384
  (no --enforce-eager)    env VLLM_NVFP4_A4Q=1
```

- **256k context** — the model is native 262 144 (no YaRN); the old 128k cap wasted half the window.
  nvf4 KV gives **11.2 M tokens / 42.7× concurrency at full 256k**, so KV capacity is a *non-issue*
  at 4–6 sessions — the constraint is *prefill*, not sessions.
- **MTP on** — +39% single-stream / +9% 4-concurrent decode, and it coexists with cudagraphs; helps
  across the whole 4–6-session range (no cutoff needed).
- **marlin** — robust across bursty load (the cliff above).

First A4Q config that's a genuine **recommendation**: **102 tok/s decode** (with MTP, at/above
vanilla 0.24.0) *and* ~2× prefill *and* the full 256k window.

### Upstream status (2026-07-06)

None of this is mainline yet. "A4Q" has **zero** upstream PRs; the *capability* is a stack of open,
unmerged PRs across two repos — vLLM #46329 (nvf4 KV on consumer Blackwell via FlashInfer FA2),
#44851 (draft), the ds4 enablement #41834 (170 commits, conflict-ridden), the SM121/GB10 foundation
#34822 (stale since April) — plus FlashInfer #3684 (the nvf4 prefill kernel, also conflicted).
Expect it piecemeal over weeks-to-months; the prebuilt `jethac` fork stays the only turnkey path.

---

## B.7 Takeaways

- The **REAP-pruned 180B fits** a single GB10; the NVIDIA 284B original does not.
- Building the ds4 vLLM from source is a multi-hour, version-brittle affair with **five distinct
  extraction gotchas**; the prebuilt-image path trades that for trusting a pseudonymous binary.
- **Always `llm-switch stop` + `BUILD_JOBS=4`** before any heavy compile — the unified-memory OOM
  hangs the box.
- DeepSeek-V4-Flash-180B **underperforms** the qwen daily-drivers on coding (REAP precision loss).
- **A4Q works on the dense qwen36 fleet** (not ds4, which needs DeepGEMM) — nvf4 KV halves KV
  memory with the native fp4 prefill kernel. The quality A/B (logprob divergence) shows the hit is
  **small (~0.6 % perplexity) and non-compounding** — divergence *shrinks* with context depth — so
  for long-context agentic use the memory halving is close to free.
- **The A4Q wheel is productionized as `llm-switch a4q`** (§B.6): CUDA graphs + MTP take decode
  **27 → 102 tok/s** single-stream (at/above vanilla vLLM 0.24.0), keeping ~2× prefill and the full
  **256k** window — the first genuinely-recommended config, for big-prompt 4–6-session agentic.
- **The scary number was a config artifact.** 27 tok/s decode was purely `enforce-eager`; dropping
  it (cudagraphs work fine on the fork) recovers most of it, and MTP spec decode adds the rest
  (+39%). Always separate "the kernel is slow" from "a feature is disabled."
- **Prefill throughput is a *curve*, not a number** — O(n²) attention makes it fall with prompt
  size: ~12k tok/s single-stream at a 6.5k prompt, but ~4.6k aggregate (4 concurrent) at 64k and
  2.7k at 128k. Quote prefill *with* its prompt size and concurrency, or it's meaningless.
- **`marlin` stays the MoE backend.** Of the native-FP4 alternatives, only `flashinfer_b12x` runs
  on sm121, and it has a **4× single-stream decode cliff** (17 vs 73) — a batch-tuned kernel that
  loses at the low-concurrency batch-1 that bursty agentic actually lives in.
- **Right model for concurrency = the MoE, not the smaller dense one.** 35B-**A3B** (3B active) is
  5–6× faster than the 27B *dense* — active params drive compute; "bigger" MoE ≠ slower.
- Methodology note: **a saturated benchmark can't measure degradation.** When SWE-bench hit 100 %
  on the easy arm64 subset, the ceiling-free logprob-divergence probe was the instrument that
  actually answered the question.
