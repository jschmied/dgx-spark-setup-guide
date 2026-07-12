# How to set up and tune a DGX Spark or ASUS Ascent GX10

A practical, end-to-end guide for turning an **NVIDIA DGX Spark** (or its OEM sibling, the **ASUS Ascent GX10** — both use the **NVIDIA GB10 Grace Blackwell Superchip** with 128 GB unified memory) into a private LLM inference server.

The reference workload is **`unsloth/Qwen3-Coder-Next-GGUF`** served with **`llama.cpp` / `llama-server`** in **router mode** (one service serving several models, swapping them on demand), but most of the material applies to any GGUF model on this hardware.

## What you get if you follow the whole guide

- A non-root Linux service running `llama-server` in router mode, serving several models (defined in a preset file) fully on the GB10 GPU, one resident at a time
- An OpenAI-compatible HTTP API on `127.0.0.1:8080` with per-user bearer keys
- SSH key-only administration and a deny-by-default host firewall
- Optional public HTTPS access via Cloudflare Tunnel
- A tuned `llama-server` configuration that sustains the GPU baseline under multi-user load
- A Prometheus + Grafana + NVIDIA DCGM Exporter monitoring stack with a 22-panel dashboard

## Index

| # | Page | What's in it |
|---|---|---|
| 1 | [Overview & architecture](01-overview.md) | Hardware, software, security model, reference architecture |
| 2 | [Base system & SSH hardening](02-base-system.md) | OS packages, service user, SSH key auth, UFW |
| 3 | [Build llama.cpp with CUDA](03-llama-cpp-build.md) | Clone, configure with `-DGGML_CUDA=ON`, build |
| 4 | [Download the GGUF model](04-model-download.md) | Hugging Face CLI, quant selection (UD-Q4_K_XL etc.) |
| 5 | [First runtime test](05-first-run.md) | Manual server start, smoke tests, API key file |
| 6 | [Run as systemd service (router mode)](06-systemd-service.md) | router mode, model preset `.ini`, env file, baseline unit, hardening flags |
| 7 | [Public access via Cloudflare Tunnel](07-public-access-cloudflare.md) | `cloudflared`, edge auth, end-to-end client test (optional) |
| 8 | [Performance tuning](08-performance-tuning.md) | `--mlock`, `--kv-unified`, `--ubatch-size`, `--parallel`, prompt cache reuse, with measured benchmarks |
| 9 | [Monitoring (Prometheus + Grafana + DCGM)](09-monitoring.md) | Stack via Docker Compose, provisioned dashboard, what GB10 does and doesn't expose |
| 10 | [Operations](10-operations.md) | Start/stop, logs, API key rotation, updating `llama.cpp` and the model, WiFi-drops troubleshooting |
| 11 | [Security checklist](11-security-checklist.md) | Final hardening summary by layer |
| 12 | [Standardized coding test](12-model-coding-test.md) | Reproducible harness to compare any served model's coding ability — two tasks: Go concurrency (`go test -race`) and a Spring Boot/Hibernate/Mockito/JUnit5 stack (`mvn test`) |
| 13 | [Model evaluation](13-model-evaluation.md) | Latest results across both tasks (qwen3-coder-next, qwen36-35b-a3b, gemma-4-26B-A4B, gemma-4-31B, Sonnet 4.6) — clean scorecards + recommendation |
| 14 | [Sampling & variance](14-sampling-and-variance.md) | How the eval was tuned: the fixed-temp mistake, recommended-temp variance, the neutral-suite fix, and the top-p → min-p win that took Gemma's Go from 1/4 to 4/4 |
| 15 | [SWE-bench (mini-SWE-agent)](15-swebench.md) | Standardized real-issue eval against the local router: install, OpenAI-compatible wiring, the aarch64 image-arch fixes (+ idempotent re-patcher), rollouts → `preds.json`, harness scoring, and the aarch64 reality — arm64 image coverage (Lite 17 %, Java 0 %), why emulation and self-built arm64 don't pay off, and when to use x86/cloud |
| A | [vLLM switch (NVFP4)](appendix/vllm-switch.md) | Add vLLM as a second backend on the same `:8080` and API keys, mutually exclusive with the router via systemd `Conflicts=`; worked example serves `nvidia/Qwen3.6-35B-A3B-NVFP4` (`--quantization modelopt`), with a one-command `llm-switch` and the monitoring/auth caveats |
| B | [DeepSeek-V4-Flash-180B & A4Q](appendix/deepseek-v4-flash-180b-and-a4q.md) | Bare-metal vLLM forks on GB10: fitting/building the REAP-pruned 180B (5 extraction gotchas + the unified-memory build-OOM lesson) — then **decommissioning** it (underperformed) and **productionizing A4Q** for the qwen36 NVFP4 fleet as `llm-switch a4q`. The tuning campaign: the sm121 MoE-backend map (marlin vs the b12x single-stream cliff), **CUDA graphs + MTP taking decode 27→102 tok/s**, and the quality reckoning — **real-task use showed prod (fp8 KV) considerably better than A4Q's nvf4 KV**, so run prod for daily/agentic work and reserve A4Q for big-prompt speed bursts. Also: agentic tool failures traced to the **temp-1.0 default** (not the backend), and why greedy logprob divergence understated the nvf4 quality gap |
| C | [Per-phase temperature for agentic coding (Pi)](appendix/pi-agent-temperature.md) | Wire **per-phase sampling** into the [Pi](https://github.com/badlogic/pi-mono) coding agent against the local `:8080` endpoint: a **hot planner (~0.7)** for exploration/loop-recovery + a **cold worker (~0.2)** for tool-call reliability (7/8 vs 1/8 at temp 1.0) and higher MTP acceptance — via two model entries + `pi-subagents`, **not** a per-turn knob (a reasoning model can't split thinking-vs-toolcall temp in one generation). Includes the `--override-generation-config` server-default gotcha (the client must actually *send* `temperature`) and how to verify it |
| D | [GB10 NVFP4 benchmarks (kernel & KV)](appendix/gb10-nvfp4-benchmarks.md) | Measured on a single GB10: MoE kernel **marlin vs b12x** — b12x wins the synthetic decode sweep (~5% aggregate + lower TTFT, detailed in Appendix B) but is **worse on real SWE-bench agents** (6 timeouts vs 0, from W4A4-activation decision drift — *not* slower turns); the **NVIDIA-weights turn-efficiency win** (78 vs 124 turns/task); and the **KV-cache quality ladder** bf16 ≈ fp8 ≫ nvfp4 (fp8 near-lossless, +0.13% perplexity; nvfp4 ~9× worse *and* `sm100f`-gated off GB10). Verdict: **NVIDIA·marlin·fp8-KV** is agentic-optimal *(revised for coding agents by Appendix E)* |
| E | [NVFP4 for agentic coding — weights × kernel](appendix/nvfp4-agentic-coding.md) · [visual guide](nvfp4-agentic-guide.html) | The **scored, multi-language** re-run of the 2×2 (Verified-30 Python + Multilingual-28 Java/JS, temp 0.6, real harness *resolution*). Flips Appendix D: the *"NVIDIA wins"* verdict was a **Python-only artifact** — on Java/JS **NVIDIA·b12x gives up 39%** (empty patches) and **Unsloth wins overall (34 vs 29/58)**. A **marlin control** proves the give-up is a **b12x activation-quant artifact**, not the checkpoint (same weights on marlin: ~3 give-ups vs 11) — driven by **uniform FP4 activations** + NVIDIA's **English-news calibration** (Unsloth is data-free/dynamic). A **prefill microbench** corrects the folklore: **marlin ≈ b12x** in speed (within ~5%); b12x's "agentic speed" was partly **quitting early**. Ends with the ordered decision points and the **bad-by-default traps** (b12x not auto-selected, temp-1.0, BF16 won't serve). **Pick: Unsloth·b12x for polyglot; NVIDIA→marlin not b12x** |
| F | [Runbook: Qwen3.6 NVFP4 + b12x on vLLM 0.25](appendix/vllm-025-qwen36-b12x-runbook.md) | Copy-paste deploy of **Unsloth Qwen3.6-35B-A3B NVFP4 on b12x, vLLM 0.25, GB10** — full 262k ctx, image input, cudagraphs, MTP, ~90 tok/s. One page of **gotcha→fix** for the 0.25 startup regressions on this hybrid-GDN+multimodal model: `ffmpeg` for torchcodec, the FP8-oracle shim, b12x opt-in (`--moe-backend flashinfer_b12x`+`CUTE_DSL_ARCH=sm_121a`, not auto on sm_121), `--limit-mm` video-off + `max_pixels`, **`--kv-cache-memory-bytes` pin for vLLM #44209 (GDN KV over-alloc)**, PIECEWISE cudagraphs, and **warming the `cute_dsl` kernel cache once** (cold-boot OOM; warmup-off→a few requests→warmup-on = ~90 vs ~12 tok/s). Plus the b12x preflight |

## Conventions

Throughout the guide, the following placeholders appear. Replace them with values for your environment.

| Placeholder | Meaning |
|---|---|
| `SERVER_IP` | LAN IP of the DGX Spark / GX10 |
| `SERVER_HOST` | Short SSH alias you'll use locally |
| `SERVICE_USER` | Linux user that runs `llama-server` (the guide uses `llm` as an example) |
| `ADMIN_USER` | Your interactive admin Linux user (must be in `sudo` group) |
| `PUBLIC_HOSTNAME` | `llm.example.com` style hostname you'll expose via Cloudflare |
| `MODEL_NAME` | Short id for a model (e.g. `qwen3-coder-next`) — used as its directory name, its preset section name, and the `"model"` value clients request |
| `MODEL_FILE` | The downloaded GGUF filename |
| `STRONG_PASSWORD` | A password from a password manager — never reuse anything from this guide |
| `sk-USER-…` | One bearer API key per consumer of the model API |

## Audience and assumptions

The guide assumes:

- You have physical or LAN access to a fresh DGX Spark or ASUS Ascent GX10 with the NVIDIA driver and CUDA stack already installed by the vendor image.
- You're comfortable on a Linux shell.
- You administer the box from macOS or Linux via SSH.
- The box is on a trusted LAN. Pages 4 and 7 cover what to do before any of it leaves that LAN.

## License

You may copy, adapt and redistribute this guide under the terms of CC BY 4.0. No warranty.
