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
| 10 | [Operations](10-operations.md) | Start/stop, logs, API key rotation, updating `llama.cpp` and the model |
| 11 | [Security checklist](11-security-checklist.md) | Final hardening summary by layer |

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
