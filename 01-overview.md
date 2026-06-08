# 1. Overview & architecture

[← Index](README.md) · [Next: Base system →](02-base-system.md)

## Hardware

| | |
|---|---|
| Product | NVIDIA DGX Spark **or** ASUS Ascent GX10 (same silicon) |
| SoC | NVIDIA **GB10** Grace Blackwell Superchip |
| CPU | 20 ARM Cortex-X925/A725 cores (aarch64) |
| GPU | Blackwell-class iGPU with native FP4 path (`BLACKWELL_NATIVE_FP4=1` in llama.cpp's CUDA build) |
| Memory | **128 GB unified LPDDR5X**, shared by CPU and GPU |
| Memory bandwidth | ~273 GB/s shared |
| Storage | Vendor SSD |

The unified-memory architecture has a big implication for tuning: the model weights and the KV cache live in the same physical pool as page cache and other processes. See [page 8](08-performance-tuning.md) for what to do about page eviction under pressure (spoiler: `--mlock` + `LimitMEMLOCK=infinity`).

## Software stack

```
Vendor base image (Ubuntu + NVIDIA driver + CUDA)
  └── llama.cpp built with -DGGML_CUDA=ON                    ← page 3
      └── llama-server in router mode                        ← page 6
          (systemd, non-root, localhost-only, one port)
          ├── models defined in a preset .ini file          ← page 6
          │   (GGUFs on the GPU with --n-gpu-layers 999)     ← page 4
          ├── --models-max 1: one model resident, swap on    ← page 6
          │   demand by the "model" field in each request
          ├── --api-key-file with per-consumer bearer keys   ← page 5
          └── per-model /metrics → Prometheus → Grafana      ← page 9

Optional layer for public exposure:
  Cloudflare Tunnel + Cloudflare Access (edge auth)          ← page 7
```

## Reference architecture (local LAN deployment)

```
Admin laptop  ──ssh key──▶  GB10 box (SSH, ufw deny-by-default)
                            │
LAN client  ───HTTP+API key─▶  llama-server router (127.0.0.1:8080)
                            │       routes on the "model" field
                            ├── qwen3-coder-next   ─┐
                            ├── qwen36-35b-a3b      ─┤
                            ├── ornstein36-27B      ─┤
                            └── ornstein36-35b-a3b  ─┴─▶ one resident
                                                        at a time on GB10 GPU
```

LAN clients reach the `llama-server` router via SSH tunnel or a same-host proxy. The router itself never binds to a public interface, and only one model is resident at a time (`--models-max 1`).

## Reference architecture (public deployment, optional)

```
Internet client ─https─▶ Cloudflare Edge ──┐
                                           │
                          Access policy /  │
                          Cloudflare API   │
                          key auth         │
                                           ▼
                          Cloudflare Tunnel (outbound from box)
                                           │
                                           ▼
                          GB10 box  ─▶ llama-server (127.0.0.1:8080)
```

Page 7 covers this. The model is never directly exposed; the box dials *out* to Cloudflare.

## Security model (layered)

```
1. SSH key-only, AllowUsers limited, root login disabled
2. UFW: deny inbound by default, allow only OpenSSH
3. llama-server binds to 127.0.0.1 only
4. llama-server requires Authorization: Bearer sk-… (one key per user/app)
5. systemd service runs as a non-root SERVICE_USER, with sandboxing flags
6. (optional) Cloudflare Tunnel as the only public ingress, with edge auth
7. (optional) Cloudflare API key authentication per consumer
```

A valid request from outside, in the public deployment, must satisfy **both** edge auth and the model bearer key.

## What this guide does not cover

- Replacing the vendor's CUDA/driver stack (treat that as provided).
- Multi-GPU / multi-host setups — GB10 is a single iGPU; the guide assumes one box.
- Running non-llama.cpp runtimes (vLLM, TGI, etc.). The systemd, networking, and monitoring chapters are still useful, but the model launch and tuning chapters are llama.cpp-specific.
- Fine-tuning. This is an inference-only guide.

---

[← Index](README.md) · [Next: Base system →](02-base-system.md)
