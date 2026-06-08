# 6. Run as a systemd service (router mode)

[← First run](05-first-run.md) · [Index](README.md) · [Next: Public access →](07-public-access-cloudflare.md)

This page wraps `llama-server` in a **single** systemd service that can serve **several models** through llama.cpp's built-in **router mode**, with all the per-model flags moved out of the unit and into a **preset file**.

This page gives you a **minimum viable** setup. Page 8 replaces the minimal preset with a tuned one (mlock, prompt-cache reuse, larger batches, metrics, etc.). Get the minimum working first.

## 6.1 Why router mode

In single-model mode you'd run one `llama-server` per model, each on its own port, each its own unit. Router mode collapses that into one process:

- You launch `llama-server` **without** a `--model`. It becomes a **router** that listens on one port.
- Models are described in a **preset** (`.ini`) file. Each section is one model; the section name is the model id clients ask for.
- The router spawns a child `llama-server` per model **on demand** and forwards each request to the right one, keyed off the `"model"` field in the request body.
- `--models-max 1` means **one model is resident at a time**: asking for a different model unloads the current one and loads the new one. That's the right setting for a single 128 GB box where each model is tens of GB (see the memory note in 6.2).

The payoff: one service to manage, one port, one API-key file, and adding a model is a few lines in the preset instead of a new unit.

## 6.2 The model preset file

This is where every per-model flag now lives. Create it:

```bash
sudo mkdir -p /etc/llama-server
sudo nano /etc/llama-server/models.ini
```

```ini
version = 1

; Shared defaults applied to every model the router spawns.
; A per-model section below can override any of these keys.
; Keys are long-form CLI flags without the leading dashes
; (e.g. --n-gpu-layers 999  ->  n-gpu-layers = 999).
[*]
ctx-size      = 131072
n-gpu-layers  = 999
flash-attn    = on
parallel      = 4
cont-batching = true
cache-type-k  = q8_0
cache-type-v  = q8_0
metrics       = true

; ---- Model 1: clients request "model": "qwen3-coder-next" ----
[qwen3-coder-next]
model           = /opt/llm/models/qwen3-coder-next/Qwen3-Coder-Next-UD-Q4_K_XL.gguf
load-on-startup = true

; ---- Model 2: clients request "model": "qwen36-35b-a3b" ----
[qwen36-35b-a3b]
model           = /opt/llm/models/qwen36-35b-a3b/Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf
load-on-startup = false
```

Notes on the format:

- The `[*]` section is **global**: its keys are passed to every model instance. A key set in a model section overrides the global one.
- A flag that takes no value (`mlock`, `cont-batching`, `metrics`, …) is written `= true`.
- `load-on-startup` and `stop-timeout` are **preset-only** keys (not CLI flags). With `--models-max 1`, mark exactly **one** model `load-on-startup = true`; the others load on first request.
- `--metrics` lives **in the preset**, not on the unit — it's the child instances that expose `/metrics`, and the router proxies it. Page 9 relies on this.
- Host, port, API key and model alias are **controlled by the router** and ignored if you put them here — set them on the unit (6.4) instead.

> **Memory budget — why `--models-max 1`.** On the 128 GB GB10 the model weights, the KV cache and page cache all share one pool. The Q4 coder model is ~50 GB and the Q8 35B is ~36 GB; with KV cache and headroom you cannot safely hold both resident at once. `--models-max 1` makes the router unload the current model before loading the next. If all your models are small enough to coexist (and you've done the arithmetic), raise it — but the default for this guide is swap.

Lock the file down. The service runs as `SERVICE_USER`, which must be able to read it:

```bash
sudo chmod 640 /etc/llama-server/models.ini
sudo chown root:SERVICE_USER /etc/llama-server/models.ini
```

## 6.3 Environment file

Settings that vary between hosts (the bind address and port) go in an env file:

```bash
sudo nano /etc/llama-server/router.env
```

```ini
LLAMA_HOST=127.0.0.1
LLAMA_PORT=8080
```

Lock it down (read by systemd as root before it drops to `SERVICE_USER`):

```bash
sudo chmod 600 /etc/llama-server/router.env
sudo chown root:root /etc/llama-server/router.env
```

## 6.4 Baseline unit file

```bash
sudo nano /etc/systemd/system/llama-router.service
```

```ini
[Unit]
Description=llama-server (router)
After=network-online.target
Wants=network-online.target

[Service]
User=SERVICE_USER
Group=SERVICE_USER
EnvironmentFile=/etc/llama-server/router.env

ExecStart=/opt/llm/runtime/llama.cpp/build/bin/llama-server \
  --host ${LLAMA_HOST} \
  --port ${LLAMA_PORT} \
  --models-preset /etc/llama-server/models.ini \
  --models-max 1 \
  --api-key-file /etc/llama-server/api_keys.txt

Restart=always
RestartSec=5
LimitNOFILE=1048576

# Sandboxing — verify these don't break your setup, then enable
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=full
ProtectHome=true
ReadWritePaths=/opt/llm /tmp

[Install]
WantedBy=multi-user.target
```

The unit has **no `--model` and no per-model flags** — that's the whole point. Launching `llama-server` without a model is what puts it in router mode; everything model-specific is in `models.ini`. The router forks the **same** binary for each child instance, so the sandboxing flags apply to the children too.

> If the service fails to start after you enable the sandboxing flags, remove them one at a time. `ProtectHome=true` in particular conflicts with running as a user whose `$HOME` is under `/home/SERVICE_USER` if anything in the runtime touches that path — for the standard layout in this guide (`/opt/llm/...`) it's fine.

## 6.5 Enable and start

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now llama-router.service
sudo systemctl status llama-router --no-pager
```

Tail the logs while the router starts and loads the `load-on-startup` model (≈ 30 s on first start, much faster on subsequent restarts because the GGUF is in page cache):

```bash
sudo journalctl -u llama-router -f
```

You'll see the router come up first, then the child instance for `qwen3-coder-next` load and report `server is listening` on an internal port. The router itself listens on `127.0.0.1:8080`.

## 6.6 Confirm the binding

```bash
ss -tulpn | grep 8080
```

Must show `127.0.0.1:8080`, **never** `0.0.0.0:8080` or `:::8080`. If it shows the latter, the env file is wrong — fix `LLAMA_HOST=127.0.0.1` and restart. (The child instances bind to ephemeral loopback ports; that's internal to the router and never exposed.)

## 6.7 Health, model list, and routing

```bash
# router health — public, no key, no model param
curl http://127.0.0.1:8080/health

# list models and their load status — public, no key
curl http://127.0.0.1:8080/v1/models | jq '.data[] | {id, status: .status.value}'
```

You should see both `qwen3-coder-next` and `qwen36-35b-a3b`, with the startup model `loaded` and the other `unloaded`.

Route a request to a specific model with the `"model"` field:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Authorization: Bearer sk-alice-REPLACE_ME" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-coder-next",
    "messages": [{"role": "user", "content": "Write a minimal Java hello world program."}],
    "temperature": 0.2,
    "max_tokens": 256
  }' | jq -r '.choices[0].message.content'
```

Now ask for the **other** model. With `--models-max 1` the router unloads the coder model and loads the 35B one — so this first call carries the load latency (tens of seconds), then runs normally:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Authorization: Bearer sk-alice-REPLACE_ME" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen36-35b-a3b",
    "messages": [{"role": "user", "content": "Say hello in one sentence."}],
    "max_tokens": 64
  }' | jq -r '.choices[0].message.content'
```

Re-run the `/v1/models` query and you'll see the loaded/unloaded states have swapped. A request without a valid `"model"` returns `400 model name is missing from the request`.

## 6.8 What's next

- **For LAN-only use** → skip page 7 and go to page 8 (tuning) or page 9 (monitoring).
- **For public exposure** → page 7 sets up Cloudflare Tunnel.

Page 8 replaces the minimal `[*]` section in `models.ini` with a tuned one and adds `LimitMEMLOCK=infinity` to this unit (needed once `--mlock` is in play).

---

[← First run](05-first-run.md) · [Index](README.md) · [Next: Public access →](07-public-access-cloudflare.md)
