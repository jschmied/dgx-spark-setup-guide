# Appendix: switching the `:8080` endpoint between llama.cpp and vLLM

The main guide runs everything through `llama-server` in router mode (page 6). Some
models ship in formats llama.cpp can't serve — notably **NVFP4**, NVIDIA's 4-bit
format for Blackwell, which only vLLM runs today. This appendix adds a **second
backend** (vLLM) on the *same* port and *same* API keys, and a one-command switch
between the two.

It is a **switch, not a parallel deployment.** On the 128 GB GB10 the model weights,
the KV cache and the page cache share one unified pool — the same reason the router
uses `--models-max 1` (page 6.2). Running vLLM alongside the router would have vLLM
statically reserve a slice of that pool (it allocates up front and holds it for its
lifetime), which collides with the router's load-on-demand swapping and OOMs the box.
So we run **exactly one backend at a time**, enforced by systemd, and clients never
change their endpoint or key.

```
                        ┌── llama-router.service ──┐  (GGUF models, router mode)
client ─▶ 127.0.0.1:8080 ┤   mutually exclusive     │
                        └── vllm.service ──────────┘  (NVFP4 model)
                              (systemd Conflicts=)
```

> **References.** The vLLM flags below follow NVIDIA's model card for
> [`nvidia/Qwen3.6-35B-A3B-NVFP4`](https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4).
> [MiaAI-Lab/Qwen3.6-27B-NVFP4-vLLM](https://github.com/MiaAI-Lab/Qwen3.6-27B-NVFP4-vLLM)
> is a working reference for the *sibling* 27B NVFP4 model — same flag set, delivered
> via the `vllm/vllm-openai:nightly` container. If the bare-metal venv here fails at
> serve time (Blackwell FP4 kernel / flashinfer mismatch), lift its known-good version
> combo with `docker run --rm vllm/vllm-openai:nightly pip freeze | grep -iE 'vllm|torch|flashinfer|cutlass'`
> and pin those in `vllm-venv`. Note that repo **omits** `--quantization modelopt`
> (auto-detecting NVFP4 from `hf_quant_config.json`); see A.3.

Worked example: **`nvidia/Qwen3.6-35B-A3B-NVFP4`** — the NVFP4 build of the same
35B-A3B model page 13 evaluates as a GGUF, served by vLLM with `--quantization modelopt`.

---

## A.1 Install vLLM into its own venv

vLLM runs as the same service user (`llm`) as the router, from a dedicated venv next
to the other runtimes so it never perturbs `convert-venv` or the llama.cpp build.

**System build deps first.** vLLM does not ship every kernel prebuilt — Triton
JIT-compiles a `cuda_utils` helper on first import, and the CUDA-graph / `torch.compile`
path shells out to `ninja`. Both run at **first serve**, not install, so a venv that
`pip install`s cleanly still crash-loops at startup without these. Install them once:

```bash
# python3-dev  → provides Python.h so Triton can compile its runtime helper
# ninja-build  → /usr/bin/ninja for CUDA-graph / torch.compile builds
sudo apt-get install -y python3-dev ninja-build
```

```bash
sudo -u llm python3 -m venv /opt/llm/runtime/vllm-venv
sudo -u llm /opt/llm/runtime/vllm-venv/bin/pip install -U pip
# vLLM 0.24 pins torch==2.11.0 built for CUDA 13 (flashinfer + cutlass cu13);
# pull the CUDA aarch64 torch from the pytorch index, not the CPU wheel PyPI serves.
sudo -u llm /opt/llm/runtime/vllm-venv/bin/pip install \
  vllm==0.24.0 --extra-index-url https://download.pytorch.org/whl/cu130
```

This pulls several GB (CUDA torch, flashinfer, cutlass). Confirm it landed a **CUDA**
torch, not CPU:

```bash
sudo -u llm /opt/llm/runtime/vllm-venv/bin/python - <<'PY'
import torch; print(torch.__version__, "cuda", torch.version.cuda, torch.cuda.is_available())
PY
# want e.g.  2.11.0  cuda 13.0  True   — a "+cpu" version means the wrong index won
```

> **GB10 is compute capability 12.1 (Blackwell).** NVFP4 kernels need a recent CUDA
> (13.x here) and a matching flashinfer. If `torch.cuda.is_available()` is `False`,
> you got the CPU wheel — reinstall torch explicitly from
> `https://download.pytorch.org/whl/cu130` before continuing.

---

## A.2 Download the NVFP4 model

The repo is public (Apache 2.0), ~23 GB across three safetensors shards. HF's Xet CDN
rejects aria2's multi-range requests, so pull each shard on a **single connection**;
`aria2c` gives a resumable download with a bandwidth cap that won't starve the box.

```bash
MODEL_DIR=/opt/llm/models/qwen36-35b-a3b-nvfp4
BASE=https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4/resolve/main

# small config/tokenizer files first (fast)
sudo -u llm env HF_HOME=/opt/llm/hf-cache /opt/llm/hf-venv/bin/hf download \
  nvidia/Qwen3.6-35B-A3B-NVFP4 --local-dir "$MODEL_DIR" \
  --exclude "*.safetensors"

# an aria2 input file pins the output names — the CDN redirect otherwise saves
# files under a content hash, and vLLM loads shards by the names in index.json.
sudo -u llm tee "$MODEL_DIR/.nvfp4.aria" >/dev/null <<EOF
$BASE/model-00001-of-00003.safetensors
  out=model-00001-of-00003.safetensors
$BASE/model-00002-of-00003.safetensors
  out=model-00002-of-00003.safetensors
$BASE/model-00003-of-00003.safetensors
  out=model-00003-of-00003.safetensors
EOF

sudo -u llm /usr/bin/aria2c -c -x1 -s1 -j1 --split=1 --max-connection-per-server=1 \
  --max-overall-download-limit=6M --auto-file-renaming=false --allow-overwrite=true \
  --retry-wait=5 --max-tries=0 -d "$MODEL_DIR" -i "$MODEL_DIR/.nvfp4.aria"
```

For a long download that must survive an SSH drop, wrap it in `tmux` or run it as a
transient unit that outlives your login:

```bash
sudo systemd-run --unit=nvfp4-dl --uid=llm --gid=llm \
  /usr/bin/aria2c -c -x1 -s1 -j1 --split=1 --max-connection-per-server=1 \
  --max-overall-download-limit=6M --auto-file-renaming=false --allow-overwrite=true \
  --retry-wait=5 --max-tries=0 -d "$MODEL_DIR" -i "$MODEL_DIR/.nvfp4.aria"
journalctl -u nvfp4-dl -f      # watch progress; unit ends when the download finishes
```

---

## A.3 The launch wrapper — shared API keys

vLLM can't read llama.cpp's `--api-key-file`, but it accepts **multiple** keys as
args (any one validates — useful for rotation). A thin wrapper expands the *same*
key file the router uses, so `/etc/llama-server/api_keys.txt` stays the single source
of truth: rotate a key there and both backends pick it up on their next start.

`/usr/local/bin/vllm-launch`:

```bash
#!/usr/bin/env bash
# vLLM launcher for the NVFP4 Qwen3.6-35B-A3B model on the GB10 box.
# Shares the router's key file; managed by vllm.service (mutually exclusive with the router).
set -euo pipefail

KEY_FILE=/etc/llama-server/api_keys.txt
MODEL_DIR=/opt/llm/models/qwen36-35b-a3b-nvfp4

# one key per line; strip blanks and #/; comments -> space-separated --api-key list
mapfile -t KEYS < <(grep -vE '^[[:space:]]*([#;]|$)' "$KEY_FILE")

exec /opt/llm/runtime/vllm-venv/bin/vllm serve "$MODEL_DIR" \
  --served-model-name qwen36-35b-nvfp4 \
  --host 127.0.0.1 --port 8080 \
  --api-key "${KEYS[@]}" \
  --quantization modelopt \
  --trust-remote-code \
  --kv-cache-dtype fp8 \
  --attention-backend flashinfer \
  --moe-backend marlin \
  --gpu-memory-utilization 0.4 \
  --max-model-len 262144 \
  --max-num-seqs 4 \
  --max-num-batched-tokens 8192 \
  --enable-chunked-prefill \
  --async-scheduling \
  --enable-prefix-caching \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3,"moe_backend":"triton"}' \
  --load-format fastsafetensors \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_xml \
  --enable-auto-tool-choice
```

```bash
sudo chmod 750 /usr/local/bin/vllm-launch
sudo chown root:llm /usr/local/bin/vllm-launch
```

The flags follow NVIDIA's DGX Spark recipe for this model, with two changes for this
guide: **`--host 127.0.0.1`** (loopback only, like the router — never `0.0.0.0`) and
**`--port 8080`** (same port as the router, so the switch is transparent to clients).
`--speculative-config … mtp` mirrors the MTP/DFlash speculative decoding the GGUF
build uses. `--gpu-memory-utilization 0.4` is NVIDIA's conservative default for the
shared 128 GB pool; because nothing else is resident while vLLM runs you can raise it
if you need more KV headroom at 262K context — watch `nvidia-smi` and total RAM.

> **`--quantization modelopt`.** NVIDIA's 35B card sets it explicitly; the MiaAI-Lab
> 27B reference omits it and lets vLLM auto-detect NVFP4 from `hf_quant_config.json`
> (which our download includes). We keep it per NVIDIA — but if vLLM errors on the
> flag at startup, drop it.

---

## A.4 The systemd unit + mutual exclusion

`/etc/systemd/system/vllm.service`:

```ini
[Unit]
Description=vLLM (OpenAI-compatible) — NVFP4 backend on :8080
After=network-online.target
Wants=network-online.target
# starting vLLM stops the llama.cpp router (shared :8080 + GPU pool)
Conflicts=llama-router.service

[Service]
User=llm
Group=llm
Environment=HF_HOME=/opt/llm/hf-cache
# ProtectHome=true (below) masks /home/llm, but Triton/torch/flashinfer want to write
# compile caches under $HOME. Redirect HOME and every cache into /opt/llm (writable via
# ReadWritePaths) or the engine crash-loops with PermissionError: '/home/llm'.
Environment=HOME=/opt/llm
Environment=XDG_CACHE_HOME=/opt/llm/.cache
Environment=TRITON_CACHE_DIR=/opt/llm/.cache/triton
Environment=VLLM_CACHE_ROOT=/opt/llm/.cache/vllm
Environment=TORCHINDUCTOR_CACHE_DIR=/opt/llm/.cache/torchinductor
Environment=FLASHINFER_WORKSPACE_BASE=/opt/llm/.cache/flashinfer
ExecStart=/usr/local/bin/vllm-launch
Restart=on-failure
RestartSec=5
LimitNOFILE=1048576
LimitMEMLOCK=infinity
# first cold start compiles kernels + captures CUDA graphs — ~6 min on the GB10
TimeoutStartSec=900

# Sandboxing — mirror llama-router.service (page 6.4)
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=full
ProtectHome=true
ReadWritePaths=/opt/llm /tmp

[Install]
WantedBy=multi-user.target
```

Add the reverse conflict to the router's unit so exclusion holds whichever one you
start first. In the `[Unit]` block of `/etc/systemd/system/llama-router.service`:

```ini
# mutual exclusion with vllm.service (shared :8080 + GPU pool)
Conflicts=vllm.service
```

Create the cache dirs the unit points at (owned by `llm`), then reload:

```bash
sudo -u llm mkdir -p /opt/llm/.cache/{triton,vllm,torchinductor,flashinfer}
sudo systemctl daemon-reload
```

`Conflicts=` is bidirectional in effect once both units declare it: `systemctl start
vllm` stops `llama-router` and vice-versa, so the port and the GPU pool are never
double-booked. **Enable only one** for boot — the router (`load-on-startup` in the
preset) is the guide's default; leave `vllm.service` disabled and start it on demand.

---

## A.5 The switch script

`/usr/local/bin/llm-switch`:

```bash
#!/usr/bin/env bash
# Switch the :8080 endpoint between the llama.cpp router and vLLM (mutually exclusive).
set -euo pipefail

show() {
  for u in llama-router vllm; do
    printf '%-14s %s\n' "$u" "$(systemctl is-active "$u.service" 2>/dev/null || echo unknown)"
  done
  echo '--- models on :8080 ---'
  curl -sf http://127.0.0.1:8080/v1/models 2>/dev/null | jq -r '.data[].id' 2>/dev/null \
    || echo '(endpoint not answering yet)'
}

wait_up() {
  echo 'waiting for :8080/health ...'
  for _ in $(seq 1 300); do
    curl -sf http://127.0.0.1:8080/health >/dev/null 2>&1 && { echo 'up.'; return 0; }
    sleep 2
  done
  echo 'timed out waiting for :8080' >&2; return 1
}

case "${1:-status}" in
  llama|llamacpp|router) sudo systemctl start llama-router.service; wait_up ;;
  vllm)                  sudo systemctl start vllm.service;        wait_up ;;
  stop)                  sudo systemctl stop vllm.service llama-router.service; echo 'both stopped.' ;;
  status)                show; exit 0 ;;
  *) echo 'usage: llm-switch {llama|vllm|stop|status}'; exit 2 ;;
esac
echo; show
```

```bash
sudo chmod 755 /usr/local/bin/llm-switch
```

Usage:

```bash
llm-switch status     # which backend is live + models on :8080
llm-switch vllm       # stop router, start vLLM, wait for health
llm-switch llama      # stop vLLM, start router, wait for health
llm-switch stop       # both down
```

A switch to vLLM carries a **cold start**: vLLM allocates its memory pool and loads
the NVFP4 weights up front (tens of seconds to a couple of minutes), comparable to a
large-model swap in the router. The client sees the same URL and key throughout;
only the served model id changes (`qwen36-35b-nvfp4`).

Verify the endpoint after a switch:

```bash
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H "Authorization: Bearer sk-alice-REPLACE_ME" -H "Content-Type: application/json" \
  -d '{"model":"qwen36-35b-nvfp4","messages":[{"role":"user","content":"hi"}],"max_tokens":32}' \
  | jq -r '.choices[0].message.content'
```

---

## A.6 Caveats vs. the rest of the guide

- **Auth surface differs (page 11).** The router requires the bearer key on the
  chat/completions paths and leaves `/health` and `/v1/models` public. vLLM only
  authenticates the `/v1`, `/v2`, `/inference` prefixes — `/health` and `/metrics`
  are open. Same posture (loopback-only bind, per-consumer keys), but note it if a
  checklist item assumes every path is keyed.
- **Keys are exposed in argv and the journal (page 11).** Unlike the router's
  `--api-key-file`, vLLM takes keys as CLI args, so every key is visible in `ps aux`
  to any local user and is echoed verbatim into the systemd journal on each start
  (vLLM logs "non-default args"). On a single-admin box with a root-only journal
  that's usually acceptable, but treat it as a real difference: keep the box's local
  users trusted, and **rotate these keys** (page 10) if the journal is ever shipped
  off-box or shared. There is no multi-key file option in vLLM 0.24 (`VLLM_API_KEY`
  is single-key only).
- **Monitoring gaps while vLLM is live (page 9).** The Prometheus job scrapes the
  router's proxied `/metrics` (llama.cpp metric names). vLLM exposes its *own*
  `/metrics` on `:8080` with different metric names, so the router's Grafana panels
  won't populate while vLLM is up. If you want continuity, add a vLLM scrape job and
  a small vLLM panel row; otherwise expect a gap for the duration of the switch.
- **`--models-max 1` logic doesn't apply.** vLLM serves a single model per process.
  Switching *models* on the vLLM side means editing `vllm-launch` and restarting the
  unit — there's no router-style on-demand model swap.
- **Cloudflare (page 7) is unaffected.** The tunnel points at `127.0.0.1:8080`
  regardless of which backend answers.

---

## A.7 A second NVFP4 model — the 27B

The obvious second vLLM model is **`nvidia/Qwen3.6-27B-NVFP4`**, the dense-27B
sibling of the 35B-A3B above. Its **direct reference deployment** is
[MiaAI-Lab/Qwen3.6-27B-NVFP4-vLLM](https://github.com/MiaAI-Lab/Qwen3.6-27B-NVFP4-vLLM)
— the same flag set as A.3, and the source we fall back to for a known-good version
combo (see the References callout above).

vLLM serves one model per process (A.6), so a second model is not a router-style
swap. Two ways to add the 27B:

1. **Parameterise the wrapper.** Change `vllm-launch` to take the model dir and
   served name as `$1`/`$2` (defaulting to the 35B), then add a second unit
   `vllm-27b.service` with `ExecStart=/usr/local/bin/vllm-launch /opt/llm/models/qwen36-27b-nvfp4 qwen36-27b-nvfp4`
   and the same `Conflicts=llama-router.service` **plus** `Conflicts=vllm.service`
   (only one vLLM model resident at a time — same 128 GB pool). Extend `llm-switch`
   with a `vllm-27b` case.
2. **Second wrapper + unit** if the flags diverge (e.g. the 27B omits
   `--quantization modelopt` per MiaAI-Lab, or wants a different `--max-model-len`).

Either way the download follows A.2 (aria2c, 6 MB/s cap, pinned shard names) into
`/opt/llm/models/qwen36-27b-nvfp4`, and the shared `api_keys.txt` covers it for free.
