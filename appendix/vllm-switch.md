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
>
> **SM_120 ≠ SM_121 — don't mistake desktop-Blackwell fixes for GB10 wins.** GB10 is
> **sm_121**; RTX 50-series / RTX PRO 6000 desktop Blackwell is **sm_120**. Upstream
> CUTLASS FP8/FP4 grouped-GEMM kernels are guarded `enable_sm120_only` and **trap on
> sm_121** (`"This kernel only supports sm120"`), so nightly changelog items that say
> "SM120 blockwise FP8 GEMM" or "NVFP4 CUTLASS MoE" do **not** light up on this box until
> SM_121 registration lands. Also note "non-gated MoE" NVFP4 support = the Nemotron
> `is_act_and_mul=False` path — irrelevant to the gated (SwiGLU) Qwen MoE experts here.
> Tracking: vllm [#43507](https://github.com/vllm-project/vllm/issues/43507) (CUTLASS MoE
> unavailable sm_120/121), [#43906](https://github.com/vllm-project/vllm/issues/43906)
> (MXFP8 MoE → Marlin fallback on sm_121), CUTLASS
> [#2800](https://github.com/NVIDIA/cutlass/issues/2800) (FP4 restricted to sm_100a).
>
> **Update (2026-07-11): the *FlashInfer cute-dsl* path is the sm_121 exception.** The gate above is
> about the **CUTLASS** grouped-GEMM kernels. A *separate* NVFP4 MoE path — FlashInfer's
> `cute_dsl/blackwell_sm12x` fused kernel (vLLM [#40082](https://github.com/vllm-project/vllm/pull/40082),
> `--moe-backend flashinfer_b12x`) — **does run on GB10**: `nvidia-cutlass-dsl` 4.5.2 already accepts
> `sm_121a`, and on CUDA 13 the FP4 `block_scale` PTX is valid (CUTLASS #3227 is CUDA-12-only). Opt in
> with `--moe-backend flashinfer_b12x` + `CUTE_DSL_ARCH=sm_121a` (it's excluded from *auto*-select, not
> broken). A matched benchmark found it **marginally faster than marlin** at concurrency ≥4 — see
> [Appendix B, "b12x on mainline vLLM"](deepseek-v4-flash-180b-and-a4q.md). So the takeaway is narrower
> than "no FP4 MoE on GB10": the *CUTLASS* route is gated; the *cute-dsl* route works.

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
  --max-num-seqs 12 \
  --max-num-batched-tokens 16384 \
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

> **Sizing `--max-num-seqs` / `--max-num-batched-tokens` for a developer fleet.**
> NVIDIA's card ships `4` / `8192`, tuned for a single interactive user. For a shared
> fleet these were the throttle: with several developers the admission cap (4) queued
> requests long before memory did. At `--gpu-memory-utilization 0.4` the fp8 KV pool
> holds **~3.55M tokens** (`vllm:cache_config_info{kv_cache_size_tokens}`), which vLLM
> reports as **~13.5 concurrent *full* 262K-context sequences** — and far more at the
> ~48K-token contexts real coding traffic averages. Observed peak KV usage was **6.6%**,
> so the pool had ~15× headroom while admission sat at 4. We raise `--max-num-seqs` to
> **12** (fits the KV budget with margin, covers ~8 developers through interactive
> bursts) and `--max-num-batched-tokens` to **16384** so the large agentic prefills
> (p95 ~134K tokens) clear in fewer scheduler steps under contention. Raising seqs
> costs **no extra memory** — the KV pool is pre-allocated inside the 0.4 budget; it
> just admits more of it. Decode is memory-bandwidth-bound on the single GB10, so
> aggregate tok/s saturates past a point — 12 buys queueing relief, not linear
> throughput. Watch the **admission panels** (queue depth, KV headroom) added to the
> vLLM dashboard to confirm you're no longer hitting the `max-num-seqs` wall.

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
- **Separate metrics namespace + its own dashboard (page 9).** The router's
  Prometheus job scrapes its proxied `/metrics` (llama.cpp metric names); vLLM
  exposes its *own* `/metrics` on `:8080` under the `vllm:*` namespace, so the
  router's Grafana panels stay flat while vLLM is up — that's expected, not a fault.
  The box already handles this: `prometheus.yml` carries a dedicated `vllm` scrape
  job (labelled `service: vllm-nvfp4`) and *drops* `vllm:*` from the `llama-server`
  job so the series aren't double-ingested under fake model labels. `up{job="vllm"}`
  is `1` only while `vllm.service` is live and `0` under the router — the switch is
  visible in Prometheus itself. A companion board,
  [appendix/vllm-dashboard.json](vllm-dashboard.json), drops into
  `grafana/dashboards/` (provisioner picks it up within 10 s) and covers the vLLM
  serving metrics — E2E/TTFT/ITL latency, throughput, scheduler state, KV-cache use —
  plus panels specific to this deployment: a **spec-decode (MTP) row** (acceptance
  rate overall and per draft position, effective tokens/step), **prefix-cache hit
  rate** (`--enable-prefix-caching`), **preemptions** (KV-pressure early warning),
  and a **GPU (DCGM) row** so GPU load reads on the same board. The dashboard is
  `model_name`-templated, so it re-targets automatically if you switch the served
  model per A.7.
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

### A.7.1 As deployed — parameterised wrapper + 27B-specific tuning

We took option 1. `vllm-launch` reads the backend from the environment (35B defaults,
so `vllm.service` needs no env): `VLLM_MODEL_DIR`, `VLLM_SERVED_NAMES`, `VLLM_MOE`
(1 = MoE → adds `--moe-backend marlin` + MTP `moe_backend`; 0 = dense), plus two
tunables added below — `VLLM_GPU_MEM_UTIL` (default 0.4) and `VLLM_MTP_NSPEC`
(default 3). `vllm-27b.service` sets the dense backend and overrides both tunables in
a `vllm-27b.service.d/optimize.conf` drop-in.

> **Quote space-separated `Environment=` values.** systemd splits an unquoted
> `Environment=VLLM_SERVED_NAMES=qwen36-27b qwen36-27b-nvfp4` on whitespace and drops
> the second token (`systemd-analyze verify` flags it: *"Invalid environment
> assignment, ignoring: qwen36-27b-nvfp4"*), so the model silently answers to only the
> first alias. Wrap the whole value in quotes: `Environment="VLLM_SERVED_NAMES=…"`.

**The 27B is dense — do not inherit the 35B's KV/util sizing.** Both models are hybrid
Gated-DeltaNet (KV grows only on the full-attention layers), but the 27B has 16 full-attn
layers × 4 KV heads vs the 35B-A3B's 10 × 2 → **~3.2× the KV bytes per token**. At the
shared-launcher default `--gpu-memory-utilization 0.4` that leaves the 27B only ~3.8
full-262K sequences (vs the 35B's measured 13.3). Since the dense weights (~21 GB) leave
the box nearly empty when the 27B is the sole resident model, the drop-in raises it to
**`VLLM_GPU_MEM_UTIL=0.6`** — measured `kv_cache_size_tokens=1.85M` →
**7.06 full-context concurrency** (`vllm:cache_config_info{kv_cache_max_concurrency}`).

**The 27B is memory-bandwidth-bound; MTP is a big win.** A dense 27B reads all ~27B
params/token (~15 GB at NVFP4) against GB10's ~273 GB/s, so decode floors at ~12 t/s —
**~7× slower than the 35B-A3B MoE** (which reads only ~3B active params/token, ~87 t/s).
That makes MTP speculative decoding unusually valuable here: it verifies several draft
tokens per weight-read pass. A dedicated nspec sweep (`bench/sweep-27b.sh`,
`results-nspec-27b-2026-07-10/`, util 0.6, temp 0.6, 256-tok outputs):

| MTP nspec | c=1 decode t/s | accept | c=4 aggregate t/s | accept |
|---:|---:|---:|---:|---:|
| 0 (off) | 11.8 | — | 13.8 | — |
| 2 | 24.0 | 0.834 | 15.7 | 0.797 |
| **3** | **25.9** | 0.671 | **17.0** | 0.702 |
| 4 | 28.6 | 0.659 | 15.7 | 0.584 |

MTP up to **2.4× single-stream** (11.8 → 28.6). Unlike the 35B (which collapses at
nspec=4), the 27B's *single-stream* decode keeps rising with nspec — the marginal cost
of verifying extra drafts is cheap against the weight read. But at the **c=4 concurrency
the fleet runs, aggregate peaks at nspec=3** (17.0, +23% vs off); nspec=4's acceptance
collapses (0.584) and it loses aggregate under load. The drop-in ships
**`VLLM_MTP_NSPEC=3`** — best under concurrency, within 10 % of the single-stream
optimum, and consistent with the 35B.

**Bottom line:** the 27B is a fine *fallback* but a poor primary on GB10 — at ~26 t/s
(nspec=3) it is still ~3× slower than the 35B-A3B's MTP decode. Keep the 35B as prod;
the 27B unit stays `disabled`.

---

## A.8 Runtime environment layout (rebuilt 2026-08-27)

Experimental venvs accumulate fast on this box — each vLLM branch build, PR overlay and
nightly is ~10 GB, and they are easy to leave behind. This is the state after a cleanup,
and the rules that keep it from re-accumulating.

### What exists

| venv | vLLM | role |
|---|---|---|
| `vllm-venv-pr52816` | `0.26.1rc1.dev912+g19c935190` | **PROD** — `vllm-qwen38.service`, DFlash2 branch build |
| `vllm-venv-027` | `0.27.1` | stable fallback; several `/opt/llm/*.sh` tools reference it |
| `vllm-venv-fnext` | `0.1.dev20073+g8e685d198` | Qwen3.8-Flash-Next (PR #53896, unmerged) |
| `convert-venv` | — | model conversion tooling, no vLLM |

Caches follow the venv, not the model: `.cache-027` (20 GB) is prod's `VLLM_CACHE_ROOT`,
set in `serve-qwen38.sh`; `.cache` is shared. **Delete a venv and its cache goes too** —
an orphaned JIT cache is pure disk cost.

### Building an env: clone or pin, never resolve loosely

`torch` here is a **local version** (`2.13.0+cu130`) that does not exist on PyPI. A plain
`python3 -m venv` followed by `pip install vllm` resolves bare `torch` and silently installs
a build that will not run on this hardware. Two valid approaches:

1. **Clone an existing venv** and upgrade the delta — see `/opt/llm/build-027.sh`. Note the
   trap: `cp -a` keeps the **original's** absolute shebang in every `bin/` script, so the copy
   silently runs the source venv's interpreter and site-packages. Rewrite `bin/*` shebangs,
   `pyvenv.cfg` and `lib/**/*.pth`, then prove it:
   ```bash
   head -1 $VENV/bin/vllm                       # must point at THIS venv
   $VENV/bin/python -c 'import vllm,os;print(os.path.dirname(vllm.__file__))'
   ```
2. **Build fresh with the index pinned explicitly** — `/opt/llm/build-fnext.sh`:
   ```bash
   pip install --index-url https://download.pytorch.org/whl/cu130 \
     torch==2.13.0+cu130 torchvision==0.28.0+cu130 torchaudio==2.11.0+cu130
   ```
   Assert `+cu130` **before and after** installing everything else — a later dependency can
   quietly pull a different torch.

### When the model is not in any released wheel

Flash-Next lives in PR #53896, which is open: not in mainline, not in a nightly wheel. The
only build that has it is the official image `vllm/vllm-openai:qwen38-flash-next`. The recipe
that keeps this honest rather than turning into a container copy:

- pin **every** dependency to that image's `pip freeze`, installed normally from PyPI and the
  cu130 index;
- copy only what is genuinely not publicly resolvable — `vllm` and `deep_ep`, both local
  wheels in the image;
- record why, in `PROVENANCE.txt` inside the venv, with the condition for undoing it (*if
  #53896 merges, install a normal wheel and delete the copy step*).

Three of the image's 263 packages are Ubuntu **system-python** artifacts that pip cannot build
(`dbus-python`, `PyGObject`, `python-apt`). Exclude them **by name**, never by ignoring errors:
a silent skip is indistinguishable from a missing dependency later.

Install the pinned set **one package at a time with `--no-deps`**. The freeze is a complete
closure, so per-package installation is correct — and unlike `pip install -r`, it does not
abort the entire run on the first failure. You get every unresolvable package in one pass
instead of discovering them one rebuild at a time. (`flashinfer-cubin==0.6.17` is one: the
image built with `uv` across several indexes, and PyPI only has up to 0.6.13.)

### Dropping envs safely

`/opt/llm/purge-stale-envs.sh` — dry run by default, `--apply` to delete. It refuses to run if
prod is down, if any delete-list path is what prod is currently running, or if a keep-listed
path appears in the delete arrays.

Before deleting any venv, **check it for local modifications**, because a patched file that
exists nowhere else is lost with it:

```bash
P=$VENV/lib/python3.12/site-packages/vllm
ref=$(stat -c %Y $P/__init__.py)
find $P -name '*.py' -newermt "@$((ref+3600))"     # edited well after install
find $P \( -name '*.orig' -o -name '*.bak' -o -name '*.rej' \)
```

Then confirm each modification is recoverable — a public PR can be re-fetched; **our own**
work must be committed *and pushed*:

```bash
git -C ~/vllm-fork log --branches --not --remotes --oneline   # must be empty
```

The 2026-08-27 cleanup removed `vllm-venv-026`, `-maintest`, `-nightly`, `.cache-026`,
`.cache-flashnext` and the dead units `vllm.service`, `vllm-27b-dflash.service.d` and
`qwen3-coder-next.service`, recovering **33 GB**. `-nightly` held the DFlash FlashInfer fix;
it was verified committed on `dflash-quantized-drafter` in `jschmied/vllm` with no unpushed
commits before deletion.

### PLE CPU offload needs CAP_SYS_PTRACE (yama), and torchcodec must be absent

Two things bite on a bare-metal Flash-Next unit that do not bite in a container run by root:

**1. `pidfd_getfd: Operation not permitted`.** `kernel.yama.ptrace_scope=1` (the DGX OS default)
restricts `PTRACE_MODE_ATTACH` to descendants. vLLM's `PleOffloadWorker` and its GPU worker are
*siblings*, so the CUDA-IPC tensor handoff is refused and the engine dies **after** loading all
206 shards, reporting only `Failed core proc(s): {}`. Having `cap_sys_ptrace` in
`CapabilityBoundingSet` is not enough — a `User=llm` service holds no effective capabilities:

```ini
[Service]
AmbientCapabilities=CAP_SYS_PTRACE
```

Prefer that over `sysctl kernel.yama.ptrace_scope=0`, which weakens ptrace machine-wide.

**2. Do not install `torchcodec`.** It needs system ffmpeg, which this host lacks. vLLM guards
the import with `except (ImportError, RuntimeError)`, but an unloadable `.so` raises **`OSError`**
— so an installed-but-broken torchcodec kills the server at import, while an *absent* one falls
back cleanly to a placeholder. Only video input is affected.

Verified: the clean bare-metal venv reproduces the container result exactly — 17.0–17.1 tok/s
against the container's 17.1–17.3, same checkpoint and flags.
