# 10. Operations

[← Monitoring](09-monitoring.md) · [Index](README.md) · [Next: Security checklist →](11-security-checklist.md)

Day-to-day tasks. Everything assumes the router unit name is `llama-router.service`, the model preset is `/etc/llama-server/models.ini`, and the env file is `/etc/llama-server/router.env` (page 6).

## 10.1 Service control

```bash
sudo systemctl start    llama-router
sudo systemctl stop     llama-router
sudo systemctl restart  llama-router
sudo systemctl status   llama-router --no-pager
sudo systemctl reload-or-restart llama-router    # after editing the unit
```

After editing `/etc/systemd/system/llama-router.service` you also need:

```bash
sudo systemctl daemon-reload
```

Restarting the router restarts **all** model instances. Editing `models.ini` also requires a restart for the change to take effect (the preset is read at startup).

## 10.2 Logs

```bash
sudo journalctl -u llama-router -f             # follow (router + all child instances)
sudo journalctl -u llama-router --since "1 hour ago"
sudo journalctl -u llama-router -p warning -n 100   # only warnings/errors
```

Child model instances inherit the router's journal, so per-request timings show up here too. Pipe through `grep -E "eval time|prompt eval|tg ="` to extract per-request timing summaries that llama-server emits at the end of each completion.

## 10.3 Add / remove an API key

API keys are enforced by the router and shared across all models. Append a new key:

```bash
key="sk-newuser-$(openssl rand -hex 24)"
echo "$key" | sudo tee -a /etc/llama-server/api_keys.txt >/dev/null
sudo systemctl restart llama-router
echo "Issue this to the user once: $key"
```

Revoke a key:

```bash
sudo nano /etc/llama-server/api_keys.txt   # delete the line
sudo systemctl restart llama-router
```

Backup the keyfile (root-owned, off the box):

```bash
sudo cat /etc/llama-server/api_keys.txt | gpg --armor --encrypt -r your-gpg-id > api_keys.txt.gpg
# move api_keys.txt.gpg to a secure offline location
```

Rotation rules of thumb:

- One key per person or integration. No "team key".
- Rotate immediately when someone leaves or a key appears anywhere unintended (a log file, a screenshot, a ticket).
- Treat the keyfile like the `/etc/shadow` of the model server.

## 10.4 Manage models at runtime

List models and their load status (public, no key):

```bash
curl -s http://127.0.0.1:8080/v1/models \
  | jq -r '.data[] | "\(.id)\t\(.status.value)"'
```

With `--models-max 1`, the router swaps automatically: just send a request with the `"model"` you want and the current model is unloaded to make room. You can also drive it explicitly (these endpoints require the API key):

```bash
# load a model without sending an inference request
curl -X POST http://127.0.0.1:8080/models/load \
  -H "Authorization: Bearer sk-alice-REPLACE_ME" \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen36-35b-a3b"}'

# unload a model to free memory
curl -X POST http://127.0.0.1:8080/models/unload \
  -H "Authorization: Bearer sk-alice-REPLACE_ME" \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen36-35b-a3b"}'
```

The first inference request after a swap carries the load latency (tens of seconds for a ~50 GB GGUF, much less when it's warm in page cache).

## 10.5 Add or change a model

Models are defined in `/etc/llama-server/models.ini`. To **add** a model, append a section (the section name is the id clients will request):

```ini
[my-new-model]
model           = /opt/llm/models/my-new-model/MODEL_FILE
load-on-startup = false
```

Then:

```bash
sudo nano /etc/llama-server/models.ini
sudo systemctl restart llama-router
curl -s http://127.0.0.1:8080/v1/models | jq -r '.data[].id'   # confirm it appears
```

To **update** a model's weights in place:

```bash
sudo systemctl stop llama-router

sudo -iu SERVICE_USER bash -c '
  cd /opt/llm/models/qwen3-coder-next
  huggingface-cli download unsloth/Qwen3-Coder-Next-GGUF \
    --include "*UD-Q4_K_XL*.gguf" \
    --local-dir /opt/llm/models/qwen3-coder-next \
    --local-dir-use-symlinks False
'

# If the new file has a different name, update the section's `model =` line:
sudo nano /etc/llama-server/models.ini

sudo systemctl start llama-router
```

Confirm with a smoke test:

```bash
curl http://127.0.0.1:8080/v1/models | jq
```

## 10.6 Update llama.cpp

```bash
sudo systemctl stop llama-router

sudo -iu SERVICE_USER bash -c '
  cd /opt/llm/runtime/llama.cpp
  git pull
  cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
  cmake --build build --config Release -j"$(nproc)"
'

sudo systemctl start llama-router
sudo journalctl -u llama-router -f
```

Verify the new build counter / commit in the startup log. Keep an eye out for new optimization flags that subsequent llama.cpp releases add — the project is fast-moving, and router-mode is itself a recent feature.

## 10.7 Useful one-liners

| What | Command |
|---|---|
| Is the port locally bound only? | `ss -tulpn \| grep 8080` (expect `127.0.0.1:8080`) |
| Which model is loaded right now | `curl -s http://127.0.0.1:8080/v1/models \| jq -r '.data[] \| select(.status.value=="loaded").id'` |
| GPU state right now | `nvidia-smi` |
| GPU process list | `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv` |
| Service uptime | `systemctl show llama-router -p ActiveEnterTimestamp --value` |
| Memory locked vs resident | `grep -E "VmRSS\|VmLck\|VmSize" /proc/$(systemctl show llama-router -p MainPID --value)/status` |
| RLIMIT_MEMLOCK of the router process | `cat /proc/$(systemctl show llama-router -p MainPID --value)/limits \| grep locked` |
| Tail eval timings from journal | `sudo journalctl -u llama-router -n 200 \| grep -E "prompt eval time\|eval time"` |
| Current scrape health (Prometheus) | `curl -sS http://127.0.0.1:9090/api/v1/targets?state=active \| jq` |

## 10.8 Rollback playbook

If a change (new llama.cpp build, new flags, new model) regresses performance or stability:

1. **Service won't start** — check journal, fall back to last-known-good unit and preset:
   ```bash
   sudo cp /etc/systemd/system/llama-router.service.bak /etc/systemd/system/llama-router.service
   sudo cp /etc/llama-server/models.ini.bak /etc/llama-server/models.ini
   sudo systemctl daemon-reload
   sudo systemctl restart llama-router
   ```
   (Both `.bak` files are created in page 8 before the tuning change.)
2. **Service starts but is slower** — compare dashboards before/after. The `Throughput (tokens/sec, log)` panel from page 9 is the most direct regression indicator.
3. **GPU fell back** — XID errors on the dashboard, or memory pressure visible in `free -h`. Check `mlock = true` is in the preset's `[*]` section and `LimitMEMLOCK=infinity` is set on the unit.

---

[← Monitoring](09-monitoring.md) · [Index](README.md) · [Next: Security checklist →](11-security-checklist.md)
