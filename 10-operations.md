# 10. Operations

[← Monitoring](09-monitoring.md) · [Index](README.md) · [Next: Security checklist →](11-security-checklist.md)

Day-to-day tasks. Everything assumes the systemd unit name is `MODEL_NAME.service` (page 6).

## 10.1 Service control

```bash
sudo systemctl start    MODEL_NAME
sudo systemctl stop     MODEL_NAME
sudo systemctl restart  MODEL_NAME
sudo systemctl status   MODEL_NAME --no-pager
sudo systemctl reload-or-restart MODEL_NAME    # after editing the unit
```

After editing `/etc/systemd/system/MODEL_NAME.service` you also need:

```bash
sudo systemctl daemon-reload
```

## 10.2 Logs

```bash
sudo journalctl -u MODEL_NAME -f             # follow
sudo journalctl -u MODEL_NAME --since "1 hour ago"
sudo journalctl -u MODEL_NAME -p warning -n 100   # only warnings/errors
```

Pipe through `grep -E "eval time|prompt eval|tg ="` to extract per-request timing summaries that llama-server emits at the end of each completion.

## 10.3 Add / remove an API key

Append a new key:

```bash
key="sk-newuser-$(openssl rand -hex 24)"
echo "$key" | sudo tee -a /etc/llama-server/api_keys.txt >/dev/null
sudo systemctl restart MODEL_NAME
echo "Issue this to the user once: $key"
```

Revoke a key:

```bash
sudo nano /etc/llama-server/api_keys.txt   # delete the line
sudo systemctl restart MODEL_NAME
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

## 10.4 Update llama.cpp

```bash
sudo systemctl stop MODEL_NAME

sudo -iu SERVICE_USER bash -c '
  cd /opt/llm/runtime/llama.cpp
  git pull
  cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
  cmake --build build --config Release -j"$(nproc)"
'

sudo systemctl start MODEL_NAME
sudo journalctl -u MODEL_NAME -f
```

Verify the new build counter / commit in the startup log. Keep an eye out for new optimization flags that subsequent llama.cpp releases add — the project is fast-moving.

## 10.5 Update the model

```bash
sudo systemctl stop MODEL_NAME

sudo -iu SERVICE_USER bash -c '
  cd /opt/llm/models/MODEL_NAME
  huggingface-cli download unsloth/Qwen3-Coder-Next-GGUF \
    --include "*UD-Q4_K_XL*.gguf" \
    --local-dir /opt/llm/models/MODEL_NAME \
    --local-dir-use-symlinks False
'

# If the new file has a different name, update the env file:
sudo nano /etc/llama-server/MODEL_NAME.env       # change MODEL_PATH=

sudo systemctl start MODEL_NAME
```

Confirm with a smoke test:

```bash
curl http://127.0.0.1:8080/v1/models \
  -H "Authorization: Bearer sk-alice-REPLACE_ME" | jq
```

## 10.6 Useful one-liners

| What | Command |
|---|---|
| Is the port locally bound only? | `ss -tulpn \| grep 8080` (expect `127.0.0.1:8080`) |
| GPU state right now | `nvidia-smi` |
| GPU process list | `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv` |
| Service uptime | `systemctl show MODEL_NAME -p ActiveEnterTimestamp --value` |
| Memory locked vs resident | `grep -E "VmRSS\|VmLck\|VmSize" /proc/$(systemctl show MODEL_NAME -p MainPID --value)/status` |
| RLIMIT_MEMLOCK of the running process | `cat /proc/$(systemctl show MODEL_NAME -p MainPID --value)/limits \| grep locked` |
| Tail eval timings from journal | `sudo journalctl -u MODEL_NAME -n 200 \| grep -E "prompt eval time\|eval time"` |
| Current scrape health (Prometheus) | `curl -sS http://127.0.0.1:9090/api/v1/targets?state=active \| jq` |

## 10.7 Rollback playbook

If a change (new llama.cpp build, new flags, new model) regresses performance or stability:

1. **Service won't start** — check journal, fall back to last-known-good unit:
   ```bash
   sudo cp /etc/systemd/system/MODEL_NAME.service.bak /etc/systemd/system/MODEL_NAME.service
   sudo systemctl daemon-reload
   sudo systemctl restart MODEL_NAME
   ```
2. **Service starts but is slower** — compare dashboards before/after. The `Throughput (tokens/sec, log)` panel from page 9 is the most direct regression indicator.
3. **GPU fell back** — XID errors on the dashboard, or memory pressure visible in `free -h`. Check `--mlock` is in place and `LimitMEMLOCK=infinity` is set on the unit.

---

[← Monitoring](09-monitoring.md) · [Index](README.md) · [Next: Security checklist →](11-security-checklist.md)
