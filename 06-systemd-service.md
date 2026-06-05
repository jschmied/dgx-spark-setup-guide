# 6. Run as a systemd service

[← First run](05-first-run.md) · [Index](README.md) · [Next: Public access →](07-public-access-cloudflare.md)

This page gives you a **minimum viable** systemd unit. Page 8 replaces it with a tuned unit (mlock, parallel 4, prompt-cache reuse, metrics endpoint, etc.). Get the minimum working first.

## 6.1 Environment file

Settings that vary between hosts go in an env file:

```bash
sudo nano /etc/llama-server/MODEL_NAME.env
```

```ini
MODEL_PATH=/opt/llm/models/MODEL_NAME/MODEL_FILE
LLAMA_HOST=127.0.0.1
LLAMA_PORT=8080
```

Lock it down:

```bash
sudo chmod 600 /etc/llama-server/MODEL_NAME.env
sudo chown root:root /etc/llama-server/MODEL_NAME.env
```

## 6.2 Baseline unit file

```bash
sudo nano /etc/systemd/system/MODEL_NAME.service
```

```ini
[Unit]
Description=llama-server (MODEL_NAME)
After=network-online.target
Wants=network-online.target

[Service]
User=SERVICE_USER
Group=SERVICE_USER
EnvironmentFile=/etc/llama-server/MODEL_NAME.env

ExecStart=/opt/llm/runtime/llama.cpp/build/bin/llama-server \
  --model ${MODEL_PATH} \
  --host ${LLAMA_HOST} \
  --port ${LLAMA_PORT} \
  --ctx-size 131072 \
  --n-gpu-layers 999 \
  --flash-attn on \
  --parallel 4 \
  --cont-batching \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
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

> If the service fails to start after you enable the sandboxing flags, remove them one at a time. `ProtectHome=true` in particular conflicts with running as a user whose `$HOME` is under `/home/SERVICE_USER` if anything in the runtime touches that path — for the standard layout in this guide (`/opt/llm/...`) it's fine.

## 6.3 Enable and start

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now MODEL_NAME.service
sudo systemctl status MODEL_NAME --no-pager
```

Tail the logs while the model loads (≈ 30 s on first start, much faster on subsequent restarts because the GGUF is in page cache):

```bash
sudo journalctl -u MODEL_NAME -f
```

You're looking for the same `server is listening on http://127.0.0.1:8080` line you saw in page 5.

## 6.4 Confirm the binding

```bash
ss -tulpn | grep 8080
```

Must show `127.0.0.1:8080`, **never** `0.0.0.0:8080` or `:::8080`. If it shows the latter, the env file is wrong — fix `LLAMA_HOST=127.0.0.1` and restart.

## 6.5 Health check from inside the box

```bash
curl http://127.0.0.1:8080/health

curl http://127.0.0.1:8080/v1/models \
  -H "Authorization: Bearer sk-alice-REPLACE_ME" | jq
```

## 6.6 What's next

- **For LAN-only use** → skip page 7 and go to page 8 (tuning) or page 9 (monitoring).
- **For public exposure** → page 7 sets up Cloudflare Tunnel.

---

[← First run](05-first-run.md) · [Index](README.md) · [Next: Public access →](07-public-access-cloudflare.md)
