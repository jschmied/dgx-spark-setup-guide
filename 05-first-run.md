# 5. First runtime test

[← Download model](04-model-download.md) · [Index](README.md) · [Next: systemd service →](06-systemd-service.md)

Goal: prove that the model loads on the GPU and answers an HTTP request. Once that works, we'll wrap it in systemd in the next page.

## 5.1 Manual launch

As `SERVICE_USER`:

```bash
MODEL_PATH=/opt/llm/models/MODEL_NAME/MODEL_FILE

/opt/llm/runtime/llama.cpp/build/bin/llama-server \
  --model "$MODEL_PATH" \
  --host 127.0.0.1 \
  --port 8080 \
  --ctx-size 131072 \
  --n-gpu-layers 999 \
  --flash-attn on \
  --parallel 4 \
  --cont-batching \
  --cache-type-k q8_0 \
  --cache-type-v q8_0
```

Watch for these lines in the startup log:

```
device_info: CUDA0 : NVIDIA GB10 (124546 MiB, 120586 MiB free)
common_init_from_params: warming up the model with an empty run
srv  llama_server: server is listening on http://127.0.0.1:8080
```

If you see `CUDA0` then the GPU is in use. If the only device listed is `CPU`, the CUDA build didn't take — go back to page 3.

Leave this process running for the smoke tests below.

## 5.2 Smoke tests

In a **second SSH session**:

```bash
# health
curl http://127.0.0.1:8080/health

# list models (no auth yet)
curl http://127.0.0.1:8080/v1/models | jq

# a small chat completion
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-coder-next",
    "messages": [{"role": "user", "content": "Write a minimal Java hello world program."}],
    "temperature": 0.2,
    "max_tokens": 256
  }' | jq -r '.choices[0].message.content'
```

If you get a code response, the runtime stack is good. Now stop the manual server with `Ctrl+C` — we'll add API keys before exposing it to anything.

## 5.3 Create the API key file

API keys are stored in a file that the `llama-server` process reads at startup. The file must be readable by `SERVICE_USER` but not by anyone else.

```bash
sudo mkdir -p /etc/llama-server
sudo touch /etc/llama-server/api_keys.txt
sudo chmod 640 /etc/llama-server/api_keys.txt
sudo chown root:SERVICE_USER /etc/llama-server/api_keys.txt
```

> The permissions are deliberately **640 root:SERVICE_USER**, not 600 root:root. The service user must be able to read the file. 600 root:root would silently break the systemd service on startup.

Generate one key per consumer (real user, application, or integration):

```bash
for consumer in alice bob ci-bot; do
  key="sk-${consumer}-$(openssl rand -hex 24)"
  echo "$key" | sudo tee -a /etc/llama-server/api_keys.txt >/dev/null
  echo "$consumer  ->  $key"
done
```

Copy the printed keys into a password manager (or hand them out via your secure-secrets workflow). After this terminal session ends, only the file remains.

The file should contain one key per line:

```
sk-alice-...
sk-bob-...
sk-ci-bot-...
```

## 5.4 Re-test with API keys enforced

Run the same launch command as in 5.1, but append:

```bash
  --api-key-file /etc/llama-server/api_keys.txt
```

In a second session:

```bash
# without auth → 401
curl -i http://127.0.0.1:8080/v1/models

# with a real key → 200
curl http://127.0.0.1:8080/v1/models \
  -H "Authorization: Bearer sk-alice-REPLACE_ME" | jq
```

Stop the server again. Next page wraps everything in systemd.

---

[← Download model](04-model-download.md) · [Index](README.md) · [Next: systemd service →](06-systemd-service.md)
