# 7. Public access via Cloudflare Tunnel (optional)

[← systemd service](06-systemd-service.md) · [Index](README.md) · [Next: Performance tuning →](08-performance-tuning.md)

> Skip this page if you only need LAN access. The model server is fine on `127.0.0.1` for SSH-tunneled use.

This page covers the architecture used to expose the model HTTPS-publicly **without** opening any ports on the box. Cloudflare Tunnel dials *out* to Cloudflare's edge; clients reach the edge, not the box.

```
Client ──https──▶ Cloudflare Edge ──auth──▶ Cloudflare Tunnel ──▶ 127.0.0.1:8080
```

## 7.1 Install `cloudflared`

```bash
sudo mkdir -p --mode=0755 /usr/share/keyrings

curl -fsSL https://pkg.cloudflare.com/cloudflare-main.gpg \
  | sudo tee /usr/share/keyrings/cloudflare-main.gpg >/dev/null

echo "deb [signed-by=/usr/share/keyrings/cloudflare-main.gpg] https://pkg.cloudflare.com/cloudflared any main" \
  | sudo tee /etc/apt/sources.list.d/cloudflared.list

sudo apt update
sudo apt install -y cloudflared
cloudflared --version
```

## 7.2 Create the tunnel from the Cloudflare dashboard

In Cloudflare → **Zero Trust → Networks → Tunnels → Create tunnel**:

1. Connector type: `cloudflared`
2. Tunnel name: e.g. `gb10-llm`
3. Copy the installation command Cloudflare gives you (one line, includes the tunnel token).
4. Run it on the box. This drops a `cloudflared` systemd service that authenticates to Cloudflare with the token and stays connected.

Then in the same UI, configure the **Public hostname**:

| Setting | Value |
|---|---|
| Subdomain + domain | `PUBLIC_HOSTNAME` (e.g. `llm.example.com`) |
| Service type | HTTP |
| URL | `127.0.0.1:8080` |

That's the entire network plumbing. No port-forwarding, no inbound rules.

## 7.3 Confirm the tunnel is up

```bash
sudo systemctl status cloudflared
sudo journalctl -u cloudflared -f
```

You should see lines about registered connections to Cloudflare's edge. Then from your workstation:

```bash
curl https://PUBLIC_HOSTNAME/health
```

If you get `OK` (no auth required at this point), the path edge → tunnel → llama-server is complete.

## 7.4 Add edge authentication

`llama-server`'s bearer key alone is enough auth *if* the public URL is on a domain you trust and the keys never leak. In practice you want a second factor at the edge so brute-force attempts never reach your tunnel.

Cloudflare offers two layers you can put in front:

| Mechanism | Use case |
|---|---|
| **Cloudflare Access** application policy | Real users from a known identity provider (Google, GitHub, OIDC, SAML). One policy per hostname. |
| **Cloudflare API Shield / API key auth** | Machine clients (CI, IDE plugins) authenticated with a Cloudflare-issued key carried as a header. |

Both are configured in the Zero Trust dashboard. Pick whichever matches your callers.

> Regardless of which you pick, **also** keep the `llama-server` bearer key. A valid request from outside must satisfy both: Cloudflare-edge auth *and* `Authorization: Bearer sk-…`. If one layer is misconfigured, the other still holds.

## 7.5 Hardening checklist for the Cloudflare side

- Per-user / per-app credentials, not one shared key
- Short expiry where the auth mechanism supports it; rotate when a user leaves
- Cloudflare access logs forwarded somewhere queryable; alert on failed auth spikes
- Rate limiting on the public hostname (Cloudflare's WAF/rate-limit rules)
- No DNS A record pointing at the box's IP — only the proxied tunnel hostname
- No router port-forward to the box

## 7.6 End-to-end test from your workstation

```bash
export LLM_BASE_URL="https://PUBLIC_HOSTNAME/v1"
export LLM_API_KEY="sk-alice-REPLACE_ME"
# plus whatever Cloudflare-side auth you configured, as another header

curl "$LLM_BASE_URL/chat/completions" \
  -H "Authorization: Bearer $LLM_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-coder-next",
    "messages": [{"role": "user", "content": "Write a Spring Boot REST controller with one GET endpoint."}],
    "max_tokens": 1024
  }' | jq
```

## 7.7 Python (OpenAI SDK) example

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://PUBLIC_HOSTNAME/v1",
    api_key="sk-alice-REPLACE_ME",
    # add whatever Cloudflare-side headers your policy expects:
    # default_headers={"CF-Access-Client-Id": "...", "CF-Access-Client-Secret": "..."}
)

resp = client.chat.completions.create(
    model="qwen3-coder-next",
    messages=[{"role": "user",
               "content": "Write a Java method that validates an email address."}],
    max_tokens=512,   # no temperature: let the server use the model's recommended sampling (page 8 §8.8)
)
print(resp.choices[0].message.content)
```

---

[← systemd service](06-systemd-service.md) · [Index](README.md) · [Next: Performance tuning →](08-performance-tuning.md)
