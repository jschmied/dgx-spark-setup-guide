# 11. Security checklist

[← Operations](10-operations.md) · [Index](README.md)

Final pass before declaring the box production-ready. Tick each item or have a documented reason for skipping it.

## Host

- [ ] System packages updated (`sudo apt update && sudo apt upgrade -y`)
- [ ] Dedicated non-root `SERVICE_USER` exists and owns `/opt/llm`
- [ ] Model service does **not** run as root (`User=SERVICE_USER` in the unit)
- [ ] SSH root login disabled (`PermitRootLogin no`)
- [ ] SSH password authentication disabled (`PasswordAuthentication no`)
- [ ] SSH key authentication enabled (`PubkeyAuthentication yes`)
- [ ] SSH access limited (`AllowUsers ADMIN_USER`)
- [ ] UFW enabled with default-deny inbound
- [ ] Only OpenSSH is allowed inbound
- [ ] Port 8080 is **not** in any UFW rule
- [ ] No router port-forward to the box

## Model runtime

- [ ] `llama-server` binds to `127.0.0.1` (verified with `ss -tulpn`)
- [ ] `llama-server` started with `--api-key-file`
- [ ] One API key per consumer; no shared "team key"
- [ ] `/etc/llama-server/api_keys.txt`: mode **640**, owner **root:SERVICE_USER**
  (so `SERVICE_USER` can read it; **not** 600 root:root)
- [ ] `/etc/llama-server/models.ini`: mode **640**, owner **root:SERVICE_USER**
  (the router runs as `SERVICE_USER` and must read the preset)
- [ ] `/etc/llama-server/router.env`: mode **600**, owner **root:root**
- [ ] systemd unit has `LimitMEMLOCK=infinity` if `mlock = true` is in the preset
- [ ] systemd unit has sandboxing flags (`NoNewPrivileges`, `PrivateTmp`, `ProtectSystem`, `ProtectHome`, `ReadWritePaths`)
- [ ] Service auto-restarts on failure (`Restart=always`)

## Public exposure (only if page 7 was followed)

- [ ] Cloudflare Tunnel is the **only** path from internet to the box
- [ ] Edge auth (Cloudflare Access or API-key auth) configured per-consumer
- [ ] `llama-server` bearer key still required in addition to edge auth
- [ ] Rate limiting / WAF rules configured on the public hostname
- [ ] No DNS record exposes the origin IP
- [ ] Cloudflare logs are reachable and reviewed
- [ ] Alert on failed-auth spikes

## Monitoring stack (page 9)

- [ ] Grafana admin password changed from defaults
- [ ] Grafana binds to `127.0.0.1` only — accessed via SSH tunnel
- [ ] Prometheus binds to `127.0.0.1` only
- [ ] Prometheus scrape job for `llama-server` uses a **dedicated** API key (not a user key)
- [ ] Dashboard provisioner pinned to a known datasource UID
- [ ] Starter alerts configured (service down, queueing, GPU temp, XID errors, throughput drop)

## Client side

- [ ] Clients use HTTPS hostname (when via Cloudflare) or SSH tunnel (when LAN-only)
- [ ] Clients carry both edge auth (if public) and `Authorization: Bearer sk-…`
- [ ] API keys stored in a password / secret manager
- [ ] API keys are **not** committed to source control
- [ ] API keys are **not** embedded in client-side frontend code

## Operational hygiene

- [ ] Backup of `/etc/llama-server/api_keys.txt` exists, encrypted, off-box
- [ ] systemd unit and model preset backed up (`llama-router.service.bak`, `models.ini.bak` from page 8)
- [ ] Documented rollback procedure (page 10)
- [ ] One known-good llama.cpp commit pinned somewhere in case `git pull` regresses
- [ ] Monitoring dashboard bookmarked by everyone who'll be on-call

## Residual risks worth knowing

| Risk | Mitigation |
|---|---|
| API key leakage via shell history / screenshots / tickets | Per-user keys, rotate on suspicion, audit access logs |
| Compromise of an admin laptop | Revoke its SSH key from `~/.ssh/authorized_keys`; rotate any API keys that laptop held |
| Accidental public exposure | The combination of UFW deny-by-default + `127.0.0.1` bind + no port-forward defeats casual mistakes; Cloudflare Tunnel is the only sanctioned path |
| Brute-force API abuse | Edge rate limiting; per-key issuance lets you ban one without affecting others |
| Excessive resource use by one consumer | Currently no per-key rate limiting in llama-server. If this matters, put a proxy (LiteLLM, etc.) in front |
| Driver / GPU fault | Surface via Grafana `XID errors` panel; alert; have a reboot runbook |
| Secrets in journal | Don't paste live keys into smoke-test commands you'll forget about. Use environment variables in your shell history-less shell or `read -s` |

---

[← Operations](10-operations.md) · [Index](README.md)
