# 2. Base system & SSH hardening

[← Overview](01-overview.md) · [Index](README.md) · [Next: Build llama.cpp →](03-llama-cpp-build.md)

## 2.1 Install required packages

On the box:

```bash
sudo apt update
sudo apt upgrade -y

sudo apt install -y \
  git git-lfs build-essential cmake \
  curl wget htop nvtop unzip jq \
  ufw openssl \
  python3 python3-pip
```

```bash
git lfs install
```

Quick sanity check:

```bash
uname -a            # expect: aarch64
uname -m            # expect: aarch64
nvidia-smi          # expect: NVIDIA GB10
```

## 2.2 Create the service user

A dedicated non-root user owns the model runtime and runs the systemd service.

```bash
sudo adduser --disabled-password --gecos "" SERVICE_USER
```

Do **not** add `SERVICE_USER` to the `sudo` group — it doesn't need it.

```bash
sudo mkdir -p /opt/llm/models /opt/llm/runtime
sudo chown -R SERVICE_USER:SERVICE_USER /opt/llm
```

## 2.3 SSH key-based access from your workstation

On your **workstation** (macOS or Linux), generate a key dedicated to this box:

```bash
ssh-keygen -t ed25519 -C "admin@workstation-to-gb10"
# Suggested path: ~/.ssh/id_ed25519_gb10
```

Copy the public key to the box. If password SSH is still temporarily enabled by the vendor image:

```bash
ssh-copy-id -i ~/.ssh/id_ed25519_gb10.pub ADMIN_USER@SERVER_IP
```

Otherwise, paste `~/.ssh/id_ed25519_gb10.pub` into `/home/ADMIN_USER/.ssh/authorized_keys` on the box (create `~/.ssh` with `chmod 700` and the file with `chmod 600` if it doesn't exist).

Add a short alias on your workstation in `~/.ssh/config`:

```sshconfig
Host SERVER_HOST
    HostName SERVER_IP
    User ADMIN_USER
    IdentityFile ~/.ssh/id_ed25519_gb10
    IdentitiesOnly yes
```

Test:

```bash
ssh SERVER_HOST
```

## 2.4 Harden the SSH server

On the box:

```bash
sudo nano /etc/ssh/sshd_config
```

Make sure these are set:

```sshconfig
PermitRootLogin no
PasswordAuthentication no
PubkeyAuthentication yes
KbdInteractiveAuthentication no
X11Forwarding no
AllowUsers ADMIN_USER
```

Validate and restart, **from a second SSH session** so a typo doesn't lock you out:

```bash
sudo sshd -t
sudo systemctl restart ssh
```

Then verify with a fresh login from a third terminal before closing your existing sessions.

## 2.5 Host firewall (UFW)

Deny-by-default, allow only SSH:

```bash
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow OpenSSH
sudo ufw enable
sudo ufw status verbose
```

Expected output ends with:

```
Status: active
22/tcp (OpenSSH)           ALLOW IN    Anywhere
22/tcp (OpenSSH (v6))      ALLOW IN    Anywhere (v6)
```

Do **not** add a rule for port 8080. `llama-server` will bind to `127.0.0.1` only and stay invisible to the network.

> If you also want to expose Grafana on the LAN later (see page 9), prefer SSH local-port-forwarding (`ssh -L 3000:127.0.0.1:3000 SERVER_HOST`) over opening port 3000 in UFW. If you must open it, scope the rule: `sudo ufw allow from 10.0.0.0/24 to any port 3000 proto tcp`.

---

[← Overview](01-overview.md) · [Index](README.md) · [Next: Build llama.cpp →](03-llama-cpp-build.md)
