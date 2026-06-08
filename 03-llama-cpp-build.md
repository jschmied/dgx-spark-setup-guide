# 3. Build `llama.cpp` with CUDA

[← Base system](02-base-system.md) · [Index](README.md) · [Next: Download model →](04-model-download.md)

## 3.1 Switch to the service user

Most build and runtime steps from here on run as `SERVICE_USER`:

```bash
sudo -iu SERVICE_USER
```

## 3.2 Clone and build

```bash
cd /opt/llm/runtime
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp

cmake -B build \
  -DGGML_CUDA=ON \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build --config Release -j"$(nproc)"
```

The CUDA build picks up the GB10's Blackwell tensor cores and the native FP4 path automatically; you don't need to set the SM architecture manually.

## 3.3 Verify the build flags

```bash
/opt/llm/runtime/llama.cpp/build/bin/llama-server --version
```

Expect lines mentioning `aarch64`, `CUDA`, and a recent build counter. On a current mainline checkout the system_info line at startup also reports:

```
CUDA : ARCHS = 1210 | USE_GRAPHS = 1 | PEER_MAX_BATCH_SIZE = 128 | BLACKWELL_NATIVE_FP4 = 1
CPU  : NEON = 1 | ARM_FMA = 1 | LLAMAFILE = 1 | OPENMP = 1 | REPACK = 1
```

If you don't see `BLACKWELL_NATIVE_FP4 = 1`, you're either on an older commit or a non-CUDA build — re-run the cmake step with `-DGGML_CUDA=ON` and check that nvcc is on `PATH`.

## 3.4 Binaries you'll use later

```bash
ls -lh build/bin/llama-server build/bin/llama-cli build/bin/llama-gguf
```

- `llama-server` — the HTTP API daemon (most of this guide).
- `llama-cli` — interactive CLI; useful for smoke-testing a new GGUF without HTTP.
- `llama-gguf` — read GGUF metadata; used in page 8 to inspect model architecture and KV layout.

## 3.5 Updating later

```bash
sudo systemctl stop llama-router

cd /opt/llm/runtime/llama.cpp
git pull

cmake -B build \
  -DGGML_CUDA=ON \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build --config Release -j"$(nproc)"

sudo systemctl start llama-router
```

(The router systemd unit `llama-router.service` and its model preset are defined on page 6. Updating the binary doesn't touch the preset, so your model definitions carry over.)

---

[← Base system](02-base-system.md) · [Index](README.md) · [Next: Download model →](04-model-download.md)
