# 4. Download the GGUF model

[← Build llama.cpp](03-llama-cpp-build.md) · [Index](README.md) · [Next: First run →](05-first-run.md)

The example model in this guide is **`unsloth/Qwen3-Coder-Next-GGUF`** — an ~80 B-parameter MoE coder model with 3 B active parameters per token and a 262 K-token native context, packaged as GGUF by Unsloth.

The instructions are generic; substitute any GGUF model that fits in 128 GB unified memory.

## 4.1 Install the Hugging Face CLI

As `SERVICE_USER`:

```bash
python3 -m pip install --user -U huggingface_hub
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

If you'll download gated or private repos, log in:

```bash
huggingface-cli login
```

## 4.2 Pick a quantization

Unsloth ships several "UD" (Unsloth Dynamic) quants. Rule of thumb for this model on GB10:

| Quant | Approx. file size | Memory budget hint | Notes |
|---|---|---|---|
| `UD-Q3_K_XL` | ~36 GB | tight | Acceptable on 64 GB systems; some quality loss |
| **`UD-Q4_K_XL`** | **~50 GB** | **comfortable on 128 GB** | **Recommended default** |
| `UD-Q5_K_XL` | ~58 GB | comfortable | Higher quality, slightly slower |
| `UD-Q6_K_XL` | ~67 GB | OK | Diminishing returns |
| `UD-Q8_K_XL` | ~85 GB | OK | Use if quality matters most |

The rest of the guide assumes `UD-Q4_K_XL`. If you choose a different one, substitute filenames accordingly.

## 4.3 Download

```bash
mkdir -p /opt/llm/models/MODEL_NAME
cd /opt/llm/models/MODEL_NAME

huggingface-cli download unsloth/Qwen3-Coder-Next-GGUF \
  --include "*UD-Q4_K_XL*.gguf" \
  --local-dir /opt/llm/models/MODEL_NAME \
  --local-dir-use-symlinks False
```

Verify:

```bash
ls -lh /opt/llm/models/MODEL_NAME/*.gguf
```

You should see a single ~50 GB file. Note the exact name — you'll reference it as `MODEL_FILE` on page 6.

## 4.4 Sanity-check the GGUF metadata

```bash
/opt/llm/runtime/llama.cpp/build/bin/llama-gguf \
  /opt/llm/models/MODEL_NAME/MODEL_FILE r 2>&1 | grep -E "qwen|general\.name|tokenizer" | head -20
```

For Qwen3-Coder-Next you'll see keys under the `qwen3next.*` namespace (SSM layers, `expert_count`, `full_attention_interval`, etc.). That confirms it's the hybrid SSM-MoE architecture rather than a standard transformer. Page 8 explains why that matters for tuning options like speculative decoding.

## 4.5 Disk and memory budget

```
GGUF file (Q4_K_XL):      ~50 GB
Process RSS at idle:      ~2 GB (model is mmap'd)
GPU memory used:          ~50 GB (model fully offloaded)
KV cache (q8_0, 131 K):   ~5 GB per slot, shared across slots if --kv-unified
Prompt cache pool:        --cache-ram MiB (tunable, default 8 GiB)
Headroom for OS/cache:    leave at least 20 GB
```

On a 128 GB box this leaves plenty of room. Page 8 raises `--cache-ram` and adds `--mlock` to lock the model in RAM.

---

[← Build llama.cpp](03-llama-cpp-build.md) · [Index](README.md) · [Next: First run →](05-first-run.md)
