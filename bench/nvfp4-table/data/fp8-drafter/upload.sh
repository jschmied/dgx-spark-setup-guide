#!/usr/bin/env bash
# Publish the FP8 DFlash2 drafter to the Hugging Face Hub.
#
# Secrets come from the environment, never from this file (see bench/sweep.sh).
#   export HF_TOKEN=hf_...            # write scope
#   export HF_REPO=<account>/Qwen3.8-27B-DFlash2-FP8
#   [export HF_PRIVATE=1]             # default is public
#
# Uploads: model.safetensors, config.json, the model card as README.md, and the
# quantization script as quantize.py. Nothing else lives in the source directory.
set -euo pipefail

: "${HF_TOKEN:?set HF_TOKEN (write scope) before running}"
: "${HF_REPO:?set HF_REPO, e.g. yourname/Qwen3.8-27B-DFlash2-FP8}"

SRC=/opt/llm/models/qwen38-27b-dflash2-fp8
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HF="/home/jschmied/vllm-venv-branch-fp8/bin/hf"
STAGE="$(mktemp -d)"; trap 'rm -rf "$STAGE"' EXIT

[ -f "$SRC/model.safetensors" ] || { echo "missing $SRC/model.safetensors" >&2; exit 1; }

# Stage: the card becomes README.md, the recipe travels with the weights.
cp "$SRC/model.safetensors" "$SRC/config.json" "$STAGE/"
cp "$HERE/MODEL_CARD.md"             "$STAGE/README.md"
cp "$HERE/quantize-dflash2-fp8.py"   "$STAGE/quantize.py"

# Refuse to publish a staged file that carries a secret.
if grep -rlIE "sk-[A-Za-z0-9_-]{16,}|hf_[A-Za-z0-9]{30,}|develop8\." "$STAGE" 2>/dev/null | grep -q .; then
    echo "ABORT: secret found in staged files" >&2; exit 1
fi

echo "staged for ${HF_REPO}:"; ls -lh "$STAGE" | tail -n +2 | awk '{printf "  %-24s %s\n",$9,$5}'

# hf reads HF_TOKEN from the environment; --token is not accepted by every subcommand.
export HF_TOKEN
"$HF" auth whoami || { echo "token rejected" >&2; exit 1; }
"$HF" repo create "$HF_REPO" --repo-type model ${HF_PRIVATE:+--private} --exist-ok
"$HF" upload "$HF_REPO" "$STAGE" . --repo-type model \
      --commit-message "FP8 (W8A8, per-channel weights, dynamic activations) DFlash2 drafter"

echo "done: https://huggingface.co/${HF_REPO}"
