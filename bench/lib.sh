#!/usr/bin/env bash
# Shared helpers for the coding benchmark (pages 12/13).
# Sourced by run-go.sh / run-java.sh / run-all.sh.
#
# Sampling policy: each model is run at its OWN recommended sampling (the value
# pinned in /etc/llama-server/models.ini and the model card / GGUF
# general.sampling), NOT a single fixed temperature. A fixed temperature is fair
# for comparison but misrepresents models whose architecture expects a different
# operating point (e.g. Gemma 4 and the dense Ornstein loop at low temp).

BENCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST="${LLAMA_HOST:-http://127.0.0.1:8080}"
RESULTS_DIR="${BENCH_RESULTS:-$BENCH_DIR/results}"
WORK_DIR="${BENCH_WORK:-$BENCH_DIR/.work}"

# --- API key resolution -----------------------------------------------------
# Prefer $LLAMA_API_KEY; else read the first key from the router's key file.
# Never echo the key.
resolve_key() {
  if [[ -n "${LLAMA_API_KEY:-}" ]]; then
    printf '%s' "$LLAMA_API_KEY"; return 0
  fi
  if [[ -r /etc/llama-server/api_keys.txt ]]; then
    head -1 /etc/llama-server/api_keys.txt; return 0
  fi
  if command -v sudo >/dev/null; then
    sudo -n head -1 /etc/llama-server/api_keys.txt 2>/dev/null && return 0
    echo "ERROR: cannot read API key. Set LLAMA_API_KEY or run with sudo." >&2
    return 1
  fi
  echo "ERROR: no API key available. Set LLAMA_API_KEY." >&2; return 1
}

# --- Per-model recommended sampling -----------------------------------------
# Echo: temp top_p top_k repeat_penalty  (repeat_penalty 0 = omit)
sampling_for() {
  case "$1" in
    qwen3-coder-next)    echo "0.7 0.8  20 1.05" ;;
    qwen36-35b-a3b)      echo "0.6 0.95 20 0"    ;;
    ornstein36-27B)      echo "1.0 0.95 20 0"    ;;
    ornstein36-35b-a3b)  echo "0.6 0.95 20 0"    ;;
    gemma-4-26B-A4B)     echo "1.0 0.95 64 0"    ;;
    *)                   echo "${BENCH_TEMP:-0.7} ${BENCH_TOPP:-0.95} ${BENCH_TOPK:-40} 0" ;;
  esac
}

# call_model MODEL PROMPT_FILE MAX_TOKENS OUT_JSON
# Sends a non-streaming chat completion at the model's recommended sampling.
call_model() {
  local model="$1" prompt="$2" maxtok="$3" out="$4"
  local key; key="$(resolve_key)" || return 1
  read -r temp topp topk rep <<<"$(sampling_for "$model")"
  echo ">>> $model  temp=$temp top_p=$topp top_k=$topk repeat=$rep  max_tokens=$maxtok" >&2
  local body
  body=$(jq -n --rawfile p "$prompt" --arg m "$model" \
    --argjson temp "$temp" --argjson topp "$topp" --argjson topk "$topk" \
    --argjson rep "$rep" --argjson mt "$maxtok" '
    {model:$m, messages:[{role:"user",content:$p}],
     temperature:$temp, top_p:$topp, top_k:$topk, max_tokens:$mt, stream:false}
    + (if $rep > 0 then {repeat_penalty:$rep} else {} end)')
  # Pass the key via a --config fd (process substitution) so it never appears
  # in the process list / argv. Only the prompt body goes on the command line.
  curl -s --max-time 2400 "$HOST/v1/chat/completions" \
    -H "Content-Type: application/json" \
    --config <(printf 'header = "Authorization: Bearer %s"\n' "$key") \
    -d "$body" > "$out"
  # surface finish reason / token usage
  jq -r '{finish:.choices[0].finish_reason,
          completion_tokens:.usage.completion_tokens,
          has_reasoning:((.choices[0].message.reasoning_content//"")|length>0),
          content_len:(.choices[0].message.content|length)}' "$out" >&2
  local finish; finish=$(jq -r '.choices[0].finish_reason // "ERROR"' "$out")
  if [[ "$finish" != "stop" ]]; then
    echo "WARN: finish_reason=$finish (not 'stop'); raise max_tokens and re-run for a fair sample." >&2
  fi
}
