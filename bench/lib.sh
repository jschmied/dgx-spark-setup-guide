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
# Echo: temp top_p top_k repeat_penalty min_p
#   repeat_penalty 0 = omit; min_p 0 = omit; top_p 1.0 = nucleus disabled.
# gemma + ornstein-35b use min-p (top-p disabled) — it fixed their Go variance
# (gemma 1/4->4/4; ornstein-35b 1/4->4/4 at temp 0.3). See page 14.
sampling_for() {
  case "$1" in
    qwen3-coder-next)    echo "0.7 0.8  20 1.05 0"   ;;
    qwen36-35b-a3b)      echo "0.6 0.95 20 0    0"   ;;
    ornstein36-27B)      echo "1.0 0.95 20 0    0"   ;;
    ornstein36-35b-a3b)  echo "0.3 1.0  20 0    0.1" ;;
    gemma-4-26B-A4B)     echo "1.0 1.0  64 0    0.1" ;;
    qwopus36-35b-a3b)    echo "0.6 0.95 20 0    0"   ;;
    *)                   echo "${BENCH_TEMP:-0.7} ${BENCH_TOPP:-0.95} ${BENCH_TOPK:-40} 0 0" ;;
  esac
}

# call_model MODEL PROMPT_FILE MAX_TOKENS OUT_JSON
# Sends a non-streaming chat completion at the model's recommended sampling.
call_model() {
  local model="$1" prompt="$2" maxtok="$3" out="$4"
  local key; key="$(resolve_key)" || return 1
  local temp topp topk rep minp
  read -r temp topp topk rep minp <<<"$(sampling_for "$model")"
  minp="${minp:-0}"
  # experiment overrides: temperature, top_p (set 1.0 to disable nucleus), top_k, min_p
  [ -n "${BENCH_FORCE_TEMP:-}" ] && temp="$BENCH_FORCE_TEMP"
  [ -n "${BENCH_FORCE_TOPP:-}" ] && topp="$BENCH_FORCE_TOPP"
  [ -n "${BENCH_FORCE_TOPK:-}" ] && topk="$BENCH_FORCE_TOPK"
  [ -n "${BENCH_FORCE_MINP:-}" ] && minp="$BENCH_FORCE_MINP"
  echo ">>> $model  temp=$temp top_p=$topp top_k=$topk min_p=$minp repeat=$rep  max_tokens=$maxtok" >&2
  local body
  body=$(jq -n --rawfile p "$prompt" --arg m "$model" \
    --argjson temp "$temp" --argjson topp "$topp" --argjson topk "$topk" \
    --argjson rep "$rep" --argjson minp "$minp" --argjson mt "$maxtok" '
    {model:$m, messages:[{role:"user",content:$p}],
     temperature:$temp, top_p:$topp, top_k:$topk, max_tokens:$mt, stream:false}
    + (if $rep  > 0 then {repeat_penalty:$rep} else {} end)
    + (if $minp > 0 then {min_p:$minp} else {} end)')
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
