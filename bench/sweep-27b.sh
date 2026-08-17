#!/usr/bin/env bash
# Isolated MTP-nspec sweep for qwen36-27b-nvfp4 (DENSE hybrid, no MoE-marlin path).
# Bounces prod vllm.service, cycles nspec configs on :8080 at the util we intend to
# ship (0.6), records decode t/s + MTP acceptance + the real KV concurrency, restores prod.
#   run as:  SUDO_PW=... ./sweep-27b.sh
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLIENT="$HERE/bench_client.py"
OUT="${BENCH_OUT:-$HERE/results-nspec-27b-2026-07-10}"; mkdir -p "$OUT"
JSONL="$OUT/sweep.jsonl"; : > "$JSONL"
LOG="$OUT/sweep.log"
CACHEINFO="$OUT/kv_cache_info.txt"; : > "$CACHEINFO"
VENV=/opt/llm/runtime/vllm-venv/bin
MODEL_DIR=/opt/llm/models/qwen36-27b-nvfp4
UTIL="${UTIL:-0.6}"
NSPECS=(${NSPECS:-0 1 2 3 4})
CONCS=(${CONCS:-1 2 4 8})
S(){ echo "${SUDO_PW:?set SUDO_PW to run the sweep}" | sudo -S -p '' "$@" 2>/dev/null; }
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

metrics_pair(){ # echoes "accepted draft" cumulative counters (0 0 if absent)
  local m; m=$(curl -s http://127.0.0.1:8080/metrics 2>/dev/null)
  local a d
  a=$(echo "$m" | grep -E '^vllm:spec_decode_num_accepted_tokens_total' | awk '{print $2}' | head -1)
  d=$(echo "$m" | grep -E '^vllm:spec_decode_num_draft_tokens_total'    | awk '{print $2}' | head -1)
  echo "${a:-0} ${d:-0}"
}

capture_cacheinfo(){ # $1=nspec ; record KV pool sizing once per config
  local m; m=$(curl -s http://127.0.0.1:8080/metrics 2>/dev/null | grep '^vllm:cache_config_info')
  local tok conc
  tok=$(echo "$m"  | grep -oE 'kv_cache_size_tokens="[0-9]+"' | grep -oE '[0-9]+')
  conc=$(echo "$m" | grep -oE 'kv_cache_max_concurrency="[0-9.]+"' | grep -oE '[0-9.]+')
  echo "nspec=$1 util=$UTIL kv_cache_size_tokens=${tok:-NA} kv_cache_max_concurrency=${conc:-NA}" | tee -a "$CACHEINFO"
}

start_server(){ # $1=nspec
  local nspec=$1
  local envs="HF_HOME=/opt/llm/hf-cache HOME=/opt/llm XDG_CACHE_HOME=/opt/llm/.cache \
TRITON_CACHE_DIR=/opt/llm/.cache/triton VLLM_CACHE_ROOT=/opt/llm/.cache/vllm \
TORCHINDUCTOR_CACHE_DIR=/opt/llm/.cache/torchinductor FLASHINFER_WORKSPACE_BASE=/opt/llm/.cache/flashinfer"
  # NOTE: single-quote the JSON. Inside the `bash -c "..."` below an unquoted
  # {"a":..,"b":..} is a bash brace-expansion (comma in braces) and gets mangled
  # to `method:mtp num_speculative_tokens:N` — vLLM then rejects the value.
  local spec_flag=""
  [ "$nspec" != "0" ] && spec_flag="--speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":${nspec}}'"
  local slog="/tmp/qwen27bench_n${nspec}.log"
  log "launch 27B nspec=$nspec util=$UTIL  (server log: $slog)"
  S -u llm bash -c "cd /opt/llm && env $envs $VENV/vllm serve $MODEL_DIR \
    --served-model-name qwen36-27b \
    --host 127.0.0.1 --port 8080 --api-key sk-bench \
    --quantization modelopt --trust-remote-code \
    --kv-cache-dtype fp8 --attention-backend flashinfer \
    --gpu-memory-utilization $UTIL --max-model-len 262144 \
    --max-num-seqs 12 --max-num-batched-tokens 16384 \
    --enable-chunked-prefill --async-scheduling --no-enable-prefix-caching \
    --load-format fastsafetensors \
    --override-generation-config '{\"temperature\":0.6,\"top_p\":0.95,\"top_k\":20,\"min_p\":0,\"presence_penalty\":0}' \
    $spec_flag \
    > $slog 2>&1" &
  # cold start is heavy: engine init (flashinfer fp8 autotune + torch.compile) ~320s,
  # plus API-server + multimodal warmup. Real systemd unit allows 600s; give 700 here.
  local i
  for i in $(seq 1 700); do
    curl -sf http://127.0.0.1:8080/health >/dev/null 2>&1 && { log "healthy after ${i}s"; return 0; }
    sleep 1
  done
  log "SERVER FAILED TO START (nspec=$nspec) -- see $slog"; return 1
}

stop_server(){ S pkill -f "vllm serve $MODEL_DIR" ; sleep 8; }

sweep(){ # $1=nspec  -> runs concurrency sweep, appends jsonl
  local nspec=$1 c reqs acc dr acc2 dr2 rate res da dd
  for c in "${CONCS[@]}"; do
    reqs=$(( c*2 )); [ $reqs -lt 4 ] && reqs=4; [ $reqs -gt 12 ] && reqs=12
    read acc dr <<<"$(metrics_pair)"
    # 27B dense is memory-bound (~12 t/s); shorter outputs keep the sweep bounded
    # while still generating enough tokens for a stable MTP acceptance estimate.
    res=$("$VENV/python3" "$CLIENT" --concurrency "$c" --requests "$reqs" \
          --output-len 256 --model qwen36-27b --key sk-bench --temp 0.6)
    read acc2 dr2 <<<"$(metrics_pair)"
    da=$(( ${acc2%.*}-${acc%.*} )); dd=$(( ${dr2%.*}-${dr%.*} ))
    rate="NA"; [ "$nspec" != "0" ] && [ "$dd" -gt 0 ] && rate=$(awk "BEGIN{printf \"%.3f\", $da/$dd}")
    echo "$res" | "$VENV/python3" -c "import sys,json; d=json.load(sys.stdin); d.update({'nspec':$nspec,'accept_rate':'$rate'}); print(json.dumps(d))" | tee -a "$JSONL"
  done
}

### MAIN
log "=== STOPPING PROD vllm.service ==="
S systemctl stop vllm.service; sleep 5
curl -sf http://127.0.0.1:8080/health >/dev/null 2>&1 && { log "port 8080 still answering after stop -- ABORT"; exit 1; }

for NS in "${NSPECS[@]}"; do
  start_server "$NS" || { stop_server; continue; }
  capture_cacheinfo "$NS"
  sweep "$NS"
  stop_server
done

log "=== RESTORING PROD vllm.service ==="
S systemctl start vllm.service
for i in $(seq 1 360); do curl -sf http://127.0.0.1:8080/health >/dev/null 2>&1 && { log "PROD RESTORED after ${i}s"; break; }; sleep 1; done
log "=== SWEEP COMPLETE -> $JSONL ==="
