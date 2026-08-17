#!/usr/bin/env bash
# Isolated bench sweep for qwen36-35b-a3b-nvfp4: atomic-add x MTP-nspec x concurrency.
# Bounces prod vllm.service, cycles configs on :8080, restores prod at the end.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLIENT="$HERE/bench_client.py"
OUT="${BENCH_OUT:-$HERE/results-sweep}"; mkdir -p "$OUT"
JSONL="$OUT/sweep.jsonl"; : > "$JSONL"
LOG="$OUT/sweep.log"
VENV=/opt/llm/runtime/vllm-venv/bin
MODEL_DIR=/opt/llm/models/qwen36-35b-a3b-nvfp4
# sudo password comes from the environment, never hard-coded: run as  SUDO_PW=... ./sweep.sh
# (or drop the wrapper and use passwordless sudo). Server logs go to /tmp (the `llm` user can write there).
S(){ echo "${SUDO_PW:?set SUDO_PW to run the sweep}" | sudo -S -p '' "$@" 2>/dev/null; }

log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

metrics_pair(){ # echoes "accepted draft" cumulative counters (0 0 if absent)
  local m; m=$(curl -s http://127.0.0.1:8080/metrics 2>/dev/null)
  local a d
  a=$(echo "$m" | grep -E '^vllm:spec_decode_num_accepted_tokens_total' | awk '{print $2}' | head -1)
  d=$(echo "$m" | grep -E '^vllm:spec_decode_num_draft_tokens_total'    | awk '{print $2}' | head -1)
  echo "${a:-0} ${d:-0}"
}

start_server(){ # $1=nspec $2=atomic(0/1)
  local nspec=$1 atomic=$2
  local spec="{\"method\":\"mtp\",\"num_speculative_tokens\":${nspec},\"moe_backend\":\"triton\"}"
  local envs="HF_HOME=/opt/llm/hf-cache HOME=/opt/llm XDG_CACHE_HOME=/opt/llm/.cache \
TRITON_CACHE_DIR=/opt/llm/.cache/triton VLLM_CACHE_ROOT=/opt/llm/.cache/vllm \
TORCHINDUCTOR_CACHE_DIR=/opt/llm/.cache/torchinductor FLASHINFER_WORKSPACE_BASE=/opt/llm/.cache/flashinfer"
  [ "$atomic" = "1" ] && envs="$envs VLLM_MARLIN_USE_ATOMIC_ADD=1"
  local slog="/tmp/qwenbench_n${nspec}_a${atomic}.log"
  log "launch nspec=$nspec atomic=$atomic  (server log: $slog)"
  S -u llm bash -c "cd /opt/llm && env $envs $VENV/vllm serve $MODEL_DIR \
    --served-model-name qwen36-35b-a3b \
    --host 127.0.0.1 --port 8080 --api-key sk-bench \
    --quantization modelopt --trust-remote-code \
    --kv-cache-dtype fp8 --attention-backend flashinfer --moe-backend marlin \
    --gpu-memory-utilization 0.4 --max-model-len 262144 \
    --max-num-seqs 12 --max-num-batched-tokens 16384 \
    --enable-chunked-prefill --async-scheduling --no-enable-prefix-caching \
    --load-format fastsafetensors \
    --override-generation-config '{\"temperature\":0.6,\"top_p\":0.95,\"top_k\":20,\"min_p\":0,\"presence_penalty\":0}' \
    --speculative-config '$spec' \
    > $slog 2>&1" &
  # wait for health
  local i
  for i in $(seq 1 360); do
    curl -sf http://127.0.0.1:8080/health >/dev/null 2>&1 && { log "healthy after ${i}s"; return 0; }
    sleep 1
  done
  log "SERVER FAILED TO START (nspec=$nspec atomic=$atomic)"; return 1
}

stop_server(){ S pkill -f "vllm serve $MODEL_DIR" ; sleep 8; }

sweep(){ # $1=nspec $2=atomic  -> runs concurrency sweep, appends jsonl
  local nspec=$1 atomic=$2 c reqs acc dr acc2 dr2 rate res
  for c in 1 2 4 8 12; do
    reqs=$(( c*3 )); [ $reqs -lt 6 ] && reqs=6; [ $reqs -gt 36 ] && reqs=36
    read acc dr <<<"$(metrics_pair)"
    res=$("$VENV/python3" "$CLIENT" --concurrency "$c" --requests "$reqs" --key sk-bench --temp 0.6)
    read acc2 dr2 <<<"$(metrics_pair)"
    local da=$(( ${acc2%.*}-${acc%.*} )) dd=$(( ${dr2%.*}-${dr%.*} ))
    rate="NA"; [ "$dd" -gt 0 ] && rate=$(awk "BEGIN{printf \"%.3f\", $da/$dd}")
    echo "$res" | "$VENV/python3" -c "import sys,json; d=json.load(sys.stdin); d.update({'nspec':$nspec,'atomic':$atomic,'accept_rate':'$rate'}); print(json.dumps(d))" | tee -a "$JSONL"
  done
}

### MAIN
log "=== STOPPING PROD vllm.service ==="
S systemctl stop vllm.service; sleep 5
curl -sf http://127.0.0.1:8080/health >/dev/null 2>&1 && { log "port 8080 still answering after stop -- ABORT"; exit 1; }

# Phase 1: atomic OFF, nspec 2/3/4  (find best nspec)
for NS in 2 3 4; do
  start_server "$NS" 0 || { stop_server; continue; }
  sweep "$NS" 0
  stop_server
done

# Phase 2: atomic ON at nspec 2 and 3 (the low-nspec range where MTP usually wins)
for NS in 2 3; do
  start_server "$NS" 1 || { stop_server; continue; }
  sweep "$NS" 1
  stop_server
done

log "=== RESTORING PROD vllm.service ==="
S systemctl start vllm.service
for i in $(seq 1 360); do curl -sf http://127.0.0.1:8080/health >/dev/null 2>&1 && { log "PROD RESTORED after ${i}s"; break; }; sleep 1; done
log "=== SWEEP COMPLETE -> $JSONL ==="
