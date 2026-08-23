#!/bin/bash
# Faehrt die verbleibenden Arme. Erwartet SUDO_PW und LLM_KEY in der Umgebung.
set -u
S="${OUT:-/tmp/blockprobe}"; mkdir -p "$S"
SP=/opt/llm/runtime/vllm-venv-maintest/lib/python3.12/site-packages/vllm
SUDO(){ printf '%s\n' "${SUDO_PW:?set SUDO_PW}" | sudo -S -p '' "$@" 2>/dev/null; }

run_arm(){  # $1 label  $2 worktree  $3 extra-args  $4 blk
  echo "=== ARM $1 ($(date +%H:%M:%S)) ==="
  SUDO rsync -a --exclude='*.so' --exclude='*.pyd' --exclude='_version.py' "$2/vllm/" "$SP/"
  # Beweis, dass der Code wirklich liegt: Datei-Identitaet, nicht Stichwortsuche
  for f in v1/core/sched/scheduler.py v1/core/kv_cache_utils.py; do
    cmp -s "$2/vllm/$f" "$SP/$f" && echo "  $f == Baum" || echo "  !! $f WEICHT AB"
  done
  SUDO systemd-run --unit=arm --collect \
    -E VLLM_QWEN38_VENV=/opt/llm/runtime/vllm-venv-maintest \
    -E VLLM_QWEN38_WEIGHTS=/opt/llm/models/qwen38-27b-radixark \
    -E VLLM_QWEN38_SPEC=mtp -E VLLM_QWEN38_NSPEC=3 -E VLLM_QWEN38_UTIL=0.76 \
    -E "VLLM_QWEN38_EXTRA=$3" \
    -E XDG_CACHE_HOME=/opt/llm/.cache-mt -E TRITON_CACHE_DIR=/opt/llm/.cache-mt/triton \
    -E VLLM_CACHE_ROOT=/opt/llm/.cache-mt/vllm -E TORCHINDUCTOR_CACHE_DIR=/opt/llm/.cache-mt/torchinductor \
    -E FLASHINFER_WORKSPACE_BASE=/opt/llm/.cache-mt/flashinfer -E MAX_JOBS=2 -E FLASHINFER_NVCC_THREADS=1 \
    bash /opt/llm/models/qwen38-27b-nvfp4/serve-qwen38.sh >/dev/null 2>&1
  ok=0
  for i in $(seq 1 50); do
    [ "$(curl -sS -m 5 -o /dev/null -w '%{http_code}' http://127.0.0.1:8080/health 2>/dev/null)" = "200" ] && { ok=1; break; }
    systemctl is-active arm >/dev/null 2>&1 || break
    sleep 10
  done
  if [ "$ok" = 1 ]; then
    SUDO journalctl -u arm --no-pager -o cat | grep -oE "attention block size to [0-9]+|prefix_match_unit=[0-9None]+" | sort -u | sed 's/^/  /'
    LLM_KEY="$LLM_KEY" BLK=$4 OFF=2000000 python3 $S/blockprobe.py "$1" "$S/probe-$1.json" 2>&1 | tail -4
  else
    echo "  ARM $1 KAM NICHT HOCH"
    SUDO journalctl -u arm --no-pager -o cat -n 15 | tail -8 | sed 's/^/    /'
  fi
  SUDO systemctl stop arm 2>/dev/null; sleep 5
}

SUDO systemctl stop radixark-serve.service; sleep 5
echo "PROD GESTOPPT $(date +%H:%M:%S)"
run_arm "50897-pmu100" /home/jschmied/wt/pr50897 '--override-generation-config {"temperature":0.6} --prefix-match-unit 100' 1600
run_arm "52244-base"   /home/jschmied/wt/base52244 '--override-generation-config {"temperature":0.6}' 1600
run_arm "52244-head"   /home/jschmied/wt/pr52244   '--override-generation-config {"temperature":0.6}' 1600
run_arm "52244-pmu100" /home/jschmied/wt/pr52244   '--override-generation-config {"temperature":0.6} --prefix-match-unit 100' 1600
echo "ARME FERTIG $(date +%H:%M:%S) -- stelle prod wieder her"
SUDO systemd-run --unit=radixark-serve --collect \
  -E VLLM_QWEN38_VENV=/opt/llm/runtime/vllm-venv-pr52816 \
  -E VLLM_QWEN38_WEIGHTS=/opt/llm/models/qwen38-27b-radixark \
  -E VLLM_QWEN38_SPEC=dflash -E VLLM_QWEN38_NSPEC=7 \
  -E VLLM_QWEN38_DRAFT=/opt/llm/models/qwen38-27b-dflash2-syvai-w4a16 \
  -E VLLM_QWEN38_UTIL=0.76 \
  -E 'VLLM_QWEN38_EXTRA=--override-generation-config {"temperature":0.6}' \
  -E XDG_CACHE_HOME=/opt/llm/.cache-pr -E TRITON_CACHE_DIR=/opt/llm/.cache-pr/triton \
  -E VLLM_CACHE_ROOT=/opt/llm/.cache-pr/vllm -E TORCHINDUCTOR_CACHE_DIR=/opt/llm/.cache-pr/torchinductor \
  -E FLASHINFER_WORKSPACE_BASE=/opt/llm/.cache-pr/flashinfer -E MAX_JOBS=2 -E FLASHINFER_NVCC_THREADS=1 \
  bash /opt/llm/models/qwen38-27b-nvfp4/serve-qwen38.sh >/dev/null 2>&1
for i in $(seq 1 50); do
  [ "$(curl -sS -m 5 -o /dev/null -w '%{http_code}' http://127.0.0.1:8080/health 2>/dev/null)" = "200" ] && { echo "PROD WIEDER OBEN $(date +%H:%M:%S)"; break; }
  sleep 10
done
echo "ALL DONE"
