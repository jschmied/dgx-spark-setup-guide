#!/bin/bash
set -u
S="${OUT:?}"
SUDO(){ printf '%s\n' "${SUDO_PW:?}" | sudo -S -p '' "$@" 2>/dev/null; }
arm(){ # $1 label  $2 retention-env ("" = Standard)
  echo "=== ARM $1 ($(date +%H:%M:%S)) ==="
  SUDO systemd-run --unit=arm --collect \
    -E VLLM_QWEN38_VENV=/opt/llm/runtime/vllm-venv-pr52816 \
    -E VLLM_QWEN38_WEIGHTS=/opt/llm/models/qwen38-27b-radixark \
    -E VLLM_QWEN38_SPEC=dflash -E VLLM_QWEN38_NSPEC=7 \
    -E VLLM_QWEN38_DRAFT=/opt/llm/models/qwen38-27b-dflash2-syvai-w4a16 \
    -E VLLM_QWEN38_UTIL=0.76 -E 'VLLM_QWEN38_EXTRA=--override-generation-config {"temperature":0.6}' \
    -E MAX_JOBS=2 -E FLASHINFER_NVCC_THREADS=1 $2 \
    bash /opt/llm/models/qwen38-27b-nvfp4/serve-qwen38.sh >/dev/null 2>&1
  ok=0; for i in $(seq 1 120); do
    [ "$(curl -sS -m 5 -o /dev/null -w '%{http_code}' http://127.0.0.1:8080/health 2>/dev/null)" = "200" ] && { ok=1; break; }
    systemctl is-active arm >/dev/null 2>&1 || break; sleep 10; done
  if [ $ok = 1 ]; then
    echo "  retention im Log: $(SUDO journalctl -u arm --no-pager -o cat --since '-20min' | grep -oE 'prefix_cache_retention_interval[=:] ?[0-9None]+' | tail -1)"
    LLM_KEY="$LLM_KEY" BLK=1648 OFF=1000000 python3 $S/blockprobe.py "$1" "$S/ret-$1.json" 2>&1 | tail -3
  else echo "  KAM NICHT HOCH"; SUDO journalctl -u arm --no-pager -o cat -n 6|tail -3|sed 's/^/    /'; fi
  SUDO systemctl stop arm 2>/dev/null; sleep 5; }
SUDO systemctl stop radixark-serve.service 2>/dev/null; sleep 5; echo "PROD GESTOPPT $(date +%H:%M:%S)"
arm "default-0"   ""
arm "retention-1648" "-E VLLM_PREFIX_CACHE_RETENTION_INTERVAL=1648"
echo "ALL DONE"
