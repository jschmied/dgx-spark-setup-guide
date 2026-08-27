#!/bin/bash
set -u
S="${OUT:?}"; M=/opt/llm/runtime/vllm-venv-maintest; SP=$M/lib/python3.12/site-packages/vllm
PP=/opt/llm/runtime/vllm-venv-pr52816/lib/python3.12/site-packages/vllm
SUDO(){ printf '%s\n' "${SUDO_PW:?}" | sudo -S -p '' "$@" 2>/dev/null; }
arm(){
  echo "=== ARM $1 ($(date +%H:%M:%S)) ==="
  SUDO rsync -a --exclude='*.so' --exclude='*.pyd' --exclude='_version.py' --exclude='__pycache__' $PP/ $SP/ 2>/dev/null
  SUDO cp "$2" $SP/model_executor/models/qwen3_dflash.py
  SUDO systemd-run --unit=arm --collect \
    -E VLLM_QWEN38_VENV=$M -E VLLM_QWEN38_WEIGHTS=/opt/llm/models/qwen38-27b-radixark \
    -E VLLM_QWEN38_SPEC=dflash -E VLLM_QWEN38_NSPEC=7 \
    -E VLLM_QWEN38_DRAFT=/opt/llm/models/qwen38-27b-dflash2-syvai-w4a16 \
    -E VLLM_QWEN38_UTIL=0.76 -E 'VLLM_QWEN38_EXTRA=--override-generation-config {"temperature":0.6}' \
    -E MAX_JOBS=2 -E FLASHINFER_NVCC_THREADS=1 \
    bash /opt/llm/models/qwen38-27b-nvfp4/serve-qwen38.sh >/dev/null 2>&1
  ok=0; for i in $(seq 1 120); do
    [ "$(curl -sS -m 5 -o /dev/null -w '%{http_code}' http://127.0.0.1:8080/health 2>/dev/null)" = "200" ] && { ok=1; break; }
    systemctl is-active arm >/dev/null 2>&1 || break; sleep 10; done
  if [ $ok = 1 ]; then
    T0=$(date --iso-8601=seconds)
    LLM_KEY="$LLM_KEY" python3 $S/ttftprobe.py "$1" "$S/ttft-$1.json" 2>&1 | tail -4
    echo "  Beweis: $(SUDO journalctl -u arm --no-pager -o cat --since "$T0" | grep -oE 'KVPROBE\[[A-Z]+\][^"]*' | sort -u | head -2 | tr '\n' ' ')"
  else echo "  KAM NICHT HOCH"; SUDO journalctl -u arm --no-pager -o cat -n 6|tail -3|sed 's/^/    /'; fi
  SUDO systemctl stop arm 2>/dev/null; sleep 5; }
SUDO systemctl stop radixark-serve.service; sleep 5; echo "PROD GESTOPPT $(date +%H:%M:%S)"
arm "FUSED"    $S/dflash.A_fused.py
arm "PERLAYER" $S/dflash.B_perlayer.py
SUDO systemd-run --unit=radixark-serve --collect \
  -E VLLM_QWEN38_VENV=/opt/llm/runtime/vllm-venv-pr52816 -E VLLM_QWEN38_WEIGHTS=/opt/llm/models/qwen38-27b-radixark \
  -E VLLM_QWEN38_SPEC=dflash -E VLLM_QWEN38_NSPEC=7 -E VLLM_QWEN38_DRAFT=/opt/llm/models/qwen38-27b-dflash2-syvai-w4a16 \
  -E VLLM_QWEN38_UTIL=0.76 -E 'VLLM_QWEN38_EXTRA=--override-generation-config {"temperature":0.6}' \
  -E MAX_JOBS=2 -E FLASHINFER_NVCC_THREADS=1 \
  bash /opt/llm/models/qwen38-27b-nvfp4/serve-qwen38.sh >/dev/null 2>&1
for i in $(seq 1 90); do
  [ "$(curl -sS -m 5 -o /dev/null -w '%{http_code}' http://127.0.0.1:8080/health 2>/dev/null)" = "200" ] && { echo "PROD WIEDER OBEN $(date +%H:%M:%S)"; break; }; sleep 10; done
echo "ALL DONE"
