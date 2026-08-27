#!/bin/bash
set -u
S="${OUT:?}"; M=/opt/llm/runtime/vllm-venv-maintest; SP=$M/lib/python3.12/site-packages/vllm
SUDO(){ printf '%s\n' "${SUDO_PW:?}" | sudo -S -p '' "$@" 2>/dev/null; }
arm(){ # $1 tag  $2 worktree  $3 extra
  echo "=== ARM $1 ($(date +%H:%M:%S)) ==="
  # 1) Wheel-Stand wiederherstellen (ohne --delete gehen sonst Dateien verloren, die
  #    nur im Wheel liegen -- das hat den ersten Anlauf zerlegt), 2) Baum darueberlegen
  SUDO rsync -a --exclude='*.so' --exclude='*.pyd' --exclude='_version.py' --exclude='__pycache__' \
       /opt/llm/runtime/vllm-venv-pr52816/lib/python3.12/site-packages/vllm/ $SP/ 2>/dev/null
  SUDO rsync -a --exclude='*.so' --exclude='*.pyd' --exclude='_version.py' --exclude='__pycache__' \
       /home/jschmied/wt/$2/vllm/ $SP/ 2>/dev/null
  SUDO cp $S/sched.$2.py $SP/v1/core/sched/scheduler.py
  SUDO systemd-run --unit=arm --collect \
    -E VLLM_QWEN38_VENV=$M -E VLLM_QWEN38_WEIGHTS=/opt/llm/models/qwen38-27b-radixark \
    -E VLLM_QWEN38_SPEC=mtp -E VLLM_QWEN38_NSPEC=3 -E VLLM_QWEN38_UTIL=0.76 \
    -E "VLLM_QWEN38_EXTRA=$3" -E MAX_JOBS=2 -E FLASHINFER_NVCC_THREADS=1 \
    bash /opt/llm/models/qwen38-27b-nvfp4/serve-qwen38.sh >/dev/null 2>&1
  ok=0; for i in $(seq 1 120); do
    [ "$(curl -sS -m 5 -o /dev/null -w '%{http_code}' http://127.0.0.1:8080/health 2>/dev/null)" = "200" ] && { ok=1; break; }
    systemctl is-active arm >/dev/null 2>&1 || break; sleep 10; done
  if [ $ok = 1 ]; then
    LLM_KEY="$LLM_KEY" BLK=1600 OFF=1000000 python3 $S/blockprobe.py "$1" "$S/probe-$1.json" 2>&1 | tail -3
    echo "  Beweis: $(SUDO journalctl -u arm --no-pager -o cat --since '-25min' | grep -oE "SCHEDPROBE\[[A-Z0-9]+\] start=0 [^\"]*" | sort -u | tail -1)"
  else echo "  KAM NICHT HOCH"; SUDO journalctl -u arm --no-pager -o cat -n 6|tail -3|sed 's/^/    /'; fi
  SUDO systemctl stop arm 2>/dev/null; sleep 5; }
SUDO systemctl stop radixark-serve.service; sleep 5; echo "PROD GESTOPPT $(date +%H:%M:%S)"
G='--override-generation-config {"temperature":0.6}'
arm "50897-base"  base50897 "$G"
arm "50897-pr"    pr50897   "$G"
arm "50897-pr-pmu" pr50897  "$G --prefix-match-unit 100"
arm "52244-base"  base52244 "$G"
arm "52244-pr"    pr52244   "$G"
SUDO systemd-run --unit=radixark-serve --collect \
  -E VLLM_QWEN38_VENV=/opt/llm/runtime/vllm-venv-pr52816 -E VLLM_QWEN38_WEIGHTS=/opt/llm/models/qwen38-27b-radixark \
  -E VLLM_QWEN38_SPEC=dflash -E VLLM_QWEN38_NSPEC=7 -E VLLM_QWEN38_DRAFT=/opt/llm/models/qwen38-27b-dflash2-syvai-w4a16 \
  -E VLLM_QWEN38_UTIL=0.76 -E "VLLM_QWEN38_EXTRA=$G" -E MAX_JOBS=2 -E FLASHINFER_NVCC_THREADS=1 \
  bash /opt/llm/models/qwen38-27b-nvfp4/serve-qwen38.sh >/dev/null 2>&1
for i in $(seq 1 90); do
  [ "$(curl -sS -m 5 -o /dev/null -w '%{http_code}' http://127.0.0.1:8080/health 2>/dev/null)" = "200" ] && { echo "PROD WIEDER OBEN $(date +%H:%M:%S)"; break; }; sleep 10; done
echo "ALL DONE"
