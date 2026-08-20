#!/usr/bin/env bash
# RadixArk/Qwen3.8-27B-NVFP4 durchmessen: Tempo unter MTP und unter DFlash2,
# dazu die Divergenz gegen BF16.
#
# Motorwahl folgt der Karte, nicht der Bequemlichkeit: MTP auf 0.27.1 wie alle
# anderen MTP-Werte, DFlash2 auf dem Zweig #52816 wie alle anderen DFlash2-Werte.
# Der Port auf 0.27.1 waere verlockend, liefert aber systematisch -0.24
# Akzeptanzlaenge (heute gemessen) und wuerde die Zelle unvergleichbar machen.
#
# Arm 1 und 2 teilen sich einen Server -- Divergenz braucht keinen eigenen Start.
#
# Der Kopf ist NVFP4 MIT input_scale, die Aktivierungen sind also mitquantisiert.
# Ungepatchtes vLLM verweigert ihn; hier laeuft er, weil der Quantkopf-Guard
# entfernt ist.
set -uo pipefail
: "${OPENAI_API_KEY:?set OPENAI_API_KEY (the vLLM server key) before running}"
OUT=/tmp/radix; mkdir -p $OUT
W=/opt/llm/models/qwen38-27b-radixark
V27=/opt/llm/runtime/vllm-venv-027
VPR=/opt/llm/runtime/vllm-venv-pr52816
log(){ echo "[$(date -u +%H:%M:%S)] $*"; }
# sudo password from the environment, never hard-coded (see bench/sweep.sh)
SUDO(){ printf '%s
' "${SUDO_PW:?set SUDO_PW}" | sudo -S -p '' "$@" 2>/dev/null; }

serve(){  # $1=Label $2=venv $3=SPEC $4=NSPEC $5=DRAFT $6=Cache
  for u in rx p27 hum df2 rc kt w8r qh ovn ds8 anch ceil gaps divg probe vllm-qwen38; do SUDO systemctl stop $u; done
  docker rm -f sgl >/dev/null 2>&1; sleep 10
  for j in $(seq 1 180); do
    a=$(pgrep -cf "bin/vll[m]"); a=${a:-0}
    m=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
    [ "$a" = "0" ] && [ "${m:-0}" -ge 90 ] && break
    [ $((j % 30)) = 0 ] && log "  [$1] warte auf Abschaltung: ${m}GB frei, $a vLLM (${j}0s)"
    [ $j = 180 ] && { log "  [$1] ABBRUCH Speicher (${m}GB, $a vLLM)"; return 1; }; sleep 10; done
  sync; SUDO sh -c 'echo 3 > /proc/sys/vm/drop_caches'
  T0=$(date +%s)
  SUDO systemd-run --unit=rx --collect \
    -E VLLM_QWEN38_VENV=$2 -E VLLM_QWEN38_WEIGHTS=$W \
    -E VLLM_QWEN38_SPEC="$3" -E VLLM_QWEN38_NSPEC="$4" -E VLLM_QWEN38_DRAFT="$5" \
    -E VLLM_QWEN38_UTIL=0.66 -E VLLM_QWEN38_MAXSEQS=4 -E VLLM_QWEN38_MAXLEN=32768 \
    -E XDG_CACHE_HOME=$6 -E TRITON_CACHE_DIR=$6/triton \
    -E VLLM_CACHE_ROOT=$6/vllm -E TORCHINDUCTOR_CACHE_DIR=$6/torchinductor \
    -E FLASHINFER_WORKSPACE_BASE=$6/flashinfer \
    -E MAX_JOBS=2 -E FLASHINFER_NVCC_THREADS=1 -E VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800 \
    bash /opt/llm/models/qwen38-27b-nvfp4/serve-qwen38.sh >/dev/null 2>&1
  for j in $(seq 1 180); do
    if [ "$(curl -s -o /dev/null -w '%{http_code}' -m 5 http://127.0.0.1:8080/v1/models \
        -H "Authorization: Bearer $OPENAI_API_KEY")" = "200" ]; then
      VER=$(journalctl -u rx --no-pager --since "@$T0" -o cat | grep -oE "V1 LLM engine \(v[^)]*\)" | head -1)
      MET=$(journalctl -u rx --no-pager --since "@$T0" -o cat | grep -oE "'method': '[a-z0-9]+'" | head -1)
      log "  [$1] geladen nach $((j*10))s  $VER $MET"
      # Erwartete Methode pruefen -- ein stiller Rueckfall auf MTP saehe nur
      # nach schlechtem DFlash2 aus.
      if [ "$3" != "off" ]; then
        echo "$MET" | grep -q "'$3'" || { log "  [$1] ABBRUCH: Methode ist $MET, erwartet $3"; return 1; }
      fi
      $V27/bin/python /opt/llm/sanity_gate.py 2>&1 | sed 's/^/    /' || { log "  [$1] SANITY"; return 1; }
      return 0; fi
    systemctl is-active --quiet rx || { log "  [$1] GESCHEITERT:"
      journalctl -u rx --no-pager --since "@$T0" | grep -oE "(ValueError|RuntimeError|AssertionError|NotImplementedError|TypeError|AttributeError|KeyError):.*" | sort -u | head -4 | sed 's/^/      /'
      return 1; }
    [ $j = 180 ] && { log "  [$1] Timeout"; return 1; }; sleep 10; done
}

bench(){  # $1=Label
  python3 /opt/llm/replay_bench.py --label $1 --out $OUT/$1.json 2>&1 | tail -2 | sed 's/^/    /'
  curl -s http://127.0.0.1:8080/metrics -H "Authorization: Bearer $OPENAI_API_KEY" \
    | grep -E "^vllm:spec_decode_num_(drafts|draft_tokens|accepted_tokens)_total" > $OUT/$1.metrics
  $V27/bin/python - "$OUT/$1.json" "$OUT/$1.metrics" "$1" <<'PY' | sed 's/^/  >> /'
import json, statistics, re, sys
r=[x for x in json.load(open(sys.argv[1])) if "error" not in x]
t={}
for l in open(sys.argv[2]):
    m=re.match(r'(vllm:\S+?)\{.*?\}\s+([\d.eE+]+)$', l.strip())
    if m: t[m.group(1)]=float(m.group(2))
d=t.get("vllm:spec_decode_num_drafts_total",0); a=t.get("vllm:spec_decode_num_accepted_tokens_total",0)
v=statistics.median(x["tok_s"] for x in r); tt=statistics.median(x["ttft"] for x in r)
L=f"  Laenge {a/d+1:.2f}" if d else ""
print(f"ERGEBNIS {sys.argv[3]}: {v:.1f} tok/s  TTFT {tt:.1f}s ({len(r)}){L}  "
      f"Zug@130 {tt+130/v:.1f}s")
PY
}

log "=== 1/3  MTP n=3 auf 0.27.1 (wie alle MTP-Werte der Karte)"
serve radix-mtp $V27 mtp 3 "" /opt/llm/.cache-027 && {
  bench radix-mtp
  log "=== 2/3  Divergenz gegen BF16, selber Server"
  MN=$(curl -s http://127.0.0.1:8080/v1/models -H "Authorization: Bearer $OPENAI_API_KEY" \
       | python3 -c "import sys,json;print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null || echo qwen38-27b)
  log "  Modellname laut Server: $MN"
  python3 /tmp/divergence.py --mode score --label radixark --model "$MN" \
    --ref /tmp/divergence/ref.json --out /tmp/divergence/radixark.json 2>&1 | tail -3 | sed 's/^/    /'
}

log "=== 3/3  DFlash2 n=7 auf dem Zweig (wie alle DFlash2-Werte der Karte)"
serve radix-dflash2 $VPR dflash 7 /opt/llm/models/qwen38-27b-dflash2 /opt/llm/.cache-pr \
  && bench radix-dflash2

SUDO systemctl stop rx
log "=== ERGEBNIS ==="
$V27/bin/python - <<'PY'
import json, statistics, re, os
def w(n):
    p=f"/tmp/radix/{n}.json"
    if not os.path.exists(p): return None
    r=[x for x in json.load(open(p)) if "error" not in x]
    if not r: return None
    t={}
    mp=f"/tmp/radix/{n}.metrics"
    if os.path.exists(mp):
        for l in open(mp):
            m=re.match(r'(vllm:\S+?)\{.*?\}\s+([\d.eE+]+)$', l.strip())
            if m: t[m.group(1)]=float(m.group(2))
    d=t.get("vllm:spec_decode_num_drafts_total",0); a=t.get("vllm:spec_decode_num_accepted_tokens_total",0)
    return (statistics.median(x["tok_s"] for x in r), statistics.median(x["ttft"] for x in r),
            a/d+1 if d else None)
for n,lbl in (("radix-mtp","MTP n=3"),("radix-dflash2","DFlash2 n=7")):
    v=w(n)
    print(f"  {lbl:<14} " + ("— nicht gelaufen" if not v else
          f"{v[0]:5.1f} tok/s  TTFT {v[1]:4.1f}s  Laenge {v[2]:.2f}  Zug@130 {v[1]+130/v[0]:.1f}s"
          if v[2] else f"{v[0]:5.1f} tok/s  TTFT {v[1]:4.1f}s  Zug@130 {v[1]+130/v[0]:.1f}s"))
p="/tmp/divergence/radixark.json"
if os.path.exists(p):
    b={x["id"]:x for x in json.load(open("/tmp/divergence/bf16.json"))}
    d={x["id"]:x for x in json.load(open(p))}
    ids=sorted(set(b)&set(d))
    if ids:
        print(f"  Divergenz      dtop1 {statistics.fmean((d[i]['top1']-b[i]['top1'])*100 for i in ids):+.2f} pp "
              f"ueber {len(ids)} Kontexte")
print("\n  Nachbarn: uns-nvfp4head 38.6/4.26 | ours-nvfp4head 39.1/4.06 | r0b0tlab 40.6/4.23")
PY
log "FERTIG"
