#!/usr/bin/env bash
# Bricht Temperatur 1.0 das Tool-Calling AUF DIESEM Modell?
#
# Die Zahl, auf die ich mich bisher berufen habe (1/8 bei 1.0 gegen 7/8 bei 0.2),
# stammt von einem anderen Checkpoint der Fleet. Fuer RadixArk war sie ungeprueft,
# und ich habe daraus trotzdem die Servervorgabe abgeleitet. Hier nachgeholt.
#
# Drei Temperaturen, je 8 Versuche, gleicher Prompt, gleicher Server. Gezaehlt
# wird, ob ein tool_call zurueckkommt -- nicht ob er inhaltlich klug ist.
set -uo pipefail
: "${OPENAI_API_KEY:?set OPENAI_API_KEY (the vLLM server key) before running}"
log(){ echo "[$(date -u +%H:%M:%S)] $*"; }

while ! systemctl is-active --quiet radixark-serve; do sleep 20; done
for i in $(seq 1 120); do
  [ "$(curl -s -o /dev/null -w '%{http_code}' -m 5 http://127.0.0.1:8080/v1/models \
      -H "Authorization: Bearer $OPENAI_API_KEY")" = "200" ] && break
  [ $i = 120 ] && { log "Server kommt nicht hoch"; exit 1; }
  sleep 10
done
log "Server bereit"

python3 - <<'PY'
import json, urllib.request, os, collections

KEY = os.environ["OPENAI_API_KEY"]
URL = "http://127.0.0.1:8080/v1/chat/completions"
TOOLS = [{"type": "function", "function": {
    "name": "run", "description": "run a shell command",
    "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}},
                   "required": ["cmd"]}}}]
FRAGE = "List the files in /tmp using the tool."
N = 8

def einmal(temp):
    body = {"model": "qwen38-27b", "max_tokens": 200,
            "messages": [{"role": "user", "content": FRAGE}], "tools": TOOLS}
    if temp is not None:
        body["temperature"] = temp
    r = urllib.request.Request(URL, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
    try:
        d = json.load(urllib.request.urlopen(r, timeout=300))
    except Exception as e:
        return ("Fehler", str(e)[:40])
    ch = d["choices"][0]
    tc = ch["message"].get("tool_calls")
    return ("ja" if tc else "nein", ch.get("finish_reason"))

print(f"  {'Temperatur':<14}{'Werkzeugaufruf':>16}{'finish_reason':>34}")
erg = {}
for temp in (0.2, 0.6, 1.0, None):
    lbl = "Servervorgabe" if temp is None else f"{temp}"
    treffer = 0; gruende = collections.Counter()
    for _ in range(N):
        ok, fr = einmal(temp)
        if ok == "ja": treffer += 1
        gruende[fr] += 1
    erg[lbl] = treffer
    top = ", ".join(f"{k}×{v}" for k, v in gruende.most_common(2))
    print(f"  {lbl:<14}{f'{treffer}/{N}':>16}{top:>34}")

print()
if erg.get("1.0", 0) >= N - 1 and erg.get("0.2", 0) >= N - 1:
    print("  -> Temperatur 1.0 bricht das Tool-Calling auf DIESEM Modell NICHT.")
    print("     Die Servervorgabe 0.6 ist damit nicht durch Tool-Calling begruendet.")
elif erg.get("1.0", 0) < erg.get("0.2", 0):
    print(f"  -> 1.0 ist schlechter als 0.2 ({erg.get('1.0')} gegen {erg.get('0.2')} von {N}).")
    print("     Die Vorgabe unter 1.0 ist begruendet.")
else:
    print("  -> kein klares Bild bei n=8; groesseres n noetig, bevor daraus etwas folgt.")
PY
log "FERTIG"
