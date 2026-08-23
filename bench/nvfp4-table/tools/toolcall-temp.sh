#!/usr/bin/env bash
# Does temperature 1.0 break tool-calling ON THIS MODEL?
#
# The figure quoted so far (1/8 at 1.0 against 7/8 at 0.2) comes from a different
# checkpoint in the fleet. It was never checked on RadixArk, and the server
# default was derived from it anyway. This closes that gap.
#
# Three temperatures, 8 attempts each, same prompt, same server. What is counted
# is whether a tool_call comes back -- not whether it is a clever one.
set -uo pipefail
: "${OPENAI_API_KEY:?set OPENAI_API_KEY (the vLLM server key) before running}"
log(){ echo "[$(date -u +%H:%M:%S)] $*"; }

while ! systemctl is-active --quiet radixark-serve; do sleep 20; done
for i in $(seq 1 120); do
  [ "$(curl -s -o /dev/null -w '%{http_code}' -m 5 http://127.0.0.1:8080/v1/models \
      -H "Authorization: Bearer $OPENAI_API_KEY")" = "200" ] && break
  [ $i = 120 ] && { log "server did not come up"; exit 1; }
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
res = {}
for temp in (0.2, 0.6, 1.0, None):
    lbl = "server default" if temp is None else f"{temp}"
    treffer = 0; gruende = collections.Counter()
    for _ in range(N):
        ok, fr = einmal(temp)
        if ok == "ja": treffer += 1
        gruende[fr] += 1
    res[lbl] = treffer
    top = ", ".join(f"{k}×{v}" for k, v in gruende.most_common(2))
    print(f"  {lbl:<14}{f'{treffer}/{N}':>16}{top:>34}")

print()
if res.get("1.0", 0) >= N - 1 and res.get("0.2", 0) >= N - 1:
    print("  -> temperature 1.0 does NOT break tool-calling on THIS model.")
    print("     so the 0.6 server default is not justified by tool-calling.")
elif res.get("1.0", 0) < res.get("0.2", 0):
    print(f"  -> 1.0 is worse than 0.2 ({res.get('1.0')} against {res.get('0.2')} of {N}).")
    print("     a default below 1.0 is justified.")
else:
    print("  -> kein klares Bild bei n=8; groesseres n noetig, bevor daraus etwas folgt.")
PY
log "FERTIG"
