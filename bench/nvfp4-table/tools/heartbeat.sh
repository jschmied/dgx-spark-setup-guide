#!/usr/bin/env bash
# Bedingungsloser Herzschlag alle 20 Minuten, solange Arbeit offen ist.
#
# Warum: die gefilterten Monitore melden nur, was ich erwartet habe. Stirbt ein
# Lauf anders -- Motor tot, Einheit weg, Port stumm -- bleibt es still, und
# Stille sieht aus wie "laeuft noch". Heute zweimal passiert. Dieser Melder
# sendet in JEDEM Fall eine Zeile; bleibt sie aus, ist der Melder selbst tot,
# und auch das ist eine Information.
#
# Endet von selbst, wenn dreimal hintereinander (also eine Stunde) nichts mehr
# offen ist -- dann meldet er das ausdruecklich und beendet sich.
set -uo pipefail
: "${OPENAI_API_KEY:?set OPENAI_API_KEY (the vLLM server key) before running}"

# Eine feste Liste waere ein Fehler: eine neu angelegte Einheit wuerde
# uebersehen, und uebersehen sieht aus wie "nichts los". Muster statt
# Aufzaehlung, damit spaetere Laeufe von selbst erfasst werden. Es deckt
# unsere Namenskonventionen ab, nicht die Systemdienste.
MUSTER='^(port027[a-z]?|p27|rc|kt|w8fix[0-9]?|radixark|restcells|fp32kontrolle|quanthead|df2|hum|dfl|sw|col|mtpc|prvf|ovn|ds8|anch|ceil|gaps|divg|probe|w8r|qh|hdiso|r0sp|dsab|mtpab|sidx|radixmess|rx|radixdauer[0-9]?|tctemp)$'
LEER=0

while true; do
  T=$(date -u +%H:%M)
  AKTIV=$(systemctl list-units --type=service --state=active --plain --no-legend 2>/dev/null \
           | awk '{print $1}' | sed 's/\.service$//' | grep -E "$MUSTER" | tr '\n' ' ')

  # Prozesse, die keine Einheit sind. Klammer-Muster, damit der Melder sich
  # nicht selbst zaehlt -- dieser Fehler ist mir heute schon zweimal passiert.
  VLLM=$(pgrep -cf "bin/vll[m]" 2>/dev/null); VLLM=${VLLM:-0}
  ARIA=$(pidof aria2c >/dev/null 2>&1 && echo ja || echo nein)

  API=$(curl -s -o /dev/null -w '%{http_code}' -m 5 \
        http://127.0.0.1:8080/v1/models -H "Authorization: Bearer $OPENAI_API_KEY" 2>/dev/null)
  API=${API:-000}
  MEM=$(awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo 2>/dev/null || echo "?")

  # letzte inhaltliche Zeile der aktiven Einheiten
  LETZTE=""
  for u in $AKTIV; do
    l=$(journalctl -u "$u" --no-pager -o cat -n 400 2>/dev/null \
        | grep -vE "pam_unix|COMMAND=|session (opened|closed)" \
        | grep -E "^\[[0-9]|ERGEBNIS|GESCHEITERT|ABBRUCH|geladen nach|SANITY" | tail -1)
    [ -n "$l" ] && LETZTE="$LETZTE | ${u}: ${l:0:90}"
  done

  # Produktion zaehlt nicht als offene Arbeit, wird aber im Zustand gezeigt.
  PROD=$(systemctl is-active vllm-qwen38 2>/dev/null); PROD=${PROD:-unbekannt}
  RDX=$(systemctl is-active radixark-serve 2>/dev/null); RDX=${RDX:-inactive}
  [ "$RDX" = "active" ] && PROD="RadixArk"
  if [ -z "$AKTIV" ] && [ "$ARIA" = "nein" ]; then
    LEER=$((LEER+1))
    echo "HERZSCHLAG $T  LEERLAUF ($LEER/3)  Produktion=$PROD  API=$API  frei=${MEM}GB"
    if [ $LEER -ge 3 ]; then
      echo "HERZSCHLAG $T  nichts mehr offen -- Melder beendet sich"
      exit 0
    fi
  else
    LEER=0
    echo "HERZSCHLAG $T  aktiv:${AKTIV:- keine}  aria2c=$ARIA  Produktion=$PROD  vLLM=$VLLM  API=$API  frei=${MEM}GB${LETZTE}"
  fi
  sleep 1200
done
