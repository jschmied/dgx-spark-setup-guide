#!/usr/bin/env bash
# Unconditional heartbeat every 20 minutes for as long as work is outstanding.
#
# Why: a filtered monitor only reports what you thought to filter for. When a run
# dies in some other way -- engine gone, unit vanished, port mute -- the filter
# stays quiet, and quiet looks exactly like "still running". That cost two
# undetected dead runs in one day. This reporter sends a line in EVERY case; if
# the line stops arriving, the reporter itself is dead, which is also news.
#
# It ends by itself once nothing has been outstanding three times in a row (one
# hour) -- and says so before exiting, rather than just falling silent.
set -uo pipefail
: "${OPENAI_API_KEY:?set OPENAI_API_KEY (the vLLM server key) before running}"

# A fixed list would be the wrong shape: a newly created unit would be missed,
# and missed looks like "nothing going on". A pattern picks up later runs on its
# own. It covers our own naming conventions, not the system services.
PATTERN='^(port027[a-z]?|p27|rc|kt|w8fix[0-9]?|radixark|restcells|fp32kontrolle|quanthead|df2|hum|dfl|sw|col|mtpc|prvf|ovn|ds8|anch|ceil|gaps|divg|probe|w8r|qh|hdiso|r0sp|dsab|mtpab|sidx|radixmess|rx|radixdauer[0-9]?|tctemp)$'
IDLE=0

while true; do
  T=$(date -u +%H:%M)
  ACTIVE=$(systemctl list-units --type=service --state=active --plain --no-legend 2>/dev/null \
           | awk '{print $1}' | sed 's/\.service$//' | grep -E "$PATTERN" | tr '\n' ' ')

  # Processes that are not units. Bracketed pattern so the reporter does not
  # count itself -- a mistake made twice in one day.
  VLLM=$(pgrep -cf "bin/vll[m]" 2>/dev/null); VLLM=${VLLM:-0}
  ARIA=$(pidof aria2c >/dev/null 2>&1 && echo yes || echo no)

  API=$(curl -s -o /dev/null -w '%{http_code}' -m 5 \
        http://127.0.0.1:8080/v1/models -H "Authorization: Bearer $OPENAI_API_KEY" 2>/dev/null)
  API=${API:-000}
  MEM=$(awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo 2>/dev/null || echo "?")

  # Last meaningful line from each active unit. These markers are the ones
  # measure-cell.sh prints -- keep the two in step if either is renamed.
  LAST=""
  for u in $ACTIVE; do
    l=$(journalctl -u "$u" --no-pager -o cat -n 400 2>/dev/null \
        | grep -vE "pam_unix|COMMAND=|session (opened|closed)" \
        | grep -E "^\[[0-9]|RESULT|FAILED|ABORT|loaded after|SANITY" | tail -1)
    [ -n "$l" ] && LAST="$LAST | ${u}: ${l:0:90}"
  done

  # Production does not count as outstanding work, but is shown in the state.
  PROD=$(systemctl is-active vllm-qwen38 2>/dev/null); PROD=${PROD:-unknown}
  RDX=$(systemctl is-active radixark-serve 2>/dev/null); RDX=${RDX:-inactive}
  [ "$RDX" = "active" ] && PROD="RadixArk"
  if [ -z "$ACTIVE" ] && [ "$ARIA" = "no" ]; then
    IDLE=$((IDLE+1))
    echo "HEARTBEAT $T  IDLE ($IDLE/3)  production=$PROD  API=$API  free=${MEM}GB"
    if [ $IDLE -ge 3 ]; then
      echo "HEARTBEAT $T  nothing outstanding -- reporter exiting"
      exit 0
    fi
  else
    IDLE=0
    echo "HEARTBEAT $T  active:${ACTIVE:- none}  aria2c=$ARIA  production=$PROD  vLLM=$VLLM  API=$API  free=${MEM}GB${LAST}"
  fi
  sleep 1200
done
