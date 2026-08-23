#!/bin/bash
# Acceptance ueber die Nebenlaeufigkeit -- prueft #53323 (SWA-Eviction) auf unserem Stack.
set -u
: "${OPENAI_API_KEY:?set OPENAI_API_KEY (the vLLM server key) before running}"
KEY="$OPENAI_API_KEY"
URL=http://127.0.0.1:8080
OUT="${OUT:-/tmp/accept-conc}"; mkdir -p "$OUT"
# Harness: the real-corpus concurrency client (random-word prompts collapse a small
# drafter, repeated-paragraph prompts flatter it -- both fake the result).
BENCH="${BENCH:-/opt/llm/bench_client_real.py}"
REQ=24; OLEN=192; ILEN=2000

snap() { curl -sS -m 15 "$URL/metrics" -H "Authorization: Bearer $KEY" \
         | grep -E '^vllm:spec_decode_num_(drafts|draft_tokens|accepted_tokens)_total|^vllm:spec_decode_num_accepted_tokens_per_pos_total' \
         | sed 's/{[^}]*}//' ; }

echo "=== Acceptance vs. Nebenlaeufigkeit  (req=$REQ out=$OLEN in=$ILEN) ==="
for C in 1 2 4 8; do
  # eigener Vorlauf, damit der klientinterne Warmlauf nicht das ganze Delta faerbt
  python3 "$BENCH" --url "$URL/v1/chat/completions" --key "$KEY" \
      --model qwen38-27b --requests 2 --concurrency "$C" --input-len $ILEN \
      --output-len 64 --temp 0.6 >/dev/null 2>&1

  snap > "$OUT/m-before-$C.txt"
  python3 "$BENCH" --url "$URL/v1/chat/completions" --key "$KEY" \
      --model qwen38-27b --requests $REQ --concurrency "$C" --input-len $ILEN \
      --output-len $OLEN --temp 0.6 > "$OUT/bench-c$C.json" 2>"$OUT/bench-c$C.err"
  snap > "$OUT/m-after-$C.txt"
  echo "  c=$C fertig"
done
echo "ALL DONE"
