#!/bin/bash
# Acceptance ueber die Nebenlaeufigkeit -- prueft #53323 (SWA-Eviction) auf unserem Stack.
set -u
KEY="${LLM_KEY:?}"
URL=http://127.0.0.1:8080
OUT=/tmp/claude-1000/-home-jschmied-git-dgx-spark-setup-guide/5a12f402-ff7f-465d-bc7b-a750fd283fe5/scratchpad
REQ=24; OLEN=192; ILEN=2000

snap() { curl -sS -m 15 "$URL/metrics" -H "Authorization: Bearer $KEY" \
         | grep -E '^vllm:spec_decode_num_(drafts|draft_tokens|accepted_tokens)_total|^vllm:spec_decode_num_accepted_tokens_per_pos_total' \
         | sed 's/{[^}]*}//' ; }

echo "=== Acceptance vs. Nebenlaeufigkeit  (req=$REQ out=$OLEN in=$ILEN) ==="
for C in 1 2 4 8; do
  # eigener Vorlauf, damit der klientinterne Warmlauf nicht das ganze Delta faerbt
  python3 /opt/llm/bench_client_real.py --url "$URL/v1/chat/completions" --key "$KEY" \
      --model qwen38-27b --requests 2 --concurrency "$C" --input-len $ILEN \
      --output-len 64 --temp 0.6 >/dev/null 2>&1

  snap > "$OUT/m-before-$C.txt"
  python3 /opt/llm/bench_client_real.py --url "$URL/v1/chat/completions" --key "$KEY" \
      --model qwen38-27b --requests $REQ --concurrency "$C" --input-len $ILEN \
      --output-len $OLEN --temp 0.6 > "$OUT/bench-c$C.json" 2>"$OUT/bench-c$C.err"
  snap > "$OUT/m-after-$C.txt"
  echo "  c=$C fertig"
done
echo "ALL DONE"
