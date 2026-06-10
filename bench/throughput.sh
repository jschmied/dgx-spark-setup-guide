#!/usr/bin/env bash
# Single-stream decode/prefill throughput via the native /completion endpoint.
# Method mirrors page 8 (8.9-8.12): a ~1800-token prompt, ignore_eos so each run
# generates exactly n_predict tokens, cache_prompt:false, warm-up run discarded,
# remaining runs averaged. Reports prefill t/s and decode t/s from server timings.
#
# Usage: ./throughput.sh <model-id> [n_predict] [runs]
set -uo pipefail
export LC_ALL=C   # force '.' decimal separator for printf/bc
cd "$(dirname "$0")"; source ./lib.sh

MODEL="${1:?usage: throughput.sh <model-id> [n_predict] [runs]}"
NPRED="${2:-512}"
RUNS="${3:-3}"
KEY="$(resolve_key)" || exit 1

# Build a ~1800-token prompt deterministically (same text every run).
PROMPT=$(python3 - <<'PY'
para=("Design a fault-tolerant distributed key-value store. Discuss replication, "
      "consensus, partitioning, failure detection, read/write quorums, conflict "
      "resolution, and the trade-offs between consistency and availability. ")
print(("".join(para))*40)
PY
)

run_once() {
  curl -s --max-time 600 "$HOST/completion" \
    -H "Content-Type: application/json" \
    --config <(printf 'header = "Authorization: Bearer %s"\n' "$KEY") \
    -d "$(jq -n --arg m "$MODEL" --arg p "$PROMPT" --argjson n "$NPRED" \
        '{model:$m, prompt:$p, n_predict:$n, ignore_eos:true, cache_prompt:false, temperature:0}')" \
    | python3 -c "import sys,json; t=json.load(sys.stdin).get('timings',{}); print(round(t.get('prompt_per_second',0),1), round(t.get('predicted_per_second',0),1), t.get('prompt_n'), t.get('predicted_n'))"
}

echo "########## $MODEL throughput (n_predict=$NPRED, $RUNS runs + warm-up) ##########"
echo "warm-up (discarded): $(run_once)"
psum=0; dsum=0
for i in $(seq 1 "$RUNS"); do
  read -r pf dec pn dn <<<"$(run_once)"
  printf '  run %s: prefill=%-7s t/s  decode=%-6s t/s  (prompt=%s gen=%s)\n' "$i" "$pf" "$dec" "$pn" "$dn"
  psum=$(echo "$psum + $pf" | bc -l); dsum=$(echo "$dsum + $dec" | bc -l)
done
printf '==> %s AVG :: prefill ~%.0f t/s  |  decode ~%.1f t/s\n' "$MODEL" \
  "$(echo "$psum/$RUNS" | bc -l)" "$(echo "$dsum/$RUNS" | bc -l)"
