#!/usr/bin/env bash
# Single-shot sampling sweep for Task B (Java/Spring). One shot per config,
# varying temp / top_k / min_p / top_p, to find sampling that yields a clean
# build. Single shots are noisy by design — read it as a scan, not a ranking.
# Usage: ./sweep-java.sh <model-id>
set -uo pipefail
cd "$(dirname "$0")"; source ./lib.sh
MODEL="${1:?usage: sweep-java.sh <model-id>}"
mkdir -p "$RESULTS_DIR" .work
export JAVA_HOME="${JAVA_HOME:-$(ls -d /usr/lib/jvm/java-17-openjdk-* 2>/dev/null | head -1)}"

# config = "temp top_p top_k min_p"   (top_p 1.0 = nucleus off; min_p 0 = omit)
configs=(
  "0.2 1.0  20 0.1"
  "0.3 1.0  40 0.05"
  "0.4 1.0  20 0.1"
  "0.5 1.0  40 0.1"
  "0.6 0.95 20 0"      # deployed top-p baseline
  "0.6 1.0  20 0.1"
  "0.6 1.0  40 0.05"
  "0.8 1.0  40 0.1"
  "1.0 1.0  20 0.1"
  "1.0 1.0  64 0.05"
)

echo "########## $MODEL  Java sweep (${#configs[@]} single shots) ##########"
i=0
for cfg in "${configs[@]}"; do
  i=$((i+1)); read -r t tp tk mp <<<"$cfg"
  out="$RESULTS_DIR/jsweep_${MODEL}_$i.json"
  proj=".work/jsweep_${MODEL}_$i"; neu="${proj}_n"
  ( export BENCH_FORCE_TEMP=$t BENCH_FORCE_TOPP=$tp BENCH_FORCE_TOPK=$tk BENCH_FORCE_MINP=$mp
    call_model "$MODEL" "$BENCH_DIR/prompts/transfer.txt" 24000 "$out" ) >/dev/null 2>&1
  rm -rf "$proj"; mkdir -p "$proj"
  python3 "$BENCH_DIR/extract_java.py" "$out" "$proj" >/dev/null 2>&1
  ( cd "$proj" && mvn -B -q test ) >/dev/null 2>&1 && own=ok || own=FAIL
  rm -rf "$neu"; mkdir -p "$neu/src/test/java/com/example/bank"
  cp "$proj/pom.xml" "$neu/pom.xml" 2>/dev/null
  cp -r "$proj/src/main" "$neu/src/main" 2>/dev/null
  cp "$BENCH_DIR/neutral/NeutralTransferTest.java" "$neu/src/test/java/com/example/bank/NeutralTransferTest.java"
  err=""
  if ( cd "$neu" && mvn -B -q test ) >/dev/null 2>&1; then neur=PASS; else
    neur=FAIL; err=$( cd "$neu" && mvn -B -q test 2>&1 | grep -m1 -E 'ERROR.*\.java|cannot find|no suitable|does not exist|Tests run.*Failures' | sed 's#.*/##' )
  fi
  tok=$(jq -r '.usage.completion_tokens // "?"' "$out" 2>/dev/null)
  mps=$([ "$mp" = 0 ] && echo "topp=$tp" || echo "minp=$mp")
  printf 'shot %2d  temp=%-3s topk=%-2s %-9s :: own=%-4s neutral=%-4s (tok=%s) %s\n' \
    "$i" "$t" "$tk" "$mps" "$own" "$neur" "$tok" "${err:+:: $err}"
done
echo "########## sweep done ##########"
