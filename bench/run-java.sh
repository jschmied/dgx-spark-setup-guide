#!/usr/bin/env bash
# Task B — Spring Boot / Hibernate / Mockito / JUnit 5 (pages 12.8).
# Usage: ./run-java.sh <model-id> [max_tokens]
# Runs at the model's recommended sampling (see lib.sh), extracts the Maven
# project, then: mvn test (the model's OWN suite) and the NEUTRAL Mockito suite
# against the model's production code.
set -uo pipefail
cd "$(dirname "$0")"; source ./lib.sh

MODEL="${1:?usage: run-java.sh <model-id> [max_tokens]}"
MAXTOK="${2:-24000}"   # generous headroom for reasoning models' thinking pass
mkdir -p "$RESULTS_DIR" "$WORK_DIR"
OUT="$RESULTS_DIR/java_${MODEL}.json"
PROJ="$WORK_DIR/java_${MODEL}"
NEU="$WORK_DIR/java_${MODEL}_neutral"

export JAVA_HOME="${JAVA_HOME:-$(ls -d /usr/lib/jvm/java-17-openjdk-* 2>/dev/null | head -1)}"

echo "=== Task B (Java) :: $MODEL  (JAVA_HOME=$JAVA_HOME) ==="
call_model "$MODEL" "$BENCH_DIR/prompts/transfer.txt" "$MAXTOK" "$OUT" || exit 1

rm -rf "$PROJ"; mkdir -p "$PROJ"
python3 "$BENCH_DIR/extract_java.py" "$OUT" "$PROJ"
echo "--- files ---"; find "$PROJ" -type f | sort

echo "--- mvn test (own suite) ---"
( cd "$PROJ" && mvn -B -q test ); OWN=$?
[ -d "$PROJ/target/surefire-reports" ] && grep -H "Tests run" "$PROJ"/target/surefire-reports/*.txt 2>/dev/null

# Neutral suite: model's production code + our Mockito test.
rm -rf "$NEU"; mkdir -p "$NEU/src/test/java/com/example/bank"
cp "$PROJ/pom.xml" "$NEU/pom.xml" 2>/dev/null
cp -r "$PROJ/src/main" "$NEU/src/main" 2>/dev/null
cp "$BENCH_DIR/neutral/NeutralTransferTest.java" "$NEU/src/test/java/com/example/bank/NeutralTransferTest.java"
echo "--- mvn test (NEUTRAL suite) ---"
( cd "$NEU" && mvn -B -q test ); NEUR=$?
[ -d "$NEU/target/surefire-reports" ] && grep -H "Tests run" "$NEU"/target/surefire-reports/*.txt 2>/dev/null

echo "=== VERDICT $MODEL (Java) :: own=$([ $OWN -eq 0 ] && echo ok || echo FAIL) neutral=$([ $NEUR -eq 0 ] && echo PASS || echo FAIL) ==="
