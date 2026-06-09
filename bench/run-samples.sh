#!/usr/bin/env bash
# Multi-sample runner: run a model N times per task and tally pass-rates.
# Single samples at the recommended (higher) temps are noisy; this reports the
# spread (e.g. "neutral 2/3") that page 13 grades on.
# Usage: ./run-samples.sh <model-id> <N> [go|java|both]
set -uo pipefail
cd "$(dirname "$0")"; source ./lib.sh

MODEL="${1:?usage: run-samples.sh <model-id> <N> [go|java|both]}"
N="${2:?N}"
WHICH="${3:-both}"
mkdir -p results .work
export GOFLAGS=-mod=mod GOCACHE="${GOCACHE:-/tmp/gocache}" GOPATH="${GOPATH:-/tmp/gopath}"
export JAVA_HOME="${JAVA_HOME:-$(ls -d /usr/lib/jvm/java-17-openjdk-* 2>/dev/null | head -1)}"

# content-based Go block selection: cache_test.go = block with 'func Test',
# cache.go = the block with 'func New'/'type Cache' that is not the test.
extract_go() {
  local md="$1" srcdir="$2"; rm -rf "$srcdir"; mkdir -p "$srcdir"
  local nblocks; nblocks=$(grep -cE '^```' "$md"); nblocks=$((nblocks/2))
  local impl="" test=""
  for b in $(seq 1 "$nblocks"); do
    awk -v n="$b" '/^```/{f=!f; if(f)c++; next} f&&c==n' "$md" > "/tmp/blk_$$_$b.go"
    if grep -qE '^\s*func Test' "/tmp/blk_$$_$b.go"; then test="/tmp/blk_$$_$b.go"
    elif grep -qE 'func New|type Cache' "/tmp/blk_$$_$b.go"; then impl="/tmp/blk_$$_$b.go"; fi
  done
  [ -z "$impl" ] && impl="/tmp/blk_$$_1.go"
  [ -z "$test" ] && test="/tmp/blk_$$_2.go"
  cp "$impl" "$srcdir/cache.go"; cp "$test" "$srcdir/cache_test.go"
  printf 'module cachetest\n\ngo 1.22\n' > "$srcdir/go.mod"
  rm -f /tmp/blk_$$_*.go
  echo "$nblocks"
}

go_sample() {
  local i="$1" out="results/go_${MODEL}_s${i}.json" md=".work/go_${MODEL}_s${i}.md"
  local src=".work/go_${MODEL}_s${i}" neu=".work/go_${MODEL}_s${i}_n"
  call_model "$MODEL" "$BENCH_DIR/prompts/cache.txt" 26000 "$out" >/dev/null 2>&1
  jq -r '.choices[0].message.content' "$out" > "$md"
  local blocks; blocks=$(extract_go "$md" "$src")
  local vet own neur err=""
  ( cd "$src" && go vet ./... ) >/dev/null 2>&1 && vet=ok || vet=FAIL
  ( cd "$src" && go test -race -count=1 -timeout 120s ./... ) >/dev/null 2>&1 && own=ok || own=FAIL
  rm -rf "$neu"; mkdir -p "$neu"; printf 'module cachetest\n\ngo 1.22\n' > "$neu/go.mod"
  cp "$src/cache.go" "$neu/"; cp "$BENCH_DIR/neutral/cache_neutral_test.go" "$neu/"
  if ( cd "$neu" && go test -race -count=1 -timeout 60s ./... ) >/dev/null 2>&1; then neur=PASS; else
    neur=FAIL; err=$( cd "$neu" && go test -race -count=1 -timeout 60s ./... 2>&1 | grep -m1 -E 'undefined|FAIL|panic|cannot|imported and not used|\.go:' )
  fi
  local tok fin; tok=$(jq -r '.usage.completion_tokens' "$out"); fin=$(jq -r '.choices[0].finish_reason' "$out")
  printf '  Go  s%s: vet=%-4s own=%-4s neutral=%-4s  (blocks=%s tok=%s fin=%s) %s\n' "$i" "$vet" "$own" "$neur" "$blocks" "$tok" "$fin" "${err:+:: $err}"
  [ "$neur" = PASS ] && echo PASS >> "/tmp/tally_go_$$"
}

java_sample() {
  local i="$1" out="results/java_${MODEL}_s${i}.json"
  local proj=".work/java_${MODEL}_s${i}" neu=".work/java_${MODEL}_s${i}_n"
  call_model "$MODEL" "$BENCH_DIR/prompts/transfer.txt" 24000 "$out" >/dev/null 2>&1
  rm -rf "$proj"; mkdir -p "$proj"
  python3 "$BENCH_DIR/extract_java.py" "$out" "$proj" >/dev/null 2>&1
  local own neur err=""
  ( cd "$proj" && mvn -B -q test ) >/dev/null 2>&1 && own=ok || own=FAIL
  rm -rf "$neu"; mkdir -p "$neu/src/test/java/com/example/bank"
  cp "$proj/pom.xml" "$neu/pom.xml" 2>/dev/null
  cp -r "$proj/src/main" "$neu/src/main" 2>/dev/null
  cp "$BENCH_DIR/neutral/NeutralTransferTest.java" "$neu/src/test/java/com/example/bank/NeutralTransferTest.java"
  if ( cd "$neu" && mvn -B -q test ) >/dev/null 2>&1; then neur=PASS; else
    neur=FAIL; err=$( cd "$neu" && mvn -B -q test 2>&1 | grep -m1 -E 'ERROR.*\.java|BUILD FAILURE|cannot find|does not exist' )
  fi
  local tok fin; tok=$(jq -r '.usage.completion_tokens' "$out"); fin=$(jq -r '.choices[0].finish_reason' "$out")
  printf '  Jav s%s: own=%-4s neutral=%-4s  (tok=%s fin=%s) %s\n' "$i" "$own" "$neur" "$tok" "$fin" "${err:+:: $err}"
  [ "$neur" = PASS ] && echo PASS >> "/tmp/tally_java_$$"
}

echo "########## $MODEL  (N=$N, $WHICH) ##########"
rm -f "/tmp/tally_go_$$" "/tmp/tally_java_$$"
for i in $(seq 1 "$N"); do
  [[ "$WHICH" == both || "$WHICH" == go   ]] && go_sample "$i"
  [[ "$WHICH" == both || "$WHICH" == java ]] && java_sample "$i"
done
gp=$( [ -f "/tmp/tally_go_$$" ] && wc -l < "/tmp/tally_go_$$" || echo 0 )
jp=$( [ -f "/tmp/tally_java_$$" ] && wc -l < "/tmp/tally_java_$$" || echo 0 )
echo "==> $MODEL TALLY :: Go neutral $gp/$N  |  Java neutral $jp/$N"
rm -f "/tmp/tally_go_$$" "/tmp/tally_java_$$"
