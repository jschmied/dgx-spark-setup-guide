#!/usr/bin/env bash
# Task A — Go concurrency cache (pages 12.1-12.7).
# Usage: ./run-go.sh <model-id> [max_tokens]
# Runs at the model's recommended sampling (see lib.sh), extracts cache.go +
# cache_test.go, then: go vet, the model's OWN suite, and the NEUTRAL suite.
set -uo pipefail
cd "$(dirname "$0")"; source ./lib.sh

MODEL="${1:?usage: run-go.sh <model-id> [max_tokens]}"
MAXTOK="${2:-26000}"   # generous: dense reasoning models (ornstein-27B) emit ~19k tok at temp 1.0
mkdir -p "$RESULTS_DIR" "$WORK_DIR"
OUT="$RESULTS_DIR/go_${MODEL}.json"
MD="$WORK_DIR/go_${MODEL}.md"
SRC="$WORK_DIR/go_${MODEL}"
NEU="$WORK_DIR/go_${MODEL}_neutral"

export GOFLAGS=-mod=mod GOCACHE="${GOCACHE:-/tmp/gocache}" GOPATH="${GOPATH:-/tmp/gopath}"

echo "=== Task A (Go) :: $MODEL ==="
call_model "$MODEL" "$BENCH_DIR/prompts/cache.txt" "$MAXTOK" "$OUT" || exit 1

jq -r '.choices[0].message.content' "$OUT" > "$MD"
block() { awk -v n="$1" '/^```/{f=!f; if(f)c++; next} f&&c==n' "$MD"; }
rm -rf "$SRC"; mkdir -p "$SRC"
block 1 > "$SRC/cache.go"
block 2 > "$SRC/cache_test.go"
printf 'module cachetest\n\ngo 1.22\n' > "$SRC/go.mod"

echo "--- extracted ---"
grep -m1 '^package' "$SRC"/cache.go "$SRC"/cache_test.go
echo "stray fences (want 0 0):"; grep -c '```' "$SRC"/cache.go "$SRC"/cache_test.go

echo "--- go vet (own) ---";              ( cd "$SRC" && go vet ./... );                         VET=$?
echo "--- go test -race (own suite) ---"; ( cd "$SRC" && go test -race -count=1 -timeout 120s ./... ); OWN=$?

rm -rf "$NEU"; mkdir -p "$NEU"
printf 'module cachetest\n\ngo 1.22\n' > "$NEU/go.mod"
cp "$SRC/cache.go" "$NEU/"; cp "$BENCH_DIR/neutral/cache_neutral_test.go" "$NEU/"
echo "--- go test -race (NEUTRAL suite) ---"
( cd "$NEU" && go test -race -count=1 -timeout 60s -v ./... ); NEUR=$?

echo "=== VERDICT $MODEL (Go) :: vet=$([ $VET -eq 0 ] && echo ok || echo FAIL) own=$([ $OWN -eq 0 ] && echo ok || echo FAIL) neutral=$([ $NEUR -eq 0 ] && echo PASS || echo FAIL) ==="
