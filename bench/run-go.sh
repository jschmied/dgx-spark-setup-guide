#!/usr/bin/env bash
# Task A — Go concurrency cache (pages 12.1-12.7).
# Usage: ./run-go.sh <model-id> [max_tokens]
# Runs at the model's recommended sampling (see lib.sh), extracts cache.go +
# cache_test.go, then: go vet, the model's OWN suite, and the NEUTRAL suite.
set -uo pipefail
cd "$(dirname "$0")"; source ./lib.sh

MODEL="${1:?usage: run-go.sh <model-id> [max_tokens]}"
MAXTOK="${2:-26000}"   # generous: reasoning models emit many thousands of tokens before code
mkdir -p "$RESULTS_DIR" "$WORK_DIR"
OUT="$RESULTS_DIR/go_${MODEL}.json"
MD="$WORK_DIR/go_${MODEL}.md"
SRC="$WORK_DIR/go_${MODEL}"
NEU="$WORK_DIR/go_${MODEL}_neutral"

export GOFLAGS=-mod=mod GOCACHE="${GOCACHE:-/tmp/gocache}" GOPATH="${GOPATH:-/tmp/gopath}"

echo "=== Task A (Go) :: $MODEL ==="
call_model "$MODEL" "$BENCH_DIR/prompts/cache.txt" "$MAXTOK" "$OUT" || exit 1

jq -r '.choices[0].message.content' "$OUT" > "$MD"
rm -rf "$SRC"; mkdir -p "$SRC"
# Select blocks by CONTENT and keep the LAST revision of each.
# The old code took fenced blocks positionally (block 1 -> cache.go, block 2 ->
# cache_test.go). Models that revise their answer in-line emit several drafts of
# the same file: laguna-s-2.1 emitted 6 blocks (3x cache.go, 3x cache_test.go),
# so block 2 was cache.go revision 2 and got written as cache_test.go — producing
# a bogus "entry redeclared in this block" cascade that scored as a model failure.
python3 - "$MD" "$SRC" <<'PY'
import os, re, sys
md = open(sys.argv[1]).read(); out = sys.argv[2]
# The language tag must be its OWN group: with `\w*` ungrouped the pattern has a single
# group, so asking for group(2) raised IndexError on every response that actually contained
# a fenced block -- i.e. this extractor crashed for any model that emitted code, and only
# appeared to "work" when the response was empty.
blocks = [m.group(2) for m in re.finditer(r"^```(\w*)\n(.*?)^```", md, re.S | re.M)]
impl = test = None
for b in blocks:
    if re.search(r'^\s*func\s+Test\w*\s*\(', b, re.M) or '"testing"' in b:
        test = b            # last test block wins
    elif re.search(r'^\s*package\s+\w+', b, re.M):
        impl = b            # last implementation block wins
open(os.path.join(out, 'cache.go'), 'w').write(((impl or '').rstrip()) + "\n")
open(os.path.join(out, 'cache_test.go'), 'w').write(((test or '').rstrip()) + "\n")
print("  extractor: %d blocks -> impl=%s test=%s" % (
    len(blocks), "yes" if impl else "MISSING", "yes" if test else "MISSING"))
PY
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
