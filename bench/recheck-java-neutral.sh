#!/usr/bin/env bash
# Re-run ONLY the neutral Java suite (current neutral/NeutralTransferTest.java)
# against already-extracted production code. No model calls, no router — just
# Maven. Use after changing the neutral suite to refresh Java-neutral verdicts.
# Usage: ./recheck-java-neutral.sh
set -uo pipefail
cd "$(dirname "$0")"; source ./lib.sh
export JAVA_HOME="${JAVA_HOME:-$(ls -d /usr/lib/jvm/java-17-openjdk-* 2>/dev/null | head -1)}"

shopt -s nullglob
for proj in .work/java_*; do
  case "$proj" in *_n) continue;; esac          # skip neutral build dirs
  [ -d "$proj/src/main" ] || continue
  neu="${proj}_n"
  rm -rf "$neu"; mkdir -p "$neu/src/test/java/com/example/bank"
  cp "$proj/pom.xml" "$neu/pom.xml" 2>/dev/null
  cp -r "$proj/src/main" "$neu/src/main" 2>/dev/null
  cp "$BENCH_DIR/neutral/NeutralTransferTest.java" "$neu/src/test/java/com/example/bank/NeutralTransferTest.java"
  if ( cd "$neu" && mvn -B -q test ) >/dev/null 2>&1; then res=PASS; err=""; else
    res=FAIL; err=$( cd "$neu" && mvn -B -q test 2>&1 | grep -m1 -E 'ERROR.*\.java|cannot find|no suitable|does not exist|Tests run.*Failures' )
  fi
  printf '%-40s neutral=%-4s %s\n' "$(basename "$proj")" "$res" "${err:+:: $err}"
done
