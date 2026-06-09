#!/usr/bin/env bash
# Run both coding tasks for one or more models at their recommended sampling.
# Usage:
#   ./run-all.sh                  # all known models (see lib.sh sampling_for)
#   ./run-all.sh gemma-4-26B-A4B  # one or more specific model ids
set -uo pipefail
cd "$(dirname "$0")"

MODELS=("$@")
if [[ ${#MODELS[@]} -eq 0 ]]; then
  MODELS=(qwen3-coder-next qwen36-35b-a3b ornstein36-27B ornstein36-35b-a3b gemma-4-26B-A4B)
fi

for m in "${MODELS[@]}"; do
  echo; echo "############################################################"
  echo "# $m"
  echo "############################################################"
  ./run-go.sh   "$m" || echo "run-go.sh $m exited non-zero"
  ./run-java.sh "$m" || echo "run-java.sh $m exited non-zero"
done

echo; echo "All done. JSON responses in results/, build dirs in .work/."
