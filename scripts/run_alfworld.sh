#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
PROFILE="${1:?Usage: scripts/run_alfworld.sh PROFILE [extra pipeline arguments]}"
shift

exec "$PYTHON_BIN" scripts/run_from_config.py \
  --config configs/main/alfworld.yaml \
  --profile "$PROFILE" \
  -- "$@"
