#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
source "$ROOT/../../../.cursor/skills/_shared/project_env.sh"
CFG="${1:-configs/smoke.yaml}"
"$PY" -u -m imlfsr.train --config "$CFG" "${@:2}"
