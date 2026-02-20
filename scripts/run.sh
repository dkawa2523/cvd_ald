#!/usr/bin/env bash
set -euo pipefail

# Repo root
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Stable runtime defaults
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp}"

mkdir -p "$ROOT_DIR/runs"
mkdir -p "$ROOT_DIR/results"

# Preflight (writes scripts/env.sh)
bash "$ROOT_DIR/scripts/preflight.sh"

# Autorun
# Exit code 42 indicates checkpoint/decision stop by design.
python_cmd="python3"
if command -v python3 >/dev/null 2>&1; then
  python_cmd="python3"
elif command -v python >/dev/null 2>&1; then
  python_cmd="python"
fi

set +e
"$python_cmd" "$ROOT_DIR/scripts/codex_autorun.py" "$@"
code=$?
set -e

exit "$code"
