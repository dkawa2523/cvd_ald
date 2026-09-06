#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_SH="$ROOT_DIR/scripts/env.sh"

ensure_env() {
  if [[ ! -f "$ENV_SH" ]]; then
    bash "$ROOT_DIR/scripts/preflight.sh"
  fi
  # shellcheck disable=SC1091
  source "$ENV_SH"
  if [[ -z "${PYTHON:-}" ]]; then
    echo "[commands] PYTHON is not set; rerun scripts/preflight.sh." >&2
    return 1
  fi
}

ensure_env
cd "$ROOT_DIR"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp}"
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"

cmd_import_check() {
  "$PYTHON" -c "import deposim_schema, deposim_sim, deposim_opt, deposim_report"
}

write_smoke_input() {
  local mode="$1"
  local path="$2"
  "$PYTHON" - "$mode" "$path" <<'PY'
from pathlib import Path
import sys
import numpy as np

mode, output = sys.argv[1:]
path = Path(output)
path.parent.mkdir(parents=True, exist_ok=True)
xy = np.array([[0.0, 0.0], [40.0, 0.0], [-40.0, 0.0]], dtype=float)

if mode == "cvd":
    cref = np.array(
        [[1.0, 0.30, 0.10, 0.0],
         [0.9, 0.25, 0.10, 0.0],
         [0.8, 0.20, 0.10, 0.0]],
        dtype=float,
    )
    np.savez(path, xy=xy, cref=cref)
elif mode == "ald":
    time = np.array([0.0, 0.25, 0.50, 0.75, 1.0], dtype=float)
    carrier = np.full(3, 0.02, dtype=float)
    a = np.array([1.0, 0.9, 0.85], dtype=float)
    b = np.array([1.0, 0.95, 0.9], dtype=float)
    inhibitor = np.array([0.05, 0.06, 0.04], dtype=float)
    residual = 0.01
    frames = [
        np.stack([a, residual * b, inhibitor, carrier], axis=1),
        np.stack([residual * a, residual * b, residual * inhibitor, carrier], axis=1),
        np.stack([residual * a, b, residual * inhibitor, carrier], axis=1),
        np.stack([residual * a, residual * b, residual * inhibitor, carrier], axis=1),
        np.stack([residual * a, residual * b, residual * inhibitor, carrier], axis=1),
    ]
    np.savez(path, xy=xy, time=time, cref=np.stack(frames, axis=0))
else:
    raise SystemExit(f"Unknown smoke mode: {mode}")
PY
}

cmd_smoke() {
  local mode="${1:-cvd}"
  case "$mode" in
    cvd)
      local input="/tmp/deposim_cvd_smoke_fluent.npz"
      write_smoke_input cvd "$input"
      "$PYTHON" -m deposim_sim.smoke --config-name cvd_steady_min \
        "sim.inputs.fluent.file=$input"
      ;;
    ald)
      local input="/tmp/deposim_ald_smoke_fluent.npz"
      write_smoke_input ald "$input"
      "$PYTHON" -m deposim_sim.smoke --config-name ald_state_min \
        "sim.inputs.fluent.file=$input"
      ;;
    *)
      echo "Usage: ./scripts/commands.sh smoke [cvd|ald]" >&2
      return 2
      ;;
  esac
}

cmd_models() {
  "$PYTHON" "$ROOT_DIR/scripts/analyze_cvd_multicond_case.py" --list-models
}

cmd_evaluate_cvd() {
  "$PYTHON" "$ROOT_DIR/scripts/analyze_cvd_multicond_case.py" \
    --data-dir "$ROOT_DIR/data" \
    --train-cases 1 2 4 5 \
    --test-case 3 \
    --response-model surface_compare \
    --models all \
    --bootstrap-samples 1000 \
    --seed 123 \
    --output "$ROOT_DIR/results/current_cvd_evaluation" \
    "$@"
}

cmd_fit() {
  local mode="${1:-}"
  if [[ -z "$mode" ]]; then
    echo "Usage: ./scripts/commands.sh fit <cvd|ald> [Hydra overrides...]" >&2
    return 2
  fi
  shift
  if ! "$PYTHON" -c "import optuna" >/dev/null 2>&1; then
    echo "[commands] fit configs use TPE. Install optimizer support with: $PIP install -e '.[optuna]'" >&2
    return 1
  fi
  local generated_dir="$ROOT_DIR/runs/generated_inputs/multicond_fit"
  "$PYTHON" "$ROOT_DIR/scripts/generate_multicond_fit_inputs.py" \
    --output-dir "$generated_dir"
  case "$mode" in
    cvd)
      "$PYTHON" -m deposim_opt.run_fit --config-name fit_cvd_multicond_min "$@"
      ;;
    ald)
      "$PYTHON" -m deposim_opt.run_fit --config-name fit_ald_state_multicond_min "$@"
      ;;
    *)
      echo "Usage: ./scripts/commands.sh fit <cvd|ald> [Hydra overrides...]" >&2
      return 2
      ;;
  esac
}

cmd_test() {
  "$PYTHON" -m unittest discover -s "$ROOT_DIR/tests" -p "test_*.py"
  "$PYTHON" -m unittest discover -s "$ROOT_DIR/src" -t "$ROOT_DIR/src" -p "test_*.py"
}

cmd_verify() {
  cmd_import_check
  cmd_smoke cvd
  cmd_smoke ald
  cmd_test
}

cmd_show_env() {
  echo "ROOT_DIR=$ROOT_DIR"
  echo "PYTHON=$PYTHON"
  echo "PYTHONPATH=${PYTHONPATH:-}"
  echo "MPLCONFIGDIR=$MPLCONFIGDIR"
}

usage() {
  cat <<'EOF'
Usage: ./scripts/commands.sh <command>

Commands:
  models          List steady, dynamic, transport, and net-film models
  evaluate_cvd    Run the fixed five-condition CVD equation census
  smoke [mode]    Run a minimal CVD or ALD state simulation
  fit <mode>      Fit generated multi-condition CVD or ALD role data
  test            Run the complete unit-test suite
  verify          Run imports, both smoke modes, and the unit-test suite
  show_env        Print the selected runtime
EOF
}

case "${1:-}" in
  models) shift; cmd_models "$@" ;;
  evaluate_cvd) shift; cmd_evaluate_cvd "$@" ;;
  smoke) shift; cmd_smoke "$@" ;;
  fit) shift; cmd_fit "$@" ;;
  test) shift; cmd_test "$@" ;;
  verify) shift; cmd_verify "$@" ;;
  show_env) shift; cmd_show_env "$@" ;;
  *) usage; exit 2 ;;
esac
