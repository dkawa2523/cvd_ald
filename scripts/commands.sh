#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_SH="$ROOT_DIR/scripts/env.sh"

python_has_core_deps() {
  "$PYTHON" - <<'PY' >/dev/null 2>&1
import importlib.util
import sys

required = ("numpy", "hydra", "omegaconf", "matplotlib")
missing = [name for name in required if importlib.util.find_spec(name) is None]
raise SystemExit(0 if not missing else 1)
PY
}

ensure_env() {
  # Auto-bootstrap environment if env.sh is missing or still placeholder.
  if [[ ! -f "$ENV_SH" ]]; then
    bash "$ROOT_DIR/scripts/preflight.sh"
  fi

  # shellcheck disable=SC1091
  source "$ENV_SH"

  if [[ -z "${PYTHON:-}" ]]; then
    bash "$ROOT_DIR/scripts/preflight.sh"
    # shellcheck disable=SC1091
    source "$ENV_SH"
  fi

  if [[ -z "${PYTHON:-}" ]]; then
    echo "[commands] ERROR: PYTHON is not set after preflight. Check scripts/preflight.sh." >&2
    exit 1
  fi

  if ! python_has_core_deps; then
    bash "$ROOT_DIR/scripts/preflight.sh"
    # shellcheck disable=SC1091
    source "$ENV_SH"
    if ! python_has_core_deps; then
      echo "[commands] ERROR: Selected PYTHON cannot import required deps (numpy, hydra, omegaconf, matplotlib)." >&2
      exit 1
    fi
  fi
}

ensure_env

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp}"
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"

cmd_import_check() {
  "$PYTHON" -c "import deposim_schema, deposim_sim, deposim_report"
}

cmd_smoke() {
  "$PYTHON" -m deposim_sim.smoke --config-name smoke
}

cmd_benchmark_wafer2d() {
  "$PYTHON" -m deposim_sim.benchmark_wafer2d --config-name smoke "$@"
}

cmd_benchmark_wafer2d_physviz() {
  "$PYTHON" -m deposim_sim.benchmark_wafer2d --config-name smoke --with-physviz --physviz-fast "$@"
}

cmd_smoke_repro_check() {
  "$PYTHON" - <<'PY'
import numpy as np

from deposim_schema import compose_sim_config
from deposim_sim.domain import build_domain_grid
from deposim_sim.synthetic_inputs import synthetic_pattern

run_spec = compose_sim_config("smoke")
grid = build_domain_grid(run_spec.domain)

pattern_a = synthetic_pattern(
    run_spec.inputs.synthetic_case,
    grid,
    random_seed=run_spec.random_seed,
)
pattern_b = synthetic_pattern(
    run_spec.inputs.synthetic_case,
    grid,
    random_seed=run_spec.random_seed,
)
if not np.array_equal(pattern_a, pattern_b):
    raise SystemExit("[commands] ERROR: smoke synthetic pattern is not deterministic for same seed.")

seeded_42 = synthetic_pattern("seeded_perturbation", grid, random_seed=42)
seeded_43 = synthetic_pattern("seeded_perturbation", grid, random_seed=43)
if np.array_equal(seeded_42, seeded_43):
    raise SystemExit("[commands] ERROR: seeded_perturbation did not change with different seeds.")
PY
}

cmd_smoke_compose_check() {
  "$PYTHON" - <<'PY'
from deposim_schema import compose_sim_config

run_spec = compose_sim_config(
    "smoke",
    overrides=[
        "domain.nr=11",
        "output.run_dir_name=smoke_check",
    ],
)
if run_spec.run_name != "smoke_synthetic":
    raise SystemExit(
        f"[commands] ERROR: expected run_name='smoke_synthetic', got {run_spec.run_name!r}"
    )
if run_spec.domain.nr != 11:
    raise SystemExit(f"[commands] ERROR: expected domain.nr=11, got {run_spec.domain.nr}")
if run_spec.output.run_dir_name != "smoke_check":
    raise SystemExit(
        f"[commands] ERROR: expected output.run_dir_name='smoke_check', got {run_spec.output.run_dir_name!r}"
    )
PY
}

cmd_require_numpy() {
  "$PYTHON" - <<'PY'
import importlib.util
import sys

if importlib.util.find_spec("numpy") is None:
    print(
        "[commands] ERROR: NumPy is required for P0-003 domain verification. "
        "Run ./scripts/preflight.sh after updating dependencies.",
        file=sys.stderr,
    )
    raise SystemExit(1)
PY
}

cmd_require_matplotlib() {
  "$PYTHON" - <<'PY'
import importlib.util
import sys

if importlib.util.find_spec("matplotlib") is None:
    print(
        "[commands] ERROR: Matplotlib is required for P0-008 run-manager/report verification.",
        file=sys.stderr,
    )
    raise SystemExit(1)
PY
}

cmd_domain_tests() {
  cmd_require_numpy
  cmd_unittest_module \
    "deposim_sim.test_domain" \
    "P0-003 domain tests reported skipped cases; treating this as verification failure."
}

cmd_mass_transfer_tests() {
  cmd_require_numpy
  cmd_unittest_module \
    "deposim_sim.test_mass_transfer" \
    "P0-004 mass-transfer tests reported skipped cases; treating this as verification failure."
}

cmd_rate_law_tests() {
  cmd_require_numpy
  cmd_unittest_module \
    "deposim_sim.test_rate_laws" \
    "P0-005 rate-law tests reported skipped cases; treating this as verification failure."
}

cmd_solver_tests() {
  cmd_require_numpy
  cmd_unittest_module \
    "deposim_sim.test_root_solve" \
    "P0-006 solver tests reported skipped cases; treating this as verification failure."
}

cmd_cvd_steady_tests() {
  cmd_require_numpy
  cmd_unittest_module \
    "deposim_sim.test_cvd_steady" \
    "P0-007 cvd_steady tests reported skipped cases; treating this as verification failure."
}

cmd_run_manager_tests() {
  cmd_require_numpy
  cmd_require_matplotlib
  cmd_unittest_module \
    "deposim_sim.test_run_manager" \
    "P0-008 run-manager tests reported skipped cases; treating this as verification failure."
}

cmd_unittest_module() {
  local module_name="${1:-}"
  local skipped_error="${2:-}"
  if [[ -z "$module_name" ]]; then
    echo "[commands] ERROR: cmd_unittest_module requires module name" >&2
    return 2
  fi
  MODULE_NAME="$module_name" SKIPPED_ERROR="$skipped_error" "$PYTHON" - <<'PY'
import os
import sys
import unittest

module_name = os.environ["MODULE_NAME"]
skipped_error = os.environ.get("SKIPPED_ERROR", "").strip()
suite = unittest.defaultTestLoader.loadTestsFromName(module_name)
if suite.countTestCases() == 0:
    print(
        f"[commands] ERROR: unittest module '{module_name}' resolved to zero tests.",
        file=sys.stderr,
    )
    raise SystemExit(1)
runner = unittest.TextTestRunner(verbosity=1)
result = runner.run(suite)
if result.skipped:
    if skipped_error:
        print(f"[commands] ERROR: {skipped_error}", file=sys.stderr)
    else:
        print(
            f"[commands] ERROR: unittest module '{module_name}' reported skipped cases.",
            file=sys.stderr,
        )
    raise SystemExit(1)
if not result.wasSuccessful():
    raise SystemExit(1)
PY
}

cmd_xy_domain_check() {
  "$PYTHON" - <<'PY'
from deposim_schema import DomainSpec
from deposim_sim.domain import build_domain_grid, radial_profile
import numpy as np

spec = DomainSpec(kind="wafer_2d_xy", wafer_radius_mm=150.0, nr=8, nx=24, ny=24, edge_exclusion_mm=5.0)
grid = build_domain_grid(spec)
if grid.kind != "wafer_2d_xy":
    raise SystemExit("[commands] ERROR: failed to build wafer_2d_xy grid.")
values = np.ones(grid.shape, dtype=float)
r_mm, prof = radial_profile(values, grid)
if r_mm.shape[0] != spec.nr:
    raise SystemExit("[commands] ERROR: XY radial profile r-size mismatch.")
if prof.shape[0] != spec.nr:
    raise SystemExit("[commands] ERROR: XY radial profile value-size mismatch.")
PY
}

cmd_registry_metadata_check() {
  "$PYTHON" - <<'PY'
from deposim_sim.models import mass_transfer, rate_laws

required = ("requires", "excludes", "time_modes", "governing_class")
for module, getter_name in (
    (mass_transfer, "get_mass_transfer_metadata"),
    (rate_laws, "get_rate_law_metadata"),
):
    getter = getattr(module, getter_name, None)
    if getter is None:
        raise SystemExit(f"[commands] ERROR: missing metadata getter {module.__name__}.{getter_name}")
    metadata = getter()
    if not isinstance(metadata, dict):
        raise SystemExit(f"[commands] ERROR: metadata getter {getter_name} must return dict")
    if not metadata:
        raise SystemExit(f"[commands] ERROR: metadata getter {getter_name} returned empty map")
    for model_name, entry in metadata.items():
        for key in required:
            if key not in entry:
                raise SystemExit(
                    f"[commands] ERROR: metadata for {module.__name__}:{model_name!r} missing key {key!r}"
                )
PY
}

cmd_compatibility_validator_check() {
  "$PYTHON" - <<'PY'
from deposim_schema import compose_sim_config
from deposim_sim.validation.compatibility import validate_run_spec

baseline = compose_sim_config("smoke")
validate_run_spec(baseline)

invalid = compose_sim_config(
    "smoke",
    overrides=[
        "model.mass_transfer_name=rotating_disk",
        "+model.mass_transfer_params.omega_zero_guard=error",
        "inputs.omega_rad_s=0.0",
    ],
)
try:
    validate_run_spec(invalid)
except ValueError:
    pass
else:
    raise SystemExit("[commands] ERROR: expected validator failure for omega=0 rotating_disk(error).")
PY
}

cmd_measurement_adapter_tests() { cmd_unittest_module "deposim_sim.test_measurement_adapter"; }
cmd_metrics_tests() { cmd_unittest_module "deposim_sim.test_metrics"; }
cmd_report_comparison_tests() { cmd_unittest_module "deposim_sim.test_report_comparison"; }
cmd_doe_tests() { cmd_unittest_module "deposim_sim.test_doe"; }
cmd_zref_tests() { cmd_unittest_module "deposim_sim.test_zref_sensitivity"; }
cmd_kinetics_net_tests() { cmd_unittest_module "deposim_sim.test_net_models"; }
cmd_phases_driver_tests() { cmd_unittest_module "deposim_sim.test_phases_driver"; }
cmd_state_closure_tests() { cmd_unittest_module "deposim_sim.test_state_closure"; }
cmd_bosanquet_pattern_tests() { cmd_unittest_module "deposim_sim.test_bosanquet_pattern"; }
cmd_identifiability_tests() { cmd_unittest_module "deposim_sim.test_identifiability"; }
cmd_jax_optional_tests() { cmd_unittest_module "deposim_sim.test_jax_optional"; }
cmd_benchmark_tests() { cmd_unittest_module "deposim_sim.test_benchmark"; }
cmd_benchmark_wafer2d_tests() { cmd_unittest_module "deposim_sim.test_benchmark_wafer2d"; }
cmd_physviz_tests() { cmd_unittest_module "deposim_sim.test_physviz"; }
cmd_opt_tests() { cmd_unittest_module "deposim_opt.test_opt_scaffold"; }
cmd_assimilation_tests() { cmd_unittest_module "deposim_opt.test_assimilation"; }
cmd_ald_tests() { cmd_unittest_module "deposim_sim.test_ald"; }
cmd_clearml_optional_tests() { cmd_unittest_module "deposim_tracking_clearml.test_optional_clearml"; }
cmd_io_plugin_tests() { cmd_unittest_module "deposim_sim.test_io_plugins"; }
cmd_zarr_optional_tests() { cmd_unittest_module "deposim_sim.test_zarr_output"; }
cmd_multiz_tests() { cmd_unittest_module "deposim_sim.test_multiz"; }

cmd_p2_io_e2e_check() {
  "$PYTHON" - <<'PY'
from pathlib import Path
from tempfile import TemporaryDirectory

from deposim_sim.smoke import main as smoke_main

with TemporaryDirectory(prefix="deposim_p2005_e2e_") as tmp:
    tmp_path = Path(tmp)
    csv_path = tmp_path / "input.csv"
    csv_path.write_text("C_ref__precursor,T\n1.6,705.0\n", encoding="utf-8")
    out_dir = tmp_path / "results"
    rc = smoke_main(
        [
            "--config-name",
            "smoke",
            "domain.nr=4",
            "domain.ntheta=8",
            "time.process_time_s=1.0",
            "inputs.source_kind=file",
            "inputs.io_loader_name=csv",
            f"inputs.field_path={csv_path}",
            f"output.project_dir={out_dir}",
            "output.run_dir_name=p2005_e2e",
        ]
    )
    if rc != 0:
        raise SystemExit("[commands] ERROR: P2-005 e2e smoke run failed.")
    runs = sorted([p for p in (out_dir / "runs").iterdir() if p.is_dir()])
    if not runs:
        raise SystemExit("[commands] ERROR: P2-005 e2e produced no run directory.")
    latest = runs[-1]
    if not (latest / "report.html").exists():
        raise SystemExit("[commands] ERROR: P2-005 e2e missing report artifact.")
PY
}

cmd_p2_zarr_doe_e2e_check() {
  "$PYTHON" - <<'PY'
from pathlib import Path
from tempfile import TemporaryDirectory

from deposim_sim.doe import run_doe
from deposim_sim.zarr_output import is_zarr_available

with TemporaryDirectory(prefix="deposim_p2006_e2e_") as tmp:
    result = run_doe(
        config_name="smoke",
        base_overrides=[
            f"output.project_dir={tmp}",
            "output.array_store=zarr",
            "domain.nr=4",
            "domain.ntheta=8",
            "time.process_time_s=1.0",
        ],
        sweep={"inputs.c_ref_mol_m3": [1.2, 1.8]},
        sampling="grid",
    )
    run_dir = result.run_dir
    if is_zarr_available():
        path = run_dir / "outputs" / "doe_cases.zarr"
    else:
        path = run_dir / "outputs" / "doe_cases.npz"
    if not path.exists():
        raise SystemExit("[commands] ERROR: P2-006 e2e missing expected DOE array-store artifact.")
PY
}

cmd_p2_multiz_smoke_e2e_check() {
  "$PYTHON" - <<'PY'
import numpy as np
from pathlib import Path
from tempfile import TemporaryDirectory

from deposim_sim.smoke import main as smoke_main

with TemporaryDirectory(prefix="deposim_p2007_e2e_") as tmp:
    out_dir = Path(tmp) / "results"
    rc = smoke_main(
        [
            "--config-name",
            "smoke",
            "domain.nr=4",
            "domain.ntheta=8",
            "time.process_time_s=1.0",
            "output.array_store=npz",
            "reference_plane.z_ref_mm=5.0",
            "reference_plane.z_ref_mm_list=[3.0,5.0]",
            f"output.project_dir={out_dir}",
            "output.run_dir_name=p2007_e2e",
        ]
    )
    if rc != 0:
        raise SystemExit("[commands] ERROR: P2-007 e2e smoke run failed.")
    runs = sorted([p for p in (out_dir / "runs").iterdir() if p.is_dir()])
    if not runs:
        raise SystemExit("[commands] ERROR: P2-007 e2e produced no run directory.")
    latest = runs[-1]
    diag = np.load(latest / "outputs" / "diagnostics.npz")
    if "plane_count" not in diag.files:
        raise SystemExit("[commands] ERROR: P2-007 e2e missing multi-z diagnostics.")
PY
}

cmd_require_state_sync() {
  local milestones_csv="${1:-P1,P2}"
  local exclude_csv="${2:-P2-999}"
  ROOT_DIR="$ROOT_DIR" MILESTONES_CSV="$milestones_csv" EXCLUDE_CSV="$exclude_csv" "$PYTHON" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["ROOT_DIR"])
milestones = {m.strip() for m in os.environ["MILESTONES_CSV"].split(",") if m.strip()}
exclude_ids = {x.strip() for x in os.environ["EXCLUDE_CSV"].split(",") if x.strip()}
tasks_path = root / "tasks" / "tasks.json"
state_path = root / "runs" / "autorun_state.json"

if not state_path.exists():
    raise SystemExit(
        "[commands] ERROR: runs/autorun_state.json is missing. "
        "Run ./scripts/commands.sh reconcile_state --milestone P1,P2 first."
    )

tasks = json.loads(tasks_path.read_text(encoding="utf-8"))["tasks"]
state = json.loads(state_path.read_text(encoding="utf-8"))
completed = state.get("completed", {})

required: list[str] = []
for task in tasks:
    tid = task.get("task_id", "")
    if not tid or tid in exclude_ids:
        continue
    if task.get("milestone") in milestones:
        required.append(tid)

missing = [tid for tid in required if completed.get(tid, {}).get("status") != "ok"]
if missing:
    missing_txt = ", ".join(missing)
    raise SystemExit(
        "[commands] ERROR: autorun_state is not synchronized for required tasks: "
        f"{missing_txt}. Run ./scripts/commands.sh reconcile_state --milestone P1,P2."
    )
PY
}

cmd_reconcile_state() {
  if [[ "$#" -lt 1 ]]; then
    echo "[commands] ERROR: reconcile_state requires task_id(s) or --milestone <P1,P2,...>." >&2
    return 2
  fi
  local task_ids=()

  if [[ "${1:-}" == "--milestone" ]]; then
    shift
    if [[ "$#" -lt 1 ]]; then
      echo "[commands] ERROR: --milestone requires comma-separated milestone names (e.g. P1,P2)." >&2
      return 2
    fi
    local milestone_csv="${1:-}"
    shift
    if [[ "$#" -gt 0 ]]; then
      echo "[commands] ERROR: unexpected extra arguments after --milestone." >&2
      return 2
    fi

    while IFS= read -r line; do
      [[ -n "$line" ]] && task_ids+=("$line")
    done < <(
      ROOT_DIR="$ROOT_DIR" MILESTONE_CSV="$milestone_csv" "$PYTHON" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["ROOT_DIR"])
milestones = {m.strip() for m in os.environ["MILESTONE_CSV"].split(",") if m.strip()}
tasks = json.loads((root / "tasks" / "tasks.json").read_text(encoding="utf-8"))["tasks"]
for task in tasks:
    tid = str(task.get("task_id", "")).strip()
    if not tid or tid.endswith("-999"):
        continue
    if task.get("milestone") in milestones:
        print(tid)
PY
    )

    if [[ "${#task_ids[@]}" -eq 0 ]]; then
      echo "[commands] ERROR: no tasks found for milestones: $milestone_csv" >&2
      return 1
    fi
  else
    task_ids=("$@")
  fi

  local tid
  for tid in "${task_ids[@]}"; do
    cmd_verify_task "$tid"
  done

  ROOT_DIR="$ROOT_DIR" TASK_IDS="$(printf "%s\n" "${task_ids[@]}")" "$PYTHON" - <<'PY'
import json
import os
import time
from pathlib import Path

root = Path(os.environ["ROOT_DIR"])
task_ids = [line.strip() for line in os.environ["TASK_IDS"].splitlines() if line.strip()]
state_path = root / "runs" / "autorun_state.json"
tasks_path = root / "tasks" / "tasks.json"
tasks = json.loads(tasks_path.read_text(encoding="utf-8"))["tasks"]
task_order = [task["task_id"] for task in tasks]

if state_path.exists():
    state = json.loads(state_path.read_text(encoding="utf-8"))
else:
    state = {"completed": {}, "created_at": time.time()}

completed = state.setdefault("completed", {})
ts = time.time()
for tid in task_ids:
    completed[tid] = {"status": "ok", "completed_at": ts}

ordered_done = [tid for tid in task_order if completed.get(tid, {}).get("status") == "ok"]
if ordered_done:
    state["last_completed"] = ordered_done[-1]
state_path.parent.mkdir(parents=True, exist_ok=True)
state_path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
print(f"[commands] reconciled autorun_state with {len(task_ids)} task(s).")
PY
}

cmd_test() {
  "$PYTHON" -m unittest discover -s tests -p "test_*.py"
  cmd_domain_tests
  cmd_mass_transfer_tests
  cmd_rate_law_tests
  cmd_solver_tests
  cmd_cvd_steady_tests
  cmd_run_manager_tests
  cmd_benchmark_wafer2d_tests
  cmd_physviz_tests
}

cmd_verify_p0() {
  cmd_import_check
  cmd_smoke
  cmd_test
}

cmd_verify_milestone() {
  local milestones_csv="${1:-}"
  local exclude_csv="${2:-}"
  if [[ -z "$milestones_csv" ]]; then
    echo "[commands] ERROR: cmd_verify_milestone requires milestone name(s)." >&2
    return 2
  fi

  local task_ids=()
  local task_lines
  task_lines="$(
    ROOT_DIR="$ROOT_DIR" MILESTONES_CSV="$milestones_csv" EXCLUDE_CSV="$exclude_csv" "$PYTHON" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["ROOT_DIR"])
milestones = {m.strip() for m in os.environ["MILESTONES_CSV"].split(",") if m.strip()}
exclude_ids = {x.strip() for x in os.environ.get("EXCLUDE_CSV", "").split(",") if x.strip()}
tasks = json.loads((root / "tasks" / "tasks.json").read_text(encoding="utf-8"))["tasks"]
for task in tasks:
    tid = str(task.get("task_id", "")).strip()
    if not tid:
        continue
    if task.get("milestone") not in milestones:
        continue
    if tid in exclude_ids:
        continue
    print(tid)
PY
  )"
  while IFS= read -r line; do
    [[ -n "$line" ]] && task_ids+=("$line")
  done <<< "$task_lines"

  if [[ "${#task_ids[@]}" -eq 0 ]]; then
    echo "[commands] ERROR: no tasks found for milestone(s): $milestones_csv" >&2
    return 1
  fi

  local tid
  for tid in "${task_ids[@]}"; do
    cmd_verify_task "$tid"
  done
}

cmd_verify_p1() {
  cmd_verify_milestone "P1"
}

cmd_verify_p2() {
  cmd_verify_milestone "P2" "P2-999"
}

cmd_verify_task_contracts() {
  "$PYTHON" "$ROOT_DIR/scripts/codex_autorun.py" --validate-task-contracts
}

cmd_verify_autorun() {
  "$PYTHON" "$ROOT_DIR/scripts/codex_autorun.py" --dry-run --git-check auto --lock-timeout-sec 1 --max-tasks 1
}

cmd_step_next() {
  "$PYTHON" "$ROOT_DIR/scripts/codex_autorun.py" --git-check auto --max-tasks 1
}

cmd_verify_repo_consistency() {
  local stale_matches
  stale_matches="$(
    rg -n --no-heading \
      -e 'python3 -m ' \
      -e 'python -m unittest' \
      -e '\./scripts/run\.sh (smoke|test|verify_p0)' \
      -e 'pytest([[:space:]]|$)' \
      "$ROOT_DIR/docs" "$ROOT_DIR/tasks" "$ROOT_DIR/README_HANDOFF.md" "$ROOT_DIR/model.md" "$ROOT_DIR/model2.md" "$ROOT_DIR/AGENTS.md" "$ROOT_DIR/PLANS.md" \
      || true
  )"
  if [[ -n "$stale_matches" ]]; then
    echo "[commands] ERROR: stale run/test commands found outside scripts/commands.sh:" >&2
    echo "$stale_matches" >&2
    return 1
  fi

  local stale_package_matches
  stale_package_matches="$(
    rg -n --no-heading \
      -e '^[[:space:]]*(from|import)[[:space:]]+(deposim_core|deposition_sim|deposim_surface|deposim_io)\\b' \
      "$ROOT_DIR/src" "$ROOT_DIR/tests" "$ROOT_DIR/scripts" \
      || true
  )"
  if [[ -n "$stale_package_matches" ]]; then
    echo "[commands] ERROR: stale package imports detected:" >&2
    echo "$stale_package_matches" >&2
    return 1
  fi

  grep -qxF "runs/" "$ROOT_DIR/.gitignore"
  grep -qxF "results/" "$ROOT_DIR/.gitignore"

  grep -qF "./scripts/commands.sh smoke" "$ROOT_DIR/docs/EVAL_PROTOCOL.md"
  grep -qF "./scripts/commands.sh verify_p0" "$ROOT_DIR/docs/EVAL_PROTOCOL.md"

  ROOT_DIR="$ROOT_DIR" "$PYTHON" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["ROOT_DIR"])
tasks = json.loads((root / "tasks/tasks.json").read_text())["tasks"]

checkpoints = [task for task in tasks if task.get("type") == "checkpoint"]
if len(checkpoints) != 1:
    raise SystemExit(
        f"[commands] ERROR: expected exactly one checkpoint task, found {len(checkpoints)}"
    )

checkpoint = checkpoints[0]
if checkpoint.get("task_id") != "P0-999":
    raise SystemExit(
        f"[commands] ERROR: checkpoint task_id must be 'P0-999', got {checkpoint.get('task_id')!r}"
    )
if not checkpoint.get("stop_after"):
    raise SystemExit("[commands] ERROR: checkpoint P0-999 must set stop_after=true")

bad_commands = []
for task in tasks:
    for cmd in task.get("verification_commands", []):
        if "./scripts/commands.sh" not in cmd:
            bad_commands.append((task.get("task_id"), cmd))

if bad_commands:
    lines = "\n".join(f"{task_id}: {cmd}" for task_id, cmd in bad_commands)
    raise SystemExit(f"[commands] ERROR: non-canonical verification command(s):\n{lines}")
PY
}

cmd_model_explain_gap_check() {
  ROOT_DIR="$ROOT_DIR" "$PYTHON" - <<'PY'
import os
from pathlib import Path

gap_path = Path(os.environ["ROOT_DIR"]) / "docs" / "MODEL_GAP.md"
text = gap_path.read_text(encoding="utf-8")
required_tokens = {
    "Stefan flow correction": "D-001",
    "Smoothing PDE": "D-002",
    "Purge residual driver": "D-003",
    "Incubation/poisoning": "D-004",
    "Chamber seasoning": "D-005",
}
for token, task_id in required_tokens.items():
    if token not in text:
        raise SystemExit(f"[commands] ERROR: MODEL_GAP missing token {token!r} for {task_id}.")
    if task_id not in text:
        raise SystemExit(f"[commands] ERROR: MODEL_GAP missing decision task id {task_id}.")
if "ADR_REQUIRED" not in text:
    raise SystemExit("[commands] ERROR: MODEL_GAP must include ADR_REQUIRED classification.")
PY
}

cmd_checkpoint_artifact_check() {
  local latest_checkpoint
  latest_checkpoint="$(ls -1t "$ROOT_DIR"/runs/P0-999-checkpoint-*.md 2>/dev/null | head -n 1 || true)"
  if [[ -z "$latest_checkpoint" ]]; then
    echo "[commands] ERROR: checkpoint summary file runs/P0-999-checkpoint-*.md is missing." >&2
    return 1
  fi

  grep -q "Verification Results" "$latest_checkpoint" || {
    echo "[commands] ERROR: checkpoint summary missing 'Verification Results' section: $latest_checkpoint" >&2
    return 1
  }
  grep -q "Next Steps" "$latest_checkpoint" || {
    echo "[commands] ERROR: checkpoint summary missing 'Next Steps' section: $latest_checkpoint" >&2
    return 1
  }
  grep -q "Known Limitations" "$latest_checkpoint" || {
    echo "[commands] ERROR: checkpoint summary missing 'Known Limitations' section: $latest_checkpoint" >&2
    return 1
  }
}

cmd_verify_task() {
  local task_id="${1:-}"
  if [[ -z "$task_id" ]]; then
    echo "[commands] ERROR: verify_task requires <task_id>" >&2
    return 2
  fi

  case "$task_id" in
    P0-000)
      cmd_verify_task_contracts
      cmd_verify_autorun
      ;;
    P0-001)
      test -f "$ROOT_DIR/pyproject.toml"
      test -d "$ROOT_DIR/src/deposim_sim"
      test -d "$ROOT_DIR/src/deposim_schema"
      test -d "$ROOT_DIR/src/deposim_report"
      test -f "$ROOT_DIR/tests/test_imports.py"
      cmd_import_check
      "$PYTHON" -m unittest discover -s "$ROOT_DIR/tests" -p "test_imports.py"
      ;;
    P0-002)
      test -f "$ROOT_DIR/configs/sim/example_cvd.yaml"
      "$PYTHON" -c "import deposim_schema"
      rm -f /tmp/p0002_resolved.yaml
      "$PYTHON" - <<'PY'
from deposim_schema import compose_and_save_sim_config

compose_and_save_sim_config("/tmp/p0002_resolved.yaml", "example_cvd")
PY
      test -f /tmp/p0002_resolved.yaml
      "$PYTHON" -m unittest discover -s "$ROOT_DIR/tests" -p "test_sim_config_compose.py"
      ;;
    P0-003)
      "$PYTHON" -c "import deposim_sim"
      cmd_domain_tests
      ;;
    P0-004)
      "$PYTHON" -c "from deposim_sim.models import mass_transfer"
      cmd_mass_transfer_tests
      ;;
    P0-005)
      "$PYTHON" -c "from deposim_sim.models import rate_laws"
      cmd_rate_law_tests
      ;;
    P0-006)
      "$PYTHON" -c "from deposim_sim.solvers import root_solve"
      cmd_solver_tests
      ;;
    P0-007)
      "$PYTHON" -c "from deposim_sim.physics import cvd_steady"
      cmd_cvd_steady_tests
      ;;
    P0-008)
      cmd_run_manager_tests
      ;;
    P0-009)
      before_count=0
      if [[ -d "$ROOT_DIR/results/runs" ]]; then
        before_count="$(find "$ROOT_DIR/results/runs" -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d ' ')"
      fi
      cmd_smoke
      after_count="$(find "$ROOT_DIR/results/runs" -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d ' ')"
      if [[ "$after_count" -le "$before_count" ]]; then
        echo "[commands] ERROR: smoke did not create a new run under results/runs" >&2
        return 1
      fi
      latest_run="$(find "$ROOT_DIR/results/runs" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)"
      test -f "$latest_run/config_resolved.yaml"
      test -f "$latest_run/report.html"
      grep -q "^run_name: smoke_synthetic$" "$latest_run/config_resolved.yaml"
      LATEST_RUN="$latest_run" "$PYTHON" - <<'PY'
import json
import os
from pathlib import Path

run_dir = Path(os.environ["LATEST_RUN"])
summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
artifact_paths = summary.get("artifact_paths", {})
required_keys = ("thickness", "diagnostics")
for key in required_keys:
    rel = artifact_paths.get(key)
    if not rel:
        raise SystemExit(f"[commands] ERROR: summary.artifact_paths.{key} is missing")
    path = run_dir / rel
    if not path.exists():
        raise SystemExit(f"[commands] ERROR: artifact path for {key} does not exist: {path}")
PY
      cmd_smoke_repro_check
      cmd_smoke_compose_check
      ;;
    P0-010)
      cmd_verify_p0
      ;;
    P0-011)
      cmd_verify_repo_consistency
      ;;
    P0-999)
      cmd_verify_p0
      cmd_checkpoint_artifact_check
      ;;
    P1-001)
      cmd_xy_domain_check
      cmd_domain_tests
      ;;
    P1-002)
      cmd_registry_metadata_check
      ;;
    P1-003)
      cmd_compatibility_validator_check
      ;;
    P1-004)
      cmd_measurement_adapter_tests
      ;;
    P1-005)
      cmd_metrics_tests
      ;;
    P1-006)
      cmd_report_comparison_tests
      ;;
    P1-007)
      cmd_doe_tests
      ;;
    P1-008)
      cmd_zref_tests
      ;;
    P1-009)
      cmd_kinetics_net_tests
      ;;
    P1-010)
      cmd_phases_driver_tests
      ;;
    P1-011)
      cmd_state_closure_tests
      ;;
    P1-012)
      cmd_bosanquet_pattern_tests
      ;;
    P1-013)
      cmd_identifiability_tests
      ;;
    P1-014)
      cmd_jax_optional_tests
      ;;
    P1-015)
      cmd_benchmark_tests
      ;;
    P2-001)
      cmd_opt_tests
      ;;
    P2-002)
      cmd_assimilation_tests
      ;;
    P2-003)
      cmd_ald_tests
      ;;
    P2-004)
      cmd_clearml_optional_tests
      ;;
    P2-005)
      cmd_io_plugin_tests
      cmd_p2_io_e2e_check
      ;;
    P2-006)
      cmd_zarr_optional_tests
      cmd_p2_zarr_doe_e2e_check
      ;;
    P2-007)
      cmd_multiz_tests
      cmd_p2_multiz_smoke_e2e_check
      ;;
    P2-999)
      cmd_require_state_sync "P1,P2" "P2-999"
      "$PYTHON" -m py_compile "$ROOT_DIR/scripts/codex_autorun.py"
      cmd_verify_p1
      cmd_verify_p2
      cmd_verify_repo_consistency
      ;;
    D-001|D-002|D-003|D-004|D-005)
      cmd_model_explain_gap_check
      ;;
    *)
      echo "[commands] ERROR: unknown task_id: $task_id" >&2
      return 2
      ;;
  esac
}

cmd_show_env() {
  echo "ROOT_DIR=$ROOT_DIR"
  echo "PYTHON=$PYTHON"
  echo "PIP=$PIP"
  echo "PYTHONPATH=${PYTHONPATH:-}"
  echo "MPLCONFIGDIR=$MPLCONFIGDIR"
}

usage() {
  cat <<EOF
Usage: ./scripts/commands.sh <command>

Commands:
  show_env       Print resolved environment
  import_check   Import deposim_* packages
  smoke          Run minimal synthetic CVD steady simulation
  benchmark_wafer2d Run wafer-2D trend benchmark (polar, synthetic+file)
  benchmark_wafer2d_physviz Run wafer-2D trend benchmark with physviz outputs
  test           Run unit tests (stdlib unittest)
  verify_p0      import_check + smoke + test
  verify_p1      Run verification gates for all P1 tasks
  verify_p2      Run verification gates for P2-001..P2-007
  step_next      Execute only the next incomplete task
  verify_autorun Run autorun in dry-run mode with lock and contract checks
  verify_task_contracts Validate tasks/tasks.json contracts
  verify_task    Run verification command set for a specific task_id
  reconcile_state Verify task(s) then mark them completed in runs/autorun_state.json
                 Supports: reconcile_state --milestone P1,P2
EOF
}

case "${1:-}" in
  show_env) shift; cmd_show_env "$@";;
  import_check) shift; cmd_import_check "$@";;
  smoke) shift; cmd_smoke "$@";;
  benchmark_wafer2d) shift; cmd_benchmark_wafer2d "$@";;
  benchmark_wafer2d_physviz) shift; cmd_benchmark_wafer2d_physviz "$@";;
  test) shift; cmd_test "$@";;
  verify_p0) shift; cmd_verify_p0 "$@";;
  verify_p1) shift; cmd_verify_p1 "$@";;
  verify_p2) shift; cmd_verify_p2 "$@";;
  step_next) shift; cmd_step_next "$@";;
  verify_autorun) shift; cmd_verify_autorun "$@";;
  verify_task_contracts) shift; cmd_verify_task_contracts "$@";;
  verify_task) shift; cmd_verify_task "$@";;
  reconcile_state) shift; cmd_reconcile_state "$@";;
  *) usage; exit 2;;
esac
