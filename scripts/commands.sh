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
  local smoke_npz="/tmp/deposim_aib_smoke_fluent.npz"
  "$PYTHON" - <<'PY'
from pathlib import Path
import numpy as np

path = Path("/tmp/deposim_aib_smoke_fluent.npz")
xy = np.array([[0.0, 0.0], [25.0, 0.0], [50.0, 0.0], [75.0, 0.0]], dtype=float)
cref = np.array(
    [
        [1.0, 0.3, 0.1, 0.0],
        [0.9, 0.3, 0.1, 0.0],
        [0.8, 0.2, 0.1, 0.0],
        [0.7, 0.2, 0.1, 0.0],
    ],
    dtype=float,
)
path.parent.mkdir(parents=True, exist_ok=True)
np.savez(path, xy=xy, cref=cref)
PY
  "$PYTHON" -m deposim_sim.smoke --config-name cvd_steady_min "sim.inputs.fluent.file=$smoke_npz"
}

cmd_benchmark_wafer2d() {
  "$PYTHON" -m deposim_sim.benchmark_wafer2d --config-name cvd_steady_min "$@"
}

cmd_benchmark_wafer2d_physviz() {
  "$PYTHON" -m deposim_sim.benchmark_wafer2d --config-name cvd_steady_min --with-physviz --physviz-fast "$@"
}

cmd_benchmark_wafer2d_flux_km() {
  "$PYTHON" -m deposim_sim.benchmark_wafer2d --config-name cvd_steady_min --compare-flux-km "$@"
}

cmd_smoke_repro_check() {
  "$PYTHON" - <<'PY'
import numpy as np

from deposim_schema import compose_sim_config
from deposim_sim.pipeline import run_aib_from_spec

xy = np.array([[0.0, 0.0], [25.0, 0.0], [50.0, 0.0], [75.0, 0.0]], dtype=float)
cref = np.array(
    [
        [1.0, 0.3, 0.1, 0.0],
        [0.9, 0.3, 0.1, 0.0],
        [0.8, 0.2, 0.1, 0.0],
        [0.7, 0.2, 0.1, 0.0],
    ],
    dtype=float,
)
np.savez("/tmp/deposim_smoke_repro_fluent.npz", xy=xy, cref=cref)

run_spec = compose_sim_config(
    "cvd_steady_min",
    overrides=["sim.inputs.fluent.file=/tmp/deposim_smoke_repro_fluent.npz"],
)
out_a = run_aib_from_spec(run_spec)
out_b = run_aib_from_spec(run_spec)
if not np.allclose(np.asarray(out_a.thickness, dtype=float), np.asarray(out_b.thickness, dtype=float)):
    raise SystemExit("[commands] ERROR: AIB smoke thickness is not reproducible for the same inputs.")
PY
}

cmd_smoke_compose_check() {
  "$PYTHON" - <<'PY'
from deposim_schema import compose_sim_config

run_spec = compose_sim_config(
    "cvd_steady_min",
    overrides=[
        "sim.output.run_name=smoke_check",
        "sim.time.t_proc_s=1.5",
        "sim.roles.A=s0",
    ],
)
if run_spec.model.name != "aib_ode":
    raise SystemExit(
        f"[commands] ERROR: expected model.name='aib_ode', got {run_spec.model.name!r}"
    )
if run_spec.time_mode != "steady":
    raise SystemExit(f"[commands] ERROR: expected time_mode='steady', got {run_spec.time_mode!r}")
if run_spec.output.run_name != "smoke_check":
    raise SystemExit(
        f"[commands] ERROR: expected output.run_name='smoke_check', got {run_spec.output.run_name!r}"
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
    "deposim_sim.test_fluent_loader" \
    "P0-003 domain/input tests reported skipped cases; treating this as verification failure."
}

cmd_mass_transfer_tests() {
  cmd_require_numpy
  cmd_unittest_module \
    "deposim_sim.test_mass_transfer" \
    "P0-004 mass-transfer tests reported skipped cases; treating this as verification failure."
}

cmd_rate_law_tests() { cmd_unittest_module "deposim_sim.test_aib_ode"; }

cmd_solver_tests() { cmd_unittest_module "deposim_sim.test_pipeline_aib"; }

cmd_cvd_steady_tests() {
  cmd_require_numpy
  cmd_unittest_module \
    "deposim_sim.test_cvd_steady_aib" \
    "P0-007 AIB steady tests reported skipped cases; treating this as verification failure."
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
  local skip_policy="${3:-auto}"
  if [[ -z "$module_name" ]]; then
    echo "[commands] ERROR: cmd_unittest_module requires module name" >&2
    return 2
  fi
  case "$skip_policy" in
    auto|warn|fail) ;;
    *)
      echo "[commands] ERROR: invalid skip policy: $skip_policy (expected: auto|warn|fail)" >&2
      return 2
      ;;
  esac
  "$PYTHON" - "$module_name" "$skipped_error" "$skip_policy" <<'PY'
import sys
import unittest

module_name = sys.argv[1]
skipped_error = sys.argv[2].strip()
skip_policy = sys.argv[3].strip() or "auto"
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
    effective_policy = skip_policy
    if effective_policy == "auto":
        effective_policy = "fail" if skipped_error else "warn"
    if effective_policy == "fail":
        if skipped_error:
            print(f"[commands] ERROR: {skipped_error}", file=sys.stderr)
        else:
            print(
                f"[commands] ERROR: unittest module '{module_name}' reported skipped cases.",
                file=sys.stderr,
            )
        raise SystemExit(1)
    print(
        f"[commands] WARN: unittest module '{module_name}' reported skipped cases (skip_policy={effective_policy}).",
        file=sys.stderr,
    )
if not result.wasSuccessful():
    raise SystemExit(1)
PY
}

cmd_xy_domain_check() {
  "$PYTHON" - <<'PY'
import numpy as np

from deposim_sim.domain import radial_profile
from deposim_sim.input_builder import build_domain_from_fluent_xy

xy = np.array([[-30.0, -10.0], [0.0, 0.0], [25.0, 12.0], [45.0, -20.0]], dtype=float)
grid = build_domain_from_fluent_xy(xy=xy, xy_unit="mm", wafer_radius_mm=150.0)
if grid.kind != "from_fluent_xy":
    raise SystemExit("[commands] ERROR: failed to build from_fluent_xy grid.")
values = np.ones(grid.shape, dtype=float)
r_mm, prof = radial_profile(values, grid)
if r_mm.size == 0 or prof.size == 0:
    raise SystemExit("[commands] ERROR: radial_profile returned empty outputs for from_fluent_xy.")
PY
}

cmd_registry_metadata_check() {
  "$PYTHON" - <<'PY'
from deposim_sim.models import mass_transfer

required = ("requires", "excludes", "time_modes", "governing_class")
getter = getattr(mass_transfer, "get_mass_transfer_metadata", None)
if getter is None:
    raise SystemExit("[commands] ERROR: missing metadata getter mass_transfer.get_mass_transfer_metadata")
metadata = getter()
if not isinstance(metadata, dict):
    raise SystemExit("[commands] ERROR: metadata getter must return dict")
if not metadata:
    raise SystemExit("[commands] ERROR: metadata getter returned empty map")
for model_name, entry in metadata.items():
    for key in required:
        if key not in entry:
            raise SystemExit(
                f"[commands] ERROR: metadata for mass_transfer:{model_name!r} missing key {key!r}"
            )
PY
}

cmd_compatibility_validator_check() {
  "$PYTHON" - <<'PY'
from deposim_schema import compose_sim_config
from deposim_sim.validation.compatibility import validate_run_spec

baseline = compose_sim_config("cvd_steady_min")
validate_run_spec(baseline)

invalid = compose_sim_config(
    "cvd_steady_min",
    overrides=[
        "sim.roles.A=s0",
        "sim.roles.I=s0",
    ],
)
try:
    validate_run_spec(invalid)
except ValueError:
    pass
else:
    raise SystemExit("[commands] ERROR: expected validator failure for duplicated A/I role.")
PY
}

cmd_measurement_adapter_tests() { cmd_unittest_module "deposim_sim.test_measurement_adapter"; }
cmd_metrics_tests() { cmd_unittest_module "deposim_sim.test_metrics"; }
cmd_report_comparison_tests() { cmd_unittest_module "deposim_sim.test_report_comparison"; }
cmd_doe_tests() { cmd_unittest_module "deposim_sim.test_doe"; }
cmd_zref_tests() { cmd_unittest_module "deposim_sim.test_zref_sensitivity"; }
cmd_kinetics_net_tests() { cmd_unittest_module "deposim_sim.test_aib_ode"; }
cmd_phases_driver_tests() { cmd_unittest_module "deposim_sim.test_phases_driver"; }
cmd_state_closure_tests() { cmd_unittest_module "deposim_sim.test_state_closure"; }
cmd_bosanquet_pattern_tests() { cmd_unittest_module "deposim_sim.test_bosanquet_pattern"; }
cmd_identifiability_tests() { cmd_unittest_module "deposim_sim.test_identifiability"; }
cmd_jax_optional_tests() { cmd_unittest_module "deposim_sim.test_jax_optional"; }
cmd_benchmark_tests() { cmd_unittest_module "deposim_sim.test_benchmark"; }
cmd_benchmark_wafer2d_tests() { cmd_unittest_module "deposim_sim.test_benchmark_wafer2d"; }
cmd_physviz_tests() { cmd_unittest_module "deposim_sim.test_physviz"; }
cmd_fit_optuna_tests() {
  cmd_unittest_module \
    "deposim_opt.test_fit_optuna" \
    "optuna optional tests skipped unexpectedly under skip_policy=warn." \
    "warn"
}
cmd_opt_tests() {
  cmd_unittest_module "deposim_opt.test_enumerate_roles"
  cmd_unittest_module "deposim_opt.test_objective"
  cmd_fit_optuna_tests
  cmd_unittest_module "deposim_opt.test_fit_diagnostics"
}
cmd_assimilation_tests() { cmd_unittest_module "deposim_opt.test_assimilation"; }
cmd_ald_tests() { cmd_unittest_module "deposim_sim.test_ald"; }
cmd_clearml_optional_tests() { cmd_unittest_module "deposim_tracking_clearml.test_optional_clearml"; }
cmd_io_plugin_tests() { cmd_unittest_module "deposim_sim.test_io_plugins"; }
cmd_zarr_optional_tests() { cmd_unittest_module "deposim_sim.test_zarr_output"; }
cmd_multiz_tests() { cmd_unittest_module "deposim_sim.test_multiz"; }
cmd_legacy_tests() {
  cmd_unittest_module "deposim_sim.test_cvd_steady"
  cmd_unittest_module "deposim_sim.test_ald"
  cmd_unittest_module "deposim_sim.test_jax_optional"
}

cmd_p2_io_e2e_check() {
  "$PYTHON" - <<'PY'
import numpy as np
from pathlib import Path
from tempfile import TemporaryDirectory

from deposim_sim.smoke import main as smoke_main

with TemporaryDirectory(prefix="deposim_p2005_e2e_") as tmp:
    tmp_path = Path(tmp)
    fluent_path = tmp_path / "fluent.npz"
    xy = np.array([[0.0, 0.0], [20.0, 0.0], [40.0, 0.0], [60.0, 0.0]], dtype=float)
    cref = np.array(
        [
            [1.0, 0.2, 0.1, 0.0],
            [0.9, 0.2, 0.1, 0.0],
            [0.8, 0.1, 0.1, 0.0],
            [0.7, 0.1, 0.1, 0.0],
        ],
        dtype=float,
    )
    np.savez(fluent_path, xy=xy, cref=cref)
    out_dir = tmp_path / "results"
    rc = smoke_main(
        [
            "--config-name",
            "cvd_steady_min",
            f"sim.inputs.fluent.file={fluent_path}",
            "sim.time.t_proc_s=1.0",
            f"sim.output.root_dir={out_dir}",
            "sim.output.project=p2_e2e",
            "sim.output.run_name=p2005_e2e",
        ]
    )
    if rc != 0:
        raise SystemExit("[commands] ERROR: P2-005 e2e smoke run failed.")
    runs_root = out_dir / "p2_e2e" / "runs"
    runs = sorted([p for p in runs_root.iterdir() if p.is_dir()])
    if not runs:
        raise SystemExit("[commands] ERROR: P2-005 e2e produced no run directory.")
    latest = runs[-1]
    if not (latest / "report.html").exists():
        raise SystemExit("[commands] ERROR: P2-005 e2e missing report artifact.")
PY
}

cmd_p2_zarr_doe_e2e_check() {
  "$PYTHON" - <<'PY'
import numpy as np
from pathlib import Path
from tempfile import TemporaryDirectory

from deposim_sim.doe import run_doe
from deposim_sim.zarr_output import is_zarr_available

with TemporaryDirectory(prefix="deposim_p2006_e2e_") as tmp:
    tmp_path = Path(tmp)
    fluent_path = tmp_path / "fluent.npz"
    xy = np.array([[0.0, 0.0], [20.0, 0.0], [40.0, 0.0], [60.0, 0.0]], dtype=float)
    cref = np.array(
        [
            [1.0, 0.2, 0.1, 0.0],
            [0.9, 0.2, 0.1, 0.0],
            [0.8, 0.1, 0.1, 0.0],
            [0.7, 0.1, 0.1, 0.0],
        ],
        dtype=float,
    )
    np.savez(fluent_path, xy=xy, cref=cref)
    result = run_doe(
        config_name="cvd_steady_min",
        base_overrides=[
            f"sim.output.root_dir={tmp}",
            "sim.output.project=p2_e2e",
            "sim.output.store.format=zarr",
            "sim.time.t_proc_s=1.0",
            f"sim.inputs.fluent.file={fluent_path}",
        ],
        sweep={"sim.model.params.kinetics.k_rxn": [0.008, 0.012]},
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

from deposim_schema import compose_sim_config
from deposim_sim.multiz import run_multi_z_synthetic

with TemporaryDirectory(prefix="deposim_p2007_e2e_") as tmp:
    tmp_path = Path(tmp)
    fluent_path = tmp_path / "fluent.npz"
    xy = np.array([[0.0, 0.0], [20.0, 0.0], [40.0, 0.0], [60.0, 0.0]], dtype=float)
    cref = np.array(
        [
            [1.0, 0.2, 0.1, 0.0],
            [0.9, 0.2, 0.1, 0.0],
            [0.8, 0.1, 0.1, 0.0],
            [0.7, 0.1, 0.1, 0.0],
        ],
        dtype=float,
    )
    np.savez(fluent_path, xy=xy, cref=cref)

    run_spec = compose_sim_config(
        "cvd_steady_min",
        overrides=[
            f"sim.inputs.fluent.file={fluent_path}",
            "sim.reference_plane.z_ref_mm=5.0",
        ],
    )
    run_spec.reference_plane.z_ref_mm_list = [3.0, 5.0]  # type: ignore[attr-defined]
    out = run_multi_z_synthetic(run_spec)
    if int(out.diagnostics.get("plane_count", 0)) != 2:
        raise SystemExit("[commands] ERROR: P2-007 e2e expected plane_count=2.")
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
  "$PYTHON" -m unittest discover -s tests -p "test_imports.py"
  "$PYTHON" -m unittest discover -s tests -p "test_sim_config_compose.py"
  cmd_unittest_module "deposim_schema.test_sim_config_v2"
  cmd_unittest_module "deposim_sim.test_fluent_loader"
  cmd_unittest_module "deposim_sim.test_domain"
  cmd_unittest_module "deposim_sim.test_role_validator"
  cmd_unittest_module "deposim_sim.test_compatibility_validator"
  cmd_unittest_module "deposim_sim.test_aib_ode"
  cmd_unittest_module "deposim_sim.test_measurement_adapter"
  cmd_unittest_module "deposim_sim.test_metrics"
  cmd_unittest_module "deposim_sim.test_transport_provider"
  cmd_unittest_module "deposim_sim.test_pipeline_aib"
  cmd_unittest_module "deposim_sim.test_cvd_steady_aib"
  cmd_unittest_module "deposim_sim.test_doe"
  cmd_unittest_module "deposim_sim.test_zarr_output"
  cmd_unittest_module "deposim_sim.test_multiz"
  cmd_unittest_module "deposim_sim.test_benchmark"
  cmd_unittest_module "deposim_sim.test_phases_driver"
  cmd_unittest_module "deposim_sim.test_zref_sensitivity"
  cmd_unittest_module "deposim_sim.test_identifiability"
  cmd_unittest_module "deposim_sim.test_io_plugins"
  cmd_unittest_module "deposim_sim.test_run_manager"
  cmd_unittest_module "deposim_sim.test_mass_transfer"
  cmd_unittest_module "deposim_sim.test_state_closure"
  cmd_unittest_module "deposim_sim.test_bosanquet_pattern"
  cmd_unittest_module "deposim_sim.test_report_comparison"
  cmd_unittest_module "deposim_sim.test_output_contract"
  cmd_unittest_module "deposim_sim.test_benchmark_wafer2d"
  cmd_unittest_module "deposim_sim.test_physviz"
  cmd_unittest_module "deposim_opt.test_enumerate_roles"
  cmd_unittest_module "deposim_opt.test_objective"
  cmd_fit_optuna_tests
  cmd_unittest_module "deposim_opt.test_fit_diagnostics"
  cmd_unittest_module "deposim_opt.test_assimilation"
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

readonly -a P3_VERIFY_P1_TASKS=(
  P3-013 P3-014 P3-015 P3-016 P3-017 P3-022 P3-029
)

readonly -a P3_VERIFY_P2_QUICK_TASKS=(
  P3-039 P3-040 P3-046 P3-047 P3-058 P3-059 P3-060 P3-061
)

readonly -a P3_VERIFY_P2_FULL_TASKS=(
  P3-039 P3-040 P3-041 P3-042 P3-043 P3-044
  P3-032 P3-033 P3-034 P3-035 P3-036 P3-037
  P3-018 P3-019 P3-020 P3-023 P3-024 P3-025 P3-026 P3-027
  P3-030
  P3-046 P3-047 P3-048 P3-049 P3-050 P3-051 P3-052 P3-053 P3-054 P3-055 P3-056 P3-057
  P3-058 P3-059 P3-060 P3-061 P3-062 P3-063
  P3-064 P3-065 P3-066 P3-067 P3-068 P3-069
)

cmd_run_task_set() {
  local set_name="${1:-task_set}"
  shift || true
  local tids=("$@")
  if [[ "${#tids[@]}" -eq 0 ]]; then
    echo "[commands] ERROR: empty task set: $set_name" >&2
    return 2
  fi
  local tid
  for tid in "${tids[@]}"; do
    cmd_verify_task "$tid"
  done
}

cmd_verify_p1() {
  cmd_run_task_set "verify_p1" "${P3_VERIFY_P1_TASKS[@]}"
}

cmd_verify_p2_quick() {
  cmd_run_task_set "verify_p2_quick" "${P3_VERIFY_P2_QUICK_TASKS[@]}"
}

cmd_verify_p2() {
  cmd_run_task_set "verify_p2" "${P3_VERIFY_P2_FULL_TASKS[@]}"
}

cmd_verify_p3() {
  cmd_verify_milestone "P3"
}

cmd_cautorun() {
  (cd "$ROOT_DIR" && "$PYTHON" "scripts/codex_autorun.py" "$@")
}

cmd_verify_task_contracts() {
  cmd_cautorun --validate-task-contracts
}

cmd_verify_autorun() {
  cmd_cautorun --dry-run --git-check auto --lock-timeout-sec 1 --max-tasks 1
}

cmd_step_next() {
  cmd_cautorun --git-check auto --max-tasks 1
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

cmd_no_legacy_runtime_refs() {
  local legacy_refs
  legacy_refs="$(
    rg -n --no-heading \
      -e 'root_solve' \
      -e 'power_law' \
      -e 'lhhw_competition' \
      "$ROOT_DIR/src/deposim_sim" \
      --glob '!**/test_*.py' \
      --glob '!**/tests/**' \
      || true
  )"
  if [[ -n "$legacy_refs" ]]; then
    echo "[commands] ERROR: legacy runtime references remain in src/deposim_sim:" >&2
    echo "$legacy_refs" >&2
    return 1
  fi
}

cmd_no_legacy_utility_refs() {
  local utility_refs
  utility_refs="$(
    rg -n --no-heading \
      -e 'run_cvd_steady' \
      -e 'build_field_bundle' \
      "$ROOT_DIR/src/deposim_sim/doe.py" \
      "$ROOT_DIR/src/deposim_sim/physviz.py" \
      "$ROOT_DIR/src/deposim_sim/identifiability.py" \
      "$ROOT_DIR/src/deposim_sim/multiz.py" \
      "$ROOT_DIR/src/deposim_sim/benchmark.py" \
      "$ROOT_DIR/src/deposim_sim/phases_driver.py" \
      "$ROOT_DIR/src/deposim_opt/assimilate.py" \
      || true
  )"
  if [[ -n "$utility_refs" ]]; then
    echo "[commands] ERROR: legacy utility references remain in migrated AIB utility modules:" >&2
    echo "$utility_refs" >&2
    return 1
  fi
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
      test -f "$ROOT_DIR/configs/sim/cvd_steady_min.yaml"
      "$PYTHON" -c "import deposim_schema"
      rm -f /tmp/p0002_resolved.yaml
      "$PYTHON" - <<'PY'
from deposim_schema import compose_and_save_sim_config

compose_and_save_sim_config("/tmp/p0002_resolved.yaml", "cvd_steady_min")
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
      "$PYTHON" -c "from deposim_sim.models import aib_ode"
      cmd_rate_law_tests
      ;;
    P0-006)
      "$PYTHON" -c "from deposim_sim import pipeline"
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
      if [[ -d "$ROOT_DIR/results/demo/runs" ]]; then
        before_count="$(find "$ROOT_DIR/results/demo/runs" -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d ' ')"
      fi
      cmd_smoke
      after_count="$(find "$ROOT_DIR/results/demo/runs" -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d ' ')"
      if [[ "$after_count" -le "$before_count" ]]; then
        echo "[commands] ERROR: smoke did not create a new run under results/demo/runs" >&2
        return 1
      fi
      latest_run="$(find "$ROOT_DIR/results/demo/runs" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)"
      test -f "$latest_run/config_resolved.yaml"
      test -f "$latest_run/report.html"
      grep -q "^    run_name: cvd_steady_min$" "$latest_run/config_resolved.yaml"
      LATEST_RUN="$latest_run" "$PYTHON" - <<'PY'
import json
import os
from pathlib import Path

run_dir = Path(os.environ["LATEST_RUN"])
summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
artifact_paths = summary.get("artifact_paths", {})
required_keys = ("fields", "metrics")
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
      cmd_verify_task P3-001
      ;;
    P1-002)
      cmd_verify_task P3-002
      ;;
    P1-003)
      cmd_verify_task P3-002
      ;;
    P1-004)
      cmd_verify_task P3-019
      ;;
    P1-005)
      cmd_verify_task P3-020
      ;;
    P1-006)
      cmd_verify_task P3-020
      ;;
    P1-007)
      cmd_verify_task P3-014
      ;;
    P1-008)
      cmd_verify_task P3-022
      ;;
    P1-009)
      cmd_verify_task P3-003
      ;;
    P1-010)
      cmd_verify_task P3-022
      ;;
    P1-011)
      cmd_verify_task P3-027
      ;;
    P1-012)
      cmd_verify_task P3-027
      ;;
    P1-013)
      cmd_verify_task P3-016
      ;;
    P1-014)
      cmd_import_check
      ;;
    P1-015)
      cmd_verify_task P3-009
      ;;
    P2-001)
      cmd_verify_task P3-018
      ;;
    P2-002)
      cmd_verify_task P3-017
      ;;
    P2-003)
      cmd_verify_task P3-004
      ;;
    P2-004)
      cmd_import_check
      ;;
    P2-005)
      cmd_verify_task P3-025
      ;;
    P2-006)
      cmd_verify_task P3-014
      ;;
    P2-007)
      cmd_verify_task P3-022
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
    D-006)
      test -f "$ROOT_DIR/docs/adr/0008-aib-ode-replacement-policy.md"
      grep -q "AIB-ODE" "$ROOT_DIR/docs/adr/0008-aib-ode-replacement-policy.md"
      ;;
    D-007)
      grep -q "AIB-ODE" "$ROOT_DIR/docs/REQUIREMENTS.md"
      grep -q "D-007" "$ROOT_DIR/docs/TRACEABILITY.md"
      ;;
    D-008)
      test -f "$ROOT_DIR/docs/adr/0011-output-visualization-contract-v1.md"
      grep -q "output.v1" "$ROOT_DIR/docs/adr/0011-output-visualization-contract-v1.md"
      ;;
    D-009)
      test -f "$ROOT_DIR/output_viz.md"
      test -f "$ROOT_DIR/docs/adr/0012-output-viz-spec-source-lock.md"
      grep -q "output.v1" "$ROOT_DIR/output_viz.md"
      grep -q "manifest.json" "$ROOT_DIR/output_viz.md"
      ;;
    D-010)
      test -f "$ROOT_DIR/docs/adr/0013-optimize-selective-adoption.md"
      grep -q "optimize.md" "$ROOT_DIR/docs/adr/0013-optimize-selective-adoption.md"
      grep -qi "multi-condition" "$ROOT_DIR/docs/adr/0013-optimize-selective-adoption.md"
      ;;
    D-011)
      test -f "$ROOT_DIR/docs/adr/0014-improve2-selective-adoption-p0.md"
      grep -q "improve2.md" "$ROOT_DIR/docs/adr/0014-improve2-selective-adoption-p0.md"
      grep -q "AIB" "$ROOT_DIR/docs/adr/0014-improve2-selective-adoption-p0.md"
      ;;
    D-012)
      test -f "$ROOT_DIR/docs/adr/0016-runtime-dependency-bootstrap-policy.md"
      grep -qi "hydra-core" "$ROOT_DIR/docs/adr/0016-runtime-dependency-bootstrap-policy.md"
      grep -qi "omegaconf" "$ROOT_DIR/docs/adr/0016-runtime-dependency-bootstrap-policy.md"
      grep -qi "Decision task: D-012" "$ROOT_DIR/docs/adr/0016-runtime-dependency-bootstrap-policy.md"
      ;;
    D-013)
      test -f "$ROOT_DIR/docs/adr/0017-staged-refactor-policy-compact-readable-aib.md"
      grep -qi "Decision task: D-013" "$ROOT_DIR/docs/adr/0017-staged-refactor-policy-compact-readable-aib.md"
      grep -qi "code-only" "$ROOT_DIR/docs/adr/0017-staged-refactor-policy-compact-readable-aib.md"
      grep -qi "summary.json" "$ROOT_DIR/docs/adr/0017-staged-refactor-policy-compact-readable-aib.md"
      ;;
    D-014)
      test -f "$ROOT_DIR/docs/adr/0018-operational-gates-skip-policy-and-generated-files.md"
      grep -qi "Decision task: D-014" "$ROOT_DIR/docs/adr/0018-operational-gates-skip-policy-and-generated-files.md"
      grep -qi "verify_p2_quick" "$ROOT_DIR/docs/adr/0018-operational-gates-skip-policy-and-generated-files.md"
      grep -qi "skip=warn" "$ROOT_DIR/docs/adr/0018-operational-gates-skip-policy-and-generated-files.md"
      grep -qi "scripts/env.sh" "$ROOT_DIR/docs/adr/0018-operational-gates-skip-policy-and-generated-files.md"
      ;;
    P3-001)
      "$PYTHON" -c "import deposim_schema"
      cmd_unittest_module "deposim_schema.test_sim_config_v2"
      ;;
    P3-002)
      cmd_unittest_module "deposim_sim.test_fluent_loader"
      cmd_unittest_module "deposim_sim.test_role_validator"
      ;;
    P3-003)
      cmd_unittest_module "deposim_sim.test_aib_ode"
      ;;
    P3-004)
      cmd_smoke
      cmd_unittest_module "deposim_sim.test_pipeline_aib"
      cmd_unittest_module "deposim_sim.test_cvd_steady_aib"
      ;;
    P3-005)
      cmd_unittest_module "deposim_sim.test_report_comparison"
      ;;
    P3-006)
      cmd_unittest_module "deposim_opt.test_enumerate_roles"
      cmd_fit_optuna_tests
      ;;
    P3-007)
      cmd_verify_task P3-001
      cmd_verify_task P3-002
      cmd_verify_task P3-003
      cmd_verify_task P3-004
      cmd_verify_task P3-005
      cmd_verify_task P3-006
      ;;
    P3-008)
      cmd_unittest_module "deposim_sim.test_report_comparison"
      cmd_unittest_module "deposim_sim.test_pipeline_aib"
      ;;
    P3-009)
      cmd_benchmark_wafer2d
      cmd_unittest_module "deposim_sim.test_benchmark_wafer2d"
      cmd_unittest_module "deposim_sim.test_physviz"
      ;;
    P3-010)
      grep -q "AIB-ODE" "$ROOT_DIR/benchmark_cvd.md"
      grep -q "phi_B" "$ROOT_DIR/benchmark_cvd.md"
      grep -q "class_compare.csv" "$ROOT_DIR/benchmark_cvd.md"
      grep -q "ranking.csv" "$ROOT_DIR/benchmark_cvd.md"
      ;;
    P3-011)
      test ! -f "$ROOT_DIR/src/deposim_sim/solvers/root_solve.py"
      test ! -f "$ROOT_DIR/src/deposim_sim/models/rate_laws.py"
      test ! -f "$ROOT_DIR/src/deposim_sim/test_root_solve.py"
      test ! -f "$ROOT_DIR/src/deposim_sim/test_rate_laws.py"
      test ! -f "$ROOT_DIR/src/deposim_sim/test_net_models.py"
      cmd_no_legacy_runtime_refs
      cmd_import_check
      ;;
    P3-012)
      cmd_verify_task P3-008
      cmd_verify_task P3-009
      cmd_verify_task P3-010
      cmd_verify_task P3-011
      cmd_test
      cmd_verify_task_contracts
      ;;
    P3-013)
      cmd_no_legacy_utility_refs
      cmd_import_check
      ;;
    P3-014)
      cmd_unittest_module "deposim_sim.test_doe"
      ;;
    P3-015)
      cmd_benchmark_wafer2d_physviz
      cmd_unittest_module "deposim_sim.test_physviz"
      ;;
    P3-016)
      cmd_unittest_module "deposim_sim.test_identifiability"
      ;;
    P3-017)
      cmd_unittest_module "deposim_opt.test_assimilation"
      ;;
    P3-018)
      cmd_unittest_module "deposim_opt.test_enumerate_roles"
      cmd_fit_optuna_tests
      ;;
    P3-019)
      cmd_unittest_module "deposim_sim.test_measurement_adapter"
      cmd_unittest_module "deposim_sim.test_pipeline_aib"
      ;;
    P3-020)
      cmd_unittest_module "deposim_sim.test_report_comparison"
      ;;
    P3-021)
      cmd_verify_p1
      cmd_verify_p2
      cmd_verify_task_contracts
      ;;
    P3-022)
      cmd_no_legacy_utility_refs
      cmd_unittest_module "deposim_sim.test_multiz"
      cmd_unittest_module "deposim_sim.test_benchmark"
      cmd_unittest_module "deposim_sim.test_phases_driver"
      ;;
    P3-023)
      cmd_fit_optuna_tests
      ;;
    P3-024)
      cmd_unittest_module "deposim_sim.test_report_comparison"
      cmd_benchmark_wafer2d_physviz
      ;;
    P3-025)
      cmd_unittest_module "deposim_sim.test_io_plugins"
      ;;
    P3-026)
      cmd_unittest_module "deposim_sim.test_run_manager"
      ;;
    P3-027)
      cmd_unittest_module "deposim_sim.test_mass_transfer"
      cmd_unittest_module "deposim_sim.test_state_closure"
      cmd_unittest_module "deposim_sim.test_bosanquet_pattern"
      ;;
    P3-028)
      cmd_verify_p1
      cmd_verify_p2
      cmd_verify_task_contracts
      ;;
    P3-029)
      cmd_unittest_module "deposim_sim.test_doe"
      cmd_unittest_module "deposim_sim.test_benchmark_wafer2d"
      cmd_unittest_module "deposim_sim.test_identifiability"
      cmd_unittest_module "deposim_sim.test_report_comparison"
      ;;
    P3-030)
      cmd_import_check
      "$PYTHON" - <<'PY'
from deposim_schema import compose_sim_config, compose_opt_config

sim = compose_sim_config("smoke")
if sim.model.name != "aib_ode":
    raise SystemExit("[commands] ERROR: smoke alias must resolve to AIB config.")
opt = compose_opt_config("stub")
if opt.sim.model.name != "aib_ode":
    raise SystemExit("[commands] ERROR: stub alias must resolve to AIB opt config.")
PY
      test ! -d "$ROOT_DIR/codex_handoff_pack (5)"
      if [[ -d "$ROOT_DIR/src/deposim.egg-info" ]]; then
        if git -C "$ROOT_DIR" ls-files --error-unmatch "src/deposim.egg-info/*" >/dev/null 2>&1; then
          echo "[commands] ERROR: tracked src/deposim.egg-info artifacts detected." >&2
          return 1
        fi
        echo "[commands] WARN: local generated src/deposim.egg-info detected (untracked)." >&2
      fi
      ;;
    P3-031)
      cmd_verify_task P0-009
      cmd_verify_p1
      cmd_verify_p2
      cmd_verify_task_contracts
      ;;
    P3-032)
      cmd_unittest_module "deposim_sim.test_run_manager"
      cmd_fit_optuna_tests
      ;;
    P3-033)
      cmd_unittest_module "deposim_sim.test_report_comparison"
      cmd_benchmark_wafer2d_physviz
      ;;
    P3-034)
      cmd_unittest_module "deposim_sim.test_aib_ode"
      cmd_unittest_module "deposim_sim.test_pipeline_aib"
      ;;
    P3-035)
      cmd_unittest_module "deposim_sim.test_doe"
      cmd_unittest_module "deposim_sim.test_benchmark_wafer2d"
      cmd_fit_optuna_tests
      ;;
    P3-036)
      cmd_unittest_module "deposim_sim.test_run_manager"
      cmd_smoke
      ;;
    P3-037)
      cmd_unittest_module "deposim_sim.test_output_contract"
      cmd_unittest_module "deposim_sim.test_report_comparison"
      cmd_unittest_module "deposim_sim.test_benchmark_wafer2d"
      ;;
    P3-038)
      cmd_verify_task_contracts
      cmd_verify_p1
      cmd_verify_p2
      ;;
    P3-039)
      cmd_unittest_module "deposim_sim.test_output_contract"
      ;;
    P3-040)
      cmd_unittest_module "deposim_sim.test_report_comparison"
      cmd_unittest_module "deposim_sim.test_run_manager"
      ;;
    P3-041)
      cmd_unittest_module "deposim_sim.test_report_comparison"
      cmd_unittest_module "deposim_sim.test_physviz"
      ;;
    P3-042)
      cmd_unittest_module "deposim_sim.test_doe"
      cmd_unittest_module "deposim_sim.test_benchmark_wafer2d"
      cmd_fit_optuna_tests
      ;;
    P3-043)
      cmd_unittest_module "deposim_sim.test_run_manager"
      ;;
    P3-044)
      cmd_unittest_module "deposim_sim.test_output_contract"
      cmd_unittest_module "deposim_sim.test_report_comparison"
      cmd_unittest_module "deposim_sim.test_benchmark_wafer2d"
      cmd_fit_optuna_tests
      ;;
    P3-045)
      cmd_verify_task_contracts
      cmd_verify_task P3-044
      ;;
    P3-046)
      cmd_unittest_module "deposim_schema.test_sim_config_v2"
      ;;
    P3-047)
      cmd_unittest_module "deposim_opt.test_objective"
      ;;
    P3-048)
      cmd_fit_optuna_tests
      ;;
    P3-049)
      cmd_fit_optuna_tests
      ;;
    P3-050)
      cmd_fit_optuna_tests
      ;;
    P3-051)
      cmd_unittest_module "deposim_opt.test_objective"
      cmd_fit_optuna_tests
      cmd_unittest_module "deposim_schema.test_sim_config_v2"
      ;;
    P3-052)
      cmd_verify_task_contracts
      cmd_verify_task P3-051
      ;;
    P3-053)
      cmd_fit_optuna_tests
      ;;
    P3-054)
      cmd_fit_optuna_tests
      ;;
    P3-055)
      cmd_fit_optuna_tests
      cmd_unittest_module "deposim_sim.test_report_comparison"
      ;;
    P3-056)
      cmd_unittest_module "deposim_schema.test_sim_config_v2"
      cmd_unittest_module "deposim_opt.test_fit_diagnostics"
      ;;
    P3-057)
      cmd_verify_task_contracts
      cmd_verify_task P3-056
      ;;
    P3-058)
      cmd_unittest_module "deposim_sim.test_run_manager"
      cmd_unittest_module "deposim_sim.test_doe"
      cmd_fit_optuna_tests
      cmd_unittest_module "deposim_sim.test_output_contract"
      ;;
    P3-059)
      cmd_unittest_module "deposim_sim.test_benchmark_wafer2d"
      cmd_unittest_module "deposim_sim.test_output_contract"
      ;;
    P3-060)
      cmd_unittest_module "deposim_sim.test_physviz"
      cmd_unittest_module "deposim_sim.test_benchmark_wafer2d"
      ;;
    P3-061)
      cmd_unittest_module "deposim_sim.test_identifiability"
      cmd_unittest_module "deposim_sim.test_benchmark_wafer2d"
      ;;
    P3-062)
      cmd_import_check
      cmd_unittest_module "deposim_sim.test_benchmark_wafer2d"
      cmd_fit_optuna_tests
      ;;
    P3-063)
      cmd_verify_task_contracts
      cmd_verify_task P3-062
      ;;
    P3-064)
      grep -q "cmd_verify_p2_quick" "$ROOT_DIR/scripts/commands.sh"
      cmd_verify_p2_quick
      ;;
    P3-065)
      grep -q "skip_policy" "$ROOT_DIR/scripts/commands.sh"
      cmd_fit_optuna_tests
      ;;
    P3-066)
      grep -q "^\\.wslbin/$" "$ROOT_DIR/.gitignore"
      grep -q "^scripts/env\\.sh$" "$ROOT_DIR/.gitignore"
      cmd_verify_task P3-030
      ;;
    P3-067)
      cmd_unittest_module "deposim_sim.test_benchmark_wafer2d"
      cmd_unittest_module "deposim_sim.test_physviz"
      ;;
    P3-068)
      cmd_unittest_module "deposim_sim.test_output_contract"
      cmd_unittest_module "deposim_sim.test_run_manager"
      cmd_unittest_module "deposim_sim.test_doe"
      cmd_fit_optuna_tests
      ;;
    P3-069)
      cmd_verify_task_contracts
      cmd_verify_p2_quick
      cmd_benchmark_wafer2d_flux_km --with-physviz
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
  benchmark_wafer2d Run wafer-2D AIB benchmark (A/AI/AB/AIB classes)
  benchmark_wafer2d_physviz Run wafer-2D AIB benchmark with physviz outputs
  benchmark_wafer2d_flux_km Run wafer-2D benchmark with free-km vs flux-km comparison
  legacy_tests   Run isolated legacy test modules (non-gating)
  test           Run unit tests (stdlib unittest)
  verify_p0      import_check + smoke + test
  verify_p1      Run AIB utility migration gates (P3-013..P3-017, P3-022, P3-029)
  verify_p2_quick Run daily quick contract/output checks (P3-039, P3-040, P3-046, P3-047, P3-058..P3-061)
  verify_p2      Run full AIB contract/output + optimization/refactor gates (P3-018..P3-020, P3-023..P3-027, P3-030, P3-032..P3-037, P3-046..P3-069)
  verify_p3      Run verification gates for P3 tasks
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
  benchmark_wafer2d_flux_km) shift; cmd_benchmark_wafer2d_flux_km "$@";;
  legacy_tests) shift; cmd_legacy_tests "$@";;
  test) shift; cmd_test "$@";;
  verify_p0) shift; cmd_verify_p0 "$@";;
  verify_p1) shift; cmd_verify_p1 "$@";;
  verify_p2_quick) shift; cmd_verify_p2_quick "$@";;
  verify_p2) shift; cmd_verify_p2 "$@";;
  verify_p3) shift; cmd_verify_p3 "$@";;
  step_next) shift; cmd_step_next "$@";;
  verify_autorun) shift; cmd_verify_autorun "$@";;
  verify_task_contracts) shift; cmd_verify_task_contracts "$@";;
  verify_task) shift; cmd_verify_task "$@";;
  reconcile_state) shift; cmd_reconcile_state "$@";;
  *) usage; exit 2;;
esac
