# CVD/ALD reaction-role assimilation

This repository maps anonymous Fluent species fields to interpretable reaction roles,
fits role-based surface models to measured film maps, and reports when a numerical fit
does not justify a chemical interpretation.

The production path is:

```text
Fluent raw species
-> candidate A / B / I assignments
-> CVD or ALD role model
-> measured thickness or deposition-rate fit
-> condition transfer and spatial validation
-> adopt / review / reject with evidence gaps
```

Raw names such as `s0`, `adn_2`, or `n2` are input labels. The software does not treat
them as known precursors, co-reactants, or inhibitors until the data distinguish those
roles.

## Current scope

- Steady CVD equation census
  - sequential AIB quasi-steady response
  - parallel A and A+B quasi-steady response
  - two-adsorbate Langmuir-Hinshelwood response
  - exact reductions such as no-inhibitor and no-finite-loss forms
- Dynamic process models
  - continuous CVD AIB surface coverage
  - Mars-van Krevelen redox reservoir
  - observation-time MvK state, pathway-rate, surface-concentration, and flux histories
  - ALD storage, conversion, and inhibitor states
- Transport closures
  - supplied wall concentration
  - fitted scalar or field mass-transfer coefficient
  - CFD transport-capacity flux converted to a mass-transfer coefficient
- Validation
  - condition-balanced fitting
  - leave-one-condition-out model selection
  - fixed-condition prediction without refitting inside the run
  - angular and radial blocked diagnostics
  - role, reduction, and equation-family stability
  - target-use readiness and the additional measurements needed to establish it

## Main commands

List the implemented model inventory:

```powershell
uv run python scripts/analyze_cvd_multicond_case.py --list-models
```

Run the current steady CVD evaluation:

```powershell
uv run python scripts/analyze_cvd_multicond_case.py `
  --data-dir data `
  --train-cases 1 2 4 5 `
  --test-case 3 `
  --response-model surface_compare `
  --reaction-input bulk_concentration `
  --models all `
  --loss mse `
  --sampler pattern `
  --bootstrap-samples 100 `
  --spatial-response radial_quartic `
  --seed 123 `
  --output results/current_cvd_separated
```

Choose the local reaction input and an optional post-selection spatial response
explicitly:

```powershell
uv run python scripts/analyze_cvd_multicond_case.py `
  --data-dir data `
  --train-cases 1 2 4 5 `
  --test-case 3 `
  --reaction-input bulk_concentration `
  --spatial-response radial_quartic `
  --wafer-temperature-k 773.15 `
  --output results/cvd_separated_response
```

`surface_concentration` and `transport_capacity_flux` are accepted when every condition
contains their documented per-species columns. Input location is fixed before chemical
candidate enumeration. The spatial response is fitted afterward and is reported
separately, so it cannot change the selected roles or reaction equation.
Each run also writes compact figures for equation-family convergence, species-to-role
assignment, assignment stability, input correlation, fitted role sensitivity, reaction
state/path fractions, reaction-step diagrams, alternative-model prediction differences,
kinetic-parameter sensitivity, and spatial residuals before and after optional correction.

Select a whole-wafer Loss and one surface sampler explicitly:

```powershell
uv run python scripts/analyze_cvd_multicond_case.py `
  --data-dir data `
  --train-cases 1 2 4 5 `
  --test-case 3 `
  --models all `
  --loss wafer_normalized_mse `
  --sampler pattern `
  --output results/cvd_wnmse
```

Compare Loss and sampler choices on one frozen role equation:

```powershell
uv sync --extra optuna
uv run python scripts/benchmark_surface_optimization.py `
  --candidate-id "<model_id from role_ranking.csv>" `
  --trials 4096 `
  --repetitions 3 `
  --workers 8 `
  --output results/surface_optimization_benchmark_4096
```

The benchmark writes `benchmark_report.md`, the complete run and condition-fold CSVs,
and two compact Loss-by-sampler heatmaps. Ranking uses training-condition transfer;
the fixed test condition appears only as a post-selection audit. Long runs write partial
CSV checkpoints; rerun the same command with `--resume` to skip completed combinations.
`mse` is squared error in the linear deposition-rate unit. Positive kinetic shape
parameters are sampled in log coordinates because their declared ranges span decades.

Run the repository verification entry points from a Bash environment:

```bash
./scripts/commands.sh smoke cvd
./scripts/commands.sh smoke ald
./scripts/commands.sh verify
```

The production fit configurations use Optuna TPE. Install the optional optimizer
backend before `fit`:

```bash
uv sync --extra optuna
bash scripts/preflight.sh
# or: python -m pip install -e '.[optuna]'
./scripts/commands.sh fit cvd
./scripts/commands.sh fit ald
```

`random`, `tpe`, `cmaes`, differential evolution (`de`), particle swarm (`pso`),
Lévy flight (`levy`), and CMA-MAE (`cma_mae`) use the same sampler boundary. The four
OptunaHub methods are optional and pinned to a reviewed registry revision. Missing
backends fail with an installation instruction; a requested sampler is never silently
replaced. CMA-MAE explores diverse response regimes and should be benchmarked before it
is used as a minimum-Loss optimizer.

Generated simulator inputs belong under `runs/generated_inputs/`; run outputs belong
under `results/`. Neither directory is a source-data store.

## Documentation

- [Technical report](docs/TECHNICAL_REPORT.md): integrated scientific account of the
  model basis, numerical method, software design, and current-data conclusions.
- [Theory and equations](docs/THEORY.md): derivations, approximations, units, model
  comparison, and literature.
- [Evaluation workflow](docs/EVALUATION_WORKFLOW.md): input-dependent execution and
  decision flow, metrics, and numerical procedures.
- [Visualization guide](docs/VISUALIZATION_GUIDE.md): examples of every steady CVD
  figure with axis definitions, source artifacts, and interpretation rules.
- [Architecture](docs/ARCHITECTURE.md): package boundaries, registries, responsibilities,
  and extension rules.
- [Current-data evaluation](docs/CURRENT_DATA_EVALUATION.md): reproducible results for
  the five supplied CVD conditions.
- [Fluent inputs](docs/inputs_fluent.md) and [transport policy](docs/transport_km.md):
  field semantics and transport assumptions.
- [Known gaps](docs/GAPS.md): current code limits and measurements needed to resolve
  them.

For every evaluated dataset, the analysis writes `data_requirements.csv`. When the
evidence is insufficient for wafer spatial correction, anonymous-species role
assignment, or elementary-rate inference, this file states the measurements and
experimental contrasts that would make each use assessable. The supplied five-condition
dataset is one worked example; these requirements are derived from evidence status and
are not tied to its species names.
