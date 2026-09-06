"""Fair loss-by-sampler evaluation for one fixed surface-role equation."""

from __future__ import annotations

import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

import numpy as np

from deposim_sim.models.aib_reductions import (
    SurfaceKineticCandidate,
    available_surface_model_families,
    enumerate_surface_kinetic_candidates,
)
from .cvd_conditions import combine_cases, condition_paths, load_case
from .metrics import prediction_metrics
from .surface_fit import (
    SurfaceOptimizationSettings,
    fit_surface_kinetic,
    predict_surface_kinetic,
)


DEFAULT_LOSSES = (
    "mse",
    "wafer_normalized_mse",
    "wafer_normalized_mae",
    "symmetric_normalized_mse",
)
DEFAULT_SAMPLERS = ("pattern", "tpe", "cmaes", "de", "pso", "levy", "cma_mae")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _candidate_for_data(candidate_id: str, data: Any) -> SurfaceKineticCandidate:
    candidates = enumerate_surface_kinetic_candidates(
        data.species,
        include_boundaries=True,
        families=available_surface_model_families(data.available_inputs()),
        available_inputs=data.available_inputs(),
        transport_modes=data.available_transport_modes(),
    )
    matched = [candidate for candidate in candidates if candidate.model_id == candidate_id]
    if not matched:
        raise ValueError(f"Candidate {candidate_id!r} is not applicable to the supplied data")
    return matched[0]


def _fit_and_score(
    candidate: SurfaceKineticCandidate,
    train: Any,
    test: Any,
    settings: SurfaceOptimizationSettings,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    started = perf_counter()
    all_indices = np.arange(train.rate.size, dtype=int)
    fitted = fit_surface_kinetic(
        candidate, train, all_indices, optimization=settings
    )
    test_prediction, _ = predict_surface_kinetic(fitted, test)
    train_metrics = prediction_metrics(train.rate, fitted.prediction)
    test_metrics = prediction_metrics(test.rate, test_prediction)

    def condition_centered_metrics(prediction: np.ndarray) -> dict[str, float]:
        observed_centered = np.empty(train.rate.shape, dtype=float)
        predicted_centered = np.empty(train.rate.shape, dtype=float)
        for condition in train.case_ids:
            mask = train.condition_id == condition
            observed_centered[mask] = train.rate[mask] - float(np.mean(train.rate[mask]))
            predicted_centered[mask] = prediction[mask] - float(np.mean(prediction[mask]))
        return prediction_metrics(observed_centered, predicted_centered)

    cv_prediction = np.full(train.rate.shape, np.nan, dtype=float)
    condition_rows: list[dict[str, Any]] = []
    for held_out in train.case_ids:
        fit_indices = all_indices[train.condition_id != held_out]
        valid_indices = all_indices[train.condition_id == held_out]
        fold_fit = fit_surface_kinetic(
            candidate, train, fit_indices, optimization=settings
        )
        cv_prediction[valid_indices] = fold_fit.prediction[valid_indices]
        fold_metrics = prediction_metrics(
            train.rate[valid_indices], cv_prediction[valid_indices]
        )
        condition_rows.append(
            {
                "loss": settings.loss_name,
                "sampler": settings.sampler,
                "seed": settings.seed,
                "held_out_condition": int(held_out),
                "rmse_nm_s": fold_metrics["rmse"],
                "relative_rmse": fold_metrics["relative_rmse"],
                "centered_r2": fold_metrics["centered_r2"],
                "spatial_correlation": fold_metrics["spatial_correlation"],
            }
        )
    cv_metrics = condition_centered_metrics(cv_prediction)
    condition_cv_rmse = float(
        np.sqrt(np.mean([row["rmse_nm_s"] ** 2 for row in condition_rows]))
    )
    result: dict[str, Any] = {
        "loss": settings.loss_name,
        "sampler": settings.sampler,
        "seed": settings.seed,
        "success": True,
        "objective_value": fitted.objective_value,
        "train_rmse_nm_s": train_metrics["rmse"],
        "condition_cv_rmse_nm_s": condition_cv_rmse,
        "condition_cv_centered_r2": cv_metrics["centered_r2"],
        "condition_cv_spatial_correlation": cv_metrics["spatial_correlation"],
        "test_rmse_nm_s": test_metrics["rmse"],
        "test_relative_rmse": test_metrics["relative_rmse"],
        "test_centered_r2": test_metrics["centered_r2"],
        "test_spatial_correlation": test_metrics["spatial_correlation"],
        "rate_scale_nm_s": fitted.rate_scale_nm_s,
        "shape_parameters": json.dumps(fitted.shape_parameters, sort_keys=True),
        "boundary_parameters": "|".join(fitted.boundary_parameters),
        "trial_count_per_fit": fitted.optimizer_trial_count,
        "elapsed_s": perf_counter() - started,
    }
    return result, condition_rows


def _benchmark_task(
    candidate: SurfaceKineticCandidate,
    train: Any,
    test: Any,
    settings: SurfaceOptimizationSettings,
    repetition: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    row, condition_rows = _fit_and_score(candidate, train, test, settings)
    row["repetition"] = repetition
    for condition_row in condition_rows:
        condition_row["repetition"] = repetition
    return row, condition_rows


def _aggregate(
    rows: list[dict[str, Any]], baseline: dict[str, Any]
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row["loss"]), str(row["sampler"])), []).append(row)
    summary: list[dict[str, Any]] = []
    for (loss, sampler), members in groups.items():
        def values(name: str) -> np.ndarray:
            return np.asarray([float(row[name]) for row in members], dtype=float)

        cv = values("condition_cv_rmse_nm_s")
        test = values("test_rmse_nm_s")
        summary.append(
            {
                "loss": loss,
                "sampler": sampler,
                "repetitions": len(members),
                "condition_cv_rmse_median_nm_s": float(np.median(cv)),
                "condition_cv_rmse_min_nm_s": float(np.min(cv)),
                "condition_cv_rmse_max_nm_s": float(np.max(cv)),
                "condition_cv_delta_vs_current_nm_s": float(
                    np.median(cv) - float(baseline["condition_cv_rmse_nm_s"])
                ),
                "test_rmse_median_nm_s": float(np.median(test)),
                "test_rmse_min_nm_s": float(np.min(test)),
                "test_rmse_max_nm_s": float(np.max(test)),
                "test_delta_vs_current_nm_s": float(
                    np.median(test) - float(baseline["test_rmse_nm_s"])
                ),
                "test_centered_r2_median": float(
                    np.median(values("test_centered_r2"))
                ),
                "test_spatial_correlation_median": float(
                    np.median(values("test_spatial_correlation"))
                ),
                "elapsed_median_s": float(np.median(values("elapsed_s"))),
                "boundary_run_fraction": float(
                    np.mean([bool(row["boundary_parameters"]) for row in members])
                ),
            }
        )
    summary.sort(
        key=lambda row: (
            float(row["condition_cv_rmse_median_nm_s"]),
            float(row["condition_cv_rmse_max_nm_s"]),
        )
    )
    for rank, row in enumerate(summary, start=1):
        row["training_selection_rank"] = rank
    return summary


def _plot_matrix(
    rows: list[dict[str, Any]],
    *,
    losses: tuple[str, ...],
    samplers: tuple[str, ...],
    value: str,
    label: str,
    path: Path,
) -> None:
    import matplotlib.pyplot as plt

    lookup = {(row["loss"], row["sampler"]): float(row[value]) for row in rows}
    matrix = np.asarray(
        [[lookup[(loss, sampler)] for sampler in samplers] for loss in losses],
        dtype=float,
    )
    sampler_labels = {
        "pattern": "Pattern",
        "tpe": "TPE",
        "cmaes": "CMA-ES",
        "de": "DE",
        "pso": "PSO",
        "levy": "Lévy",
        "cma_mae": "CMA-MAE",
    }
    loss_labels = {
        "mse": "Linear MSE",
        "wafer_normalized_mse": "Wafer-normalized MSE",
        "wafer_normalized_mae": "Wafer-normalized MAE",
        "symmetric_normalized_mse": "Symmetric normalized MSE",
    }
    figure, axis = plt.subplots(figsize=(8.2, 4.0), constrained_layout=True)
    image = axis.imshow(matrix, aspect="auto", cmap="viridis_r")
    axis.set_xticks(
        range(len(samplers)), [sampler_labels.get(name, name) for name in samplers]
    )
    axis.set_yticks(
        range(len(losses)), [loss_labels.get(name, name.replace("_", " ")) for name in losses]
    )
    axis.set_xlabel("Sampler")
    axis.set_ylabel("Loss")
    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label(label)
    figure.savefig(path, dpi=220)
    plt.close(figure)


def _write_benchmark_report(
    path: Path,
    *,
    candidate_id: str,
    summary_rows: list[dict[str, Any]],
    baseline: dict[str, Any],
    trials: int,
    repetitions: int,
    fixed_test_case_id: int,
    workers: int,
) -> None:
    lines = [
        "# Loss and sampler benchmark",
        "",
        f"Fixed equation: `{candidate_id}`.",
        "",
        (
            "Combinations are ranked by the median leave-one-identification-condition-out "
            "RMSE. The fixed test condition is evaluated after ranking and is not used "
            "for selection."
        ),
        "",
        (
            f"Stochastic fits use {trials:,} evaluations and {repetitions} seeds. "
            f"The current pattern/MSE reference uses "
            f"{int(baseline['trial_count_per_fit']):,} evaluations per full fit. "
            f"Runs used {workers} worker(s). Elapsed time covers one full fit and all "
            "condition-refits and may include concurrent CPU contention."
        ),
        "",
        "|Rank|Loss|Sampler|Condition CV RMSE (nm/s)|Seed range|Fixed test RMSE (nm/s)|Test centered R2|Median time (s)|",
        "|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    report_loss_labels = {
        "mse": "Linear MSE",
        "wafer_normalized_mse": "Wafer-normalized MSE",
        "wafer_normalized_mae": "Wafer-normalized MAE",
        "symmetric_normalized_mse": "Symmetric normalized MSE",
    }
    for row in summary_rows:
        loss_label = report_loss_labels.get(str(row["loss"]), str(row["loss"]))
        lines.append(
            "|{rank}|{loss}|{sampler}|{cv:.6g}|{lo:.6g}–{hi:.6g}|{test:.6g}|{r2:.4f}|{elapsed:.3f}|".format(
                rank=int(row["training_selection_rank"]),
                loss=loss_label,
                sampler=str(row["sampler"]),
                cv=float(row["condition_cv_rmse_median_nm_s"]),
                lo=float(row["condition_cv_rmse_min_nm_s"]),
                hi=float(row["condition_cv_rmse_max_nm_s"]),
                test=float(row["test_rmse_median_nm_s"]),
                r2=float(row["test_centered_r2_median"]),
                elapsed=float(row["elapsed_median_s"]),
            )
        )
    lines.extend(
        [
            "",
            f"The fixed test condition is condition {fixed_test_case_id}. "
            "A lower transfer RMSE does not by itself establish wafer-pattern recovery; "
            "the centered R2 column assesses that separate question.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def benchmark_surface_optimization(
    *,
    data_dir: Path,
    output_dir: Path,
    candidate_id: str,
    train_case_ids: tuple[int, ...] = (1, 2, 4, 5),
    test_case_id: int = 3,
    losses: Iterable[str] = DEFAULT_LOSSES,
    samplers: Iterable[str] = DEFAULT_SAMPLERS,
    trials: int = 256,
    repetitions: int = 3,
    seed: int = 123,
    conditions_file: Path | None = None,
    edge_uncertainty_ratio: float = 1.0,
    radial_uncertainty_power: float = 2.0,
    workers: int = 1,
    resume: bool = False,
) -> dict[str, Any]:
    """Benchmark optimizers without using the fixed test condition for selection."""

    loss_names = tuple(dict.fromkeys(str(name) for name in losses))
    sampler_names = tuple(dict.fromkeys(str(name) for name in samplers))
    if repetitions < 1:
        raise ValueError("repetitions must be >= 1")
    if workers < 1:
        raise ValueError("workers must be >= 1")
    case_ids = tuple(sorted({*train_case_ids, int(test_case_id)}))
    paths = condition_paths(Path(data_dir), case_ids, conditions_file)
    cases = {case_id: load_case(case_id, *paths[case_id]) for case_id in case_ids}
    train = combine_cases(cases[case_id] for case_id in train_case_ids)
    test = combine_cases([cases[test_case_id]])
    candidate = _candidate_for_data(candidate_id, train)

    baseline_settings = SurfaceOptimizationSettings(
        loss_name="mse",
        sampler="pattern",
        seed=seed,
        edge_uncertainty_ratio=edge_uncertainty_ratio,
        radial_power=radial_uncertainty_power,
    )
    baseline, baseline_conditions = _fit_and_score(
        candidate, train, test, baseline_settings
    )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    run_checkpoint = output / "combination_runs.partial.csv"
    condition_checkpoint = output / "condition_holdout_runs.partial.csv"
    if resume:
        rows = _read_csv(run_checkpoint)
        task_condition_rows = _read_csv(condition_checkpoint)
    else:
        rows = []
        task_condition_rows = []
        run_checkpoint.unlink(missing_ok=True)
        condition_checkpoint.unlink(missing_ok=True)
    completed = {
        (str(row["loss"]), str(row["sampler"]), int(row["repetition"]))
        for row in rows
    }
    tasks: list[tuple[SurfaceOptimizationSettings, int]] = []
    for loss in loss_names:
        for sampler in sampler_names:
            for repetition in range(repetitions):
                repetition_number = repetition + 1
                if (loss, sampler, repetition_number) in completed:
                    continue
                run_seed = int(seed + 104729 * repetition)
                settings = SurfaceOptimizationSettings(
                    loss_name=loss,
                    sampler=sampler,
                    trials=trials,
                    seed=run_seed,
                    edge_uncertainty_ratio=edge_uncertainty_ratio,
                    radial_power=radial_uncertainty_power,
                )
                tasks.append((settings, repetition_number))

    def save_checkpoint(
        row: dict[str, Any], per_condition: list[dict[str, Any]]
    ) -> None:
        rows.append(row)
        task_condition_rows.extend(per_condition)
        rows.sort(key=lambda item: (str(item["loss"]), str(item["sampler"]), int(item["repetition"])))
        task_condition_rows.sort(
            key=lambda item: (
                str(item["loss"]),
                str(item["sampler"]),
                int(item["repetition"]),
                int(item["held_out_condition"]),
            )
        )
        _write_csv(run_checkpoint, rows)
        _write_csv(condition_checkpoint, task_condition_rows)

    if workers == 1:
        for settings, repetition_number in tasks:
            save_checkpoint(
                *_benchmark_task(
                    candidate, train, test, settings, repetition_number
                )
            )
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _benchmark_task,
                    candidate,
                    train,
                    test,
                    settings,
                    repetition_number,
                ): (settings.loss_name, settings.sampler, repetition_number)
                for settings, repetition_number in tasks
            }
            for future in as_completed(futures):
                save_checkpoint(*future.result())

    summary_rows = _aggregate(rows, baseline)
    condition_rows = [*baseline_conditions, *task_condition_rows]
    baseline_row = {**baseline, "repetition": 1}
    _write_csv(output / "combination_runs.csv", [baseline_row, *rows])
    _write_csv(output / "condition_holdout_runs.csv", condition_rows)
    _write_csv(output / "combination_summary.csv", summary_rows)
    _plot_matrix(
        summary_rows,
        losses=loss_names,
        samplers=sampler_names,
        value="condition_cv_rmse_median_nm_s",
        label="Condition CV RMSE (nm/s)",
        path=output / "condition_cv_rmse.png",
    )
    _plot_matrix(
        summary_rows,
        losses=loss_names,
        samplers=sampler_names,
        value="test_rmse_median_nm_s",
        label="Fixed test RMSE (nm/s)",
        path=output / "fixed_test_rmse.png",
    )
    _write_benchmark_report(
        output / "benchmark_report.md",
        candidate_id=candidate_id,
        summary_rows=summary_rows,
        baseline=baseline,
        trials=trials,
        repetitions=repetitions,
        fixed_test_case_id=test_case_id,
        workers=workers,
    )
    payload = {
        "candidate_id": candidate_id,
        "train_case_ids": list(train_case_ids),
        "fixed_test_case_id": test_case_id,
        "losses": list(loss_names),
        "samplers": list(sampler_names),
        "stochastic_trials_per_fit": trials,
        "current_pattern_evaluations_per_full_fit": baseline["trial_count_per_fit"],
        "repetitions": repetitions,
        "workers": workers,
        "selection_metric": "median leave-one-identification-condition-out RMSE",
        "fixed_test_used_for_selection": False,
        "current_baseline": baseline,
        "training_selected_combination": summary_rows[0],
        "combination_count": len(summary_rows),
    }
    (output / "benchmark_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    run_checkpoint.unlink(missing_ok=True)
    condition_checkpoint.unlink(missing_ok=True)
    return payload


__all__ = [
    "DEFAULT_LOSSES",
    "DEFAULT_SAMPLERS",
    "benchmark_surface_optimization",
]
