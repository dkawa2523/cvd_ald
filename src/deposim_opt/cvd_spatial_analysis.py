"""Interpretable single-condition CVD spatial-response analysis.

This module intentionally estimates effective spatial-response coefficients,
not elementary kinetic constants.  With one wafer map, absolute reaction
contributions and causal reaction orders are not identifiable.  The centered
role models below attribute only the measured spatial variation around the
median concentration state.
"""

from __future__ import annotations

from dataclasses import dataclass
import csv
from hashlib import sha256
from io import StringIO
import json
import math
from pathlib import Path
from contextlib import redirect_stdout
from datetime import datetime, timezone
from typing import Any, Iterable

import numpy as np


_EPS = 1.0e-30


@dataclass(frozen=True)
class RoleResponseCandidate:
    model_id: str
    class_id: str
    A: str | None = None
    I: str | None = None
    B: str | None = None

    @property
    def effect_count(self) -> int:
        return int(self.A is not None) + int(self.I is not None) + int(self.B is not None)

    @property
    def effect_groups(self) -> dict[str, list[str]]:
        """Terms in design-column order; the product has no A/B direction."""
        groups = {}
        if self.A is not None:
            groups["A" if self.B is None else "AB"] = (
                [self.A] if self.B is None else sorted([self.A, self.B]))
        if self.I is not None:
            groups["I"] = [self.I]
        return groups

    def reductions(self) -> list[RoleResponseCandidate]:
        """Allowed term deletions, fixed before seeing any measured response."""
        out = []
        if self.I is not None:
            if self.A is None:
                out.append(RoleResponseCandidate("baseline", "baseline"))
            elif self.B is None:
                out.append(RoleResponseCandidate(f"A:{self.A}", "A", A=self.A))
            else:
                out.append(RoleResponseCandidate(f"AB:{self.A}|{self.B}", "AB", A=self.A, B=self.B))
        if self.A is not None:
            # Retain an inhibitory response without inventing a film-forming A.
            out.append(RoleResponseCandidate(f"I_response:{self.I}", "unassigned_driver", I=self.I)
                       if self.I is not None else RoleResponseCandidate("baseline", "baseline"))
        return out


@dataclass(frozen=True)
class FitResult:
    coefficients: np.ndarray
    prediction: np.ndarray
    design: np.ndarray
    effect_names: tuple[str, ...]
    references: dict[str, float]
    active_effects: tuple[bool, ...]


def _read_numeric_csv(path: Path) -> tuple[list[str], dict[str, np.ndarray]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = list(reader.fieldnames or [])
        if not headers:
            raise ValueError(f"CSV has no header: {path}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"CSV has no data rows: {path}")

    columns: dict[str, np.ndarray] = {}
    for name in headers:
        values: list[float] = []
        for row_index, row in enumerate(rows, start=2):
            raw = str(row.get(name, "")).strip()
            if raw == "":
                values.append(float("nan"))
                continue
            try:
                values.append(float(raw))
            except ValueError as exc:
                raise ValueError(f"Non-numeric value in {path}:{row_index}, column {name!r}: {raw!r}") from exc
        columns[name] = np.asarray(values, dtype=float)
    return headers, columns


def _coordinate_matrix(columns: dict[str, np.ndarray]) -> np.ndarray:
    missing = [name for name in ("x", "y", "z") if name not in columns]
    if missing:
        raise ValueError(f"Missing coordinate columns: {missing}")
    return np.column_stack([columns["x"], columns["y"], columns["z"]])


def _coordinate_keys(xyz: np.ndarray) -> list[tuple[float, float, float]]:
    return [tuple(float(value) for value in row) for row in np.asarray(xyz, dtype=float)]


def _align_validation(
    condition_xyz: np.ndarray,
    validation_xyz: np.ndarray,
    validation_rate: np.ndarray,
    *,
    coordinate_decimals: int | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    condition_match_xyz = (
        np.round(condition_xyz, decimals=coordinate_decimals)
        if coordinate_decimals is not None
        else condition_xyz
    )
    validation_match_xyz = (
        np.round(validation_xyz, decimals=coordinate_decimals)
        if coordinate_decimals is not None
        else validation_xyz
    )
    condition_keys = _coordinate_keys(condition_match_xyz)
    validation_keys = _coordinate_keys(validation_match_xyz)
    duplicate_condition = len(condition_keys) - len(set(condition_keys))
    duplicate_validation = len(validation_keys) - len(set(validation_keys))
    if duplicate_condition or duplicate_validation:
        raise ValueError(
            "Coordinate keys must be unique; "
            f"condition duplicates={duplicate_condition}, validation duplicates={duplicate_validation}"
        )

    validation_lookup = {key: idx for idx, key in enumerate(validation_keys)}
    missing = [key for key in condition_keys if key not in validation_lookup]
    extra = [key for key in validation_keys if key not in set(condition_keys)]
    if missing or extra:
        raise ValueError(
            "Condition/validation coordinates do not match at the configured precision; "
            f"missing_in_validation={len(missing)}, extra_in_validation={len(extra)}"
        )
    order = np.asarray([validation_lookup[key] for key in condition_keys], dtype=int)
    aligned_xyz = validation_xyz[order]
    max_diff = float(np.max(np.abs(condition_xyz - aligned_xyz))) if condition_xyz.size else 0.0
    return validation_rate[order], {
        "condition_row_count": int(condition_xyz.shape[0]),
        "validation_row_count": int(validation_xyz.shape[0]),
        "coordinate_exact_match": bool(max_diff == 0.0),
        "coordinate_match_decimals": coordinate_decimals,
        "coordinate_tolerance_match": True,
        "coordinate_max_abs_difference": max_diff,
        "condition_duplicate_coordinates": int(duplicate_condition),
        "validation_duplicate_coordinates": int(duplicate_validation),
    }


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size < 2 or float(np.std(a)) <= _EPS or float(np.std(b)) <= _EPS:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    residual = np.asarray(pred, dtype=float) - np.asarray(y, dtype=float)
    mse = float(np.mean(np.square(residual)))
    centered = y - float(np.mean(y))
    sst = float(np.sum(np.square(centered)))
    sse = float(np.sum(np.square(residual)))
    return {
        "mse_nm2_s2": mse,
        "rmse_nm_s": float(math.sqrt(max(mse, 0.0))),
        "mae_nm_s": float(np.mean(np.abs(residual))),
        "max_abs_nm_s": float(np.max(np.abs(residual))),
        "r2": float(1.0 - sse / sst) if sst > _EPS else float("nan"),
        "bias_nm_s": float(np.mean(residual)),
    }


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def enumerate_role_response_candidates(species: Iterable[str], *, include_reductions: bool = False) -> list[RoleResponseCandidate]:
    names = sorted(str(name) for name in species)
    candidates = [RoleResponseCandidate(model_id="baseline", class_id="baseline")]
    for a_name in names:
        candidates.append(RoleResponseCandidate(model_id=f"A:{a_name}", class_id="A", A=a_name))
    for a_name in names:
        for i_name in names:
            if i_name != a_name:
                candidates.append(
                    RoleResponseCandidate(model_id=f"AI:{a_name}|{i_name}", class_id="AI", A=a_name, I=i_name)
                )
    for index, a_name in enumerate(names):
        for b_name in names[index + 1 :]:
            candidates.append(
                RoleResponseCandidate(model_id=f"AB:{a_name}|{b_name}", class_id="AB", A=a_name, B=b_name)
            )
    if len(names) >= 3:
        for i_name in names:
            pair = [name for name in names if name != i_name]
            for left in range(len(pair)):
                for right in range(left + 1, len(pair)):
                    a_name, b_name = pair[left], pair[right]
                    candidates.append(
                        RoleResponseCandidate(
                            model_id=f"AIB:{a_name}|{i_name}|{b_name}",
                            class_id="AIB",
                            A=a_name,
                            I=i_name,
                            B=b_name,
                        )
                    )
    if include_reductions:
        candidates += [reduced for candidate in candidates for reduced in candidate.reductions()]
    # Exact model symmetry only. Fitted zero coefficients do not remove a
    # candidate from the validation folds.
    return list({tuple((key, tuple(value)) for key, value in candidate.effect_groups.items()): candidate
                 for candidate in reversed(candidates)}.values())[::-1]


def _candidate_design(
    candidate: RoleResponseCandidate,
    concentrations: dict[str, np.ndarray],
    indices: np.ndarray,
    *,
    references: dict[str, float] | None = None,
) -> tuple[np.ndarray, tuple[str, ...], dict[str, float]]:
    idx = np.asarray(indices, dtype=int)
    involved = [name for name in (candidate.A, candidate.I, candidate.B) if name is not None]
    if references is None:
        references = {
            name: float(np.median(np.asarray(concentrations[name], dtype=float)[idx])) for name in sorted(set(involved))
        }
    if any(value <= 0.0 or not math.isfinite(value) for value in references.values()):
        raise ValueError(f"Positive finite concentration references are required: {references}")

    features: list[np.ndarray] = [np.ones(idx.size, dtype=float)]
    effect_names: list[str] = []
    if candidate.B is None and candidate.A is not None:
        value = concentrations[candidate.A][idx] / references[candidate.A] - 1.0
        features.append(np.asarray(value, dtype=float))
        effect_names.append(f"A:{candidate.A}")
    elif candidate.A is not None and candidate.B is not None:
        value = (
            concentrations[candidate.A][idx]
            * concentrations[candidate.B][idx]
            / (references[candidate.A] * references[candidate.B])
            - 1.0
        )
        features.append(np.asarray(value, dtype=float))
        effect_names.append(f"AB:{candidate.A}*{candidate.B}")
    if candidate.I is not None:
        # A positive fitted coefficient represents inhibition.  The signed
        # design column makes above-reference inhibitor concentration reduce rate.
        value = -(concentrations[candidate.I][idx] / references[candidate.I] - 1.0)
        features.append(np.asarray(value, dtype=float))
        effect_names.append(f"I:{candidate.I}")
    return np.column_stack(features), tuple(effect_names), dict(references)


def _fit_nonnegative_effects(
    design: np.ndarray, target: np.ndarray, *,
    weights: np.ndarray | None = None, regularization: float = 0.0,
    penalty_matrix: np.ndarray | None = None,
) -> tuple[np.ndarray, tuple[bool, ...]]:
    """Least squares with an unconstrained intercept and nonnegative effects.

    The candidate models contain only a few effect coefficients. Enumerating
    active sets keeps NumPy as the only numerical dependency and gives the exact
    constrained least-squares solution for this small problem.
    """

    x = np.asarray(design, dtype=float)
    y = np.asarray(target, dtype=float)
    if not np.isfinite(regularization) or regularization < 0:
        raise ValueError("regularization must be finite and nonnegative")
    weight = np.ones(y.size) if weights is None else np.asarray(weights, dtype=float)
    if weight.shape != y.shape or np.any(weight < 0) or not np.isfinite(weight).all() or weight.sum() <= 0:
        raise ValueError("weights must be finite, nonnegative and match the observations")
    root_weight = np.sqrt(weight / weight.sum())
    x = x * root_weight[:, None]
    y = y * root_weight
    if regularization:
        # Penalize dimensionless effects, never the reference-rate intercept.
        operator = np.eye(x.shape[1])[1:] if penalty_matrix is None else np.asarray(penalty_matrix, dtype=float)
        if operator.ndim != 2 or operator.shape[1] != x.shape[1] or not np.isfinite(operator).all() or np.any(operator[:, 0]):
            raise ValueError("penalty_matrix must be finite, match the design and leave the intercept unpenalized")
        penalty = np.sqrt(regularization) * operator
        x = np.vstack([x, penalty])
        y = np.concatenate([y, np.zeros(penalty.shape[0])])
    effect_count = x.shape[1] - 1
    best_coef: np.ndarray | None = None
    best_mask: tuple[bool, ...] | None = None
    best_sse = float("inf")
    for bits in range(1 << effect_count):
        mask = tuple(bool(bits & (1 << j)) for j in range(effect_count))
        columns = [0] + [j + 1 for j, active in enumerate(mask) if active]
        coef_reduced, *_ = np.linalg.lstsq(x[:, columns], y, rcond=None)
        if coef_reduced.size > 1 and np.any(coef_reduced[1:] < -1.0e-12):
            continue
        full = np.zeros(x.shape[1], dtype=float)
        full[columns] = coef_reduced
        full[1:] = np.clip(full[1:], 0.0, np.inf)
        residual = x @ full - y
        sse = float(np.sum(np.square(residual)))
        if sse < best_sse:
            best_sse = sse
            best_coef = full
            best_mask = tuple(bool(value > 0) for value in full[1:])
    if best_coef is None or best_mask is None:  # pragma: no cover - intercept-only is always feasible
        raise RuntimeError("No feasible constrained least-squares solution")
    return best_coef, best_mask


def fit_candidate(
    candidate: RoleResponseCandidate,
    concentrations: dict[str, np.ndarray],
    target: np.ndarray,
    train_indices: np.ndarray,
    predict_indices: np.ndarray | None = None,
    *,
    references: dict[str, float] | None = None,
) -> FitResult:
    train_idx = np.asarray(train_indices, dtype=int)
    pred_idx = train_idx if predict_indices is None else np.asarray(predict_indices, dtype=int)
    design_train, effect_names, fitted_references = _candidate_design(
        candidate,
        concentrations,
        train_idx,
        references=references,
    )
    coefficients, active = _fit_nonnegative_effects(design_train, target[train_idx])
    design_pred, _, _ = _candidate_design(
        candidate,
        concentrations,
        pred_idx,
        references=fitted_references,
    )
    return FitResult(
        coefficients=coefficients,
        prediction=design_pred @ coefficients,
        design=design_pred,
        effect_names=effect_names,
        references=fitted_references,
        active_effects=active,
    )


def _angular_groups(xy: np.ndarray, requested_groups: int = 8) -> np.ndarray:
    angles = np.mod(np.arctan2(xy[:, 1], xy[:, 0]), 2.0 * np.pi)
    groups = np.floor(angles / (2.0 * np.pi / requested_groups)).astype(int)
    radius = np.sqrt(np.sum(np.square(xy), axis=1))
    groups[radius <= max(float(np.max(radius)) * 1.0e-10, _EPS)] = 0
    return groups


def _radial_groups(xy: np.ndarray, max_groups: int = 6) -> np.ndarray:
    radius = np.sqrt(np.sum(np.square(xy), axis=1))
    unique = np.unique(np.round(radius, decimals=10))
    if unique.size <= max_groups:
        mapping = {float(value): idx for idx, value in enumerate(unique)}
        return np.asarray([mapping[float(value)] for value in np.round(radius, decimals=10)], dtype=int)
    quantiles = np.quantile(radius, np.linspace(0.0, 1.0, max_groups + 1))
    inner = np.unique(quantiles[1:-1])
    return np.digitize(radius, inner, right=True).astype(int)


def _blocked_predictions(
    candidate: RoleResponseCandidate,
    concentrations: dict[str, np.ndarray],
    target: np.ndarray,
    groups: np.ndarray,
) -> np.ndarray:
    prediction = np.full(target.shape, np.nan, dtype=float)
    all_indices = np.arange(target.size, dtype=int)
    for group in np.unique(groups):
        validation_idx = all_indices[groups == group]
        train_idx = all_indices[groups != group]
        if train_idx.size < 3 or validation_idx.size == 0:
            continue
        fit = fit_candidate(candidate, concentrations, target, train_idx, validation_idx)
        prediction[validation_idx] = fit.prediction
    if not np.all(np.isfinite(prediction)):
        raise RuntimeError(f"Blocked CV did not predict all rows for {candidate.model_id}")
    return prediction


def _aicc(target: np.ndarray, prediction: np.ndarray, parameter_count: int) -> float:
    n = int(target.size)
    sse = max(float(np.sum(np.square(prediction - target))), _EPS)
    base = n * math.log(sse / n) + 2.0 * parameter_count
    if n <= parameter_count + 1:
        return float("inf")
    return float(base + 2.0 * parameter_count * (parameter_count + 1) / (n - parameter_count - 1))


def _coefficient_rows(
    candidate: RoleResponseCandidate,
    fit: FitResult,
    bootstrap_coefficients: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    baseline_samples = bootstrap_coefficients[:, 0]
    rows.append(
        {
            "model_id": candidate.model_id,
            "term": "reference_rate",
            "role": "baseline",
            "species": "",
            "coefficient_scaled": float(fit.coefficients[0]),
            "coefficient_physical": float(fit.coefficients[0]),
            "physical_unit": "nm/s",
            "reference_concentration": "",
            "bootstrap_scaled_p05": float(np.quantile(baseline_samples, 0.05)),
            "bootstrap_scaled_p50": float(np.quantile(baseline_samples, 0.50)),
            "bootstrap_scaled_p95": float(np.quantile(baseline_samples, 0.95)),
            "bootstrap_physical_p05": float(np.quantile(baseline_samples, 0.05)),
            "bootstrap_physical_p50": float(np.quantile(baseline_samples, 0.50)),
            "bootstrap_physical_p95": float(np.quantile(baseline_samples, 0.95)),
            "bootstrap_zero_fraction": 0.0,
        }
    )
    for effect_index, effect_name in enumerate(fit.effect_names, start=1):
        scaled = float(fit.coefficients[effect_index])
        samples = bootstrap_coefficients[:, effect_index]
        if effect_name.startswith("AB:"):
            left, right = effect_name.removeprefix("AB:").split("*")
            reference = fit.references[left] * fit.references[right]
            physical_unit = "nm*m^6/(s*kmol^2)"
            role = "AB_joint"
            species = f"{left}*{right}"
            reference_text = f"{reference:.12g} (kmol/m^3)^2"
        else:
            role, species = effect_name.split(":", maxsplit=1)
            reference = fit.references[species]
            physical_unit = "nm*m^3/(s*kmol)"
            reference_text = f"{reference:.12g} kmol/m^3"
        rows.append(
            {
                "model_id": candidate.model_id,
                "term": effect_name,
                "role": role,
                "species": species,
                "coefficient_scaled": scaled,
                "coefficient_physical": scaled / reference,
                "physical_unit": physical_unit,
                "reference_concentration": reference_text,
                "bootstrap_scaled_p05": float(np.quantile(samples, 0.05)),
                "bootstrap_scaled_p50": float(np.quantile(samples, 0.50)),
                "bootstrap_scaled_p95": float(np.quantile(samples, 0.95)),
                "bootstrap_physical_p05": float(np.quantile(samples / reference, 0.05)),
                "bootstrap_physical_p50": float(np.quantile(samples / reference, 0.50)),
                "bootstrap_physical_p95": float(np.quantile(samples / reference, 0.95)),
                "bootstrap_zero_fraction": float(np.mean(samples <= 1.0e-14)),
            }
        )
    return rows


def _bootstrap_coefficients(
    candidate: RoleResponseCandidate,
    concentrations: dict[str, np.ndarray],
    target: np.ndarray,
    groups: np.ndarray,
    references: dict[str, float],
    *,
    samples: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    unique_groups = np.unique(groups)
    rows: list[np.ndarray] = []
    for _ in range(samples):
        selected = rng.choice(unique_groups, size=unique_groups.size, replace=True)
        indices = np.concatenate([np.flatnonzero(groups == group) for group in selected])
        fit = fit_candidate(candidate, concentrations, target, indices, references=references)
        rows.append(fit.coefficients)
    return np.vstack(rows)


def _moran_i(
    residual: np.ndarray,
    xyz: np.ndarray,
    *,
    neighbors: int = 6,
    permutations: int = 499,
    seed: int = 1729,
) -> dict[str, float]:
    values = np.asarray(residual, dtype=float) - float(np.mean(residual))
    delta = xyz[:, None, :] - xyz[None, :, :]
    distances = np.sqrt(np.sum(np.square(delta), axis=2))
    np.fill_diagonal(distances, np.inf)
    k = min(max(int(neighbors), 1), max(values.size - 1, 1))
    nearest = np.argsort(distances, axis=1)[:, :k]
    weights = np.zeros_like(distances)
    for row in range(values.size):
        weights[row, nearest[row]] = 1.0 / np.maximum(distances[row, nearest[row]], _EPS)
    weights = 0.5 * (weights + weights.T)
    weight_sum = float(np.sum(weights))
    denominator = float(np.sum(np.square(values)))

    def statistic(sample: np.ndarray) -> float:
        if denominator <= _EPS or weight_sum <= _EPS:
            return float("nan")
        return float(values.size / weight_sum * np.sum(weights * np.outer(sample, sample)) / np.sum(np.square(sample)))

    observed = statistic(values)
    rng = np.random.default_rng(seed)
    permuted = np.asarray([statistic(rng.permutation(values)) for _ in range(permutations)], dtype=float)
    expected = -1.0 / max(values.size - 1, 1)
    p_two_sided = float((1 + np.sum(np.abs(permuted - expected) >= abs(observed - expected))) / (permutations + 1))
    return {"moran_i": observed, "expected_i": expected, "permutation_p_two_sided": p_two_sided}


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def _plot_results(
    output_dir: Path,
    xyz: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
    contribution_rows: list[dict[str, Any]],
    effect_names: tuple[str, ...],
) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:  # pragma: no cover
        return []

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    residual = prediction - target

    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    ax.scatter(target, prediction, color="#3266a8", edgecolor="#1f2937", linewidth=0.5)
    lo = float(min(np.min(target), np.min(prediction)))
    hi = float(max(np.max(target), np.max(prediction)))
    ax.plot([lo, hi], [lo, hi], color="#374151", linestyle="--", linewidth=1.2)
    ax.set_xlabel("Measured deposition rate [nm/s]")
    ax.set_ylabel("Predicted deposition rate [nm/s]")
    ax.set_title("Measured versus predicted deposition rate")
    ax.grid(color="#d1d5db", linewidth=0.6, alpha=0.7)
    fig.tight_layout()
    path = plot_dir / "measured_vs_predicted.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    written.append(str(path.relative_to(output_dir)).replace("\\", "/"))

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.2), constrained_layout=True)
    rate_lo = float(min(np.min(target), np.min(prediction)))
    rate_hi = float(max(np.max(target), np.max(prediction)))
    residual_limit = float(max(np.max(np.abs(residual)), _EPS))
    panels = [
        (target, "Measured rate [nm/s]", "viridis", rate_lo, rate_hi),
        (prediction, "Predicted rate [nm/s]", "viridis", rate_lo, rate_hi),
        (residual, "Residual: predicted - measured [nm/s]", "coolwarm", -residual_limit, residual_limit),
    ]
    for ax, (values, title, cmap, vmin, vmax) in zip(axes, panels):
        scatter = ax.scatter(xyz[:, 0], xyz[:, 1], c=values, cmap=cmap, vmin=vmin, vmax=vmax, s=55)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x [source coordinate unit]")
        ax.set_ylabel("y [source coordinate unit]")
        ax.set_title(title)
        fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    path = plot_dir / "spatial_fit_maps.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    written.append(str(path.relative_to(output_dir)).replace("\\", "/"))

    if effect_names:
        columns = [f"contribution_{name}" for name in effect_names]
        values_by_name = {
            name: np.asarray([float(row[column]) for row in contribution_rows], dtype=float)
            for name, column in zip(effect_names, columns)
        }
        fig, axes = plt.subplots(1, len(effect_names), figsize=(5.0 * len(effect_names), 4.2), squeeze=False)
        for ax, name in zip(axes.ravel(), effect_names):
            values = values_by_name[name]
            limit = float(max(np.max(np.abs(values)), _EPS))
            scatter = ax.scatter(
                xyz[:, 0], xyz[:, 1], c=values, cmap="coolwarm", vmin=-limit, vmax=limit, s=58
            )
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel("x [source coordinate unit]")
            ax.set_ylabel("y [source coordinate unit]")
            ax.set_title(f"{name} contribution [nm/s]")
            fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        path = plot_dir / "spatial_contributions.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        written.append(str(path.relative_to(output_dir)).replace("\\", "/"))
    return written


def _write_executed_notebook(output_dir: Path, condition_path: Path, validation_path: Path) -> Path:
    """Write a minimal executed notebook without adding a notebook dependency."""

    notebook_path = output_dir / "cvd_condition_1_analysis.ipynb"
    relative_output = output_dir.as_posix()
    cells: list[dict[str, Any]] = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## tl;dr\n",
                "This notebook is a compact, rerunnable companion to the CVD spatial-response analysis. "
                "The coefficients are effective spatial-response coefficients, not elementary kinetic constants.\n",
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Context & Methods\n",
                f"Inputs: `{condition_path.as_posix()}` and `{validation_path.as_posix()}`.\n",
                "\n### Key Assumptions\n",
                "- Concentration columns use kmol/m^3 and deposition rate uses nm/s as provided.\n",
                "- Coordinate units and independent process-condition validation are unavailable.\n",
                "- Mole fractions and density are consistency checks, not independent predictors.\n",
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Data\n", "Load the generated, reviewed result tables from the deterministic output directory.\n"],
        },
        {
            "cell_type": "code",
            "execution_count": 1,
            "metadata": {},
            "outputs": [],
            "source": [
                "from pathlib import Path\n",
                "import csv, json\n",
                f"output_dir = Path({relative_output!r})\n",
                "summary = json.loads((output_dir / 'analysis_summary.json').read_text(encoding='utf-8'))\n",
                "with (output_dir / 'model_ranking.csv').open(encoding='utf-8') as handle:\n",
                "    ranking = list(csv.DictReader(handle))\n",
                "with (output_dir / 'coefficients.csv').open(encoding='utf-8') as handle:\n",
                "    coefficients = list(csv.DictReader(handle))\n",
                "print('rows:', summary['data_quality']['row_count'])\n",
                "print('species:', ', '.join(summary['species']))\n",
            ],
        },
        {"cell_type": "markdown", "metadata": {}, "source": ["## Results\n"]},
        {
            "cell_type": "code",
            "execution_count": 2,
            "metadata": {},
            "outputs": [],
            "source": [
                "best = summary['best_model']\n",
                "print('best model:', best['model_id'])\n",
                "print('blocked CV RMSE [nm/s]:', best['blocked_cv_rmse_nm_s'])\n",
                "print('in-sample R2:', best['in_sample_r2'])\n",
                "print('assessment:', summary['validity']['overall_assessment'])\n",
                "for row in coefficients:\n",
                "    print(row['term'], row['coefficient_physical'], row['physical_unit'])\n",
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Generated figures\n",
                "![Measured versus predicted](plots/measured_vs_predicted.png)\n",
                "\n![Spatial fit maps](plots/spatial_fit_maps.png)\n",
                "\n![Spatial contributions](plots/spatial_contributions.png)\n",
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Takeaways\n",
                "See `report.md` for the reviewed interpretation, limitations, and missing information.\n",
            ],
        },
    ]

    namespace: dict[str, Any] = {}
    execution_count = 0
    for cell in cells:
        if cell["cell_type"] != "code":
            continue
        execution_count += 1
        source = "".join(cell["source"])
        output = StringIO()
        with redirect_stdout(output):
            exec(compile(source, f"{notebook_path.name}:cell{execution_count}", "exec"), namespace, namespace)
        text = output.getvalue()
        cell["execution_count"] = execution_count
        cell["outputs"] = (
            [{"name": "stdout", "output_type": "stream", "text": text.splitlines(keepends=True)}] if text else []
        )

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    notebook_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    # Minimal structural validation for environments without nbformat.
    parsed = json.loads(notebook_path.read_text(encoding="utf-8"))
    if parsed.get("nbformat") != 4 or not isinstance(parsed.get("cells"), list):
        raise RuntimeError("Generated notebook failed structural validation")
    return notebook_path


def analyze_cvd_spatial_case(
    *,
    condition_path: Path,
    validation_path: Path,
    output_dir: Path,
    bootstrap_samples: int = 1000,
    seed: int = 123,
) -> dict[str, Any]:
    condition_path = Path(condition_path)
    validation_path = Path(validation_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    condition_headers, condition = _read_numeric_csv(condition_path)
    validation_headers, validation = _read_numeric_csv(validation_path)
    if "dr_nm_per_sec" not in validation:
        raise ValueError("validation CSV must contain dr_nm_per_sec")
    concentration_columns = [name for name in condition_headers if name.startswith("concentration_")]
    if not concentration_columns:
        raise ValueError("No concentration_* columns found")
    species = [name.removeprefix("concentration_") for name in concentration_columns]
    concentrations = {species_name: condition[column] for species_name, column in zip(species, concentration_columns)}
    mole_fraction_columns = [f"molef_{name}" for name in species]
    missing_mole_fraction = [name for name in mole_fraction_columns if name not in condition]

    condition_xyz = _coordinate_matrix(condition)
    validation_xyz = _coordinate_matrix(validation)
    target, alignment = _align_validation(condition_xyz, validation_xyz, validation["dr_nm_per_sec"])
    all_values = [*condition.values(), *validation.values()]
    nonfinite_total = int(sum(np.size(values) - np.count_nonzero(np.isfinite(values)) for values in all_values))
    if nonfinite_total:
        raise ValueError(f"Input contains {nonfinite_total} non-finite numeric values")
    if np.any(target < 0.0):
        raise ValueError("Negative deposition rates are not supported")
    if any(np.any(values <= 0.0) for values in concentrations.values()):
        raise ValueError("All concentration values must be positive for normalized role-response fitting")

    concentration_sum = np.sum(np.column_stack([concentrations[name] for name in species]), axis=1)
    mole_fraction_sum = (
        np.sum(np.column_stack([condition[name] for name in mole_fraction_columns]), axis=1)
        if not missing_mole_fraction
        else np.full(target.shape, np.nan)
    )
    total_from_each_species = (
        np.column_stack(
            [concentrations[name] / np.maximum(condition[f"molef_{name}"], _EPS) for name in species]
        )
        if not missing_mole_fraction
        else np.empty((target.size, 0))
    )
    total_consistency_rel = (
        np.abs(total_from_each_species - concentration_sum[:, None]) / np.maximum(concentration_sum[:, None], _EPS)
        if total_from_each_species.size
        else np.empty((target.size, 0))
    )
    mixture_molar_mass = (
        condition["density"] / np.maximum(concentration_sum, _EPS) if "density" in condition else np.full(target.shape, np.nan)
    )

    species_associations: list[dict[str, Any]] = []
    concentration_matrix = np.column_stack([concentrations[name] for name in species])
    concentration_corr = np.corrcoef(concentration_matrix, rowvar=False)
    radius = np.sqrt(np.sum(np.square(condition_xyz[:, :2]), axis=1))
    preliminary_angular_groups = _angular_groups(condition_xyz[:, :2])
    spatial_basis = np.column_stack(
        [
            np.ones(target.size, dtype=float),
            condition_xyz[:, 0],
            condition_xyz[:, 1],
            np.square(condition_xyz[:, 0]),
            condition_xyz[:, 0] * condition_xyz[:, 1],
            np.square(condition_xyz[:, 1]),
        ]
    )
    spatial_target_coef, *_ = np.linalg.lstsq(spatial_basis, target, rcond=None)
    spatial_target_residual = target - spatial_basis @ spatial_target_coef
    for name in species:
        values = concentrations[name]
        centered = values - float(np.mean(values))
        slope = float(np.dot(centered, target - float(np.mean(target))) / max(np.dot(centered, centered), _EPS))
        univariate_pred = float(np.mean(target)) + slope * centered
        assoc_metrics = _metrics(target, univariate_pred)
        spatial_value_coef, *_ = np.linalg.lstsq(spatial_basis, values, rcond=None)
        spatial_value_residual = values - spatial_basis @ spatial_value_coef
        rng = np.random.default_rng(seed + len(species_associations) + 1)
        unique_blocks = np.unique(preliminary_angular_groups)
        slope_samples: list[float] = []
        for _ in range(max(int(bootstrap_samples), 100)):
            selected_blocks = rng.choice(unique_blocks, size=unique_blocks.size, replace=True)
            sampled_indices = np.concatenate(
                [np.flatnonzero(preliminary_angular_groups == block) for block in selected_blocks]
            )
            sampled_values = values[sampled_indices]
            sampled_target = target[sampled_indices]
            sampled_centered = sampled_values - float(np.mean(sampled_values))
            slope_samples.append(
                float(
                    np.dot(sampled_centered, sampled_target - float(np.mean(sampled_target)))
                    / max(np.dot(sampled_centered, sampled_centered), _EPS)
                )
            )
        slope_sample_array = np.asarray(slope_samples, dtype=float)
        species_associations.append(
            {
                "species": name,
                "concentration_min_kmol_m3": float(np.min(values)),
                "concentration_median_kmol_m3": float(np.median(values)),
                "concentration_max_kmol_m3": float(np.max(values)),
                "relative_range_vs_median": float((np.max(values) - np.min(values)) / np.median(values)),
                "pearson_rate_correlation": _safe_corr(values, target),
                "pearson_radius_correlation": _safe_corr(values, radius),
                "partial_rate_correlation_after_quadratic_xy": _safe_corr(
                    spatial_value_residual, spatial_target_residual
                ),
                "univariate_slope_nm_m3_s_kmol": slope,
                "univariate_slope_bootstrap_p05": float(np.quantile(slope_sample_array, 0.05)),
                "univariate_slope_bootstrap_p50": float(np.quantile(slope_sample_array, 0.50)),
                "univariate_slope_bootstrap_p95": float(np.quantile(slope_sample_array, 0.95)),
                "univariate_elasticity_at_median": float(slope * np.median(values) / max(np.median(target), _EPS)),
                "univariate_r2": assoc_metrics["r2"],
                "signed_rate_change_over_observed_concentration_range_nm_s": float(
                    slope * (np.max(values) - np.min(values))
                ),
                "interpretation": "marginal spatial association; not an isolated reaction coefficient",
            }
        )
    concentration_correlation_rows: list[dict[str, Any]] = []
    for left_index, left_name in enumerate(species):
        for right_index in range(left_index + 1, len(species)):
            right_name = species[right_index]
            concentration_correlation_rows.append(
                {
                    "species_1": left_name,
                    "species_2": right_name,
                    "pearson_correlation": float(concentration_corr[left_index, right_index]),
                }
            )

    xy = condition_xyz[:, :2]
    angular_groups = preliminary_angular_groups
    radial_groups = _radial_groups(xy)
    candidates = enumerate_role_response_candidates(species)
    all_indices = np.arange(target.size, dtype=int)
    ranking_rows: list[dict[str, Any]] = []
    fits: dict[str, FitResult] = {}
    cv_predictions: dict[str, dict[str, np.ndarray]] = {}
    for candidate in candidates:
        fit = fit_candidate(candidate, concentrations, target, all_indices)
        fits[candidate.model_id] = fit
        angular_prediction = _blocked_predictions(candidate, concentrations, target, angular_groups)
        radial_prediction = _blocked_predictions(candidate, concentrations, target, radial_groups)
        cv_predictions[candidate.model_id] = {"angular": angular_prediction, "radial": radial_prediction}
        in_sample = _metrics(target, fit.prediction)
        angular = _metrics(target, angular_prediction)
        radial = _metrics(target, radial_prediction)
        parameter_count = 1 + int(sum(fit.active_effects))
        ranking_rows.append(
            {
                "model_id": candidate.model_id,
                "class_id": candidate.class_id,
                "role_A": candidate.A or "",
                "role_I": candidate.I or "",
                "role_B": candidate.B or "",
                "effect_count": candidate.effect_count,
                "active_effect_count": int(sum(fit.active_effects)),
                "in_sample_rmse_nm_s": in_sample["rmse_nm_s"],
                "in_sample_mae_nm_s": in_sample["mae_nm_s"],
                "in_sample_max_abs_nm_s": in_sample["max_abs_nm_s"],
                "in_sample_r2": in_sample["r2"],
                "angular_cv_rmse_nm_s": angular["rmse_nm_s"],
                "angular_cv_r2": angular["r2"],
                "radial_cv_rmse_nm_s": radial["rmse_nm_s"],
                "radial_cv_r2": radial["r2"],
                "blocked_cv_rmse_nm_s": max(angular["rmse_nm_s"], radial["rmse_nm_s"]),
                "aicc": _aicc(target, fit.prediction, parameter_count),
            }
        )
    ranking_rows.sort(key=lambda row: (float(row["blocked_cv_rmse_nm_s"]), int(row["effect_count"]), float(row["aicc"])))
    for rank, row in enumerate(ranking_rows, start=1):
        row["rank"] = rank

    best_row = ranking_rows[0]
    best_candidate = next(candidate for candidate in candidates if candidate.model_id == best_row["model_id"])
    best_fit = fits[best_candidate.model_id]
    bootstrap = _bootstrap_coefficients(
        best_candidate,
        concentrations,
        target,
        angular_groups,
        best_fit.references,
        samples=max(int(bootstrap_samples), 100),
        seed=seed,
    )
    coefficient_rows = _coefficient_rows(best_candidate, best_fit, bootstrap)
    selected_roles = {name: "excluded" for name in species}
    if best_candidate.A is not None:
        selected_roles[best_candidate.A] = "A" if best_candidate.B is None else "AB_joint"
    if best_candidate.B is not None:
        selected_roles[best_candidate.B] = "AB_joint"
    if best_candidate.I is not None:
        selected_roles[best_candidate.I] = "I"
    for row in species_associations:
        row["selected_model_role"] = selected_roles[str(row["species"])]

    fold_winner_counts: dict[str, int] = {}
    fold_count = 0
    for scheme, groups in (("angular", angular_groups), ("radial", radial_groups)):
        for group in np.unique(groups):
            mask = groups == group
            fold_count += 1
            winner = min(
                candidates,
                key=lambda candidate: (
                    _metrics(target[mask], cv_predictions[candidate.model_id][scheme][mask])["rmse_nm_s"],
                    candidate.effect_count,
                ),
            )
            fold_winner_counts[winner.model_id] = fold_winner_counts.get(winner.model_id, 0) + 1
    best_fold_winner_fraction = float(fold_winner_counts.get(best_candidate.model_id, 0) / max(fold_count, 1))

    contribution_rows: list[dict[str, Any]] = []
    for row_index in range(target.size):
        row: dict[str, Any] = {
            "x": float(condition_xyz[row_index, 0]),
            "y": float(condition_xyz[row_index, 1]),
            "z": float(condition_xyz[row_index, 2]),
            "measured_rate_nm_s": float(target[row_index]),
            "predicted_rate_nm_s": float(best_fit.prediction[row_index]),
            "residual_pred_minus_meas_nm_s": float(best_fit.prediction[row_index] - target[row_index]),
            "reference_rate_contribution_nm_s": float(best_fit.coefficients[0]),
        }
        for name in species:
            row[f"concentration_{name}_kmol_m3"] = float(concentrations[name][row_index])
        for effect_index, effect_name in enumerate(best_fit.effect_names, start=1):
            row[f"contribution_{effect_name}"] = float(best_fit.design[row_index, effect_index] * best_fit.coefficients[effect_index])
        contribution_rows.append(row)
    contribution_summary_rows: list[dict[str, Any]] = [
        {
            "term": "reference_rate",
            "role": "baseline",
            "species": "",
            "min_contribution_nm_s": float(best_fit.coefficients[0]),
            "median_contribution_nm_s": float(best_fit.coefficients[0]),
            "max_contribution_nm_s": float(best_fit.coefficients[0]),
            "mean_abs_contribution_nm_s": float(abs(best_fit.coefficients[0])),
            "std_contribution_nm_s": 0.0,
            "peak_to_peak_contribution_nm_s": 0.0,
        }
    ]
    for effect_index, effect_name in enumerate(best_fit.effect_names, start=1):
        values = best_fit.design[:, effect_index] * best_fit.coefficients[effect_index]
        if effect_name.startswith("AB:"):
            role = "AB_joint"
            term_species = effect_name.removeprefix("AB:")
        else:
            role, term_species = effect_name.split(":", maxsplit=1)
        contribution_summary_rows.append(
            {
                "term": effect_name,
                "role": role,
                "species": term_species,
                "min_contribution_nm_s": float(np.min(values)),
                "median_contribution_nm_s": float(np.median(values)),
                "max_contribution_nm_s": float(np.max(values)),
                "mean_abs_contribution_nm_s": float(np.mean(np.abs(values))),
                "std_contribution_nm_s": float(np.std(values)),
                "peak_to_peak_contribution_nm_s": float(np.ptp(values)),
            }
        )

    best_residual = best_fit.prediction - target
    residual_spatial = _moran_i(best_residual, condition_xyz)
    effect_design = best_fit.design[:, 1:]
    if effect_design.shape[1] >= 2:
        feature_correlation = float(np.corrcoef(effect_design, rowvar=False)[0, 1])
        design_condition_number = float(np.linalg.cond(best_fit.design))
    elif effect_design.shape[1] == 1:
        feature_correlation = float("nan")
        design_condition_number = float(np.linalg.cond(best_fit.design))
    else:
        feature_correlation = float("nan")
        design_condition_number = 1.0

    baseline_row = next(row for row in ranking_rows if row["model_id"] == "baseline")
    baseline_cv = float(baseline_row["blocked_cv_rmse_nm_s"])
    best_cv = float(best_row["blocked_cv_rmse_nm_s"])
    cv_improvement = float((baseline_cv - best_cv) / baseline_cv) if baseline_cv > _EPS else 0.0
    second_gap = (
        float(ranking_rows[1]["blocked_cv_rmse_nm_s"] - best_cv) / max(best_cv, _EPS)
        if len(ranking_rows) > 1
        else float("inf")
    )
    max_predictor_corr = (
        float(np.max(np.abs(concentration_corr - np.eye(len(species))))) if len(species) > 1 else 0.0
    )
    coefficient_zero_issue = any(
        float(row["bootstrap_zero_fraction"]) >= 0.10 for row in coefficient_rows if row["role"] != "baseline"
    )
    validity_flags = {
        "independent_condition_validation_available": False,
        "blocked_cv_improvement_vs_constant": cv_improvement,
        "top_model_relative_gap_to_second": second_gap,
        "max_abs_concentration_correlation": max_predictor_corr,
        "best_design_condition_number": design_condition_number,
        "best_effect_feature_correlation": feature_correlation,
        "coefficient_bootstrap_zero_fraction_ge_0_10": coefficient_zero_issue,
        "residual_spatial_autocorrelation": residual_spatial,
    }
    high_collinearity = bool(max_predictor_corr >= 0.95 or design_condition_number >= 30.0)
    role_instability = bool(second_gap <= 0.05 or best_fold_winner_fraction < 0.50)
    residual_structure = bool(
        math.isfinite(residual_spatial["permutation_p_two_sided"])
        and residual_spatial["permutation_p_two_sided"] < 0.05
    )
    spatial_association_assessment = "share_with_caveats" if cv_improvement > 0.0 else "needs_revision"
    # The user's requested reaction interpretation cannot be externally
    # validated from a single condition, especially under strong collinearity.
    reaction_interpretation_assessment = "needs_revision"
    overall_assessment = reaction_interpretation_assessment
    validity = {
        "overall_assessment": overall_assessment,
        "spatial_association_assessment": spatial_association_assessment,
        "reaction_interpretation_assessment": reaction_interpretation_assessment,
        "model_is_useful_for_spatial_association": bool(cv_improvement > 0.0),
        "model_is_validated_as_elementary_kinetics": False,
        "high_collinearity": high_collinearity,
        "role_ranking_is_close": role_instability,
        "residual_spatial_structure_detected": residual_structure,
        "fold_winner_counts": fold_winner_counts,
        "fold_count": fold_count,
        "best_model_fold_winner_fraction": best_fold_winner_fraction,
        "flags": validity_flags,
    }

    data_quality = {
        "row_count": int(target.size),
        "column_count_condition": len(condition_headers),
        "column_count_validation": len(validation_headers),
        "nonfinite_value_count": nonfinite_total,
        "coordinate_alignment": alignment,
        "z_unique_count": int(np.unique(condition_xyz[:, 2]).size),
        "coordinate_ranges": {
            "x": [float(np.min(condition_xyz[:, 0])), float(np.max(condition_xyz[:, 0]))],
            "y": [float(np.min(condition_xyz[:, 1])), float(np.max(condition_xyz[:, 1]))],
            "z": [float(np.min(condition_xyz[:, 2])), float(np.max(condition_xyz[:, 2]))],
        },
        "deposition_rate_min_nm_s": float(np.min(target)),
        "deposition_rate_median_nm_s": float(np.median(target)),
        "deposition_rate_max_nm_s": float(np.max(target)),
        "deposition_rate_relative_range_vs_median": float(
            (np.max(target) - np.min(target)) / max(np.median(target), _EPS)
        ),
        "mole_fraction_columns_missing": missing_mole_fraction,
        "mole_fraction_sum_min": float(np.nanmin(mole_fraction_sum)),
        "mole_fraction_sum_max": float(np.nanmax(mole_fraction_sum)),
        "mole_fraction_sum_max_abs_error_from_one": float(np.nanmax(np.abs(mole_fraction_sum - 1.0))),
        "concentration_mole_fraction_max_relative_inconsistency": (
            float(np.max(total_consistency_rel)) if total_consistency_rel.size else None
        ),
        "derived_total_molar_concentration_min_kmol_m3": float(np.min(concentration_sum)),
        "derived_total_molar_concentration_max_kmol_m3": float(np.max(concentration_sum)),
        "derived_mixture_molar_mass_min_kg_kmol": float(np.nanmin(mixture_molar_mass)),
        "derived_mixture_molar_mass_max_kg_kmol": float(np.nanmax(mixture_molar_mass)),
        "density_min_kg_m3": float(np.min(condition["density"])) if "density" in condition else None,
        "density_max_kg_m3": float(np.max(condition["density"])) if "density" in condition else None,
        "coordinate_unit_supplied": False,
        "rate_unit": "nm/s",
        "concentration_unit": "kmol/m^3",
    }

    summary = {
        "analysis_kind": "single-condition centered CVD role-response analysis",
        "species": species,
        "data_quality": data_quality,
        "best_model": {
            **best_row,
            "reference_concentrations_kmol_m3": best_fit.references,
            "effect_names": list(best_fit.effect_names),
        },
        "validity": validity,
        "missing_information": [
            "chemical identities, molar masses, feed/byproduct roles, and stoichiometry for adn_2, idn_2, and n2",
            "coordinate unit",
            "wafer temperature map and pressure",
            "surface/site density, sticking or adsorption/desorption information",
            "species-resolved incident flux or mass-transfer coefficient",
            "measurement uncertainty and replicate deposition maps",
            "independent process conditions for external validation",
            "zero/low-feed or designed concentration perturbations for causal coefficient separation",
        ],
        "interpretation_limits": [
            "Coefficients explain spatial covariation around the median concentration state.",
            "They are not elementary reaction-rate constants or proof of chemical causality.",
            "AB is a joint product term; A-versus-B direction cannot be identified from this static map alone.",
            "Excluded species have zero contribution only inside the selected empirical model and are not proven inert.",
        ],
        "source": {
            "condition_path": str(condition_path),
            "validation_path": str(validation_path),
            "condition_sha256": _sha256_file(condition_path),
            "validation_sha256": _sha256_file(validation_path),
        },
    }

    _write_rows(output_dir / "data_quality_species.csv", species_associations)
    _write_rows(output_dir / "concentration_correlations.csv", concentration_correlation_rows)
    _write_rows(output_dir / "model_ranking.csv", ranking_rows)
    _write_rows(output_dir / "coefficients.csv", coefficient_rows)
    _write_rows(output_dir / "contribution_summary.csv", contribution_summary_rows)
    _write_rows(output_dir / "spatial_contributions.csv", contribution_rows)
    cv_rows: list[dict[str, Any]] = []
    for index in range(target.size):
        cv_rows.append(
            {
                "x": float(condition_xyz[index, 0]),
                "y": float(condition_xyz[index, 1]),
                "z": float(condition_xyz[index, 2]),
                "measured_rate_nm_s": float(target[index]),
                "angular_cv_prediction_nm_s": float(cv_predictions[best_candidate.model_id]["angular"][index]),
                "radial_cv_prediction_nm_s": float(cv_predictions[best_candidate.model_id]["radial"][index]),
                "angular_group": int(angular_groups[index]),
                "radial_group": int(radial_groups[index]),
            }
        )
    _write_rows(output_dir / "blocked_cv_predictions.csv", cv_rows)
    _write_json(output_dir / "data_quality.json", data_quality)
    _write_json(output_dir / "analysis_summary.json", summary)
    plots = _plot_results(output_dir, condition_xyz, target, best_fit.prediction, contribution_rows, best_fit.effect_names)

    best_metrics = _metrics(target, best_fit.prediction)
    formula_terms = [f"{float(coefficient_rows[0]['coefficient_physical']):.9g}"]
    for row in coefficient_rows[1:]:
        coefficient = float(row["coefficient_physical"])
        role = str(row["role"])
        if role == "I":
            reference = best_fit.references[str(row["species"])]
            formula_terms.append(f"- {coefficient:.9g}*(C_{row['species']} - {reference:.9g})")
        elif role == "AB_joint":
            left, right = str(row["species"]).split("*")
            reference = best_fit.references[left] * best_fit.references[right]
            formula_terms.append(f"+ {coefficient:.9g}*(C_{left}*C_{right} - {reference:.9g})")
        else:
            reference = best_fit.references[str(row["species"])]
            formula_terms.append(f"+ {coefficient:.9g}*(C_{row['species']} - {reference:.9g})")
    model_formula = " ".join(formula_terms)
    report_lines = [
        "# CVD condition_1 空間反応寄与解析",
        "",
        "## 結論",
        "",
        f"- 最良の空間応答候補は `{best_candidate.model_id}` です。",
        f"- 実測内 RMSE は {best_metrics['rmse_nm_s']:.6g} nm/s、R² は {best_metrics['r2']:.4f} です。",
        f"- 保守的な空間ブロックCV RMSE は {best_cv:.6g} nm/sで、定数モデル比の改善率は {100.0 * cv_improvement:.2f}% です。",
        f"- 空間相関モデルとしては `{spatial_association_assessment}`、反応機構の解釈は `{reaction_interpretation_assessment}` です。",
        "- 係数は空間変動の有効応答係数であり、素反応速度定数ではありません。",
        "",
        "## データ品質",
        "",
        f"- 条件・検証データは各 {target.size} 点で、座標は完全一致しています。",
        f"- 欠損・非有限値は {nonfinite_total} 件、重複座標はありません。",
        f"- モル分率和の最大 |sum-1| は {data_quality['mole_fraction_sum_max_abs_error_from_one']:.3e} です。",
        f"- 濃度とモル分率から得る全モル濃度の最大相対不整合は {data_quality['concentration_mole_fraction_max_relative_inconsistency']:.3e} です。",
        "- density と濃度から混合平均モル質量は算出できますが、各種の分子量がないため化学的正しさまでは検証できません。",
        "- 入力の実列名は `molef_*` です。density と molef_* は濃度と冗長なため、回帰ではなく整合性確認だけに使いました。",
        "",
        "## 濃度種ごとの単変量関連",
        "",
        "|濃度種|濃度範囲/中央値|成膜レート相関 r|XY二次傾向除去後 r|単変量傾き [nm·m³/(s·kmol)]|bootstrap 5–95%|観測範囲の符号付き変化 [nm/s]|",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in species_associations:
        report_lines.append(
            f"|{row['species']}|{100.0 * float(row['relative_range_vs_median']):.4f}%|"
            f"{float(row['pearson_rate_correlation']):+.4f}|"
            f"{float(row['partial_rate_correlation_after_quadratic_xy']):+.4f}|"
            f"{float(row['univariate_slope_nm_m3_s_kmol']):+.6g}|"
            f"{float(row['univariate_slope_bootstrap_p05']):+.6g}–{float(row['univariate_slope_bootstrap_p95']):+.6g}|"
            f"{float(row['signed_rate_change_over_observed_concentration_range_nm_s']):+.6g}|"
        )
    report_lines.extend(
        [
        "",
        "## 係数と寄与",
        "",
        "モデルは中央値状態を基準に、A項、AB共同項、I阻害項の空間偏差を加算します。",
        "絶対成膜レートの共通部分は reference_rate に残し、観測範囲外のゼロ濃度へは外挿しません。",
        f"このケースの式（C は kmol/m³、出力は nm/s）は `{model_formula}` です。",
        "",
        "|項|有効係数|物理単位|bootstrap 5–95%|ゼロ率|",
        "|---|---:|---|---:|---:|",
        ]
    )
    for row in coefficient_rows:
        report_lines.append(
            f"|{row['term']}|{float(row['coefficient_physical']):.6g}|{row['physical_unit']}|"
            f"{float(row['bootstrap_physical_p05']):.6g}–{float(row['bootstrap_physical_p95']):.6g}|"
            f"{100.0 * float(row['bootstrap_zero_fraction']):.1f}%|"
        )
    report_lines.extend(
        [
            "",
            "点ごとの符号付き寄与と予測値は `spatial_contributions.csv` に保存しています。",
            "観測範囲内の寄与幅は `contribution_summary.csv` に要約しています。",
            "AB候補の積項は共同寄与であり、AとBへ恣意的に分割していません。",
            "",
            "## 妥当性",
            "",
            f"- 最大濃度種間相関 |r|: {max_predictor_corr:.5f}",
            f"- 最良モデル設計行列の条件数: {design_condition_number:.5g}",
            f"- 首位と2位のブロックCV RMSE差: {100.0 * second_gap:.2f}%",
            f"- 各空間foldで首位候補が勝った割合: {100.0 * best_fold_winner_fraction:.1f}% ({fold_winner_counts})",
            f"- 残差 Moran's I: {residual_spatial['moran_i']:.5f} (permutation p={residual_spatial['permutation_p_two_sided']:.4f})",
            "- 独立条件の検証データはありません。validation_1 は同一空間点の教師データとして使用しています。",
            "",
            "## 不足情報",
            "",
        ]
    )
    report_lines.extend([f"- {item}" for item in summary["missing_information"]])
    report_lines.extend(
        [
            "",
            "## 成果物",
            "",
            "- `model_ranking.csv`: 全役割候補の比較",
            "- `concentration_correlations.csv`: 濃度種間の共線性",
            "- `coefficients.csv`: 最良候補の係数とブートストラップ区間",
            "- `contribution_summary.csv`: 最良候補の観測範囲内寄与幅",
            "- `spatial_contributions.csv`: 座標別の濃度・寄与・予測・残差",
            "- `blocked_cv_predictions.csv`: 空間ブロックCV予測",
            "- `analysis_summary.json`: 機械可読サマリー",
            "- `cvd_condition_1_analysis.ipynb`: 再現可能なコンパニオンノートブック",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    notebook_path = _write_executed_notebook(output_dir, condition_path, validation_path)

    source_metadata = {
        "label": "CVD condition_1 concentration field and validation_1 deposition-rate map",
        "files": [str(condition_path), str(validation_path)],
        "filters": [f"Exact coordinate inner join; {target.size} matched rows", "No rows excluded"],
    }
    report_snapshot = {
        "title": "CVD空間相関は予測に有効だが、反応役割の確定には追加条件が必要",
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "status": "authored",
        "filters": [],
        "queries": {
            "fit_summary": {
                "rows": [
                    {
                        "model": best_candidate.model_id,
                        "points": int(target.size),
                        "rmse": best_metrics["rmse_nm_s"],
                        "r2": best_metrics["r2"],
                        "blockedCvRmse": best_cv,
                        "blockedCvImprovement": cv_improvement,
                        "foldWinnerFraction": best_fold_winner_fraction,
                        "assessment": reaction_interpretation_assessment,
                    }
                ],
                "source": {
                    **source_metadata,
                    "metricDefinitions": [
                        {
                            "label": "Blocked CV RMSE",
                            "definition": "Larger of angular-sector and radial-band cross-validation RMSE, in nm/s.",
                            "componentIds": ["fit-quality", "report-summary"],
                            "sourceLineage": [{"files": [str(condition_path), str(validation_path)]}],
                        },
                        {
                            "label": "Reaction interpretation assessment",
                            "definition": "Needs revision when a single condition cannot externally identify reaction roles or kinetic constants.",
                            "componentIds": ["fit-quality", "report-summary"],
                            "sourceLineage": [{"files": [str(condition_path), str(validation_path)]}],
                        },
                    ],
                },
            },
            "species_associations": {
                "rows": species_associations,
                "source": {
                    **source_metadata,
                    "metricDefinitions": [
                        {
                            "label": "Marginal concentration slope",
                            "definition": "Single-species spatial regression slope; association only, not a causal rate constant.",
                            "componentIds": ["species-association", "species-table"],
                            "sourceLineage": [{"files": [str(condition_path), str(validation_path)]}],
                        }
                    ],
                },
            },
            "model_ranking": {
                "rows": ranking_rows,
                "source": {
                    **source_metadata,
                    "metricDefinitions": [
                        {
                            "label": "Role candidate ranking",
                            "definition": "A/AI/AB/AIB candidates ranked by conservative spatial blocked-CV RMSE.",
                            "componentIds": ["model-ranking"],
                            "sourceLineage": [{"files": [str(condition_path), str(validation_path)]}],
                        }
                    ],
                },
            },
            "coefficients": {
                "rows": coefficient_rows,
                "source": {
                    **source_metadata,
                    "metricDefinitions": [
                        {
                            "label": "Effective spatial-response coefficient",
                            "definition": "Centered local-response coefficient around median concentration; not an elementary kinetic constant.",
                            "componentIds": ["coefficient-table", "report-methods"],
                            "sourceLineage": [{"files": [str(condition_path), str(validation_path)]}],
                        }
                    ],
                },
            },
            "contribution_summary": {
                "rows": contribution_summary_rows,
                "source": {
                    **source_metadata,
                    "metricDefinitions": [
                        {
                            "label": "Signed spatial contribution",
                            "definition": "Term contribution to predicted deposition-rate deviation from the median reference state, in nm/s.",
                            "componentIds": ["contribution-range"],
                            "sourceLineage": [{"files": [str(condition_path), str(validation_path)]}],
                        }
                    ],
                },
            },
            "fit_points": {
                "rows": contribution_rows,
                "source": {
                    **source_metadata,
                    "metricDefinitions": [
                        {
                            "label": "Measured deposition rate",
                            "definition": "dr_nm_per_sec from validation_1.csv at each matched coordinate.",
                            "componentIds": ["fit-scatter"],
                            "sourceLineage": [{"files": [str(validation_path)]}],
                        },
                        {
                            "label": "Predicted deposition rate",
                            "definition": "Full-data fitted centered role-response prediction in nm/s; not a held-out prediction.",
                            "componentIds": ["fit-scatter"],
                            "sourceLineage": [{"files": [str(condition_path), str(validation_path)]}],
                        },
                    ],
                },
            },
            "data_quality": {
                "rows": [{**data_quality, "coordinate_ranges": json.dumps(data_quality["coordinate_ranges"])}],
                "source": {
                    **source_metadata,
                    "metricDefinitions": [
                        {
                            "label": "Concentration/mole-fraction consistency",
                            "definition": "Relative agreement between sum of concentration_* and concentration_i/molef_i.",
                            "componentIds": ["data-quality"],
                            "sourceLineage": [{"files": [str(condition_path)]}],
                        }
                    ],
                },
            },
        },
    }
    _write_json(output_dir / "report_snapshot.json", report_snapshot)

    manifest = {
        "analysis": summary["analysis_kind"],
        "source": summary["source"],
        "artifacts": [
            "analysis_summary.json",
            "data_quality.json",
            "data_quality_species.csv",
            "model_ranking.csv",
            "concentration_correlations.csv",
            "coefficients.csv",
            "contribution_summary.csv",
            "spatial_contributions.csv",
            "blocked_cv_predictions.csv",
            "report.md",
            "report_snapshot.json",
            notebook_path.name,
            *plots,
        ],
    }
    _write_json(output_dir / "manifest.json", manifest)
    return summary


__all__ = [
    "FitResult",
    "RoleResponseCandidate",
    "analyze_cvd_spatial_case",
    "enumerate_role_response_candidates",
    "fit_candidate",
]
