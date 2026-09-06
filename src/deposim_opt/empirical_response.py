"""Compatibility empirical role responses used by the CVD analyses."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np


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
        groups: dict[str, list[str]] = {}
        if self.A is not None:
            groups["A" if self.B is None else "AB"] = (
                [self.A] if self.B is None else sorted([self.A, self.B])
            )
        if self.I is not None:
            groups["I"] = [self.I]
        return groups

    def reductions(self) -> list[RoleResponseCandidate]:
        out: list[RoleResponseCandidate] = []
        if self.I is not None:
            if self.A is None:
                out.append(RoleResponseCandidate("baseline", "baseline"))
            elif self.B is None:
                out.append(RoleResponseCandidate(f"A:{self.A}", "A", A=self.A))
            else:
                out.append(
                    RoleResponseCandidate(
                        f"AB:{self.A}|{self.B}", "AB", A=self.A, B=self.B
                    )
                )
        if self.A is not None:
            out.append(
                RoleResponseCandidate(
                    f"I_response:{self.I}", "unassigned_driver", I=self.I
                )
                if self.I is not None
                else RoleResponseCandidate("baseline", "baseline")
            )
        return out


@dataclass(frozen=True)
class FitResult:
    coefficients: np.ndarray
    prediction: np.ndarray
    design: np.ndarray
    effect_names: tuple[str, ...]
    references: dict[str, float]
    active_effects: tuple[bool, ...]


def enumerate_role_response_candidates(
    species: Iterable[str], *, include_reductions: bool = False
) -> list[RoleResponseCandidate]:
    names = sorted(str(name) for name in species)
    candidates = [RoleResponseCandidate(model_id="baseline", class_id="baseline")]
    for a_name in names:
        candidates.append(RoleResponseCandidate(model_id=f"A:{a_name}", class_id="A", A=a_name))
    for a_name in names:
        for i_name in names:
            if i_name != a_name:
                candidates.append(
                    RoleResponseCandidate(
                        model_id=f"AI:{a_name}|{i_name}", class_id="AI", A=a_name, I=i_name
                    )
                )
    for index, a_name in enumerate(names):
        for b_name in names[index + 1 :]:
            candidates.append(
                RoleResponseCandidate(
                    model_id=f"AB:{a_name}|{b_name}", class_id="AB", A=a_name, B=b_name
                )
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
    unique = {
        tuple((key, tuple(value)) for key, value in candidate.effect_groups.items()): candidate
        for candidate in reversed(candidates)
    }
    return list(unique.values())[::-1]


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
            name: float(np.median(np.asarray(concentrations[name], dtype=float)[idx]))
            for name in sorted(set(involved))
        }
    if any(value <= 0.0 or not math.isfinite(value) for value in references.values()):
        raise ValueError(f"Positive finite concentration references are required: {references}")

    features: list[np.ndarray] = [np.ones(idx.size, dtype=float)]
    effect_names: list[str] = []
    if candidate.B is None and candidate.A is not None:
        features.append(concentrations[candidate.A][idx] / references[candidate.A] - 1.0)
        effect_names.append(f"A:{candidate.A}")
    elif candidate.A is not None and candidate.B is not None:
        features.append(
            concentrations[candidate.A][idx]
            * concentrations[candidate.B][idx]
            / (references[candidate.A] * references[candidate.B])
            - 1.0
        )
        effect_names.append(f"AB:{candidate.A}*{candidate.B}")
    if candidate.I is not None:
        features.append(-(concentrations[candidate.I][idx] / references[candidate.I] - 1.0))
        effect_names.append(f"I:{candidate.I}")
    return np.column_stack(features), tuple(effect_names), dict(references)


def fit_nonnegative_effects(
    design: np.ndarray,
    target: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    regularization: float = 0.0,
    penalty_matrix: np.ndarray | None = None,
) -> tuple[np.ndarray, tuple[bool, ...]]:
    """Fit an unconstrained intercept and exact nonnegative effect active sets."""

    x = np.asarray(design, dtype=float)
    y = np.asarray(target, dtype=float)
    if not np.isfinite(regularization) or regularization < 0:
        raise ValueError("regularization must be finite and nonnegative")
    weight = np.ones(y.size) if weights is None else np.asarray(weights, dtype=float)
    if (
        weight.shape != y.shape
        or np.any(weight < 0)
        or not np.isfinite(weight).all()
        or weight.sum() <= 0
    ):
        raise ValueError("weights must be finite, nonnegative and match the observations")
    root_weight = np.sqrt(weight / weight.sum())
    x = x * root_weight[:, None]
    y = y * root_weight
    if regularization:
        operator = (
            np.eye(x.shape[1])[1:]
            if penalty_matrix is None
            else np.asarray(penalty_matrix, dtype=float)
        )
        if (
            operator.ndim != 2
            or operator.shape[1] != x.shape[1]
            or not np.isfinite(operator).all()
            or np.any(operator[:, 0])
        ):
            raise ValueError(
                "penalty_matrix must be finite, match the design and leave the intercept unpenalized"
            )
        x = np.vstack([x, np.sqrt(regularization) * operator])
        y = np.concatenate([y, np.zeros(operator.shape[0])])
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
        sse = float(np.sum(np.square(x @ full - y)))
        if sse < best_sse:
            best_sse = sse
            best_coef = full
            best_mask = tuple(bool(value > 0) for value in full[1:])
    if best_coef is None or best_mask is None:  # pragma: no cover
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
        candidate, concentrations, train_idx, references=references
    )
    coefficients, active = fit_nonnegative_effects(
        design_train, target[train_idx]
    )
    design_pred, _, _ = _candidate_design(
        candidate, concentrations, pred_idx, references=fitted_references
    )
    return FitResult(
        coefficients=coefficients,
        prediction=design_pred @ coefficients,
        design=design_pred,
        effect_names=effect_names,
        references=fitted_references,
        active_effects=active,
    )


__all__ = [
    "FitResult",
    "RoleResponseCandidate",
    "enumerate_role_response_candidates",
    "fit_candidate",
    "fit_nonnegative_effects",
]
