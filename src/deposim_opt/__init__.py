"""Optimization/assimilation package (separate from deposim_sim)."""

from .class_compare import build_class_compare
from .enumerate_orders import enumerate_orders
from .enumerate_roles import RoleCandidate, class_id_from_roles, enumerate_roles
from .fit_optuna import fit_candidate_with_optuna
from .objective import evaluate_candidate_score
from .assimilate import run_synthetic_assimilation
from .run_fit import run_fit
from .transforms import (
    inverse_transform_value,
    positive_to_unconstrained,
    transform_value,
    unconstrained_to_positive,
    unconstrained_to_unit,
    unit_to_unconstrained,
)

__all__ = [
    "RoleCandidate",
    "build_class_compare",
    "class_id_from_roles",
    "enumerate_orders",
    "enumerate_roles",
    "evaluate_candidate_score",
    "fit_candidate_with_optuna",
    "inverse_transform_value",
    "run_fit",
    "run_synthetic_assimilation",
    "positive_to_unconstrained",
    "transform_value",
    "unconstrained_to_positive",
    "unconstrained_to_unit",
    "unit_to_unconstrained",
]
