"""Structured config for optimization runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


@dataclass
class ParameterSpec:
    name: str
    path: str
    initial: float
    transform: str = "identity"
    lower: float | None = None
    upper: float | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("parameter.name must be non-empty")
        if not self.path:
            raise ValueError("parameter.path must be non-empty")
        if self.lower is not None and self.upper is not None and self.lower > self.upper:
            raise ValueError("parameter.lower must be <= parameter.upper")


@dataclass
class ObjectiveSpec:
    metric: str = "thickness_rmse"
    robust_loss: str = "huber"

    def __post_init__(self) -> None:
        if not self.metric:
            raise ValueError("objective.metric must be non-empty")
        if not self.robust_loss:
            raise ValueError("objective.robust_loss must be non-empty")


@dataclass
class OptRunSpec:
    run_name: str = "opt_baseline"
    sim_config_name: str = "smoke"
    max_iters: int = 20
    parameters: list[ParameterSpec] = field(default_factory=list)
    objective: ObjectiveSpec = field(default_factory=ObjectiveSpec)

    def __post_init__(self) -> None:
        if not self.run_name:
            raise ValueError("run_name must be non-empty")
        if self.max_iters < 1:
            raise ValueError(f"max_iters must be >= 1, got {self.max_iters}")
        if not self.parameters:
            raise ValueError("parameters must be non-empty")


def _as_opt_run_spec(data: dict[str, Any]) -> OptRunSpec:
    params = [ParameterSpec(**entry) for entry in data.get("parameters", [])]
    objective_data = data.get("objective", {})
    objective = ObjectiveSpec(**objective_data)
    return OptRunSpec(
        run_name=str(data.get("run_name", "opt_baseline")),
        sim_config_name=str(data.get("sim_config_name", "smoke")),
        max_iters=int(data.get("max_iters", 20)),
        parameters=params,
        objective=objective,
    )


def load_opt_run_spec(path: str | Path) -> OptRunSpec:
    cfg = OmegaConf.load(Path(path))
    data = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(data, dict):
        raise ValueError("opt config must resolve to a mapping")
    return _as_opt_run_spec(data)


__all__ = ["ObjectiveSpec", "OptRunSpec", "ParameterSpec", "load_opt_run_spec"]
