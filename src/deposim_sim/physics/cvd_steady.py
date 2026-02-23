"""Legacy cvd_steady interface retired after AIB-ODE replacement."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class FieldBundle:
    C_ref: dict[str, Any]
    U: Any | None = None
    T: Any | None = None
    scalars: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CVDSteadyResult:
    thickness: Any
    deposition_rate: Any
    R: Any
    Cs: dict[str, Any]
    diagnostics: dict[str, Any]


def run_cvd_steady(*args: Any, **kwargs: Any) -> CVDSteadyResult:
    raise RuntimeError(
        "run_cvd_steady legacy path was retired. Use deposim_sim.pipeline.run_aib_from_spec with sim.model.name='aib_ode'."
    )


__all__ = ["FieldBundle", "CVDSteadyResult", "run_cvd_steady"]
