"""Deterministic smoke run for CVD/ALD/multi-z output generation."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from deposim_schema import compose_sim_config

from .domain import build_domain_grid
from .pipeline import run_from_run_spec
from .run_manager import save_run_outputs
from .validation import validate_run_spec

try:  # pragma: no cover
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover
    np = None  # type: ignore[assignment]


def _require_numpy() -> None:
    if np is None:
        raise RuntimeError("NumPy is required for smoke run.")


def main(argv: Sequence[str] | None = None) -> int:
    _require_numpy()
    parser = argparse.ArgumentParser(description="Run deterministic deposim smoke simulation.")
    parser.add_argument("--config-name", default="smoke")
    parser.add_argument("overrides", nargs="*", help="Hydra-style key=value overrides")
    args = parser.parse_args(list(argv) if argv is not None else None)

    run_spec = compose_sim_config(args.config_name, overrides=args.overrides)
    validate_run_spec(run_spec)
    result = run_from_run_spec(run_spec)
    grid = build_domain_grid(run_spec.domain)
    run_dir = save_run_outputs(
        run_spec=run_spec,
        config_name=args.config_name,
        config_overrides=args.overrides,
        grid=grid,
        result=result,
    )
    print(f"[smoke] wrote run artifacts to: {run_dir}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
