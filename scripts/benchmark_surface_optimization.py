from __future__ import annotations

import argparse
import json
from pathlib import Path

from deposim_opt.surface_optimization_benchmark import (
    DEFAULT_LOSSES,
    DEFAULT_SAMPLERS,
    benchmark_surface_optimization,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare whole-wafer losses and samplers for one fixed reaction-role equation."
    )
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--conditions-file")
    parser.add_argument("--train-cases", type=int, nargs="+", default=(1, 2, 4, 5))
    parser.add_argument("--test-case", type=int, default=3)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--losses", nargs="+", default=DEFAULT_LOSSES)
    parser.add_argument("--samplers", nargs="+", default=DEFAULT_SAMPLERS)
    parser.add_argument("--trials", type=int, default=256)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume completed combinations from partial CSV checkpoints.",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--edge-uncertainty-ratio", type=float, default=1.0)
    parser.add_argument("--radial-uncertainty-power", type=float, default=2.0)
    parser.add_argument("--output", default="results/surface_optimization_benchmark")
    args = parser.parse_args()
    summary = benchmark_surface_optimization(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output),
        candidate_id=args.candidate_id,
        train_case_ids=tuple(args.train_cases),
        test_case_id=args.test_case,
        losses=tuple(args.losses),
        samplers=tuple(args.samplers),
        trials=args.trials,
        repetitions=args.repetitions,
        seed=args.seed,
        conditions_file=Path(args.conditions_file) if args.conditions_file else None,
        edge_uncertainty_ratio=args.edge_uncertainty_ratio,
        radial_uncertainty_power=args.radial_uncertainty_power,
        workers=args.workers,
        resume=args.resume,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[surface-optimization-benchmark] wrote artifacts to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
