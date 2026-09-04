from __future__ import annotations

import argparse
import json
from pathlib import Path

from deposim_opt.cvd_multicond_analysis import analyze_cvd_multicond_case


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fit multiple CVD conditions and evaluate one no-refit held-out condition."
    )
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--train-cases", type=int, nargs="+", default=(1, 2, 4, 5))
    parser.add_argument("--test-case", type=int, default=3)
    parser.add_argument("--response-structure", choices=("shared", "within_between", "select"), default="shared",
                        help="Empirical-power compatibility option: shared, separate within/between, or compare both.")
    parser.add_argument(
        "--response-model",
        choices=("surface_qss", "empirical_power"),
        default="surface_qss",
        help="Quasi-steady site-balance reductions (default) or the empirical power compatibility model.",
    )
    parser.add_argument(
        "--output",
        default="results/cvd_conditions_1_2_4_5_train_3_test",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    summary = analyze_cvd_multicond_case(
        data_dir=Path(args.data_dir),
        train_case_ids=tuple(args.train_cases),
        test_case_id=args.test_case,
        response_structure=args.response_structure,
        response_model=args.response_model,
        output_dir=Path(args.output),
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    print(json.dumps(summary["primary_split"], ensure_ascii=False, indent=2))
    print(f"[cvd-multicond-analysis] wrote artifacts to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
