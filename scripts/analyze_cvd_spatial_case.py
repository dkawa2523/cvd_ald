from __future__ import annotations

import argparse
import json
from pathlib import Path

from deposim_opt.cvd_spatial_analysis import analyze_cvd_spatial_case


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze one CVD spatial concentration/rate case.")
    parser.add_argument("--condition", default="data/condition_1.csv")
    parser.add_argument("--validation", default="data/validation_1.csv")
    parser.add_argument("--output", default="results/cvd_condition_1_analysis")
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    summary = analyze_cvd_spatial_case(
        condition_path=Path(args.condition),
        validation_path=Path(args.validation),
        output_dir=Path(args.output),
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    print(json.dumps(summary["best_model"], ensure_ascii=False, indent=2))
    print(f"[cvd-spatial-analysis] wrote artifacts to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
