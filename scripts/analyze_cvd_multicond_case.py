from __future__ import annotations

import argparse
import json
from pathlib import Path

from deposim_opt.cvd_multicond_analysis import analyze_cvd_multicond_case
from deposim_sim.models.aib_reductions import (
    available_surface_model_families,
    get_surface_model_family,
)
from deposim_sim.models.mass_transfer import (
    available_mass_transfer_models,
    get_mass_transfer_metadata,
)
from deposim_sim.models.net_models import available_net_models
from deposim_sim.models.process_models import primary_process_models


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fit multiple CVD conditions and evaluate one no-refit held-out condition."
    )
    parser.add_argument("--data-dir", default="data")
    parser.add_argument(
        "--conditions-file",
        help=(
            "Optional JSON manifest mapping condition ids to condition and validation CSV paths. "
            "Relative paths are resolved from the manifest directory."
        ),
    )
    parser.add_argument("--train-cases", type=int, nargs="+", default=(1, 2, 4, 5))
    parser.add_argument("--test-case", type=int, default=3)
    parser.add_argument(
        "--reaction-input",
        choices=(
            "bulk_concentration",
            "surface_concentration",
            "transport_capacity_flux",
        ),
        default="bulk_concentration",
        help=(
            "One explicitly selected local field supplied to every steady role equation. "
            "Input representation is not selected by chemical-model ranking."
        ),
    )
    parser.add_argument(
        "--spatial-response",
        choices=("none", "radial_quadratic", "radial_quartic"),
        default="none",
        help=(
            "Optional mean-preserving residual model fitted only after chemical "
            "model selection. It never changes role or equation ranking."
        ),
    )
    parser.add_argument(
        "--wafer-temperature-k",
        type=float,
        help=(
            "Optional uniform wafer temperature recorded as provenance; no radial "
            "temperature field is fitted."
        ),
    )
    parser.add_argument("--response-structure", choices=("shared", "within_between", "select"), default="shared",
                        help="Empirical-power compatibility option: shared, separate within/between, or compare both.")
    parser.add_argument(
        "--response-model",
        choices=("surface_compare", "empirical_power"),
        default="surface_compare",
        help="Compare physical QSS families (default) or use the empirical compatibility model.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help=(
            "Optional equation-family subset, for example --models aib_qss, "
            "or --models all for the registered equation census."
        ),
    )
    parser.add_argument(
        "--candidate-id",
        help="Run one exact candidate ID from a prior role_ranking.csv.",
    )
    parser.add_argument(
        "--output",
        default="results/cvd_conditions_1_2_4_5_train_3_test",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--loss",
        choices=(
            "mse",
            "wafer_normalized_mse",
            "wafer_normalized_mae",
            "symmetric_normalized_mse",
        ),
        default="mse",
        help="One whole-wafer objective used to fit every shared parameter set.",
    )
    parser.add_argument(
        "--sampler",
        choices=("pattern", "random", "tpe", "cmaes", "de", "pso", "levy", "cma_mae"),
        default="pattern",
        help="Shape-parameter optimizer; the separable nonnegative rate scale is profiled.",
    )
    parser.add_argument("--sampler-trials", type=int, default=256)
    parser.add_argument(
        "--sampler-options",
        default="{}",
        help="JSON object passed to the selected sampler.",
    )
    parser.add_argument(
        "--edge-uncertainty-ratio",
        type=float,
        default=1.0,
        help="Relative edge/center standard uncertainty; 1 disables radial reweighting.",
    )
    parser.add_argument("--radial-uncertainty-power", type=float, default=2.0)
    parser.add_argument(
        "--list-models",
        action="store_true",
        help=(
            "List steady response families, state-process models, transport "
            "closures, and net-film models, then exit."
        ),
    )
    args = parser.parse_args()
    sampler_options = json.loads(args.sampler_options)
    if not isinstance(sampler_options, dict):
        parser.error("--sampler-options must decode to a JSON object")

    if args.list_models:
        rows = []
        for name in available_surface_model_families():
            family = get_surface_model_family(name)
            rows.append(
                {
                    "equation_family": family.name,
                    "required_inputs": list(family.required_inputs),
                    "supported_processes": list(family.supported_processes),
                    "observables": list(family.observables),
                    "enabled_by_default": family.enabled_by_default,
                    "role_classes": list(family.required_classes),
                    "reduction_ids": list(family.reduction_ids),
                    "physical_question": family.physical_question,
                    "evidence_requirements": list(family.evidence_requirements),
                    "mechanism": family.mechanism,
                    "pathways": list(family.pathways),
                    "state_variables": list(family.state_variables),
                    "parameter_units": {
                        "rate_scale_nm_s": "nm/s",
                        "normalized_shape_parameters": "1",
                    },
                    "description": family.description,
                }
            )
        state_models = [
            {
                "model_name": info.name,
                "implementation": info.implementation,
                "processes": list(info.processes),
                "time_modes": list(info.time_modes),
                "mechanism": info.mechanism,
                "pathways": list(info.pathways),
                "state_variables": list(info.state_variables),
                "required_roles": list(info.required_roles),
                "quantity_units": dict(info.quantity_units),
                "steady_observable_equivalence": info.steady_observable_equivalence,
                "description": info.description,
            }
            for info in primary_process_models()
        ]
        print(
            json.dumps(
                {
                    "steady_response_families": rows,
                    "state_process_models": state_models,
                    "transport_closures": {
                        "steady_reaction_inputs": [
                            "bulk_concentration",
                            "surface_concentration",
                            "transport_capacity_flux",
                        ],
                        "active_role_pipeline": [
                            "direct_surface",
                            "fit_scalar",
                            "from_cfd_flux_sink",
                        ],
                        "supporting_km_utilities": list(
                            available_mass_transfer_models()
                        ),
                        "supporting_km_metadata": get_mass_transfer_metadata(),
                        "diffusivity_options": ["direct", "constant", "bosanquet"],
                        "units": {
                            "concentration": "kmol/m^3",
                            "km": "m/s",
                            "flux": "kmol/(m^2 s)",
                        },
                        "workflow_note": (
                            "the steady census fixes one reaction input before chemical "
                            "ranking; supporting_km_utilities are registered calculators "
                            "for the state-model pipeline"
                        ),
                    },
                    "net_film_models": {
                        "models": list(available_net_models()),
                        "rate_unit": "nm/s",
                        "workflow_note": (
                            "registered signed-rate composition utilities; not a "
                            "surface-reaction mechanism or role-selection candidate"
                        ),
                    },
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    summary = analyze_cvd_multicond_case(
        data_dir=Path(args.data_dir),
        train_case_ids=tuple(args.train_cases),
        test_case_id=args.test_case,
        response_structure=args.response_structure,
        response_model=args.response_model,
        output_dir=Path(args.output),
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
        conditions_file=Path(args.conditions_file) if args.conditions_file else None,
        model_families=tuple(args.models) if args.models else None,
        candidate_id=args.candidate_id,
        surface_loss=args.loss,
        surface_sampler=args.sampler,
        surface_trials=args.sampler_trials,
        surface_sampler_options=sampler_options,
        edge_uncertainty_ratio=args.edge_uncertainty_ratio,
        radial_uncertainty_power=args.radial_uncertainty_power,
        reaction_input=args.reaction_input,
        spatial_response=args.spatial_response,
        wafer_temperature_k=args.wafer_temperature_k,
    )
    print(json.dumps(summary["primary_split"], ensure_ascii=False, indent=2))
    print(f"[cvd-multicond-analysis] wrote artifacts to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
