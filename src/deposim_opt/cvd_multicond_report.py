"""Artifact rendering for the multi-condition CVD analysis.

This module formats already-computed results. It does not fit candidates, choose
roles, or inspect held-out targets during selection.
"""

from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
from typing import Any

import numpy as np

from deposim_sim.models.aib_reductions import surface_formula
from .surface_fit import SurfaceKineticFit
from .spatial_validation import EPS


def _uses_surface_response(response_model: str) -> bool:
    return response_model == "surface_compare"


def _effect_names(candidate: Any) -> tuple[str, ...]:
    names = ["common_total_order"]
    if candidate.A is not None and candidate.B is None:
        names.append(f"A:{candidate.A}")
    elif candidate.A is not None and candidate.B is not None:
        names.append(f"AB:{candidate.A}*{candidate.B}")
    if candidate.I is not None:
        names.append(f"I:{candidate.I}")
    return tuple(names)


def _write_notebook(
    output_dir: Path,
    train_case_ids: tuple[int, ...],
    test_case_id: int,
) -> Path:
    notebook_path = output_dir / "cvd_multicond_transfer_analysis.ipynb"
    evaluation = json.loads((output_dir / "analysis_summary.json").read_text(encoding="utf-8"))
    validity = evaluation["validity"]
    surface = _uses_surface_response(evaluation["primary_split"]["response_model"])
    method_text = (
        "Quasi-steady site-balance families use absolute species concentrations normalized by "
        "identification-data references. Training-condition CV selects the equation family, "
        "exact reduction, and anonymous-species role assignment. "
        if surface
        else
        "Log-rate uses total-concentration and optional species-fraction role terms. "
        "Training-condition CV selects roles, shared or within/between responses, and regularization. "
    )
    cells: list[dict[str, Any]] = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# CVD multi-condition transfer analysis\n",
                "\n",
                "## tl;dr\n",
                f"Conditions {list(train_case_ids)} identify the response model; condition {test_case_id} is held out without refitting. "
                f"Numerical prediction winner: {evaluation['primary_split']['selected_model']}. "
                f"Adopted model: {validity.get('adopted_model') or 'none'}. "
                f"Response structure: {evaluation['primary_split']['response_structure']}. "
                f"Decision: {validity['decision']}; role support: {validity['species_role_assessment']}. "
                "Outer condition folds assess the selection procedure; each fold fits its own model.\n",
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Context & Methods\n",
                "\n",
                method_text,
                "Term removal and alternative assignments are compared on the same conditions; crossing losses leave role support unresolved. "
                "Numerical loss ties prefer fewer effects and parameters. Angular/radial blocked CV and design rank are diagnostics.\n",
                "\n",
                "### Key Assumptions\n",
                "- Response coefficients transfer across conditions; there are no measured-rate corrections for an unseen condition.\n",
                "- The test condition is not used for fitting or model selection.\n",
                "- Coefficients are effective transfer responses, not elementary kinetics.\n",
                "- The selected concentration mode is recorded in analysis_summary.json; absolute wall flux is not calculated.\n",
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Data\n"],
        },
        {
            "cell_type": "code",
            "execution_count": 1,
            "metadata": {},
            "outputs": [],
            "source": [
                "import csv, json\n",
                "from pathlib import Path\n",
                "output = Path('.')\n",
                "with (output / 'analysis_summary.json').open(encoding='utf-8') as handle:\n",
                "    summary = json.load(handle)\n",
                "with (output / 'condition_quality.csv').open(encoding='utf-8') as handle:\n",
                "    quality = list(csv.DictReader(handle))\n",
                "print('train/test:', summary['primary_split']['train_cases'], '->', summary['primary_split']['test_case'])\n",
                "print('condition rows:', [(row['condition'], row['rows'], row['rate_unique_count']) for row in quality])\n",
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Results\n"],
        },
        {
            "cell_type": "code",
            "execution_count": 2,
            "metadata": {},
            "outputs": [],
            "source": [
                "primary = summary['primary_split']\n",
                "print('numerical prediction winner:', primary['selected_model'])\n",
                "print('adopted model:', summary['validity'].get('adopted_model') or 'none')\n",
                "print('common order:', primary['common_total_order'])\n",
                "print('test RMSE [nm/s]:', primary['test_rmse_nm_s'])\n",
                "print('test relative RMSE:', primary['test_relative_rmse_vs_test_mean'])\n",
                "print('test spatial R2:', primary['test_centered_spatial_r2'])\n",
                "print('species-role assessment:', summary['validity']['species_role_assessment'])\n",
                "print('model-structure envelope / test mean:', summary['model_structure_uncertainty']['mean_envelope_width_relative_to_test_mean'])\n",
                "print('equation families:')\n",
                "for row in summary.get('equation_family_assessments', []):\n",
                "    print(row['equation_family'], row['applicability_status'], row['condition_cv_rmse_nm_s'], row['outer_selection_frequency'])\n",
                "print('reaction mechanisms:')\n",
                "for row in summary.get('reaction_mechanism_assessments', []):\n",
                "    print(row['mechanism_id'], row['evaluation_status'], row['steady_representation'])\n",
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Figures\n",
                "\n",
                "![Condition transfer](plots/condition_mean_transfer.png)\n",
                "\n",
                "![Held-out fit](plots/test_measured_vs_predicted.png)\n",
                "\n",
                "![Held-out maps](plots/test_spatial_maps.png)\n",
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Takeaways\n",
                "\n",
                f"- Fixed-model holdout: {validity['fixed_model_assessment']['prediction_status']}; "
                f"spatial shape: {validity['fixed_model_assessment']['spatial_status']}.\n",
                f"- Outer selection procedure: {validity['procedure_assessment']['prediction_status']}. "
                f"Application criteria: {validity['application_status']}.\n",
                "- Raw species are candidate inputs. An unresolved steady AB response does not determine its A/B direction.\n",
                f"- Decision evidence: {validity['reason']}.\n",
            ],
        },
    ]
    execution_globals: dict[str, Any] = {"__name__": "__notebook__"}
    for cell in cells:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        buffer = StringIO()
        previous_cwd = Path.cwd()
        try:
            import os

            os.chdir(output_dir)
            with redirect_stdout(buffer):
                exec(compile(source, str(notebook_path), "exec"), execution_globals)
        finally:
            os.chdir(previous_cwd)
        text = buffer.getvalue()
        cell["outputs"] = (
            [{"name": "stdout", "output_type": "stream", "text": text.splitlines(keepends=True)}]
            if text
            else []
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
    parsed = json.loads(notebook_path.read_text(encoding="utf-8"))
    if parsed.get("nbformat") != 4 or not isinstance(parsed.get("cells"), list):
        raise RuntimeError("Generated notebook failed structural validation")
    return notebook_path


def _fit_formula(fit: Any | SurfaceKineticFit) -> str:
    if isinstance(fit, SurfaceKineticFit):
        return surface_formula(fit.candidate)
    if fit.response_structure == "within_between":
        return ("log(rate) = log(reference_rate) + mean_map(x) @ beta_between + "
                "(x - mean_map(x)) @ beta_within; x = " + str(list(_effect_names(fit.candidate))) +
                "; x uses log(total/reference), log(A fraction/reference) or log(AB fraction product/reference), "
                "and -log(I fraction/reference). Map means use Fluent inputs only.")
    formula = (
        "rate = reference_rate * (total_concentration / reference_total_concentration) "
        "** common_total_order"
    )
    candidate = fit.candidate
    if candidate.A is not None and candidate.B is None:
        formula += (
            f" * (fraction_{candidate.A} / reference_fraction_{candidate.A}) "
            f"** elasticity_A_{candidate.A}"
        )
    elif candidate.A is not None and candidate.B is not None:
        formula += (
            f" * ((fraction_{candidate.A} * fraction_{candidate.B}) / "
            f"(reference_fraction_{candidate.A} * reference_fraction_{candidate.B})) "
            f"** elasticity_AB_{candidate.A}_{candidate.B}"
        )
    if candidate.I is not None:
        formula += (
            f" * (fraction_{candidate.I} / reference_fraction_{candidate.I}) "
            f"** (-elasticity_I_{candidate.I})"
        )
    return formula


def _write_markdown_report(
    output_dir: Path, summary: dict[str, Any], coefficient_rows: list[dict[str, Any]],
    split_rows: list[dict[str, Any]],
) -> None:
    """Render measured results; interpretation is computed before presentation."""
    primary, validity = summary["primary_split"], summary["validity"]
    response_line = (
        f"Response model: `{primary['response_model']}`; parameters are observable dimensionless groups of the quasi-steady site balance."
        if _uses_surface_response(str(primary.get("response_model", "")))
        else (f"Response structure: `{primary['response_structure']}`; between/within total orders: "
              f"{primary['common_total_order']:.6g} / {primary['within_total_order']:.6g}.")
    )
    lines = [
        "# CVD multi-condition role evaluation", "",
        f"Training conditions: {primary['train_cases']}; fixed no-refit evaluation condition: {primary['test_case']}.",
        f"Numerical prediction winner: `{primary['selected_model']}`. Decision: `{validity['decision']}`.",
        f"Adopted model/candidate: `{validity.get('adopted_model') or 'none'}`.",
        response_line,
        f"Role evidence: `{validity['species_role_assessment']}`.",
        f"Chemical-model spatial prediction: `{validity['chemical_spatial_prediction']}`.",
        f"Post-selection spatial response: `{validity['spatial_response_assessment']}`.", "",
        "Selection uses condition refits in deposition-rate units; numerical loss ties prefer fewer effects and parameters.",
        "The fixed evaluation condition does not select coefficients, roles, or thresholds within this run. Repeated development use requires a new external condition for a final unbiased test.", "",
        f"Decision evidence: {validity['evaluation_scope']}. {validity['reason']}",
        "Outer condition refits evaluate the selection procedure, with a separately fitted model in each fold.", "",
        "## Prediction", "",
        f"Test RMSE: {primary['test_rmse_nm_s']:.6g} nm/s; centered R2: {primary['test_centered_spatial_r2']:.6g}.",
        f"Test spatial correlation: {primary['test_spatial_correlation']:.6g}; predicted/observed range: {primary['test_range_capture_fraction']:.6g}.",
        f"Condition-CV RMSE: {primary['train_condition_cv_rmse_nm_s']:.6g} nm/s.", "",
        "The outer-fold selected structures give a mean held-out prediction-envelope width of "
        f"{summary['model_structure_uncertainty']['mean_envelope_width_nm_s']:.6g} nm/s "
        f"({summary['model_structure_uncertainty']['mean_envelope_width_relative_to_test_mean']:.4%} of the test mean). "
        "This is model-selection sensitivity, not a confidence interval.", "",
        "![Measured, predicted, and residual maps](plots/test_spatial_maps.png)",
        "",
        "![Radial mean profile](plots/test_radial_profile.png)",
        "",
        "![Prediction spread across selected equations](plots/model_structure_prediction_spread.png)",
        "",
    ]
    workflow_rows = summary.get("workflow_layers", [])
    if workflow_rows:
        lines += [
            "## Workflow scope", "",
            "|Layer|Responsibility|Models|Current execution scope|Units|",
            "|---|---|---|---|---|",
        ]
        for row in workflow_rows:
            models = ", ".join(row.get("models", []))
            supporting = ", ".join(row.get("supporting_models", []))
            if supporting:
                models += f"; supporting: {supporting}"
            units = "; ".join(
                f"{key.removesuffix('_unit')}={value}"
                for key, value in row.items()
                if key.endswith("_unit")
            ) or "not applicable"
            lines.append(
                f"|{row['layer']}|{row['responsibility']}|{models}|"
                f"{row.get('execution_scope', '')}|{units}|"
            )
        lines += [
            "",
            "Surface equations, dynamic states, transport closure, and net-film "
            "composition answer different questions and are evaluated in their own layers.",
            "",
        ]
    family_rows = summary.get("equation_family_assessments", [])
    if family_rows:
        lines += [
            "## Equation families", "",
            "|Family|Use|Best condition-CV RMSE [nm/s]|Gap from best|Outer selection|Contrast|",
            "|---|---|---:|---:|---:|---|",
        ]
        for row in family_rows:
            rmse = row["condition_cv_rmse_nm_s"]
            gap = row["relative_rmse_gap_to_best"]
            lines.append(
                f"|{row['equation_family']}|{row['applicability_status']}|"
                f"{rmse:.6g}|{gap:.3%}|{row['outer_selection_frequency']:.1%}|"
                f"{row.get('contrast_status', 'not_assessed')}|"
            )
        lines += [
            "",
            "![Equation-family prediction error and selection frequency](plots/equation_family_comparison.png)",
            "",
            "![Optimization convergence for the best fit in each equation family](plots/optimization_convergence.png)",
            "",
            "![Best raw-species assignment in each equation family](plots/best_model_role_assignments.png)",
            "",
            "![Reaction steps represented by each fitted equation](plots/reaction_pathway_models.png)",
            "",
            "![Held-out predictions from alternative reaction models](plots/reaction_model_prediction_agreement.png)",
            "",
            "![Role assignments across condition refits](plots/role_selection_stability.png)",
            "",
            "![Correlation of condition-mean reaction inputs](plots/reaction_input_correlation.png)",
        ]
        lines += ["", "Physical reading by family:"]
        for row in family_rows:
            supported_text = "; ".join(row.get("supported_claims", [])) or "none"
            missing_text = "; ".join(row.get("missing_evidence", [])) or "none"
            lines.append(
                f"- `{row['equation_family']}` — {row['physical_question']} "
                f"Supported: {supported_text}. Unresolved: {missing_text}."
            )
        selected = summary["selected_model"]
        supported = selected.get("supported_claims", [])
        missing = selected.get("missing_evidence", [])
        lines += ["", "Supported by the numerical winner's equation:"]
        lines += [f"- {item}" for item in supported] or ["- No reaction-role effect is independently established."]
        lines += ["", "Evidence still required:"]
        lines += [f"- {item}" for item in missing] or ["- No model-specific evidence gap was detected."]
        lines.append("")
        prediction_comparison = summary.get("reaction_model_prediction_comparison", [])
        if prediction_comparison:
            lines += [
                "### Held-out prediction consequence of family choice", "",
                "|Family|Assigned A|Assigned B|Assigned I|Held-out RMSE [nm/s]|RMS difference from selected [nm/s]|Difference / selected RMSE|",
                "|---|---|---|---|---:|---:|---:|",
            ]
            for row in prediction_comparison:
                lines.append(
                    f"|{row['equation_family']}|{row.get('role_A') or '—'}|"
                    f"{row.get('role_B') or '—'}|{row.get('role_I') or '—'}|"
                    f"{float(row['held_out_rmse_nm_s']):.6g}|"
                    f"{float(row['rms_difference_from_selected_nm_s']):.6g}|"
                    f"{float(row['rms_difference_to_selected_rmse_ratio']):.3g}|"
                )
            lines += [
                "",
                "The difference ratio measures the predictive consequence of choosing another "
                "fitted family. It is neither a mechanism probability nor an uncertainty interval.",
                "",
            ]
    mechanism_rows = summary.get("reaction_mechanism_assessments", [])
    if mechanism_rows:
        lines += [
            "## Reaction mechanisms", "",
            "The pathway diagram shows the adsorption, blocking, and conversion stages represented by each fitted equation. Its arrows define the candidate mechanism; they do not by themselves establish elementary reactions.",
            "",
            "|Mechanism|Pathways|State|Evaluation|Steady representation|Condition-CV RMSE [nm/s]|",
            "|---|---|---|---|---|---:|",
        ]
        for row in mechanism_rows:
            lines.append(
                f"|{row['mechanism_id']}|{', '.join(row.get('pathways', []))}|"
                f"{', '.join(row.get('state_variables', [])) or 'none'}|"
                f"{row['evaluation_status']}|{row.get('steady_representation', '')}|"
                f"{float(row.get('condition_cv_rmse_nm_s', float('nan'))):.6g}|"
            )
        mvk = next(
            (row for row in mechanism_rows if row["mechanism_id"] == "mars_van_krevelen"),
            None,
        )
        if mvk is not None:
            lines += [
                "",
                "The steady Mars-van Krevelen projection is algebraically equivalent to the "
                "AIB AB no-desorption response for the present concentration-only data. It is "
                "therefore represented once and does not receive a duplicate model-selection vote.",
                "Time-resolved evidence required for Mars-van Krevelen discrimination:",
            ]
            lines += [f"- {item}" for item in mvk.get("missing_evidence", [])]
            lines.append("")
        state_comparison = summary.get("reaction_model_state_comparison", [])
        if state_comparison:
            lines += [
                "### Model-conditional surface states and pathways", "",
                "|Family|Quantity|Component|Mean|Minimum|Maximum|",
                "|---|---|---|---:|---:|---:|",
            ]
            for row in state_comparison:
                lines.append(
                    f"|{row['equation_family']}|{str(row['quantity']).replace('_', ' ')}|"
                    f"{str(row['component']).replace('_', ' ')}|"
                    f"{float(row['mean_fraction']):.6g}|"
                    f"{float(row['minimum_fraction']):.6g}|"
                    f"{float(row['maximum_fraction']):.6g}|"
                )
            lines += [
                "",
                "These fractions are latent quantities calculated within each fitted equation. "
                "They are not direct measurements of surface coverage or elementary pathway flux.",
                "",
            ]
    capability_rows = summary.get("capability_assessments", [])
    data_requirements = summary.get("data_requirements", [])
    if capability_rows:
        lines += [
            "## Data required for each target use",
            "",
            "|Target use|Current evidence|Measurements to add|Evidence required for use|",
            "|---|---|---|---|",
        ]
        for capability in capability_rows:
            name = str(capability["capability"])
            measurements = [
                str(row["required_measurement"])
                for row in data_requirements
                if row["capability"] == name
                and bool(row["needed_for_current_evidence"])
            ]
            lines.append(
                f"|{name.replace('_', ' ')}|{capability['current_status']}|"
                f"{'<br>'.join(measurements) or 'no additional measurement identified'}|"
                f"{capability['ready_when']}|"
            )
        lines += [
            "",
            "`data_requirements.csv` records the experimental variation, the ambiguity "
            "resolved by each measurement, and how it enters the workflow.",
            "",
            "![Condition reaction-input contrast](plots/condition_reaction_input_contrast.png)",
            "",
        ]
    lines += [
        "## Coefficients", "", f"`{summary['selected_model']['formula']}`", "",
        "|Term|Value|Conditional spatial bootstrap 5-95%|", "|---|---:|---|",
    ]
    for row in coefficient_rows:
        lines.append(f"|{row['term']}|{row['value']:.6g}|{row['bootstrap_p05']:.6g} - {row['bootstrap_p95']:.6g}|")
    lines += [
        "",
        "Intervals condition on the numerical prediction winner and supplied conditions; they do not include model-selection uncertainty.",
        "",
    ]
    if _uses_surface_response(str(primary.get("response_model", ""))):
        role_importance = summary.get("role_importance_and_stability", [])
        if role_importance:
            lines += [
                "### Role importance and assignment stability", "",
                "|Role|Raw species|Outer selection|RMS prediction change [nm/s]|Change / held-out RMSE|Reading|",
                "|---|---|---:|---:|---:|---|",
            ]
            for row in role_importance:
                ratio = float(row["prediction_change_to_rmse_ratio"])
                frequency = float(row["selection_frequency"])
                if ratio < 1.0 and frequency < 0.8:
                    reading = "unstable; small prediction consequence in the tested range"
                elif ratio >= 1.0 and frequency < 0.8:
                    reading = "influential assignment; unresolved across condition refits"
                elif ratio < 1.0:
                    reading = "stable; small prediction consequence in the tested range"
                else:
                    reading = "stable and influential in the tested range"
                lines.append(
                    f"|{row['role']}|{row['species']}|{frequency:.1%}|"
                    f"{float(row['rms_prediction_change_nm_s']):.6g}|{ratio:.3g}|{reading}|"
                )
            lines += [
                "",
                "The ratio of one-at-a-time prediction change to held-out RMSE is a scale "
                "comparison; 1 is a visual reference rather than a statistical cutoff.",
                "",
            ]
        parameter_sensitivity = summary.get("parameter_sensitivity_correlations", [])
        if parameter_sensitivity:
            diagonal = [
                row for row in parameter_sensitivity
                if row.get("parameter_1") == row.get("parameter_2")
            ]
            unique_pairs = [
                row for row in parameter_sensitivity
                if str(row.get("parameter_1")) < str(row.get("parameter_2"))
            ]
            lines += [
                "### Local kinetic-parameter sensitivity", "",
                "|Parameter|RMS log-rate sensitivity|Mean log-rate sensitivity|",
                "|---|---:|---:|",
            ]
            for row in diagonal:
                lines.append(
                    f"|{row['parameter_1']}|"
                    f"{float(row['rms_log_rate_sensitivity_1']):.6g}|"
                    f"{float(row['mean_log_rate_sensitivity_1']):.6g}|"
                )
            if unique_pairs:
                lines += [
                    "", "|Parameter pair|Correlation of local log-rate sensitivities|",
                    "|---|---:|",
                ]
                for row in unique_pairs:
                    lines.append(
                        f"|{row['parameter_1']} / {row['parameter_2']}|"
                        f"{float(row['pearson_correlation']):.6g}|"
                    )
            lines += [
                "",
                "Small sensitivity magnitude identifies a locally inactive fitted direction; "
                "strong correlation identifies coupled response directions. These are local "
                "diagnostics rather than global parameter intervals.",
                "",
            ]
        lines += [
            "The input sensitivity is the RMS change in prediction when one local role input is replaced by its fitted reference value. Because the reaction equations are nonlinear, these changes are not additive rate fractions.",
            "The importance-versus-stability figure compares that prediction change with the held-out RMSE and with the frequency of the same raw-species assignment across condition refits. A low and unstable point is predictively negligible over the supplied range; a high and unstable point is influential but not identified.",
            "Parameter loss slices re-optimize only the overall rate scale while one kinetic ratio is varied. The other kinetic ratios remain fixed, so the curves diagnose local flatness but are not joint confidence intervals.",
            "",
        ]
        for caption, name in (
            ("Sensitivity of predicted rate to each assigned input", "role_input_sensitivity.png"),
            ("Prediction importance and assignment stability", "role_importance_and_stability.png"),
            ("Predicted response while varying one assigned input", "role_response_curves.png"),
            ("Mean fitted site and pathway fractions on the held-out wafer", "reaction_state_summary.png"),
            ("Local sensitivity and correlation of kinetic parameters", "kinetic_parameter_sensitivity.png"),
            ("Loss when one kinetic parameter is varied", "parameter_loss_slices.png"),
            ("Selected-equation surface states", "selected_surface_state_maps.png"),
        ):
            if (output_dir / "plots" / name).exists():
                lines += [f"![{caption}](plots/{name})", ""]
    spatial = summary.get("spatial_response", {})
    if spatial.get("model") not in {None, "none"}:
        fixed = spatial["fixed_holdout"]
        lines += [
            "## Post-selection spatial response", "",
            f"Model: `{spatial['model']}`. It preserves the chemical condition mean and does not participate in reaction-role or equation selection.",
            f"Fixed-holdout centered R2: chemical {fixed['chemical_centered_r2']:.6g}; chemical + spatial {fixed['corrected_centered_r2']:.6g}.",
            f"Fixed-holdout RMSE: chemical {fixed['chemical_rmse_nm_s']:.6g} nm/s; chemical + spatial {fixed['corrected_rmse_nm_s']:.6g} nm/s.",
            f"Positive centered R2 on every outer condition: `{spatial['outer_condition_transfer_supported']}`.", "",
            "![Spatial-correction performance across held-out conditions](plots/spatial_correction_performance.png)", "",
            "![Centered chemical and spatial predictions](plots/test_spatial_response.png)", "",
            "![Residual maps before and after spatial correction](plots/spatial_residuals.png)", "",
            "![Spatial correction versus wafer radius](plots/spatial_correction_profile.png)", "",
        ]
    lines += [
        "## Condition refits",
        "",
        "|Held-out condition|Numerical winner|Response structure|Relative RMSE|Centered R2|",
        "|---|---|---|---:|---:|",
    ]
    for row in split_rows:
        model_label = str(row['selected_model']).replace("|", "\\|")
        lines.append(f"|{row['test_case']}|{model_label}|{row['response_structure']}|{row['test_relative_rmse_vs_test_mean']:.4%}|{row['test_centered_spatial_r2']:.4g}|")
    lines += ["", "## Interpretation", "",
              "Raw species are candidate inputs, not established chemical identities. Indistinguishable assignments remain unresolved.",
              ("Bulk concentrations are used as surface-response inputs; absolute wall flux is not calculated."
               if summary["model_inputs"].get("bulk_as_surface_approximation") else
               "Measured surface concentrations are used directly; absolute wall flux requires independent transport and stoichiometric information."),
              "Measurement uncertainty and independent process conditions are needed to assess practical identifiability.", "",
              "See role_summary.csv, role_ranking.csv, role_stability.csv, condition_scores.csv, model_structure_uncertainty.csv, and data_requirements.csv for decisions and evidence."]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")



def plot_multicond_results(
    output_dir: Path,
    cases: list[Any],
    condition_predictions: dict[int, np.ndarray],
    primary_train: Any,
    primary_test: Any,
    primary_prediction: np.ndarray,
    ranking: list[dict[str, Any]],
    equation_family_assessments: list[dict[str, Any]],
    split_sensitivity_rows: list[dict[str, Any]],
    model_uncertainty_rows: list[dict[str, Any]],
    test_prediction_rows: list[dict[str, Any]],
    spatial_prediction: np.ndarray | None = None,
    reaction_input_mode: str = "bulk_as_surface",
    optimization_history_rows: list[dict[str, Any]] | None = None,
    family_role_rows: list[dict[str, Any]] | None = None,
    input_correlation_rows: list[dict[str, Any]] | None = None,
    role_sensitivity_rows: list[dict[str, Any]] | None = None,
    role_importance_rows: list[dict[str, Any]] | None = None,
    role_response_rows: list[dict[str, Any]] | None = None,
    reaction_state_rows: list[dict[str, Any]] | None = None,
    family_prediction_rows: list[dict[str, Any]] | None = None,
    family_state_rows: list[dict[str, Any]] | None = None,
    parameter_sensitivity_rows: list[dict[str, Any]] | None = None,
    parameter_loss_rows: list[dict[str, Any]] | None = None,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    optimization_history_rows = optimization_history_rows or []
    family_role_rows = family_role_rows or []
    input_correlation_rows = input_correlation_rows or []
    role_sensitivity_rows = role_sensitivity_rows or []
    role_importance_rows = role_importance_rows or []
    role_response_rows = role_response_rows or []
    reaction_state_rows = reaction_state_rows or []
    family_prediction_rows = family_prediction_rows or []
    family_state_rows = family_state_rows or []
    parameter_sensitivity_rows = parameter_sensitivity_rows or []
    parameter_loss_rows = parameter_loss_rows or []
    family_labels = {
        "aib_qss": "Sequential A + B",
        "parallel_a_ab_qss": "Parallel A and A + B",
        "langmuir_hinshelwood_qss": "Langmuir–Hinshelwood",
    }
    role_labels = {
        "A": "Surface reactant A",
        "I": "Inhibitor I",
        "B": "Co-reactant B",
    }

    def case_reaction_inputs(case: Any) -> dict[str, np.ndarray]:
        if reaction_input_mode == "bulk_as_surface":
            return case.bulk_concentrations
        if reaction_input_mode == "direct_surface":
            return case.surface_concentrations
        if reaction_input_mode == "direct_flux":
            return case.transport_capacity_flux
        return case.bulk_concentrations

    input_is_flux = reaction_input_mode == "direct_flux"
    input_quantity_label = "transport-capacity flux" if input_is_flux else "concentration"
    input_unit_label = "kmol/(m² s)" if input_is_flux else "kmol/m³"

    figure, axis = plt.subplots(figsize=(8.0, 5.5), constrained_layout=True)
    for case in cases:
        color = "#2563eb" if case.case_id in primary_train.case_ids else "#d97706"
        case_prediction = np.asarray(condition_predictions[case.case_id], dtype=float)
        selected_fields = case_reaction_inputs(case)
        mean_total = float(
            np.mean(sum(np.asarray(values, dtype=float) for values in selected_fields.values()))
        )
        mean_measured = float(np.mean(case.rate))
        mean_prediction = float(np.mean(case_prediction))
        axis.plot(
            [mean_total, mean_total],
            [mean_measured, mean_prediction],
            color=color,
            alpha=0.45,
            linewidth=1.2,
        )
        axis.scatter(
            mean_total,
            mean_measured,
            s=80,
            marker="o",
            color=color,
            edgecolor="#111827",
            linewidth=0.6,
            label="measured identification" if case.case_id == primary_train.case_ids[0] else (
                "measured held-out" if case.case_id == primary_test.case_ids[0] else None
            ),
        )
        axis.scatter(
            mean_total,
            mean_prediction,
            s=85,
            marker="x",
            color=color,
            linewidth=1.8,
            label="model prediction" if case.case_id == cases[0].case_id else None,
        )
        axis.annotate(
            f"condition {case.case_id}",
            (mean_total, mean_measured),
            xytext=(7, 7),
            textcoords="offset points",
        )
    axis.set_xlabel(f"Mean total {input_quantity_label} [{input_unit_label}]")
    axis.set_ylabel("Mean deposition rate [nm/s]")
    axis.set_title("Condition-mean transfer of the numerical prediction winner")
    handles, labels = axis.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    axis.legend(unique.values(), unique.keys(), frameon=False)
    axis.grid(alpha=0.25)
    path = plot_dir / "condition_mean_transfer.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    outputs.append(path)

    family_rows = [
        row
        for row in equation_family_assessments
        if np.isfinite(float(row.get("condition_cv_rmse_nm_s", float("nan"))))
    ]
    if family_rows:
        family_rows = sorted(
            family_rows, key=lambda row: float(row["condition_cv_rmse_nm_s"])
        )
        labels = [family_labels.get(str(row["equation_family"]), str(row["equation_family"])) for row in family_rows]
        figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), constrained_layout=True)
        axes[0].barh(
            labels[::-1],
            [float(row["condition_cv_rmse_nm_s"]) for row in family_rows[::-1]],
            color="#4f83b5",
            edgecolor="#263746",
            linewidth=0.6,
        )
        axes[0].set_xlabel("Leave-one-condition-out RMSE [nm/s]")
        axes[0].set_title("Prediction error")
        axes[0].grid(axis="x", alpha=0.2)
        axes[1].barh(
            labels[::-1],
            [float(row["outer_selection_frequency"]) for row in family_rows[::-1]],
            color="#d8e4ee",
            edgecolor="#263746",
            linewidth=0.6,
        )
        axes[1].set_xlim(0.0, 1.0)
        axes[1].set_xlabel("Selection frequency")
        axes[1].set_title("Outer condition refits")
        axes[1].grid(axis="x", alpha=0.2)
        path = plot_dir / "equation_family_comparison.png"
        figure.savefig(path, dpi=180)
        plt.close(figure)
        outputs.append(path)

    if family_role_rows:
        families = list(
            dict.fromkeys(str(row["equation_family"]) for row in family_role_rows)
        )
        roles_by_family = {
            family: {
                str(row["role"]): str(row.get("species", ""))
                for row in family_role_rows
                if str(row["equation_family"]) == family
            }
            for family in families
        }
        states_by_family = {
            family: {
                str(row["component"]): float(row["mean_fraction"])
                for row in family_state_rows
                if str(row["equation_family"]) == family
            }
            for family in families
        }

        def state_node_label(
            family: str, component: str, symbol: str
        ) -> str:
            value = states_by_family.get(family, {}).get(component)
            return symbol if value is None else f"{symbol}\n{value:.3g}"

        def draw_node(axis: Any, x: float, y: float, label: str, color: str) -> None:
            axis.text(
                x,
                y,
                label,
                ha="center",
                va="center",
                fontsize=10,
                bbox={
                    "boxstyle": "round,pad=0.35",
                    "facecolor": color,
                    "edgecolor": "#334155",
                    "linewidth": 0.8,
                },
            )

        def draw_arrow(
            axis: Any,
            start: tuple[float, float],
            end: tuple[float, float],
            label: str = "",
            *,
            rad: float = 0.0,
        ) -> None:
            axis.annotate(
                "",
                xy=end,
                xytext=start,
                arrowprops={
                    "arrowstyle": "->",
                    "color": "#334155",
                    "linewidth": 1.4,
                    "connectionstyle": f"arc3,rad={rad}",
                    "shrinkA": 14,
                    "shrinkB": 14,
                },
            )
            if label:
                midpoint = (
                    0.5 * (start[0] + end[0]),
                    0.5 * (start[1] + end[1]) + 0.35 * rad,
                )
                axis.text(
                    midpoint[0],
                    midpoint[1],
                    label,
                    ha="center",
                    va="bottom" if rad >= 0 else "top",
                    fontsize=8.5,
                    color="#111827",
                )

        figure, axes = plt.subplots(
            1,
            len(families),
            figsize=(5.0 * len(families), 4.1),
            constrained_layout=True,
            squeeze=False,
        )
        for axis, family in zip(axes[0], families):
            roles = roles_by_family[family]
            state = states_by_family.get(family, {})
            axis.set_xlim(-0.07, 1.03)
            axis.set_ylim(0.0, 1.0)
            axis.axis("off")
            axis.set_title(family_labels.get(family, family), fontsize=12)
            if family == "langmuir_hinshelwood_qss":
                draw_node(
                    axis,
                    0.08,
                    0.50,
                    state_node_label(family, "theta_free", "vacant site *"),
                    "#f8fafc",
                )
                draw_node(
                    axis,
                    0.42,
                    0.72,
                    state_node_label(family, "theta_A", "A*"),
                    "#fed7aa",
                )
                draw_node(
                    axis,
                    0.42,
                    0.28,
                    state_node_label(family, "theta_B", "B*"),
                    "#bfdbfe",
                )
                draw_node(axis, 0.90, 0.50, "film + 2*", "#dcfce7")
                draw_arrow(axis, (0.12, 0.55), (0.37, 0.68), f"A: {roles.get('A', '')}")
                draw_arrow(axis, (0.12, 0.45), (0.37, 0.32), f"B: {roles.get('B', '')}")
                draw_arrow(axis, (0.48, 0.68), (0.84, 0.54))
                draw_arrow(axis, (0.48, 0.32), (0.84, 0.46), "A* + B*")
            else:
                draw_node(
                    axis,
                    0.08,
                    0.55,
                    state_node_label(family, "theta_free", "vacant site *"),
                    "#f8fafc",
                )
                draw_node(
                    axis,
                    0.47,
                    0.55,
                    state_node_label(family, "theta_A", "A*"),
                    "#fed7aa",
                )
                draw_node(axis, 0.90, 0.55, "film + *", "#dcfce7")
                draw_arrow(axis, (0.13, 0.57), (0.41, 0.57), f"A: {roles.get('A', '')}")
                draw_arrow(axis, (0.41, 0.49), (0.14, 0.49), "desorption", rad=-0.25)
                if family == "parallel_a_ab_qss":
                    fraction_a = state.get("path_A_fraction", float("nan"))
                    fraction_ab = state.get("path_AB_fraction", float("nan"))
                    label_a = "A-only" if not np.isfinite(fraction_a) else f"A-only  {fraction_a:.2f}"
                    label_ab = (
                        f"B: {roles.get('B', '')}"
                        if not np.isfinite(fraction_ab)
                        else f"B: {roles.get('B', '')}  {fraction_ab:.2f}"
                    )
                    draw_arrow(axis, (0.53, 0.55), (0.84, 0.55), label_a, rad=0.30)
                    draw_arrow(axis, (0.53, 0.55), (0.84, 0.55), label_ab, rad=-0.30)
                else:
                    draw_arrow(
                        axis,
                        (0.53, 0.55),
                        (0.84, 0.55),
                        f"B: {roles.get('B', '')}",
                    )
                inhibitor = roles.get("I", "")
                if inhibitor:
                    draw_node(
                        axis,
                        0.18,
                        0.16,
                        state_node_label(family, "theta_I", "I*"),
                        "#dcfce7",
                    )
                    draw_arrow(axis, (0.10, 0.48), (0.17, 0.23), f"I: {inhibitor}")
        figure.suptitle("Reaction steps represented by each fitted equation", fontsize=15)
        figure.text(
            0.5,
            0.01,
            "Arrows are terms in the fitted equation; they are not independently confirmed elementary steps.",
            ha="center",
            fontsize=9,
            color="#475569",
        )
        path = plot_dir / "reaction_pathway_models.png"
        figure.savefig(path, dpi=180)
        plt.close(figure)
        outputs.append(path)

    if family_prediction_rows:
        rows = sorted(
            family_prediction_rows,
            key=lambda row: float(row["held_out_rmse_nm_s"]),
        )
        labels = [
            family_labels.get(str(row["equation_family"]), str(row["equation_family"]))
            for row in rows
        ]
        positions = np.arange(len(rows), dtype=float)
        height = 0.34
        figure, axis = plt.subplots(figsize=(8.0, 4.2), constrained_layout=True)
        measurement_bars = axis.barh(
            positions + height / 2,
            [float(row["held_out_rmse_nm_s"]) for row in rows],
            height=height,
            color="#4f83b5",
            label="RMSE versus measurement",
        )
        difference_bars = axis.barh(
            positions - height / 2,
            [float(row["rms_difference_from_selected_nm_s"]) for row in rows],
            height=height,
            color="#d97706",
            label="RMS difference from selected model",
        )
        axis.set_yticks(positions, labels)
        axis.invert_yaxis()
        axis.set_xlabel("Rate difference [nm/s]")
        axis.set_title("Held-out predictions from alternative reaction models")
        axis.grid(axis="x", alpha=0.2)
        axis.legend(frameon=False)
        maximum_bar = max(
            max(float(row["held_out_rmse_nm_s"]) for row in rows),
            max(float(row["rms_difference_from_selected_nm_s"]) for row in rows),
        )
        axis.set_xlim(0.0, maximum_bar * 1.12)
        axis.bar_label(measurement_bars, fmt="%.2g", padding=3, fontsize=8)
        axis.bar_label(
            difference_bars,
            labels=[
                f"{float(row['rms_difference_from_selected_nm_s']):.2g}"
                if float(row["rms_difference_from_selected_nm_s"]) > EPS
                else ""
                for row in rows
            ],
            padding=3,
            fontsize=8,
        )
        path = plot_dir / "reaction_model_prediction_agreement.png"
        figure.savefig(path, dpi=180)
        plt.close(figure)
        outputs.append(path)

    if optimization_history_rows:
        figure, axis = plt.subplots(figsize=(7.2, 4.7), constrained_layout=True)
        for family in dict.fromkeys(
            str(row["equation_family"]) for row in optimization_history_rows
        ):
            rows = [
                row
                for row in optimization_history_rows
                if str(row["equation_family"]) == family
            ]
            rows.sort(key=lambda row: int(row["trial"]))
            axis.plot(
                [int(row["trial"]) for row in rows],
                [float(row["best_error"]) for row in rows],
                linewidth=1.7,
                label=family_labels.get(family, family),
            )
        axis.set_xlabel("Objective evaluations")
        error_name = str(optimization_history_rows[0].get("best_error_name", ""))
        axis.set_ylabel(
            "Best training RMSE [nm/s]"
            if error_name == "training_rmse_nm_s"
            else "Best normalized fitting error"
        )
        axis.set_yscale("log")
        axis.set_title("Optimization convergence")
        axis.grid(alpha=0.2)
        axis.legend(frameon=False)
        path = plot_dir / "optimization_convergence.png"
        figure.savefig(path, dpi=180)
        plt.close(figure)
        outputs.append(path)

    if family_role_rows:
        families = list(
            dict.fromkeys(str(row["equation_family"]) for row in family_role_rows)
        )
        roles = ("A", "I", "B")
        species_for_color = list(cases[0].species)
        matrix = np.zeros((len(families), len(roles)), dtype=int)
        text_matrix = np.full(matrix.shape, "—", dtype=object)
        for row in family_role_rows:
            i = families.index(str(row["equation_family"]))
            j = roles.index(str(row["role"]))
            species_name = str(row.get("species", ""))
            text_matrix[i, j] = species_name or "—"
            if species_name in species_for_color:
                matrix[i, j] = species_for_color.index(species_name) + 1
        colors = ["#f8fafc", "#c6dbef", "#fdd0a2", "#c7e9c0", "#dadaeb"]
        cmap = ListedColormap(colors[: len(species_for_color) + 1])
        figure, axis = plt.subplots(
            figsize=(8.0, max(3.2, 0.75 * len(families) + 1.6)),
            constrained_layout=True,
        )
        axis.imshow(
            matrix,
            cmap=cmap,
            vmin=-0.5,
            vmax=len(species_for_color) + 0.5,
            aspect="auto",
        )
        axis.set_xticks(np.arange(3), [role_labels[role] for role in roles])
        axis.set_yticks(
            np.arange(len(families)),
            [family_labels.get(family, family) for family in families],
        )
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                axis.text(j, i, text_matrix[i, j], ha="center", va="center")
        axis.set_title("Best species assignment in each reaction model")
        axis.tick_params(length=0)
        for spine in axis.spines.values():
            spine.set_visible(False)
        path = plot_dir / "best_model_role_assignments.png"
        figure.savefig(path, dpi=180)
        plt.close(figure)
        outputs.append(path)

    species = list(cases[0].species)
    condition_matrix = np.asarray(
        [
            [float(np.mean(case_reaction_inputs(case)[name])) for name in species]
            for case in cases
        ],
        dtype=float,
    )
    reference = np.exp(np.mean(np.log(np.maximum(condition_matrix, EPS)), axis=0))
    contrast = np.log10(np.maximum(condition_matrix / reference, EPS))
    limit = max(float(np.max(np.abs(contrast))), EPS)
    figure, axis = plt.subplots(figsize=(6.8, 4.4), constrained_layout=True)
    image = axis.imshow(contrast, cmap="coolwarm", vmin=-limit, vmax=limit, aspect="auto")
    axis.set_xticks(np.arange(len(species)), species)
    axis.set_yticks(np.arange(len(cases)), [str(case.case_id) for case in cases])
    axis.set_xlabel("Fluent species")
    axis.set_ylabel("Condition")
    axis.set_title(f"Condition {input_quantity_label} contrast")
    colorbar = figure.colorbar(image, ax=axis, shrink=0.9)
    colorbar.set_label("log₁₀(condition mean / geometric mean)")
    path = plot_dir / "condition_reaction_input_contrast.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    outputs.append(path)

    if input_correlation_rows:
        correlation = np.full((len(species), len(species)), np.nan, dtype=float)
        for row in input_correlation_rows:
            i = species.index(str(row["species_1"]))
            j = species.index(str(row["species_2"]))
            correlation[i, j] = float(row["pearson_correlation"])
        figure, axis = plt.subplots(figsize=(5.6, 4.7), constrained_layout=True)
        image = axis.imshow(correlation, cmap="coolwarm", vmin=-1.0, vmax=1.0)
        axis.set_xticks(np.arange(len(species)), species)
        axis.set_yticks(np.arange(len(species)), species)
        for i in range(len(species)):
            for j in range(len(species)):
                value = correlation[i, j]
                axis.text(
                    j,
                    i,
                    "n/a" if not np.isfinite(value) else f"{value:.3f}",
                    ha="center",
                    va="center",
                    color="white" if np.isfinite(value) and abs(value) > 0.65 else "#111827",
                )
        axis.set_title(f"Correlation of condition-mean {input_quantity_label}")
        colorbar = figure.colorbar(image, ax=axis, shrink=0.88)
        colorbar.set_label("Pearson correlation")
        path = plot_dir / "reaction_input_correlation.png"
        figure.savefig(path, dpi=180)
        plt.close(figure)
        outputs.append(path)

    if split_sensitivity_rows:
        columns = ["none", *species]
        matrix = np.zeros((3, len(columns)), dtype=float)
        for row_index, role in enumerate(("A", "I", "B")):
            selected = [
                str(row.get(f"selected_role_{role}") or "none")
                for row in split_sensitivity_rows
            ]
            for value in columns:
                matrix[row_index, columns.index(value)] = selected.count(value) / len(selected)
        figure, axis = plt.subplots(figsize=(7.2, 3.6), constrained_layout=True)
        image = axis.imshow(matrix, cmap="Blues", vmin=0.0, vmax=1.0, aspect="auto")
        axis.set_xticks(np.arange(len(columns)), columns)
        axis.set_yticks(np.arange(3), [role_labels[role] for role in ("A", "I", "B")])
        axis.set_xlabel("Assigned species")
        axis.set_ylabel("Reaction role")
        axis.set_title("Species assignment across held-out conditions")
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if matrix[i, j] > 0.0:
                    axis.text(
                        j,
                        i,
                        f"{matrix[i, j]:.0%}",
                        ha="center",
                        va="center",
                        color="white" if matrix[i, j] >= 0.55 else "#111827",
                    )
        colorbar = figure.colorbar(image, ax=axis, shrink=0.9)
        colorbar.set_label("Selection frequency")
        path = plot_dir / "role_selection_stability.png"
        figure.savefig(path, dpi=180)
        plt.close(figure)
        outputs.append(path)

    radius = np.sqrt(np.sum(np.square(primary_test.xyz[:, :2]), axis=1))
    wafer_radius = max(float(np.max(radius)), EPS)
    normalized_radius = radius / wafer_radius
    normalized_xy = primary_test.xyz[:, :2] / wafer_radius
    if float(np.ptp(radius)) <= EPS:
        radial_group_ids = np.zeros(radius.shape, dtype=int)
    else:
        shell_count = min(6, max(3, int(round(np.sqrt(radius.size)))))
        edges = np.linspace(float(np.min(radius)), float(np.max(radius)), shell_count + 1)
        radial_group_ids = np.digitize(radius, edges[1:-1], right=True)
    radial_rows = []
    for group in np.unique(radial_group_ids):
        mask = radial_group_ids == group
        radial_rows.append(
            (
                float(np.mean(normalized_radius[mask])),
                float(np.mean(primary_test.rate[mask])),
                float(np.std(primary_test.rate[mask])),
                float(np.mean(primary_prediction[mask])),
                float(np.std(primary_prediction[mask])),
            )
        )
    figure, axis = plt.subplots(figsize=(7.2, 4.7), constrained_layout=True)
    axis.errorbar(
        [row[0] for row in radial_rows],
        [row[1] for row in radial_rows],
        yerr=[row[2] for row in radial_rows],
        marker="o",
        color="#263746",
        capsize=3,
        label="Measured",
    )
    axis.errorbar(
        [row[0] for row in radial_rows],
        [row[3] for row in radial_rows],
        yerr=[row[4] for row in radial_rows],
        marker="s",
        markerfacecolor="white",
        color="#4f83b5",
        capsize=3,
        label="Prediction",
    )
    axis.set_xlabel("Normalized wafer radius, r/R")
    axis.set_ylabel("Mean deposition rate [nm/s]")
    axis.set_title(f"Condition {primary_test.case_ids[0]} radial profile")
    axis.legend(frameon=False)
    axis.grid(alpha=0.2)
    path = plot_dir / "test_radial_profile.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    outputs.append(path)

    if model_uncertainty_rows:
        ux = np.asarray([float(row["x"]) for row in model_uncertainty_rows])
        uy = np.asarray([float(row["y"]) for row in model_uncertainty_rows])
        width = np.asarray(
            [float(row["structure_envelope_width_nm_s"]) for row in model_uncertainty_rows]
        )
        figure, axis = plt.subplots(figsize=(5.8, 4.8), constrained_layout=True)
        image = axis.scatter(
            ux / wafer_radius,
            uy / wafer_radius,
            c=width,
            cmap="viridis",
            s=65,
        )
        axis.set_xlabel("x/R")
        axis.set_ylabel("y/R")
        axis.set_aspect("equal", adjustable="box")
        axis.set_title("Prediction spread across selected equations")
        colorbar = figure.colorbar(image, ax=axis, shrink=0.9)
        colorbar.set_label("Prediction range [nm/s]")
        path = plot_dir / "model_structure_prediction_spread.png"
        figure.savefig(path, dpi=180)
        plt.close(figure)
        outputs.append(path)

    if test_prediction_rows:
        state_fields = [
            ("theta_A", "Adsorbed-state fraction θA"),
            ("theta_I", "Blocked-site fraction θI"),
            ("dimensionless_response", "Normalized rate"),
        ]
        available = [
            (key, label)
            for key, label in state_fields
            if key in test_prediction_rows[0]
            and np.all(np.isfinite([float(row[key]) for row in test_prediction_rows]))
        ]
        if available:
            figure, axes = plt.subplots(
                1, len(available), figsize=(4.5 * len(available), 4.3), constrained_layout=True
            )
            axes = np.atleast_1d(axes)
            xy = normalized_xy
            for axis, (key, label) in zip(axes, available):
                values = np.asarray([float(row[key]) for row in test_prediction_rows])
                limits = {"vmin": 0.0, "vmax": 1.0} if key in {"theta_A", "theta_I"} else {}
                image = axis.scatter(
                    xy[:, 0], xy[:, 1], c=values, cmap="viridis", s=65, **limits
                )
                axis.set_title(label)
                axis.set_xlabel("x/R")
                axis.set_ylabel("y/R")
                axis.set_aspect("equal", adjustable="box")
                figure.colorbar(image, ax=axis, shrink=0.82)
            path = plot_dir / "selected_surface_state_maps.png"
            figure.savefig(path, dpi=180)
            plt.close(figure)
            outputs.append(path)

    if reaction_state_rows:
        state_group_specs = [
            ("site_fraction", "Mean site fraction"),
            ("pathway_fraction", "Mean reaction-path fraction"),
        ]
        available_groups = [
            (key, title)
            for key, title in state_group_specs
            if any(str(row["quantity"]) == key for row in reaction_state_rows)
        ]
        figure, axes = plt.subplots(
            1,
            len(available_groups),
            figsize=(5.0 * len(available_groups), 4.1),
            constrained_layout=True,
        )
        axes = np.atleast_1d(axes)
        for axis, (group, title) in zip(axes, available_groups):
            rows = [row for row in reaction_state_rows if str(row["quantity"]) == group]
            positions = np.arange(len(rows))
            mean = np.asarray([float(row["mean_fraction"]) for row in rows])
            low = np.asarray([float(row["minimum_fraction"]) for row in rows])
            high = np.asarray([float(row["maximum_fraction"]) for row in rows])
            axis.errorbar(
                mean,
                positions,
                xerr=np.vstack((mean - low, high - mean)),
                fmt="o",
                color="#2563eb",
                ecolor="#94a3b8",
                capsize=3,
            )
            axis.set_yticks(positions, [str(row["component"]) for row in rows])
            axis.set_xlim(-0.03, 1.03)
            axis.set_xlabel("Fraction")
            axis.set_title(title)
            axis.grid(axis="x", alpha=0.2)
        path = plot_dir / "reaction_state_summary.png"
        figure.savefig(path, dpi=180)
        plt.close(figure)
        outputs.append(path)

    if role_sensitivity_rows:
        labels = [
            f"{role_labels.get(str(row['role']), row['role'])}: {row['species']}"
            for row in role_sensitivity_rows
        ]
        values = [float(row["rms_prediction_change_nm_s"]) for row in role_sensitivity_rows]
        figure, axis = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
        bars = axis.barh(labels[::-1], values[::-1], color="#4f83b5", edgecolor="#263746")
        axis.set_xlabel("RMS change in predicted rate [nm/s]")
        axis.set_title(f"Sensitivity to observed {input_quantity_label} variation")
        axis.grid(axis="x", alpha=0.2)
        if values:
            axis.set_xlim(0.0, max(values) * 1.16)
        for bar, value in zip(bars, values[::-1]):
            axis.text(
                bar.get_width(),
                bar.get_y() + bar.get_height() / 2,
                f" {value:.2g}",
                va="center",
                ha="left",
            )
        path = plot_dir / "role_input_sensitivity.png"
        figure.savefig(path, dpi=180)
        plt.close(figure)
        outputs.append(path)

    if role_importance_rows:
        figure, axis = plt.subplots(figsize=(7.0, 4.7), constrained_layout=True)
        for row in role_importance_rows:
            x = float(row["selection_frequency"])
            y = max(float(row["prediction_change_to_rmse_ratio"]), EPS)
            label = f"{role_labels.get(str(row['role']), row['role'])}: {row['species']}"
            axis.scatter(x, y, s=85, color="#2563eb", edgecolor="#1f2937")
            axis.annotate(label, (x, y), xytext=(7, 5), textcoords="offset points")
        axis.axhline(
            1.0,
            color="#64748b",
            linestyle="--",
            linewidth=1.2,
        )
        axis.text(
            1.0,
            1.12,
            "equal to held-out RMSE",
            ha="right",
            va="bottom",
            fontsize=9,
            color="#475569",
        )
        axis.set_xlim(-0.03, 1.03)
        axis.set_yscale("log")
        axis.set_xlabel("Selection frequency across held-out conditions")
        axis.set_ylabel("RMS prediction change / held-out RMSE")
        axis.set_title("Prediction importance and assignment stability")
        axis.grid(alpha=0.2)
        axis.margins(y=0.12)
        path = plot_dir / "role_importance_and_stability.png"
        figure.savefig(path, dpi=180)
        plt.close(figure)
        outputs.append(path)

    if parameter_sensitivity_rows:
        parameters = list(
            dict.fromkeys(str(row["parameter_1"]) for row in parameter_sensitivity_rows)
        )
        labels = [name.replace("_", " ") for name in parameters]
        rms_values = [
            float(
                next(
                    row["rms_log_rate_sensitivity_1"]
                    for row in parameter_sensitivity_rows
                    if str(row["parameter_1"]) == name
                )
            )
            for name in parameters
        ]
        correlation = np.full((len(parameters), len(parameters)), np.nan, dtype=float)
        for row in parameter_sensitivity_rows:
            i = parameters.index(str(row["parameter_1"]))
            j = parameters.index(str(row["parameter_2"]))
            correlation[i, j] = float(row["pearson_correlation"])
        figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.3), constrained_layout=True)
        axes[0].barh(labels[::-1], rms_values[::-1], color="#4f83b5")
        axes[0].set_xlabel("RMS |∂ ln(rate) / ∂ ln(parameter)|")
        axes[0].set_title("Local parameter sensitivity")
        axes[0].grid(axis="x", alpha=0.2)
        image = axes[1].imshow(correlation, cmap="coolwarm", vmin=-1.0, vmax=1.0)
        axes[1].set_xticks(np.arange(len(parameters)), labels, rotation=30, ha="right")
        axes[1].set_yticks(np.arange(len(parameters)), labels)
        for i in range(len(parameters)):
            for j in range(len(parameters)):
                value = correlation[i, j]
                axes[1].text(
                    j,
                    i,
                    "n/a" if not np.isfinite(value) else f"{value:.2f}",
                    ha="center",
                    va="center",
                    color="white" if np.isfinite(value) and abs(value) > 0.65 else "#111827",
                )
        axes[1].set_title("Correlation of parameter sensitivities")
        colorbar = figure.colorbar(image, ax=axes[1], shrink=0.84)
        colorbar.set_label("Pearson correlation")
        path = plot_dir / "kinetic_parameter_sensitivity.png"
        figure.savefig(path, dpi=180)
        plt.close(figure)
        outputs.append(path)

    if parameter_loss_rows:
        figure, axis = plt.subplots(figsize=(7.2, 4.7), constrained_layout=True)
        for parameter in dict.fromkeys(
            str(row["parameter"]) for row in parameter_loss_rows
        ):
            rows = [
                row for row in parameter_loss_rows if str(row["parameter"]) == parameter
            ]
            rows.sort(key=lambda row: float(row["factor_from_fitted_value"]))
            axis.plot(
                [float(row["factor_from_fitted_value"]) for row in rows],
                [max(float(row["fitting_error"]), EPS) for row in rows],
                linewidth=1.8,
                label=parameter.replace("_", " "),
            )
        axis.axvline(1.0, color="#64748b", linestyle="--", linewidth=1.0)
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel("Parameter value / fitted value")
        error_name = str(parameter_loss_rows[0].get("fitting_error_name", ""))
        axis.set_ylabel(
            "Training RMSE [nm/s]"
            if error_name == "training_rmse_nm_s"
            else "Normalized fitting error"
        )
        axis.set_title("Loss when one kinetic parameter is varied")
        axis.grid(alpha=0.2)
        axis.legend(frameon=False)
        path = plot_dir / "parameter_loss_slices.png"
        figure.savefig(path, dpi=180)
        plt.close(figure)
        outputs.append(path)

    if role_response_rows:
        roles = list(dict.fromkeys(str(row["role"]) for row in role_response_rows))
        figure, axes = plt.subplots(
            1,
            len(roles),
            figsize=(4.5 * len(roles), 4.0),
            constrained_layout=True,
            squeeze=False,
        )
        all_rates = np.asarray(
            [float(row["predicted_rate_nm_s"]) for row in role_response_rows]
        )
        rate_margin = max(float(np.ptp(all_rates)) * 0.05, EPS)
        common_rate_limits = (
            float(np.min(all_rates)) - rate_margin,
            float(np.max(all_rates)) + rate_margin,
        )
        for axis, role in zip(axes[0], roles):
            rows = [row for row in role_response_rows if str(row["role"]) == role]
            species_name = str(rows[0]["species"])
            axis.plot(
                [float(row["reaction_input"]) for row in rows],
                [float(row["predicted_rate_nm_s"]) for row in rows],
                color="#2563eb",
                linewidth=2.0,
            )
            axis.axvline(
                float(rows[0]["reference_input"]),
                color="#64748b",
                linestyle="--",
                linewidth=1.0,
            )
            axis.set_xlabel(f"{species_name} [{rows[0]['reaction_input_unit']}]")
            axis.set_ylabel("Predicted deposition rate [nm/s]")
            axis.set_title(role_labels.get(role, role))
            axis.set_ylim(*common_rate_limits)
            axis.ticklabel_format(
                axis="x",
                style="sci",
                scilimits=(0, 0),
                useMathText=True,
            )
            axis.grid(alpha=0.2)
        path = plot_dir / "role_response_curves.png"
        figure.savefig(path, dpi=180)
        plt.close(figure)
        outputs.append(path)

    figure, axis = plt.subplots(figsize=(6.6, 6.0), constrained_layout=True)
    axis.scatter(primary_test.rate, primary_prediction, s=55, color="#2563eb", edgecolor="#1f2937", linewidth=0.6)
    low = min(float(np.min(primary_test.rate)), float(np.min(primary_prediction)))
    high = max(float(np.max(primary_test.rate)), float(np.max(primary_prediction)))
    axis.plot([low, high], [low, high], linestyle="--", color="#334155", linewidth=1.5)
    axis.set_xlabel("Measured deposition rate [nm/s]")
    axis.set_ylabel("Held-out prediction [nm/s]")
    axis.set_title(
        f"Condition {primary_test.case_ids[0]}: measured versus no-refit prediction"
    )
    axis.grid(alpha=0.25)
    path = plot_dir / "test_measured_vs_predicted.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    outputs.append(path)

    figure, axes = plt.subplots(1, 3, figsize=(14.0, 4.5), constrained_layout=True)
    xy = normalized_xy
    rate_low = min(float(np.min(primary_test.rate)), float(np.min(primary_prediction)))
    rate_high = max(float(np.max(primary_test.rate)), float(np.max(primary_prediction)))
    measured_plot = axes[0].scatter(xy[:, 0], xy[:, 1], c=primary_test.rate, cmap="viridis", vmin=rate_low, vmax=rate_high, s=65)
    predicted_plot = axes[1].scatter(xy[:, 0], xy[:, 1], c=primary_prediction, cmap="viridis", vmin=rate_low, vmax=rate_high, s=65)
    residual = primary_prediction - primary_test.rate
    residual_limit = max(float(np.max(np.abs(residual))), EPS)
    residual_plot = axes[2].scatter(xy[:, 0], xy[:, 1], c=residual, cmap="coolwarm", vmin=-residual_limit, vmax=residual_limit, s=65)
    axes[0].set_title("Measured rate [nm/s]")
    axes[1].set_title("Held-out prediction [nm/s]")
    axes[2].set_title("Residual: predicted - measured [nm/s]")
    for axis_item in axes:
        axis_item.set_xlabel("x/R")
        axis_item.set_ylabel("y/R")
        axis_item.set_aspect("equal", adjustable="box")
    figure.colorbar(measured_plot, ax=axes[:2], shrink=0.86)
    figure.colorbar(residual_plot, ax=axes[2], shrink=0.86)
    path = plot_dir / "test_spatial_maps.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    outputs.append(path)

    if spatial_prediction is not None:
        corrected = np.asarray(spatial_prediction, dtype=float)
        measured_centered = primary_test.rate - np.mean(primary_test.rate)
        chemical_centered = primary_prediction - np.mean(primary_prediction)
        corrected_centered = corrected - np.mean(corrected)
        limit = max(
            float(np.max(np.abs(measured_centered))),
            float(np.max(np.abs(chemical_centered))),
            float(np.max(np.abs(corrected_centered))),
            EPS,
        )
        figure, axes = plt.subplots(1, 3, figsize=(13.2, 4.3), constrained_layout=True)
        values_and_titles = (
            (measured_centered, "Measured"),
            (chemical_centered, "Chemical model"),
            (corrected_centered, "Chemical + spatial response"),
        )
        image = None
        for axis_item, (values, title) in zip(axes, values_and_titles):
            image = axis_item.scatter(
                xy[:, 0],
                xy[:, 1],
                c=values,
                cmap="coolwarm",
                vmin=-limit,
                vmax=limit,
                s=65,
            )
            axis_item.set_title(title)
            axis_item.set_xlabel("x/R")
            axis_item.set_ylabel("y/R")
            axis_item.set_aspect("equal", adjustable="box")
        assert image is not None
        figure.colorbar(image, ax=axes, shrink=0.86, label="Centered rate [nm/s]")
        path = plot_dir / "test_spatial_response.png"
        figure.savefig(path, dpi=180)
        plt.close(figure)
        outputs.append(path)

        chemical_residual = primary_prediction - primary_test.rate
        corrected_residual = corrected - primary_test.rate
        residual_limit = max(
            float(np.max(np.abs(chemical_residual))),
            float(np.max(np.abs(corrected_residual))),
            EPS,
        )
        figure, axes = plt.subplots(1, 2, figsize=(9.2, 4.3), constrained_layout=True)
        residual_image = None
        for axis_item, values, title in (
            (axes[0], chemical_residual, "Before spatial correction"),
            (axes[1], corrected_residual, "After spatial correction"),
        ):
            residual_image = axis_item.scatter(
                xy[:, 0],
                xy[:, 1],
                c=values,
                cmap="coolwarm",
                vmin=-residual_limit,
                vmax=residual_limit,
                s=65,
            )
            axis_item.set_title(title)
            axis_item.set_xlabel("x/R")
            axis_item.set_ylabel("y/R")
            axis_item.set_aspect("equal", adjustable="box")
        assert residual_image is not None
        figure.colorbar(
            residual_image,
            ax=axes,
            shrink=0.86,
            label="Predicted − measured rate [nm/s]",
        )
        path = plot_dir / "spatial_residuals.png"
        figure.savefig(path, dpi=180)
        plt.close(figure)
        outputs.append(path)

        correction = corrected - primary_prediction
        radial_correction = []
        for group in np.unique(radial_group_ids):
            mask = radial_group_ids == group
            radial_correction.append(
                (
                    float(np.mean(radius[mask])),
                    float(np.mean(correction[mask])),
                    float(np.std(correction[mask])),
                )
            )
        normalized_radial_correction = [
            (row[0] / wafer_radius, row[1], row[2]) for row in radial_correction
        ]
        figure, axis = plt.subplots(figsize=(6.8, 4.3), constrained_layout=True)
        axis.scatter(
            normalized_radius,
            correction,
            s=24,
            color="#94a3b8",
            alpha=0.55,
        )
        axis.errorbar(
            [row[0] for row in normalized_radial_correction],
            [row[1] for row in normalized_radial_correction],
            yerr=[row[2] for row in normalized_radial_correction],
            marker="o",
            color="#2563eb",
            capsize=3,
        )
        axis.axhline(0.0, color="#334155", linewidth=1.0)
        axis.set_xlabel("Normalized wafer radius, r/R")
        axis.set_ylabel("Change in predicted rate [nm/s]")
        axis.set_title("Spatial correction versus wafer radius")
        axis.grid(alpha=0.2)
        path = plot_dir / "spatial_correction_profile.png"
        figure.savefig(path, dpi=180)
        plt.close(figure)
        outputs.append(path)

        if split_sensitivity_rows and all(
            "spatial_response_test_centered_spatial_r2" in row
            for row in split_sensitivity_rows
        ):
            ordered = sorted(
                split_sensitivity_rows, key=lambda row: int(row["test_case"])
            )
            positions = np.arange(len(ordered))
            labels = [str(row["test_case"]) for row in ordered]
            chemical_r2 = np.asarray(
                [float(row["test_centered_spatial_r2"]) for row in ordered]
            )
            corrected_r2 = np.asarray(
                [
                    float(row["spatial_response_test_centered_spatial_r2"])
                    for row in ordered
                ]
            )
            chemical_rmse = np.asarray(
                [float(row["test_rmse_nm_s"]) for row in ordered]
            )
            corrected_rmse = np.asarray(
                [float(row["spatial_response_test_rmse_nm_s"]) for row in ordered]
            )
            figure, axes = plt.subplots(
                1, 2, figsize=(10.0, 4.5), constrained_layout=True
            )
            for i in positions:
                axes[0].plot(
                    [chemical_r2[i], corrected_r2[i]],
                    [i, i],
                    color="#cbd5e1",
                    linewidth=1.5,
                )
                axes[1].plot(
                    [chemical_rmse[i], corrected_rmse[i]],
                    [i, i],
                    color="#cbd5e1",
                    linewidth=1.5,
                )
            axes[0].scatter(chemical_r2, positions, label="Chemical model", color="#64748b")
            axes[0].scatter(corrected_r2, positions, label="With spatial correction", color="#2563eb")
            axes[0].axvline(0.0, color="#334155", linewidth=1.0)
            axes[0].set_xlabel("R² for within-wafer variation")
            axes[0].set_title("Wafer-pattern prediction")
            axes[1].scatter(chemical_rmse, positions, color="#64748b")
            axes[1].scatter(corrected_rmse, positions, color="#2563eb")
            axes[1].set_xlabel("RMSE [nm/s]")
            axes[1].set_title("Rate prediction error")
            axes[1].ticklabel_format(style="sci", axis="x", scilimits=(-3, -3))
            for axis_item in axes:
                axis_item.set_yticks(positions, labels)
                axis_item.set_ylabel("Held-out condition")
                axis_item.grid(axis="x", alpha=0.2)
            axes[0].legend(frameon=False)
            path = plot_dir / "spatial_correction_performance.png"
            figure.savefig(path, dpi=180)
            plt.close(figure)
            outputs.append(path)

    top = ranking[:8]
    figure, axis = plt.subplots(figsize=(10.5, 5.8), constrained_layout=True)

    def ranking_label(row: dict[str, Any]) -> str:
        roles = row.get("effective_roles", row.get("roles", {}))
        role_text = ", ".join(
            f"{name}={value}" for name, value in dict(roles or {}).items() if value
        )
        family = str(row.get("equation_family", row.get("model_family", "model")))
        family = family_labels.get(family, family)
        reduction = str(row.get("reduction_id", row.get("class_id", "")))
        reduction_text = "" if reduction in {"", "full"} else f" · {reduction}"
        return f"{family}{reduction_text}" + (f" · {role_text}" if role_text else "")

    labels = [ranking_label(row) for row in reversed(top)]
    values = [float(row["condition_cv_rmse_nm_s"]) for row in reversed(top)]
    colors = ["#2563eb" if bool(row.get("selected", False)) else "#cbd5e1" for row in reversed(top)]
    axis.barh(labels, values, color=colors, edgecolor="#475569", linewidth=0.5)
    axis.set_xlabel("Leave-one-condition-out RMSE [nm/s]")
    axis.set_title("Training-condition candidate ranking")
    axis.grid(axis="x", alpha=0.25)
    path = plot_dir / "training_candidate_ranking.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    outputs.append(path)
    return outputs


__all__ = [
    "_fit_formula",
    "_write_markdown_report",
    "_write_notebook",
    "plot_multicond_results",
]
