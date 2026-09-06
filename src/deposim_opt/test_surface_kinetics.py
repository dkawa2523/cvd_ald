from __future__ import annotations

from dataclasses import replace
import unittest

import numpy as np

from deposim_sim.models.aib_reductions import (
    AIB_QSS,
    LH_QSS,
    PARALLEL_QSS,
    TOTAL_POWER_BASELINE,
    SurfaceKineticCandidate,
    available_surface_model_families,
    default_surface_model_families,
    enumerate_surface_kinetic_candidates,
    candidate_evidence_requirements,
    candidate_physical_question,
    reduction_removed_effects,
    response_shape,
    surface_state,
)
from .surface_fit import (
    _profile_rate_scale,
    condition_balanced_weights,
    fit_surface_kinetic,
    parameter_loss_slice_rows,
    parameter_design_diagnostics,
    parameter_sensitivity_rows,
    predict_surface_kinetic,
    role_input_sensitivity_rows,
    role_response_curve_rows,
)
from .role_fields import (
    DIRECT_FLUX,
    DIRECT_SURFACE,
    RoleFieldSet,
    condition_contrast_summary,
)


def _data() -> RoleFieldSet:
    x = np.linspace(0.25, 2.5, 30)
    concentrations = {
        "s0": x,
        "s1": 0.3 + 1.2 * x[::-1],
        "s2": 0.4 + 0.35 * np.sin(np.linspace(0.0, 3.0, x.size)) ** 2,
    }
    rate = np.ones(x.size)
    return RoleFieldSet(
        case_ids=(1, 2, 3),
        xyz=np.zeros((x.size, 3)),
        species=("s0", "s1", "s2"),
        bulk_concentrations=concentrations,
        species_fractions={
            name: values / sum(concentrations.values())
            for name, values in concentrations.items()
        },
        total_concentration=sum(concentrations.values()),
        condition_id=np.repeat((1, 2, 3), 10),
        rate=rate,
    )


class SurfaceKineticTests(unittest.TestCase):
    def test_available_inputs_exposes_selected_driver_concept(self):
        self.assertIn("reaction_driver", _data().available_inputs())

    def test_total_concentration_baseline_ignores_composition(self):
        candidate = SurfaceKineticCandidate(TOTAL_POWER_BASELINE)
        refs = {"s0": 1.0, "s1": 1.0}
        concentrations = {
            "s0": np.array([1.0, 0.2, 2.0]),
            "s1": np.array([1.0, 1.8, 2.0]),
        }
        response = response_shape(
            candidate, concentrations, refs, {"common_total_order": 2.0}
        )
        np.testing.assert_allclose(response, [1.0, 1.0, 4.0])
        self.assertEqual(candidate.effect_groups, {})

    def test_practical_identifiability_is_separate_from_structural_rank(self):
        x = np.linspace(-1.0, 1.0, 40)
        weak = np.column_stack([np.ones_like(x), x, x + 1.0e-6 * x**2])
        diagnostic = parameter_design_diagnostics(weak)
        self.assertTrue(diagnostic["full_rank"])
        self.assertEqual(diagnostic["status"], "weak")
        self.assertGreater(diagnostic["condition_number"], 1.0e4)

        strong = np.column_stack([np.ones_like(x), x, x**2])
        self.assertEqual(parameter_design_diagnostics(strong)["status"], "sufficient")

    def test_condition_contrast_separates_independent_and_confounded_roles(self):
        condition_id = np.repeat((1, 2, 3, 4), 2)
        first = np.repeat((1.0, 2.0, 3.0, 4.0), 2)
        confounded = 5.0 * first
        independent = np.repeat((1.0, 3.0, 2.0, 5.0), 2)
        data = RoleFieldSet(
            case_ids=(1, 2, 3, 4),
            xyz=np.zeros((8, 3)),
            condition_id=condition_id,
            species=("s0", "s1", "s2"),
            bulk_concentrations={
                "s0": first,
                "s1": confounded,
                "s2": independent,
            },
            species_fractions={},
            total_concentration=first + confounded + independent,
            rate=np.ones(8),
        )
        limited = condition_contrast_summary(data, ("s0", "s1"))
        sufficient = condition_contrast_summary(data, ("s0", "s2"))
        self.assertEqual(limited["status"], "limited")
        self.assertEqual(limited["rank"], 1)
        self.assertEqual(limited["confounded_species"], ["s0", "s1"])
        self.assertEqual(sufficient["status"], "sufficient")
        self.assertEqual(sufficient["rank"], 2)

    def test_model_definition_owns_physical_question_and_evidence(self):
        candidate = SurfaceKineticCandidate(
            "AB", A="s0", B="s1", family=PARALLEL_QSS
        )
        self.assertIn("parallel", candidate_physical_question(candidate))
        self.assertTrue(
            any("near-zero B" in item for item in candidate_evidence_requirements(candidate))
        )

    def test_condition_weights_use_measurement_uncertainty_without_point_count_bias(self):
        condition_id = np.array([1, 1, 2, 2, 2])
        indices = np.arange(5)
        sigma = np.array([1.0, 2.0, 2.0, 4.0, 8.0])
        weights = condition_balanced_weights(condition_id, indices, sigma)
        np.testing.assert_allclose(weights[:2], [0.4, 0.1])
        np.testing.assert_allclose(weights[2:], np.array([16.0, 4.0, 1.0]) / 42.0)
        self.assertAlmostEqual(float(np.sum(weights[condition_id == 1])), 0.5)
        self.assertAlmostEqual(float(np.sum(weights[condition_id == 2])), 0.5)

    def test_optional_radial_uncertainty_reduces_edge_weight_only_within_condition(self):
        condition_id = np.array([1, 1, 1, 2, 2, 2])
        indices = np.arange(6)
        xyz = np.array(
            [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0]] * 2
        )
        weights = condition_balanced_weights(
            condition_id,
            indices,
            xyz=xyz,
            edge_uncertainty_ratio=2.0,
            radial_power=2.0,
        )
        self.assertAlmostEqual(float(np.sum(weights[:3])), 0.5)
        self.assertAlmostEqual(float(np.sum(weights[3:])), 0.5)
        self.assertGreater(weights[1], weights[0])
        self.assertGreater(weights[1], weights[2])

    def test_qss_coverages_close_the_site_balance(self):
        data = _data()
        candidate = SurfaceKineticCandidate("AIB", A="s0", I="s2", B="s1")
        refs = {name: float(np.median(values)) for name, values in data.bulk_concentrations.items()}
        parameters = {
            "desorption_ratio": 0.4,
            "conversion_ratio": 1.7,
            "inhibition_ratio": 2.2,
        }
        state = surface_state(candidate, data.bulk_concentrations, refs, parameters)
        np.testing.assert_allclose(
            state["theta_free"] + state["theta_A"] + state["theta_I"],
            1.0,
            rtol=1.0e-13,
            atol=1.0e-13,
        )

    def test_fit_recovers_aib_observable_response(self):
        data = _data()
        candidate = SurfaceKineticCandidate("AIB", A="s0", I="s2", B="s1")
        train = np.arange(data.rate.size)
        refs = {name: float(np.median(values)) for name, values in data.bulk_concentrations.items()}
        parameters = {
            "desorption_ratio": 1.0,
            "conversion_ratio": 0.1,
            "inhibition_ratio": 0.01,
        }
        truth = 0.23 * response_shape(candidate, data.bulk_concentrations, refs, parameters)
        fitted = fit_surface_kinetic(candidate, replace(data, rate=truth), train)
        prediction, _ = predict_surface_kinetic(fitted, data)
        self.assertLess(float(np.max(np.abs(prediction - truth))), 1.0e-7)

    def test_recorded_optimization_and_role_response_are_physical_outputs(self):
        data = _data()
        candidate = SurfaceKineticCandidate("AIB", A="s0", I="s2", B="s1")
        refs = {
            name: float(np.median(values))
            for name, values in data.bulk_concentrations.items()
        }
        truth = 0.23 * response_shape(
            candidate,
            data.bulk_concentrations,
            refs,
            {
                "desorption_ratio": 1.0,
                "conversion_ratio": 0.1,
                "inhibition_ratio": 0.4,
            },
        )
        fitted = fit_surface_kinetic(
            candidate,
            replace(data, rate=truth),
            np.arange(data.rate.size),
            record_optimization_history=True,
        )
        history = fitted.optimization_history
        self.assertEqual(len(history), fitted.optimizer_trial_count)
        self.assertAlmostEqual(float(history[-1]["best_score"]), fitted.objective_value)
        best = np.asarray([float(row["best_score"]) for row in history])
        self.assertTrue(np.all(np.diff(best) <= 0.0))

        interpreted = replace(data, rate=truth)
        sensitivities = role_input_sensitivity_rows(fitted, interpreted)
        self.assertEqual({row["role"] for row in sensitivities}, {"A", "I", "B"})
        self.assertTrue(
            all(float(row["rms_prediction_change_nm_s"]) >= 0.0 for row in sensitivities)
        )
        curves = role_response_curve_rows(fitted, interpreted, points=12)
        self.assertEqual(len(curves), 36)
        inhibitor = [row for row in curves if row["role"] == "I"]
        self.assertLess(
            float(inhibitor[-1]["predicted_rate_nm_s"]),
            float(inhibitor[0]["predicted_rate_nm_s"]),
        )

        parameter_sensitivity = parameter_sensitivity_rows(fitted)
        parameter_names = set(candidate.parameter_names)
        self.assertEqual(
            len(parameter_sensitivity), len(parameter_names) ** 2
        )
        self.assertEqual(
            {row["parameter_1"] for row in parameter_sensitivity},
            parameter_names,
        )
        loss_slices = parameter_loss_slice_rows(
            fitted, interpreted, span_decades=1.0, points=11
        )
        for name in parameter_names:
            rows = [row for row in loss_slices if row["parameter"] == name]
            fitted_row = min(
                rows, key=lambda row: abs(float(row["factor_from_fitted_value"]) - 1.0)
            )
            self.assertAlmostEqual(
                float(fitted_row["objective"]), fitted.objective_value, places=12
            )

    def test_all_whole_wafer_losses_profile_the_rate_scale(self):
        data = _data()
        candidate = SurfaceKineticCandidate("AI", A="s0", I="s2")
        refs = {
            name: float(np.median(values))
            for name, values in data.bulk_concentrations.items()
        }
        truth = 0.31 * response_shape(
            candidate,
            data.bulk_concentrations,
            refs,
            {"half_saturation_ratio": 0.8, "inhibition_ratio": 0.2},
        )
        shape = response_shape(
            candidate,
            data.bulk_concentrations,
            refs,
            {"half_saturation_ratio": 0.8, "inhibition_ratio": 0.2},
        )
        weights = condition_balanced_weights(
            data.condition_id, np.arange(truth.size)
        )
        for loss_name in (
            "mse",
            "wafer_normalized_mse",
            "wafer_normalized_mae",
            "symmetric_normalized_mse",
        ):
            scale, objective = _profile_rate_scale(
                shape,
                truth,
                data.condition_id,
                weights,
                loss_name,
            )
            self.assertAlmostEqual(scale, 0.31, places=9, msg=loss_name)
            self.assertLess(objective, 1.0e-15, msg=loss_name)

    def test_symmetric_scale_profile_matches_direct_loss_search(self):
        shape = np.array([0.3, 1.1, 2.4, 0.7, 1.8, 3.2])
        target = np.array([0.5, 1.4, 2.1, 0.9, 1.6, 2.7])
        condition_id = np.array([1, 1, 1, 2, 2, 2])
        weights = condition_balanced_weights(condition_id, np.arange(shape.size))
        scale, objective = _profile_rate_scale(
            shape,
            target,
            condition_id,
            weights,
            "symmetric_normalized_mse",
        )
        trial_scales = np.geomspace(0.01, 100.0, 20001)
        direct = np.array(
            [
                np.mean(
                    [
                        2.0
                        * np.mean(np.square(value * shape[condition_id == group] - target[condition_id == group]))
                        / np.mean(
                            np.square(target[condition_id == group])
                            + np.square(value * shape[condition_id == group])
                        )
                        for group in (1, 2)
                    ]
                )
                for value in trial_scales
            ]
        )
        self.assertLessEqual(objective, float(np.min(direct)) + 1.0e-9)
        self.assertAlmostEqual(scale, float(trial_scales[np.argmin(direct)]), places=3)

    def test_references_make_concentration_units_invariant(self):
        data = _data()
        candidate = SurfaceKineticCandidate("AI", A="s0", I="s2")
        refs = {name: float(np.median(values)) for name, values in data.bulk_concentrations.items()}
        parameters = {"half_saturation_ratio": 0.8, "inhibition_ratio": 1.3}
        first = response_shape(candidate, data.bulk_concentrations, refs, parameters)
        factors = {"s0": 1.0e3, "s1": 1.0e-2, "s2": 1.0e6}
        scaled = replace(
            data,
            bulk_concentrations={name: values * factors[name] for name, values in data.bulk_concentrations.items()},
        )
        scaled_refs = {name: refs[name] * factors[name] for name in refs}
        second = response_shape(candidate, scaled.bulk_concentrations, scaled_refs, parameters)
        np.testing.assert_allclose(first, second, rtol=1.0e-14, atol=0.0)

    def test_no_inhibitor_ab_response_has_a_b_exchange_symmetry(self):
        data = _data()
        refs = {name: float(np.median(values)) for name, values in data.bulk_concentrations.items()}
        forward = SurfaceKineticCandidate("AB", A="s0", B="s1")
        reverse = SurfaceKineticCandidate("AB", A="s1", B="s0")
        delta, conversion, scale = 0.7, 2.5, 0.31
        first = scale * response_shape(
            forward, data.bulk_concentrations, refs,
            {"desorption_ratio": delta, "conversion_ratio": conversion},
        )
        second = (scale * conversion) * response_shape(
            reverse, data.bulk_concentrations, refs,
            {
                "desorption_ratio": delta / conversion,
                "conversion_ratio": 1.0 / conversion,
            },
        )
        np.testing.assert_allclose(first, second, rtol=1.0e-14, atol=1.0e-14)

    def test_candidate_reductions_are_explicit_physical_boundaries(self):
        candidates = enumerate_surface_kinetic_candidates(("s0", "s1", "s2"))
        ids = {candidate.model_id for candidate in candidates}
        self.assertIn(
            "cvd:aib_qss:AB:no_desorption:bulk_as_surface:A=s0,B=s1", ids
        )
        full = next(
            candidate for candidate in candidates
            if candidate.model_id
            == "cvd:aib_qss:AIB:full:bulk_as_surface:A=s0,I=s2,B=s1"
        )
        reductions = {candidate.model_id for candidate in full.reductions()}
        self.assertEqual(
            reductions,
            {
                "cvd:aib_qss:AIB:no_desorption:bulk_as_surface:A=s0,I=s2,B=s1",
                "cvd:aib_qss:AB:full:bulk_as_surface:A=s0,B=s1",
            },
        )
        by_id = {candidate.model_id: candidate for candidate in full.reductions()}
        self.assertEqual(
            reduction_removed_effects(
                full,
                by_id["cvd:aib_qss:AB:full:bulk_as_surface:A=s0,B=s1"],
            ),
            ("I",),
        )

    def test_prediction_reuses_identification_references(self):
        data = _data()
        candidate = SurfaceKineticCandidate("A", A="s0")
        train = np.arange(20)
        target = 0.2 * data.bulk_concentrations["s0"] / (data.bulk_concentrations["s0"] + 0.5)
        fitted = fit_surface_kinetic(candidate, replace(data, rate=target), train)
        changed = replace(
            data,
            bulk_concentrations={
                **data.bulk_concentrations,
                "s0": data.bulk_concentrations["s0"] * 10.0,
            },
        )
        predict_surface_kinetic(fitted, changed)
        self.assertEqual(
            fitted.reference_concentrations["s0"],
            float(np.median(data.bulk_concentrations["s0"][train])),
        )

    def test_direct_surface_candidate_uses_surface_fields_and_stable_id(self):
        data = _data()
        surface = {
            name: values * (0.35 + 0.1 * index)
            for index, (name, values) in enumerate(data.bulk_concentrations.items())
        }
        capacity = {
            name: np.full(values.shape, 10.0)
            for name, values in surface.items()
        }
        realized = {
            name: capacity[name]
            * (1.0 - surface[name] / data.bulk_concentrations[name])
            for name in surface
        }
        data = replace(
            data,
            surface_concentrations=surface,
            transport_capacity_flux=capacity,
            realized_reactive_flux=realized,
        )
        candidate = SurfaceKineticCandidate(
            "AI", A="s0", I="s2", transport_mode=DIRECT_SURFACE
        )
        refs = {name: float(np.median(values)) for name, values in surface.items()}
        truth = 0.17 * response_shape(
            candidate,
            surface,
            refs,
            {"half_saturation_ratio": 1.0, "inhibition_ratio": 0.01},
        )
        fit = fit_surface_kinetic(
            candidate, replace(data, rate=truth), np.arange(truth.size)
        )
        prediction, state = predict_surface_kinetic(fit, data)
        self.assertLess(float(np.max(np.abs(prediction - truth))), 1.0e-6)
        self.assertEqual(
            candidate.model_id,
            "cvd:aib_qss:AI:full:direct_surface:A=s0,I=s2",
        )
        np.testing.assert_allclose(state["surface_to_bulk_A"], 0.35)
        np.testing.assert_allclose(state["flux_closure_residual_A"], 0.0)

    def test_direct_flux_is_an_explicit_reaction_driver_not_a_surface_concentration(self):
        data = _data()
        fluxes = {
            name: 1.0e-4 * (index + 1) * np.asarray(values)
            for index, (name, values) in enumerate(data.bulk_concentrations.items())
        }
        realized = {name: 0.4 * values for name, values in fluxes.items()}
        data = replace(
            data,
            transport_capacity_flux=fluxes,
            realized_reactive_flux=realized,
        )
        candidate = SurfaceKineticCandidate(
            "AI", A="s0", I="s2", transport_mode=DIRECT_FLUX
        )
        refs = {name: float(np.median(values)) for name, values in fluxes.items()}
        truth = 0.19 * response_shape(
            candidate,
            fluxes,
            refs,
            {"half_saturation_ratio": 0.8, "inhibition_ratio": 0.2},
        )
        fit = fit_surface_kinetic(
            candidate, replace(data, rate=truth), np.arange(truth.size)
        )
        prediction, state = predict_surface_kinetic(fit, data)

        self.assertLess(float(np.max(np.abs(prediction - truth))), 2.0e-3)
        self.assertEqual(
            data.resolve_reaction_input_mode("transport_capacity_flux"), DIRECT_FLUX
        )
        self.assertNotIn("surface_to_bulk_A", state)
        np.testing.assert_allclose(state["reaction_input_A"], fluxes["s0"])
        np.testing.assert_allclose(state["reactive_to_capacity_flux_A"], 0.4)

    def test_parallel_family_contains_sequential_and_a_only_limits(self):
        data = _data()
        refs = {name: float(np.median(values)) for name, values in data.bulk_concentrations.items()}
        parallel = SurfaceKineticCandidate(
            "AIB", A="s0", I="s2", B="s1", family=PARALLEL_QSS
        )
        sequential = SurfaceKineticCandidate("AIB", A="s0", I="s2", B="s1")
        common = {
            "desorption_ratio": 0.4,
            "conversion_ratio": 1.7,
            "inhibition_ratio": 2.2,
        }
        parallel_shape = response_shape(
            parallel,
            data.bulk_concentrations,
            refs,
            {**common, "single_conversion_ratio": 0.0},
        )
        np.testing.assert_allclose(
            parallel_shape,
            response_shape(sequential, data.bulk_concentrations, refs, common),
            rtol=1.0e-14,
            atol=0.0,
        )

        a_only = SurfaceKineticCandidate("AI", A="s0", I="s2")
        single_conversion = 0.8
        no_b = response_shape(
            parallel,
            data.bulk_concentrations,
            refs,
            {
                **common,
                "single_conversion_ratio": single_conversion,
                "conversion_ratio": 0.0,
            },
        )
        reduced = response_shape(
            a_only,
            data.bulk_concentrations,
            refs,
            {
                "half_saturation_ratio": common["desorption_ratio"] + single_conversion,
                "inhibition_ratio": common["inhibition_ratio"],
            },
        )
        np.testing.assert_allclose(no_b, single_conversion * reduced, rtol=1.0e-14)

    def test_parallel_descriptor_is_complete_for_parameter_fitting(self):
        data = _data()
        candidate = SurfaceKineticCandidate(
            "AB", A="s0", B="s1", family=PARALLEL_QSS
        )
        refs = {
            name: float(np.median(values))
            for name, values in data.bulk_concentrations.items()
        }
        truth = 0.2 * response_shape(
            candidate,
            data.bulk_concentrations,
            refs,
            {
                "single_conversion_ratio": 1.0,
                "desorption_ratio": 1.0,
                "conversion_ratio": 0.1,
            },
        )
        fit = fit_surface_kinetic(
            candidate, replace(data, rate=truth), np.arange(truth.size)
        )
        self.assertEqual(
            set(fit.shape_parameters),
            {
                "single_conversion_ratio",
                "desorption_ratio",
                "conversion_ratio",
            },
        )

    def test_family_enumeration_is_explicit_and_unique(self):
        sequential = enumerate_surface_kinetic_candidates(("s0", "s1", "s2"))
        compared = enumerate_surface_kinetic_candidates(
            ("s0", "s1", "s2"), families=(AIB_QSS, PARALLEL_QSS)
        )
        sequential_ids = {candidate.model_id for candidate in sequential}
        compared_ids = [candidate.model_id for candidate in compared]
        self.assertFalse(any(":parallel_a_ab_qss:" in name for name in sequential_ids))
        self.assertIn(
            "cvd:parallel_a_ab_qss:AB:full:bulk_as_surface:A=s0,B=s1",
            compared_ids,
        )
        self.assertIn(
            "cvd:parallel_a_ab_qss:AB:full:bulk_as_surface:A=s1,B=s0",
            compared_ids,
        )
        self.assertEqual(len(compared_ids), len(set(compared_ids)))

    def test_sequential_family_collapses_only_the_exact_ab_symmetry(self):
        candidates = enumerate_surface_kinetic_candidates(("s0", "s1", "s2"))
        ab = [candidate for candidate in candidates if candidate.class_id == "AB"
              and candidate.reduction_id == "full"]
        full_aib = [candidate for candidate in candidates if candidate.class_id == "AIB"
                    and candidate.reduction_id == "full"]
        self.assertEqual(len(ab), 3)
        self.assertEqual(len(full_aib), 6)
        ids = {candidate.model_id for candidate in full_aib}
        self.assertIn(
            "cvd:aib_qss:AIB:full:bulk_as_surface:A=s0,I=s2,B=s1", ids
        )
        self.assertIn(
            "cvd:aib_qss:AIB:full:bulk_as_surface:A=s1,I=s2,B=s0", ids
        )
        self.assertNotIn(
            "cvd:aib_qss:AB:full:bulk_as_surface:A=s1,B=s0",
            {candidate.model_id for candidate in ab},
        )

    def test_family_applicability_uses_declared_inputs(self):
        self.assertEqual(available_surface_model_families({"concentration"}),
                         (AIB_QSS, PARALLEL_QSS, LH_QSS))
        self.assertEqual(
            default_surface_model_families({"concentration"}),
            (AIB_QSS, PARALLEL_QSS),
        )
        self.assertEqual(available_surface_model_families({"realized_reactive_flux"}), ())
        self.assertEqual(
            enumerate_surface_kinetic_candidates(
                ("s0", "s1", "s2"),
                families=(AIB_QSS, PARALLEL_QSS),
                available_inputs=("realized_reactive_flux",),
            ),
            [SurfaceKineticCandidate("baseline")],
        )

    def test_lh_family_closes_site_balance_and_declares_only_exact_reduction(self):
        data = _data()
        refs = {
            name: float(np.median(values))
            for name, values in data.bulk_concentrations.items()
        }
        candidate = SurfaceKineticCandidate(
            "AIB", A="s0", I="s2", B="s1", family=LH_QSS
        )
        parameters = {
            "adsorption_ratio_A": 0.7,
            "adsorption_ratio_B": 1.4,
            "inhibition_ratio": 0.3,
        }
        state = surface_state(
            candidate, data.bulk_concentrations, refs, parameters
        )
        np.testing.assert_allclose(
            state["theta_free"]
            + state["theta_A"]
            + state["theta_B"]
            + state["theta_I"],
            1.0,
            rtol=1.0e-14,
        )
        reverse = SurfaceKineticCandidate(
            "AIB", A="s1", I="s2", B="s0", family=LH_QSS
        )
        np.testing.assert_allclose(
            response_shape(candidate, data.bulk_concentrations, refs, parameters),
            response_shape(
                reverse,
                data.bulk_concentrations,
                refs,
                {
                    "adsorption_ratio_A": parameters["adsorption_ratio_B"],
                    "adsorption_ratio_B": parameters["adsorption_ratio_A"],
                    "inhibition_ratio": parameters["inhibition_ratio"],
                },
            ),
            rtol=1.0e-14,
        )
        reductions = candidate.reductions()
        self.assertEqual(len(reductions), 1)
        self.assertEqual(reductions[0].family, LH_QSS)
        candidates = enumerate_surface_kinetic_candidates(
            data.species, families=(LH_QSS,)
        )
        self.assertFalse(
            any(
                row.family == LH_QSS and row.reduction_id == "no_desorption"
                for row in candidates
            )
        )


if __name__ == "__main__":
    unittest.main()
