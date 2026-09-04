"""Generate deterministic multi-condition CVD/ALD fit fixtures."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from deposim_schema import compose_sim_config
from deposim_sim.pipeline import run_sim_from_spec


SPECIES = np.asarray(("s0", "s1", "s2", "s3"))


def _xy_points() -> np.ndarray:
    points = [[0.0, 0.0]]
    for radius, count in ((45.0, 8), (95.0, 12), (135.0, 16)):
        for idx in range(count):
            angle = 2.0 * np.pi * idx / count
            points.append([radius * np.cos(angle), radius * np.sin(angle)])
    return np.asarray(points, dtype=float)


def _spatial_terms(xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r = np.sqrt(np.sum(np.square(xy), axis=1))
    r_norm = r / max(float(np.max(r)), 1.0)
    angle = np.arctan2(xy[:, 1], xy[:, 0])
    return r_norm, angle


def _write_measurement_from_sim(*, config_name: str, fluent_path: Path, meas_path: Path, overrides: list[str]) -> None:
    spec = compose_sim_config(
        config_name,
        overrides=[f"sim.inputs.fluent.file={fluent_path}", "sim.measurement.enabled=false", *overrides],
    )
    spec.measurement.enabled = False
    spec.measurement.file = ""
    result = run_sim_from_spec(spec)
    xy = np.asarray(result.diagnostics["xy_mm"], dtype=float)
    h_nm = np.asarray(result.fields["h_nm"], dtype=float)
    np.savez(meas_path, xy=xy, h_nm=h_nm)


def write_cvd_conditions(output_dir: Path) -> None:
    xy = _xy_points()
    r_norm, angle = _spatial_terms(xy)
    conditions = [
        ("base", 1.0, 1.0),
        ("high_feed", 1.8, 0.95),
        ("edge_depleted", 1.0, 1.45),
    ]
    for name, feed_scale, edge_loss in conditions:
        s0 = np.clip(feed_scale * (1.0 - 0.24 * edge_loss * r_norm + 0.08 * np.cos(angle)), 0.05, np.inf)
        s1 = np.clip(0.70 + 0.28 * r_norm + 0.10 * np.sin(angle), 0.05, np.inf)
        s2 = np.clip(0.10 + 0.35 * r_norm + 0.04 * np.cos(2.0 * angle), 0.01, np.inf)
        s3 = np.full_like(s0, 0.04)
        cref = np.stack([s0, s1, s2, s3], axis=1)
        velocity = np.clip(0.05 - 0.012 * r_norm + 0.004 * np.cos(angle), 0.004, np.inf)
        flux_sink = np.clip(cref * velocity[:, None], 0.0, np.inf)
        fluent_path = output_dir / f"cvd_{name}_fluent.npz"
        meas_path = output_dir / f"cvd_{name}_meas.npz"
        np.savez(fluent_path, xy=xy, cref=cref, flux_sink=flux_sink, species=SPECIES)
        _write_measurement_from_sim(
            config_name="cvd_steady_min",
            fluent_path=fluent_path,
            meas_path=meas_path,
            overrides=[
                "sim.roles.A=s0",
                "sim.model.params.kinetics.k_rxn=0.035",
                "sim.model.params.transport.km_A.value=0.045",
                "sim.model.params.thickness.alpha_h=1.0",
                "sim.time.t_proc_s=12.0",
                "sim.time.dt_s=0.02",
            ],
        )


def _ald_frames(
    *,
    xy: np.ndarray,
    dose_scale: float,
    purge_duration_s: float,
    purge_residual: float,
    cycles: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r_norm, angle = _spatial_terms(xy)
    precursor = np.clip(dose_scale * (1.0 - 0.20 * r_norm + 0.06 * np.cos(angle)), 0.03, np.inf)
    coreactant = np.clip(0.90 - 0.08 * r_norm + 0.05 * np.sin(angle), 0.03, np.inf)
    blocker = np.clip(0.12 + 0.05 * r_norm + 0.02 * np.sin(2.0 * angle), 0.0, np.inf)
    carrier = np.full_like(precursor, 0.03)

    frames: list[np.ndarray] = []
    times: list[float] = []
    phase_code: list[int] = []
    t = 0.0
    for _ in range(cycles):
        frames.append(np.stack([precursor, 0.02 * coreactant, 0.02 * blocker, carrier], axis=1))
        times.append(t)
        phase_code.append(1)
        t += 0.18
        frames.append(np.stack([purge_residual * precursor, purge_residual * coreactant, purge_residual * blocker, carrier], axis=1))
        times.append(t)
        phase_code.append(2)
        t += purge_duration_s
        frames.append(np.stack([purge_residual * precursor, coreactant, 0.02 * blocker, carrier], axis=1))
        times.append(t)
        phase_code.append(3)
        t += 0.18
        frames.append(np.stack([purge_residual * precursor, purge_residual * coreactant, purge_residual * blocker, carrier], axis=1))
        times.append(t)
        phase_code.append(4)
        t += purge_duration_s
    frames.append(np.stack([purge_residual * precursor, purge_residual * coreactant, purge_residual * blocker, carrier], axis=1))
    times.append(t)
    phase_code.append(0)

    cref = np.stack(frames, axis=0)
    velocity = np.clip(0.04 - 0.010 * r_norm + 0.003 * np.cos(angle), 0.004, np.inf)[None, :, None]
    flux_sink = np.clip(cref * velocity, 0.0, np.inf)
    return np.asarray(times, dtype=float), cref, flux_sink


def write_ald_conditions(output_dir: Path) -> None:
    xy = _xy_points()
    conditions = [
        ("low_dose", 0.40, 0.20, 0.015),
        ("nominal", 1.00, 0.20, 0.015),
        ("high_dose", 3.00, 0.20, 0.015),
        ("short_purge", 1.00, 0.06, 0.080),
        ("long_purge", 1.00, 0.45, 0.003),
    ]
    for name, dose_scale, purge_duration_s, purge_residual in conditions:
        time, cref, flux_sink = _ald_frames(
            xy=xy,
            dose_scale=dose_scale,
            purge_duration_s=purge_duration_s,
            purge_residual=purge_residual,
            cycles=4,
        )
        fluent_path = output_dir / f"ald_{name}_fluent.npz"
        meas_path = output_dir / f"ald_{name}_meas.npz"
        np.savez(
            fluent_path,
            xy=xy,
            time=time,
            cref=cref,
            flux_sink=flux_sink,
            phase_code=np.asarray([1, 2, 3, 4] * 4 + [0], dtype=int),
            species=SPECIES,
        )
        _write_measurement_from_sim(
            config_name="ald_state_min",
            fluent_path=fluent_path,
            meas_path=meas_path,
            overrides=[
                "sim.roles.A=s0",
                "sim.roles.B=s1",
                "sim.model.params.transport.km_A.value=10.0",
                "sim.model.params.transport.km_B.value=0.04",
                "sim.model.params.kinetics.k_store_A=10.0",
                "sim.model.params.kinetics.k_release_A=0.05",
                "sim.model.params.kinetics.k_convert_A=0.30",
                "sim.model.params.kinetics.k_convert_AB=0.30",
                "sim.model.params.thickness.alpha_h=1.0",
            ],
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate multi-condition CVD/ALD fit fixtures.")
    parser.add_argument("--output-dir", default="runs/generated_inputs/multicond_fit")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_cvd_conditions(output_dir)
    write_ald_conditions(output_dir)
    print(f"[multicond_fit_inputs] wrote CVD/ALD multi-condition inputs to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
