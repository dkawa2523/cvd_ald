"""Generate deterministic ALD-like transient inputs for role-model benchmarks.

The generated data intentionally stays within the current Fluent-like transient
input contract. It is a reduced readiness fixture, not a substitute for measured
ALD process data.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np


SPECIES = ("s0", "s1", "s2", "s3")
WAFER_RADIUS_MM = 150.0


@dataclass(frozen=True)
class Scenario:
    name: str
    dose_scale: float
    purge_duration_s: float
    purge_residual: float
    cycles: int = 4


SCENARIOS: tuple[Scenario, ...] = (
    Scenario("nominal", dose_scale=1.0, purge_duration_s=0.20, purge_residual=0.015),
    Scenario("low_dose", dose_scale=0.35, purge_duration_s=0.20, purge_residual=0.015),
    Scenario("high_dose", dose_scale=3.00, purge_duration_s=0.20, purge_residual=0.015),
    Scenario("short_purge", dose_scale=1.0, purge_duration_s=0.05, purge_residual=0.080),
    Scenario("long_purge", dose_scale=1.0, purge_duration_s=0.45, purge_residual=0.003),
)


def build_wafer_points() -> np.ndarray:
    """Return center + eight 44-point rings: 353 wafer points total."""

    points: list[tuple[float, float]] = [(0.0, 0.0)]
    radii = np.linspace(18.0, 144.0, 8, dtype=float)
    angles = np.linspace(0.0, 2.0 * np.pi, 44, endpoint=False, dtype=float)
    for radius in radii:
        for angle in angles:
            points.append((float(radius * np.cos(angle)), float(radius * np.sin(angle))))
    return np.asarray(points, dtype=float)


def spatial_fields(xy_mm: np.ndarray) -> dict[str, np.ndarray]:
    r = np.sqrt(np.sum(np.square(xy_mm), axis=1))
    r_norm = r / max(float(np.max(r)), 1.0)
    theta = np.arctan2(xy_mm[:, 1], xy_mm[:, 0])

    precursor = np.clip(1.00 - 0.28 * r_norm + 0.07 * np.cos(theta), 0.05, np.inf)
    coreactant = np.clip(0.92 - 0.12 * r_norm + 0.05 * np.sin(theta), 0.05, np.inf)
    blocker = np.clip(0.18 + 0.08 * np.sin(2.0 * theta) + 0.05 * r_norm, 0.0, np.inf)
    carrier = np.full_like(precursor, 0.04)
    velocity = np.clip(0.045 - 0.012 * r_norm + 0.004 * np.cos(2.0 * theta), 0.006, np.inf)
    target = 0.045 * (1.0 - 0.03 * r_norm + 0.01 * np.cos(theta))
    return {
        "r_norm": r_norm,
        "precursor": precursor,
        "coreactant": coreactant,
        "blocker": blocker,
        "carrier": carrier,
        "velocity": velocity,
        "target_h_nm": target,
    }


def build_transient_payload(scenario: Scenario) -> dict[str, np.ndarray]:
    xy = build_wafer_points()
    fields = spatial_fields(xy)
    n_pts = int(xy.shape[0])

    zero = np.zeros(n_pts, dtype=float)
    carrier = fields["carrier"]
    precursor = fields["precursor"] * float(scenario.dose_scale)
    coreactant = fields["coreactant"]
    blocker = fields["blocker"]
    residual = float(scenario.purge_residual)

    a_pulse = np.stack([precursor, residual * coreactant, zero, carrier], axis=1)
    purge_after_a = np.stack([residual * precursor, residual * coreactant, zero, carrier], axis=1)
    b_pulse = np.stack([residual * precursor, coreactant, zero, carrier], axis=1)
    purge_after_b = np.stack([residual * precursor, residual * coreactant, zero, carrier], axis=1)

    # A reserved inhibited frame is not used by the nominal config, but keeping
    # s2 spatially meaningful makes future site-blocking scenarios reproducible.
    a_pulse[:, 2] = 0.05 * blocker
    purge_after_a[:, 2] = residual * blocker
    b_pulse[:, 2] = 0.05 * blocker
    purge_after_b[:, 2] = residual * blocker

    intervals: list[tuple[float, np.ndarray, int]] = []
    t = 0.0
    for _cycle in range(int(scenario.cycles)):
        intervals.append((0.30, a_pulse, 1))
        intervals.append((float(scenario.purge_duration_s), purge_after_a, 2))
        intervals.append((0.30, b_pulse, 3))
        intervals.append((float(scenario.purge_duration_s), purge_after_b, 4))

    times: list[float] = [0.0]
    frames: list[np.ndarray] = []
    phase_codes: list[int] = []
    for duration_s, frame, phase_code in intervals:
        if duration_s <= 0.0:
            raise ValueError("duration_s must be > 0")
        frames.append(np.asarray(frame, dtype=float))
        phase_codes.append(int(phase_code))
        t = float(t + duration_s)
        times.append(t)

    # The loader requires cref.shape[0] == time.shape[0]. The final frame is not
    # used as an interval by the transient solver, but records the terminal phase.
    frames.append(np.asarray(intervals[-1][1], dtype=float))
    phase_codes.append(int(intervals[-1][2]))

    cref = np.stack(frames, axis=0)
    velocity = fields["velocity"][None, :, None]
    flux_sink = np.clip(cref * velocity, 0.0, np.inf)
    return {
        "xy": xy,
        "time": np.asarray(times, dtype=float),
        "cref": cref,
        "flux_sink": flux_sink,
        "species": np.asarray(SPECIES),
        "phase_code": np.asarray(phase_codes, dtype=int),
    }


def write_scenario(output_dir: Path, scenario: Scenario) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = build_transient_payload(scenario)
    out = output_dir / f"ald_like_{scenario.name}.npz"
    np.savez(out, **payload)

    xy = np.asarray(payload["xy"], dtype=float)
    target = spatial_fields(xy)["target_h_nm"] * float(scenario.dose_scale) ** 0.15
    meas_out = output_dir / f"ald_like_{scenario.name}_meas.npz"
    np.savez(meas_out, xy=xy, h_nm=target)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ALD-like transient benchmark inputs.")
    parser.add_argument("--output-dir", default="runs/generated_inputs/ald_like_reduced")
    parser.add_argument(
        "--scenario",
        default="all",
        choices=["all", *[scenario.name for scenario in SCENARIOS]],
        help="Scenario to generate.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    selected = SCENARIOS if args.scenario == "all" else tuple(s for s in SCENARIOS if s.name == args.scenario)
    for scenario in selected:
        write_scenario(output_dir, scenario)
        print(f"[ald_like_inputs] wrote scenario={scenario.name!r} to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
