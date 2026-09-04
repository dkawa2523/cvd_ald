"""Generate small deterministic Fluent/measurement fixtures for role fitting."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


SPECIES = np.asarray(("s0", "s1", "s2", "s3"))


def _xy_points() -> np.ndarray:
    return np.asarray(
        [
            [0.0, 0.0],
            [30.0, 0.0],
            [-30.0, 0.0],
            [0.0, 45.0],
            [0.0, -45.0],
            [70.0, 30.0],
            [-70.0, -30.0],
            [110.0, 0.0],
        ],
        dtype=float,
    )


def _spatial_base(xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r = np.sqrt(np.sum(np.square(xy), axis=1))
    r_norm = r / max(float(np.max(r)), 1.0)
    angle = np.arctan2(xy[:, 1], xy[:, 0])
    return r_norm, angle


def write_cvd(output_dir: Path) -> None:
    xy = _xy_points()
    r_norm, angle = _spatial_base(xy)
    s0 = np.clip(1.00 - 0.25 * r_norm + 0.08 * np.cos(angle), 0.05, np.inf)
    s1 = np.clip(0.30 + 0.10 * np.sin(2.0 * angle), 0.02, np.inf)
    s2 = np.clip(0.55 + 0.20 * r_norm, 0.02, np.inf)
    s3 = np.full_like(s0, 0.05)
    cref = np.stack([s0, s1, s2, s3], axis=1)
    velocity = np.clip(0.04 - 0.01 * r_norm + 0.004 * np.cos(angle), 0.004, np.inf)
    flux_sink = np.clip(cref * velocity[:, None], 0.0, np.inf)
    h_meas = 0.040 * (1.0 - 0.12 * r_norm + 0.04 * np.cos(angle))
    np.savez(output_dir / "fluent_cvd_steady.npz", xy=xy, cref=cref, flux_sink=flux_sink, species=SPECIES)
    np.savez(output_dir / "meas_cvd_steady.npz", xy=xy, h_nm=h_meas)


def write_ald(output_dir: Path) -> None:
    xy = _xy_points()
    r_norm, angle = _spatial_base(xy)
    precursor = np.clip(1.00 - 0.20 * r_norm + 0.05 * np.cos(angle), 0.05, np.inf)
    coreactant = np.clip(0.55 + 0.25 * r_norm + 0.08 * np.sin(angle), 0.05, np.inf)
    blocker = np.clip(0.15 + 0.05 * np.sin(2.0 * angle), 0.0, np.inf)
    carrier = np.full_like(precursor, 0.04)
    residual = 0.02
    frames = [
        np.stack([precursor, residual * coreactant, 0.02 * blocker, carrier], axis=1),
        np.stack([residual * precursor, residual * coreactant, residual * blocker, carrier], axis=1),
        np.stack([residual * precursor, coreactant, 0.02 * blocker, carrier], axis=1),
        np.stack([residual * precursor, residual * coreactant, residual * blocker, carrier], axis=1),
        np.stack([residual * precursor, residual * coreactant, residual * blocker, carrier], axis=1),
    ]
    time = np.asarray([0.0, 0.25, 0.45, 0.70, 0.90], dtype=float)
    cref = np.stack(frames, axis=0)
    velocity = np.clip(0.04 - 0.01 * r_norm, 0.004, np.inf)[None, :, None]
    flux_sink = np.clip(cref * velocity, 0.0, np.inf)
    h_meas = 0.045 * precursor / max(float(np.mean(precursor)), 1.0e-12)
    np.savez(output_dir / "fluent_ald_transient.npz", xy=xy, time=time, cref=cref, flux_sink=flux_sink, species=SPECIES)
    np.savez(output_dir / "meas_ald_final.npz", xy=xy, h_nm=h_meas)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate role-fit inputs.")
    parser.add_argument("--output-dir", default="runs/generated_inputs/role_fit")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_cvd(output_dir)
    write_ald(output_dir)
    print(f"[role_fit_inputs] wrote CVD/ALD fit inputs to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
