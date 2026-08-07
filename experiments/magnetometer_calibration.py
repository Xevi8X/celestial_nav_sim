"""Magnetometer calibration experiment from the article."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from imu import (
    Magnetometer,
    MagnetometerCalibration,
    MagnetometerParameters,
    WMM2025Provider,
    magnetic_heading_deg,
)
from sky_render import Camera

from .common import (
    Location,
    angle_difference_deg,
    body_from_ned,
    noisy_attitude,
    seed,
    validate_simulation_settings,
)


__all__ = ["Input", "Output", "run"]


_HEADING_STEP_DEG = 15.0
_CONSTRAINED_ROLL_DEG = 2.0
_CONSTRAINED_PITCH_DEG = 3.0
_FULL_SWEEPS_DEG = (
    (30.0, 30.0, 1.0),
    (-30.0, 45.0, -1.0),
    (30.0, 60.0, 1.0),
    (-30.0, 75.0, -1.0),
)

_HELD_OUT_ORIENTATIONS = 96
_INDICATOR_LIMIT = 0.03
_OFFSET_INDICATOR_LIMIT = 0.3
_MAXIMUM_CONDITION_NUMBER = 100.0
_CONSECUTIVE_UPDATES = 3
_REFERENCE_SCALE_UT = 50.0
_HANDOFF_PROCEDURE = "Complete pattern"


@dataclass(frozen=True)
class Input:
    camera: Camera
    attitude_noise_std_deg: float
    true_parameters: MagnetometerParameters
    locations: tuple[Location, ...]
    repetitions: int
    random_seed: int


@dataclass(frozen=True)
class Output:
    models: Mapping[tuple[str, int], MagnetometerParameters]
    results: tuple[Mapping[str, object], ...]
    updates: tuple[Mapping[str, object], ...]


def _validate_input(value: Input) -> None:
    if not isinstance(value, Input):
        raise TypeError("input must be a magnetometer calibration Input")
    validate_simulation_settings(
        value.camera,
        value.attitude_noise_std_deg,
        value.locations,
        value.repetitions,
        value.random_seed,
    )
    if not isinstance(value.true_parameters, MagnetometerParameters):
        raise TypeError("true_parameters must be MagnetometerParameters")
    parameter_values = (
        value.true_parameters.transformation_matrix,
        value.true_parameters.offset,
        value.true_parameters.noise_stddev,
    )
    if not all(np.isfinite(part).all() for part in parameter_values):
        raise ValueError("magnetometer parameters must be finite")


def _orientations(constrained: bool):
    headings = np.arange(0.0, 360.0, _HEADING_STEP_DEG)
    if constrained:
        for yaw in headings:
            yield (
                _CONSTRAINED_ROLL_DEG,
                _CONSTRAINED_PITCH_DEG,
                float(yaw),
            )
        return

    for roll, pitch, direction in _FULL_SWEEPS_DEG:
        sweep = headings if direction > 0 else headings[::-1]
        for yaw in sweep:
            yield roll, pitch, float(yaw)


def run(input: Input) -> Output:
    """Run both maneuvers and return their fitted models and raw records."""

    _validate_input(input)
    wmm = WMM2025Provider()
    actual = input.true_parameters
    results = []
    updates = []
    models = {}

    for repetition in range(input.repetitions):
        for location in input.locations:
            field = wmm.field(
                location.latitude_deg,
                location.longitude_deg,
                location.elevation_m,
                location.observation_time,
            )
            field_ned = field.ned_uT
            fitted = {}

            for procedure, constrained in (
                (_HANDOFF_PROCEDURE, False),
                ("One constrained turn", True),
            ):
                run_seed = seed(
                    input.random_seed,
                    "magnetometer",
                    repetition,
                    location.name,
                    procedure,
                )
                sensor = Magnetometer(actual, seed=run_seed)
                calibration = MagnetometerCalibration(
                    constrained=constrained,
                    indicator_limit=_INDICATOR_LIMIT,
                    offset_indicator_limit=_OFFSET_INDICATOR_LIMIT,
                    max_condition_number=_MAXIMUM_CONDITION_NUMBER,
                    consecutive_updates=_CONSECUTIVE_UPDATES,
                    reference_scale_ut=_REFERENCE_SCALE_UT,
                )

                accepted = 0
                for index, (roll, pitch, yaw) in enumerate(
                    _orientations(constrained)
                ):
                    actual_attitude = body_from_ned(roll, pitch, yaw)
                    reading = sensor.measure(actual_attitude @ field_ned)
                    estimated_attitude = noisy_attitude(
                        actual_attitude,
                        seed(
                            input.random_seed,
                            run_seed,
                            index,
                            "attitude",
                        ),
                        input.attitude_noise_std_deg,
                    )
                    calibration.update(
                        estimated_attitude @ field_ned,
                        reading,
                    )
                    accepted += 1

                    condition_number = calibration.condition_number
                    updates.append({
                        "location": location.name,
                        "repetition": repetition,
                        "procedure": procedure,
                        "observation": index,
                        "accepted": True,
                        "initialized": calibration.initialized,
                        "converged": calibration.converged,
                        "condition_number": (
                            condition_number
                            if np.isfinite(condition_number) else None
                        ),
                        "stable_updates": calibration.stable_updates,
                    })
                    if calibration.converged:
                        break

                fitted[procedure] = (calibration, accepted)
                if (
                    procedure == _HANDOFF_PROCEDURE
                    and calibration.initialized
                ):
                    models[(location.name, repetition)] = (
                        calibration.get_parameters()
                    )

            evaluations = {
                procedure: ([], []) for procedure in fitted
            }
            test_rng = np.random.default_rng(seed(
                input.random_seed,
                "magnetometer-tests",
                repetition,
                location.name,
            ))
            test_sensor = Magnetometer(
                actual,
                seed=seed(
                    input.random_seed,
                    "magnetometer-test-noise",
                    repetition,
                    location.name,
                ),
            )

            for _ in range(_HELD_OUT_ORIENTATIONS):
                attitude = body_from_ned(
                    test_rng.uniform(-60.0, 60.0),
                    test_rng.uniform(-70.0, 70.0),
                    test_rng.uniform(0.0, 360.0),
                )
                reference_body = attitude @ field_ned
                reading = test_sensor.measure(reference_body)

                for procedure, (calibration, _accepted) in fitted.items():
                    corrected_body = calibration.correct(reading)
                    vector_error = float(np.linalg.norm(
                        corrected_body - reference_body
                    ))
                    evaluations[procedure][0].append(vector_error)

                    if field.in_blackout_zone:
                        continue
                    corrected_ned = attitude.T @ corrected_body
                    try:
                        heading_error = angle_difference_deg(
                            magnetic_heading_deg(corrected_ned),
                            magnetic_heading_deg(field_ned),
                        )
                    except ValueError:
                        continue
                    evaluations[procedure][1].append(heading_error)

            for procedure, (calibration, accepted) in fitted.items():
                vector_errors, heading_errors = evaluations[procedure]
                results.append({
                    "location": location.name,
                    "repetition": repetition,
                    "procedure": procedure,
                    "vector_error_ut": float(np.sqrt(np.mean(
                        np.square(vector_errors)
                    ))),
                    "heading_error_deg": (
                        None if not heading_errors
                        else float(np.sqrt(np.mean(
                            np.square(heading_errors)
                        )))
                    ),
                    "test_orientation_count": len(vector_errors),
                    "accepted_observations": accepted,
                    "rejected_observations": 0,
                    "wmm_blackout_zone": field.in_blackout_zone,
                    "status": (
                        "heading_rejected_wmm_blackout"
                        if field.in_blackout_zone
                        else "converged"
                        if calibration.converged
                        else "maneuver_complete"
                    ),
                })

    return Output(models, tuple(results), tuple(updates))
