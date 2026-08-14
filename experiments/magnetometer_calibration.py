"""Magnetometer calibration experiment from the article."""

from __future__ import annotations

import datetime as dt
import logging
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from celestial_nav import Navigator
from common import Sky
from imu import (
    Magnetometer,
    MagnetometerCalibration,
    MagnetometerParameters,
    WMM2025Provider,
    magnetic_heading_deg,
)
from sky_render import Camera, Renderer

from .common import (
    Location,
    angle_difference_deg,
    body_from_ned,
    render_and_solve_body_attitude,
    seed,
    validate_simulation_settings,
)


__all__ = ["Input", "Output", "run"]


LOGGER = logging.getLogger(__name__)


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
    sky: Sky
    camera_to_body: np.ndarray
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
        value.locations,
        value.repetitions,
        value.random_seed,
    )
    if not isinstance(value.sky, Sky):
        raise TypeError("sky must be a Sky")
    camera_to_body = np.asarray(value.camera_to_body)
    if camera_to_body.shape != (3, 3) or not np.isfinite(camera_to_body).all():
        raise ValueError("camera_to_body must be a finite 3x3 matrix")
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


def _rms(values):
    return None if not values else float(np.sqrt(np.mean(np.square(values))))


def run(input: Input) -> Output:
    """Run both maneuvers and return their fitted models and raw records."""

    _validate_input(input)
    wmm = WMM2025Provider()
    actual = input.true_parameters
    results = []
    updates = []
    models = {}

    fov = input.camera.geometry.fov
    fov_range = (max(1, round(fov) - 2), min(179, round(fov) + 2))
    navigator = Navigator(input.sky, fov_range=fov_range)
    renderer = Renderer(input.sky, input.camera)
    exposure_s = input.camera.image_model.exposure_time

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
                poses = tuple(_orientations(constrained))
                LOGGER.info(
                    "Magnetometer calibration: %s, repetition %d/%d, %s "
                    "(%d samples)",
                    location.name,
                    repetition + 1,
                    input.repetitions,
                    procedure,
                    len(poses),
                )
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

                accepted_observations = 0
                rejected_observations = 0
                for index, (roll, pitch, yaw) in enumerate(poses):
                    timestamp = location.observation_time + dt.timedelta(
                        seconds=index * exposure_s
                    )
                    actual_attitude = body_from_ned(roll, pitch, yaw)
                    reading = sensor.measure(actual_attitude @ field_ned)

                    estimated_attitude = render_and_solve_body_attitude(
                        renderer,
                        navigator,
                        input.camera_to_body,
                        location,
                        timestamp,
                        actual_attitude,
                        seed(run_seed, index, "render"),
                    )
                    accepted = estimated_attitude is not None
                    if accepted:
                        calibration.update(
                            estimated_attitude @ field_ned,
                            reading,
                        )
                        accepted_observations += 1
                    else:
                        rejected_observations += 1

                    condition_number = calibration.condition_number
                    updates.append({
                        "location": location.name,
                        "repetition": repetition,
                        "procedure": procedure,
                        "observation": index,
                        "roll_deg": roll,
                        "pitch_deg": pitch,
                        "yaw_deg": yaw,
                        "accepted": accepted,
                        "initialized": calibration.initialized,
                        "converged": calibration.converged,
                        "condition_number": (
                            condition_number
                            if np.isfinite(condition_number) else None
                        ),
                        "stable_updates": calibration.stable_updates,
                    })
                    completed = index + 1
                    if (
                        completed == 1
                        or completed % 10 == 0
                        or completed == len(poses)
                        or calibration.converged
                    ):
                        LOGGER.info(
                            "Magnetometer calibration %s/%s: %d/%d samples, "
                            "%d accepted, %d rejected",
                            location.name,
                            procedure,
                            completed,
                            len(poses),
                            accepted_observations,
                            rejected_observations,
                        )
                    if calibration.converged:
                        break

                LOGGER.info(
                    "Magnetometer calibration %s/%s finished: %s, "
                    "%d accepted, %d rejected",
                    location.name,
                    procedure,
                    "converged" if calibration.converged else "maneuver complete",
                    accepted_observations,
                    rejected_observations,
                )

                fitted[procedure] = (
                    calibration,
                    accepted_observations,
                    rejected_observations,
                )
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
            uncalibrated_vector_errors = []
            uncalibrated_heading_errors = []
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

                uncalibrated_vector_errors.append(float(np.linalg.norm(
                    reading - reference_body
                )))
                if not field.in_blackout_zone:
                    raw_ned = attitude.T @ reading
                    try:
                        uncalibrated_heading_errors.append(
                            angle_difference_deg(
                                magnetic_heading_deg(raw_ned),
                                magnetic_heading_deg(field_ned),
                            )
                        )
                    except ValueError:
                        pass

                for procedure, (calibration, _accepted, _rejected) in (
                    fitted.items()
                ):
                    if not calibration.initialized:
                        continue
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

            results.append({
                "location": location.name,
                "repetition": repetition,
                "procedure": "Uncalibrated",
                "vector_error_ut": _rms(uncalibrated_vector_errors),
                "heading_error_deg": _rms(uncalibrated_heading_errors),
                "test_orientation_count": len(uncalibrated_vector_errors),
                "accepted_observations": None,
                "rejected_observations": None,
                "wmm_blackout_zone": field.in_blackout_zone,
                "status": (
                    "heading_rejected_wmm_blackout"
                    if field.in_blackout_zone
                    else "baseline"
                ),
            })
            for procedure, (
                calibration,
                accepted_observations,
                rejected_observations,
            ) in fitted.items():
                vector_errors, heading_errors = evaluations[procedure]
                results.append({
                    "location": location.name,
                    "repetition": repetition,
                    "procedure": procedure,
                    "vector_error_ut": _rms(vector_errors),
                    "heading_error_deg": _rms(heading_errors),
                    "test_orientation_count": len(vector_errors),
                    "accepted_observations": accepted_observations,
                    "rejected_observations": rejected_observations,
                    "wmm_blackout_zone": field.in_blackout_zone,
                    "status": (
                        "unobservable"
                        if not calibration.initialized
                        else "heading_rejected_wmm_blackout"
                        if field.in_blackout_zone
                        else "converged"
                        if calibration.converged
                        else "maneuver_complete"
                    ),
                })

    return Output(models, tuple(results), tuple(updates))
