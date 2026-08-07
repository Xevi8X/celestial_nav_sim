"""Accelerometer calibration along a synchronized figure-eight maneuver."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from imu import Accelerometer, AccelerometerCalibration, AccelerometerParameters
from sky_render import Camera

from .common import (
    Location,
    body_from_ned,
    noisy_attitude,
    rotation_error_deg,
    seed,
    validate_simulation_settings,
)


__all__ = ["Input", "Output", "run"]


_INITIAL_AZIMUTHS_DEG = (0.0, 90.0, 180.0, 270.0)
# Use one explicitly chosen run as the calibration passed to later experiments.
_HANDOFF_AZIMUTH_DEG = 0.0
_PATTERN_PERIOD_S = 10.0
_AZIMUTH_AMPLITUDE_DEG = 15.0
_ELEVATION_CENTER_DEG = 50.0
_ELEVATION_AMPLITUDE_DEG = 30.0
_ROLL_AMPLITUDE_DEG = 45.0
_MATRIX_INDICATOR_LIMIT = 0.04
_BIAS_INDICATOR_LIMIT = 0.30
_MAXIMUM_CONDITION_NUMBER = 30.0
_CONSECUTIVE_UPDATES = 3
_MAXIMUM_OBSERVATIONS = 10000


@dataclass(frozen=True)
class Input:
    camera: Camera
    attitude_noise_std_deg: float
    true_parameters: AccelerometerParameters
    locations: tuple[Location, ...]
    repetitions: int
    random_seed: int


@dataclass(frozen=True)
class Output:
    models: Mapping[tuple[str, int], AccelerometerParameters]
    results: tuple[Mapping[str, object], ...]
    updates: tuple[Mapping[str, object], ...]


def _validate_input(value: Input) -> None:
    if not isinstance(value, Input):
        raise TypeError("input must be an accelerometer calibration Input")
    validate_simulation_settings(
        value.camera,
        value.attitude_noise_std_deg,
        value.locations,
        value.repetitions,
        value.random_seed,
    )
    if not isinstance(value.true_parameters, AccelerometerParameters):
        raise TypeError("true_parameters must be AccelerometerParameters")
    parameter_values = (
        value.true_parameters.assembly_matrix,
        value.true_parameters.axis_scale,
        value.true_parameters.bias,
        value.true_parameters.noise_stddev,
    )
    if not all(np.isfinite(part).all() for part in parameter_values):
        raise ValueError("accelerometer parameters must be finite")


def _maneuver_pose(
    time_s: float,
    center_azimuth_deg: float,
) -> tuple[float, float, float]:
    """Return roll, camera elevation, and azimuth on the figure eight."""

    phase = 2.0 * np.pi * time_s / _PATTERN_PERIOD_S
    camera_azimuth_deg = (
        center_azimuth_deg
        + _AZIMUTH_AMPLITUDE_DEG * np.cos(phase)
    )
    camera_elevation_deg = (
        _ELEVATION_CENTER_DEG
        + _ELEVATION_AMPLITUDE_DEG * np.sin(2.0 * phase)
    )
    roll_deg = -_ROLL_AMPLITUDE_DEG * np.sin(phase)
    return (
        float(roll_deg),
        float(camera_elevation_deg),
        float(camera_azimuth_deg),
    )


def run(input: Input) -> Output:
    """Run the maneuver and return its fitted models and raw records."""

    _validate_input(input)
    actual = input.true_parameters
    results = []
    updates = []
    models = {}

    for repetition in range(input.repetitions):
        for location in input.locations:
            for initial_azimuth in _INITIAL_AZIMUTHS_DEG:
                run_seed = seed(
                    input.random_seed,
                    "accelerometer",
                    repetition,
                    location.name,
                    initial_azimuth,
                )
                sensor = Accelerometer(actual, seed=run_seed)
                calibration = AccelerometerCalibration(
                    matrix_indicator_limit=_MATRIX_INDICATOR_LIMIT,
                    bias_indicator_limit=_BIAS_INDICATOR_LIMIT,
                    max_condition_number=_MAXIMUM_CONDITION_NUMBER,
                    consecutive_updates=_CONSECUTIVE_UPDATES,
                )
                elapsed = 0.0
                estimate = None
                exposure_s = input.camera.image_model.exposure_time

                for observation in range(_MAXIMUM_OBSERVATIONS):
                    time_s = observation * exposure_s
                    roll, elevation, azimuth = _maneuver_pose(
                        time_s,
                        initial_azimuth,
                    )
                    actual_attitude = body_from_ned(
                        roll,
                        elevation,
                        azimuth,
                    )
                    reading = sensor.measure(actual_attitude)
                    reference_attitude = noisy_attitude(
                        actual_attitude,
                        seed(
                            input.random_seed,
                            run_seed,
                            observation,
                            "attitude",
                        ),
                        input.attitude_noise_std_deg,
                    )
                    estimate = calibration.update(reference_attitude, reading)
                    elapsed = time_s + exposure_s

                    condition_number = calibration.condition_number
                    updates.append({
                        "location": location.name,
                        "repetition": repetition,
                        "azimuth_deg": initial_azimuth,
                        "observation": observation,
                        "elapsed_s": elapsed,
                        "roll_deg": roll,
                        "camera_elevation_deg": elevation,
                        "camera_azimuth_deg": azimuth,
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
                        print("Converged after", observation, "observations")
                        break

                initialized = calibration.initialized and estimate is not None
                if initialized:
                    rotation_error = rotation_error_deg(
                        actual.assembly_matrix,
                        estimate.assembly_matrix,
                    )
                    scale_error = float(np.sqrt(np.mean(np.square(
                        100.0
                        * (estimate.axis_scale - actual.axis_scale)
                        / actual.axis_scale
                    ))))
                    bias_error = float(np.linalg.norm(
                        estimate.bias - actual.bias
                    ))
                else:
                    rotation_error = None
                    scale_error = None
                    bias_error = None

                if calibration.converged:
                    status = "converged"
                elif initialized:
                    status = "maximum_observations"
                else:
                    status = "unobservable"

                results.append({
                    "location": location.name,
                    "repetition": repetition,
                    "azimuth_deg": initial_azimuth,
                    "rotation_error_deg": rotation_error,
                    "scale_error_percent": scale_error,
                    "bias_error_mps2": bias_error,
                    "calibration_time_s": elapsed,
                    "accepted_observations": calibration.fit.count,
                    "rejected_observations": 0,
                    "status": status,
                    "true_rotation": actual.assembly_matrix.tolist(),
                    "estimated_rotation": (
                        estimate.assembly_matrix.tolist()
                        if initialized else None
                    ),
                    "true_scale": actual.axis_scale.tolist(),
                    "estimated_scale": (
                        estimate.axis_scale.tolist() if initialized else None
                    ),
                    "true_bias_mps2": actual.bias.tolist(),
                    "estimated_bias_mps2": (
                        estimate.bias.tolist() if initialized else None
                    ),
                })

                if (
                    initial_azimuth == _HANDOFF_AZIMUTH_DEG
                    and initialized
                    and calibration.converged
                ):
                    models[(location.name, repetition)] = estimate

    return Output(models, tuple(results), tuple(updates))
