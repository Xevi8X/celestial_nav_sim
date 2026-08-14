"""Accelerometer calibration along a synchronized figure-eight maneuver."""

from __future__ import annotations

import datetime as dt
import logging
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from celestial_nav import Navigator
from common import Config, Sky
from imu import Accelerometer, AccelerometerCalibration, AccelerometerParameters
from sky_render import Camera, Renderer

from .common import (
    Location,
    body_from_camera_observer,
    body_from_ned,
    camera_observer_from_body,
    rotation_error_deg,
    seed,
    validate_simulation_settings,
)


__all__ = ["Input", "Output", "run"]


LOGGER = logging.getLogger(__name__)


_INITIAL_AZIMUTHS_DEG = (0.0, 90.0, 180.0, 270.0)
# Use one explicitly chosen run as the calibration passed to later experiments.
_HANDOFF_AZIMUTH_DEG = 0.0
_PATTERN_PERIOD_S = 10.0
_AZIMUTH_AMPLITUDE_DEG = 15.0
_ELEVATION_CENTER_DEG = 50.0
_ELEVATION_AMPLITUDE_DEG = 30.0
_ROLL_AMPLITUDE_DEG = 45.0
_MATRIX_INDICATOR_LIMIT = 0.12
_BIAS_INDICATOR_LIMIT = 1.2
_MAXIMUM_CONDITION_NUMBER = np.inf
_CONSECUTIVE_UPDATES = 3
_MAXIMUM_OBSERVATIONS = 200


@dataclass(frozen=True)
class Input:
    camera: Camera
    sky: Sky
    camera_to_body: np.ndarray
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
        value.locations,
        value.repetitions,
        value.random_seed,
    )
    if not isinstance(value.sky, Sky):
        raise TypeError("sky must be a Sky")
    camera_to_body = np.asarray(value.camera_to_body)
    if camera_to_body.shape != (3, 3) or not np.isfinite(camera_to_body).all():
        raise ValueError("camera_to_body must be a finite 3x3 matrix")
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

    fov = input.camera.geometry.fov
    fov_range = (max(1, round(fov) - 2), min(179, round(fov) + 2))
    navigator = Navigator(input.sky, fov_range=fov_range)
    renderer = Renderer(input.sky, input.camera)

    for repetition in range(input.repetitions):
        for location in input.locations:
            for initial_azimuth in _INITIAL_AZIMUTHS_DEG:
                LOGGER.info(
                    "Accelerometer calibration: %s, repetition %d/%d, "
                    "initial azimuth %.0f deg",
                    location.name,
                    repetition + 1,
                    input.repetitions,
                    initial_azimuth,
                )
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
                rejected_observations = 0

                for observation in range(_MAXIMUM_OBSERVATIONS):
                    time_s = observation * exposure_s
                    timestamp = location.observation_time + dt.timedelta(
                        seconds=time_s
                    )
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

                    observer = camera_observer_from_body(
                        actual_attitude,
                        input.camera_to_body,
                        location,
                        timestamp,
                    )
                    render_seed = seed(
                        input.random_seed, run_seed, observation, "render"
                    )
                    tic = dt.datetime.now()
                    image = renderer.render(observer, noise_seed=render_seed)
                    render_time = dt.datetime.now() - tic
                    estimated_observer = navigator.estimate_orientation(
                        image,
                        timestamp,
                        location.latitude_deg,
                        location.longitude_deg,
                        location.elevation_m,
                    )
                    plate_solving_time = dt.datetime.now() - tic - render_time
                    LOGGER.debug(
                        "Render %.1f ms; plate solve %.1f ms",
                        render_time.total_seconds() * 1000.0,
                        plate_solving_time.total_seconds() * 1000.0,
                    )

                    accepted = estimated_observer.observer_matrix is not None
                    elapsed = time_s + exposure_s

                    if accepted:
                        reference_attitude = body_from_camera_observer(
                            estimated_observer, input.camera_to_body
                        )
                        estimate = calibration.update(reference_attitude, reading)
                    else:
                        rejected_observations += 1

                    indicators = calibration.indicators
                    condition_number = calibration.condition_number

                    valid_indicators = (
                        indicators is not None
                        and len(indicators) == 4
                        and np.all(np.isfinite(indicators))
                    )
                    if valid_indicators:
                        LOGGER.debug(
                            "Indicators: %s",
                            ", ".join(f"{value:.3f}" for value in indicators),
                        )
                        i = {
                            "i1": float(indicators[0]),
                            "i2": float(indicators[1]),
                            "i3": float(indicators[2]),
                            "i4": float(indicators[3]),
                        }
                    else:
                        i = {
                            "i1": None,
                            "i2": None,
                            "i3": None,
                            "i4": None,
                        }

                    updates.append({
                        "location": location.name,
                        "repetition": repetition,
                        "azimuth_deg": initial_azimuth,
                        "observation": observation,
                        "elapsed_s": elapsed,
                        "roll_deg": roll,
                        "camera_elevation_deg": elevation,
                        "camera_azimuth_deg": azimuth,
                        "accepted": accepted,
                        "initialized": calibration.initialized,
                        **i,
                        "converged": calibration.converged,
                        "condition_number": (
                            condition_number
                            if np.isfinite(condition_number) else None
                        ),
                        "stable_updates": calibration.stable_updates,
                    })

                    completed = observation + 1
                    if (
                        completed == 1
                        or completed % 10 == 0
                        or calibration.converged
                    ):
                        LOGGER.info(
                            "Accelerometer calibration %s/%.0f deg: "
                            "%d/%d samples, %d accepted, %d rejected",
                            location.name,
                            initial_azimuth,
                            completed,
                            _MAXIMUM_OBSERVATIONS,
                            calibration.fit.count,
                            rejected_observations,
                        )
                    if calibration.converged:
                        LOGGER.info(
                            "Accelerometer calibration converged after %d "
                            "accepted samples",
                            calibration.fit.count,
                        )
                        break

                initialized = calibration.initialized and estimate is not None
                if initialized:
                    rotation_error = rotation_error_deg(
                        actual.assembly_matrix,
                        estimate.assembly_matrix,
                    )
                    scale_rmse = float(np.sqrt(np.mean(np.square(
                        estimate.axis_scale - actual.axis_scale
                    ))))
                    bias_error = float(np.linalg.norm(
                        estimate.bias - actual.bias
                    ))
                else:
                    rotation_error = None
                    scale_rmse = None
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
                    "scale_rmse": scale_rmse,
                    "bias_error_mps2": bias_error,
                    "accepted_observations": calibration.fit.count,
                    "rejected_observations": rejected_observations,
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
