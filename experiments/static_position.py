"""Static-position experiment from four camera azimuths."""

from __future__ import annotations

import datetime as dt
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from celestial_nav import Navigator
from common import ECEF, Sky
from imu import Accelerometer, AccelerometerParameters
from sky_render import Camera, Renderer

from .common import (
    Location,
    body_from_ned,
    render_and_solve_body_attitude,
    seed,
    validate_simulation_settings,
)


__all__ = ["Input", "Output", "run"]


LOGGER = logging.getLogger(__name__)


_AZIMUTHS_DEG = (0.0, 90.0, 180.0, 270.0)
_SAMPLES_PER_POSE = 25
_MAXIMUM_ATTEMPTS_PER_DIRECTION = 50
_ACCELEROMETER_READINGS_PER_IMAGE = 20
_PATTERN_CHECKING_STARS = 12
_ROLL_DEG = 0.0
_PITCH_DEG = 60.0
_OUTLIER_THRESHOLD_DEG = 1.0
_SINGULAR_HORIZONTAL_NORM = 1e-8
_CARDINAL_DIRECTION_NAMES = {
    0.0: "North",
    90.0: "East",
    180.0: "South",
    270.0: "West",
}


@dataclass(frozen=True)
class Input:
    """Camera and calibrated accelerometers used for position estimation."""

    camera: Camera
    sky: Sky
    camera_to_body: np.ndarray
    true_accelerometer_parameters: AccelerometerParameters
    accelerometer_models: Mapping[
        tuple[str, int], AccelerometerParameters
    ]
    locations: tuple[Location, ...]
    repetitions: int
    random_seed: int


@dataclass(frozen=True)
class Output:
    results: tuple[Mapping[str, object], ...]
    observations: tuple[Mapping[str, object], ...]


def _validate_input(input: Input) -> None:
    if not isinstance(input, Input):
        raise TypeError("input must be a static-position Input")
    validate_simulation_settings(
        input.camera,
        input.locations,
        input.repetitions,
        input.random_seed,
    )
    if not isinstance(input.sky, Sky):
        raise TypeError("sky must be a Sky")
    camera_to_body = np.asarray(input.camera_to_body)
    if camera_to_body.shape != (3, 3) or not np.isfinite(camera_to_body).all():
        raise ValueError("camera_to_body must be a finite 3x3 matrix")
    if not isinstance(
        input.true_accelerometer_parameters,
        AccelerometerParameters,
    ):
        raise TypeError(
            "true_accelerometer_parameters must be AccelerometerParameters"
        )
    required_keys = tuple(
        (location.name, repetition)
        for repetition in range(input.repetitions)
        for location in input.locations
    )
    _require_models(
        input.accelerometer_models,
        required_keys,
        AccelerometerParameters,
        "accelerometer_models",
    )


def _require_models(models, keys, model_type, name: str) -> None:
    if not isinstance(models, Mapping):
        raise TypeError(f"{name} must be a mapping")
    missing = [key for key in keys if not isinstance(models.get(key), model_type)]
    if missing:
        raise ValueError(f"{name} is missing calibrated runs: {missing}")


def _zenith_location(zenith_ecef: np.ndarray):
    vector = np.asarray(zenith_ecef, dtype=float)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        return None

    norm = np.linalg.norm(vector)
    if norm <= np.finfo(float).eps:
        return None
    vector = vector / norm

    horizontal = float(np.hypot(vector[0], vector[1]))
    if horizontal <= _SINGULAR_HORIZONTAL_NORM:
        return None

    latitude = np.degrees(np.arctan2(vector[2], horizontal))
    longitude = np.degrees(np.arctan2(vector[1], vector[0]))
    return float(latitude), float(longitude)


def _position_error_m(
    reference_latitude_deg: float,
    reference_longitude_deg: float,
    estimated_position: Sequence[float],
) -> float:
    north_east = ECEF.north_east_vector(
        reference_latitude_deg,
        reference_longitude_deg,
        float(estimated_position[0]),
        float(estimated_position[1]),
    )
    return float(np.linalg.norm(north_east))


def _mean_position_error_m(
    reference_latitude_deg: float,
    reference_longitude_deg: float,
    positions: Sequence[Sequence[float]],
) -> float:
    if not positions:
        raise ValueError("at least one individual position is required")
    return float(np.mean([
        _position_error_m(
            reference_latitude_deg,
            reference_longitude_deg,
            position,
        )
        for position in positions
    ]))


def _inlier_mask(zeniths):
    vectors = np.asarray(zeniths)
    center = np.sum(vectors, axis=0)
    center /= np.linalg.norm(center)
    angles = np.degrees(np.arccos(np.clip(vectors @ center, -1.0, 1.0)))
    inliers = angles <= _OUTLIER_THRESHOLD_DEG
    if np.count_nonzero(inliers) < 2:
        return np.ones(len(vectors), dtype=bool)
    return inliers


def run(input: Input) -> Output:
    """Estimate positions using only explicitly supplied pipeline models."""

    _validate_input(input)
    results = []
    observations = []

    fov = input.camera.geometry.fov
    fov_range = (max(1, round(fov) - 2), min(179, round(fov) + 2))
    navigator = Navigator(
        input.sky,
        fov_range=fov_range,
        pattern_checking_stars=_PATTERN_CHECKING_STARS,
    )
    renderer = Renderer(input.sky, input.camera)
    exposure_s = input.camera.image_model.exposure_time

    for repetition in range(input.repetitions):
        for location in input.locations:
            target_samples = len(_AZIMUTHS_DEG) * _SAMPLES_PER_POSE
            maximum_attempts = (
                len(_AZIMUTHS_DEG) * _MAXIMUM_ATTEMPTS_PER_DIRECTION
            )
            LOGGER.info(
                "Static position: %s, repetition %d/%d "
                "(target %d valid samples, at most %d attempts)",
                location.name,
                repetition + 1,
                input.repetitions,
                target_samples,
                maximum_attempts,
            )
            correction = input.accelerometer_models[
                (location.name, repetition)
            ]
            sensor = Accelerometer(
                input.true_accelerometer_parameters,
                seed=seed(
                    input.random_seed,
                    "position",
                    repetition,
                    location.name,
                ),
            )
            ecef_to_ned = ECEF.ecef_to_ned(
                location.latitude_deg,
                location.longitude_deg,
            )
            zeniths = []
            positions = []
            location_observations = []
            solved_images = 0
            attempted_images = 0
            direction_attempted_images = {}
            direction_valid_images = {}
            outliers = 0

            for azimuth in _AZIMUTHS_DEG:
                direction = _CARDINAL_DIRECTION_NAMES.get(
                    float(azimuth),
                    f"{float(azimuth):g} deg",
                )
                direction_attempts = 0
                direction_valid = 0
                direction_zeniths = []
                direction_positions = []
                direction_observations = []
                direction_inliers = np.empty(0, dtype=bool)
                while (
                    direction_attempts < _MAXIMUM_ATTEMPTS_PER_DIRECTION
                    and direction_valid < _SAMPLES_PER_POSE
                ):
                    sample = attempted_images
                    direction_attempt = direction_attempts
                    attempted_images += 1
                    direction_attempts += 1
                    timestamp = location.observation_time + dt.timedelta(
                        seconds=sample * exposure_s
                    )
                    attitude = body_from_ned(
                        _ROLL_DEG,
                        _PITCH_DEG,
                        azimuth,
                    )
                    reading = np.mean(
                        [
                            sensor.measure(attitude)
                            for _ in range(
                                _ACCELEROMETER_READINGS_PER_IMAGE
                            )
                        ],
                        axis=0,
                    )
                    estimated_attitude = render_and_solve_body_attitude(
                        renderer,
                        navigator,
                        input.camera_to_body,
                        location,
                        timestamp,
                        attitude,
                        seed(
                            input.random_seed,
                            "position",
                            repetition,
                            location.name,
                            sample,
                            "render",
                        ),
                    )
                    if estimated_attitude is not None:
                        solved_images += 1
                        try:
                            zenith_body = correction.correct(reading)
                            zenith_body /= np.linalg.norm(zenith_body)
                            zenith_ecef = (
                                ecef_to_ned.T
                                @ estimated_attitude.T
                                @ zenith_body
                            )
                            zenith_ecef /= np.linalg.norm(zenith_ecef)
                            position = _zenith_location(zenith_ecef)
                        except (ValueError, np.linalg.LinAlgError):
                            position = None

                        if position is not None:
                            direction_zeniths.append(zenith_ecef)
                            direction_positions.append(position)
                            direction_observations.append({
                                "location": location.name,
                                "repetition": repetition,
                                "sample": sample,
                                "direction_attempt": direction_attempt,
                                "direction": direction,
                                "azimuth_deg": azimuth,
                                "latitude_deg": position[0],
                                "longitude_deg": position[1],
                                "position_error_m": _position_error_m(
                                    location.latitude_deg,
                                    location.longitude_deg,
                                    position,
                                ),
                            })
                            direction_inliers = _inlier_mask(
                                direction_zeniths
                            )
                            direction_valid = min(
                                int(np.count_nonzero(direction_inliers)),
                                _SAMPLES_PER_POSE,
                            )

                    if (
                        direction_attempts == 1
                        or direction_attempts % 10 == 0
                        or direction_valid == _SAMPLES_PER_POSE
                    ):
                        LOGGER.info(
                            "Static position %s/%s: %d/%d attempts, "
                            "%d/%d valid; %d images solved, %d total valid",
                            location.name,
                            direction,
                            direction_attempts,
                            _MAXIMUM_ATTEMPTS_PER_DIRECTION,
                            direction_valid,
                            _SAMPLES_PER_POSE,
                            solved_images,
                            len(zeniths) + direction_valid,
                        )

                retained_indices = np.flatnonzero(
                    direction_inliers
                )[:_SAMPLES_PER_POSE]
                outliers += int(np.count_nonzero(~direction_inliers))
                zeniths.extend(
                    direction_zeniths[index]
                    for index in retained_indices
                )
                positions.extend(
                    direction_positions[index]
                    for index in retained_indices
                )
                location_observations.extend(
                    direction_observations[index]
                    for index in retained_indices
                )
                direction_valid = len(retained_indices)
                direction_attempted_images[direction] = direction_attempts
                direction_valid_images[direction] = direction_valid

            incomplete_directions = tuple(
                direction
                for direction, count in direction_valid_images.items()
                if count < _SAMPLES_PER_POSE
            )
            if not zeniths:
                individual_error = None
                combined_error = None
                status = "insufficient_direction_samples"
            else:
                combined_position = _zenith_location(np.sum(
                    zeniths,
                    axis=0,
                ))
                individual_error = _mean_position_error_m(
                    location.latitude_deg,
                    location.longitude_deg,
                    positions,
                )
                if combined_position is None:
                    combined_error = None
                    status = (
                        "insufficient_direction_samples"
                        if incomplete_directions
                        else "rejected_invalid_combined_position"
                    )
                else:
                    combined_error = _position_error_m(
                        location.latitude_deg,
                        location.longitude_deg,
                        combined_position,
                    )
                    if incomplete_directions:
                        status = "insufficient_direction_samples"
                    else:
                        status = "complete"

            observations.extend(location_observations)
            rejected_images = attempted_images - len(zeniths)

            results.append({
                "location": location.name,
                "repetition": repetition,
                "latitude_deg": location.latitude_deg,
                "longitude_deg": location.longitude_deg,
                "individual_image_mean_error_m": individual_error,
                "four_pose_error_m": combined_error,
                "target_accepted_images": target_samples,
                "attempted_images": attempted_images,
                "accepted_images": len(zeniths),
                "rejected_images": rejected_images,
                "outlier_images": outliers,
                "direction_attempted_images": direction_attempted_images,
                "direction_valid_images": direction_valid_images,
                "incomplete_directions": incomplete_directions,
                "status": status,
            })
            LOGGER.info(
                "Static position %s finished: %s, %d accepted, %d rejected",
                location.name,
                status,
                len(zeniths),
                rejected_images,
            )

    return Output(tuple(results), tuple(observations))
