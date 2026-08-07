"""Static-position experiment from four camera azimuths."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from common import ECEF
from imu import (
    Accelerometer,
    AccelerometerParameters,
    MagnetometerParameters,
)
from sky_render import Camera

from .common import (
    Location,
    body_from_ned,
    noisy_attitude,
    seed,
    validate_simulation_settings,
)


__all__ = ["Input", "Output", "run"]


_AZIMUTHS_DEG = (0.0, 90.0, 180.0, 270.0)
_SAMPLES_PER_POSE = 15
_ROLL_DEG = 0.0
_PITCH_DEG = 60.0
_OUTLIER_THRESHOLD_DEG = 1.0
_SINGULAR_HORIZONTAL_NORM = 1e-8


@dataclass(frozen=True)
class Input:
    """Full pipeline handoff for static-position estimation.

    Position is calculated from the camera-derived attitude uncertainty and
    calibrated accelerometer.  The camera and selected magnetometer models
    remain explicit here so an incomplete upstream calibration cannot be
    mistaken for a complete experiment run.
    """

    camera: Camera
    attitude_noise_std_deg: float
    true_accelerometer_parameters: AccelerometerParameters
    accelerometer_models: Mapping[
        tuple[str, int], AccelerometerParameters
    ]
    true_magnetometer_parameters: MagnetometerParameters
    magnetometer_models: Mapping[
        tuple[str, int], MagnetometerParameters
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
        input.attitude_noise_std_deg,
        input.locations,
        input.repetitions,
        input.random_seed,
    )
    if not isinstance(
        input.true_accelerometer_parameters,
        AccelerometerParameters,
    ):
        raise TypeError(
            "true_accelerometer_parameters must be AccelerometerParameters"
        )
    if not isinstance(
        input.true_magnetometer_parameters,
        MagnetometerParameters,
    ):
        raise TypeError(
            "true_magnetometer_parameters must be MagnetometerParameters"
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
    _require_models(
        input.magnetometer_models,
        required_keys,
        MagnetometerParameters,
        "magnetometer_models",
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


def _reject_outliers(zeniths, positions):
    vectors = np.asarray(zeniths)
    center = np.sum(vectors, axis=0)
    center /= np.linalg.norm(center)
    angles = np.degrees(np.arccos(np.clip(vectors @ center, -1.0, 1.0)))
    inliers = angles <= _OUTLIER_THRESHOLD_DEG
    if np.count_nonzero(inliers) < 2:
        return vectors, positions, 0
    return (
        vectors[inliers],
        [position for position, keep in zip(positions, inliers) if keep],
        int(np.count_nonzero(~inliers)),
    )


def run(input: Input) -> Output:
    """Estimate positions using only explicitly supplied pipeline models."""

    _validate_input(input)
    results = []
    observations = []

    for repetition in range(input.repetitions):
        for location in input.locations:
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
            rejected = 0

            azimuths = (
                azimuth
                for azimuth in _AZIMUTHS_DEG
                for _ in range(_SAMPLES_PER_POSE)
            )
            for sample, azimuth in enumerate(azimuths):
                attitude = body_from_ned(_ROLL_DEG, _PITCH_DEG, azimuth)
                reading = sensor.measure(attitude)
                try:
                    zenith_body = correction.correct(reading)
                    zenith_body /= np.linalg.norm(zenith_body)
                except (ValueError, np.linalg.LinAlgError):
                    rejected += 1
                    continue

                estimated_attitude = noisy_attitude(
                    attitude,
                    seed(
                        input.random_seed,
                        "position",
                        repetition,
                        location.name,
                        sample,
                    ),
                    input.attitude_noise_std_deg,
                )
                zenith_ecef = (
                    ecef_to_ned.T
                    @ estimated_attitude.T
                    @ zenith_body
                )
                zenith_ecef /= np.linalg.norm(zenith_ecef)
                position = _zenith_location(zenith_ecef)
                if position is None:
                    rejected += 1
                    continue

                zeniths.append(zenith_ecef)
                positions.append(position)
                observations.append({
                    "location": location.name,
                    "repetition": repetition,
                    "sample": sample,
                    "azimuth_deg": azimuth,
                    "latitude_deg": position[0],
                    "longitude_deg": position[1],
                    "position_error_m": _position_error_m(
                        location.latitude_deg,
                        location.longitude_deg,
                        position,
                    ),
                })

            if not zeniths:
                results.append({
                    "location": location.name,
                    "repetition": repetition,
                    "latitude_deg": location.latitude_deg,
                    "longitude_deg": location.longitude_deg,
                    "individual_image_mean_error_m": None,
                    "four_pose_error_m": None,
                    "accepted_images": 0,
                    "rejected_images": rejected,
                    "status": "no_valid_images",
                })
                continue

            zeniths, positions, outliers = _reject_outliers(
                zeniths,
                positions,
            )
            rejected += outliers
            combined_position = _zenith_location(np.sum(zeniths, axis=0))

            if combined_position is None:
                combined_error = None
                status = "rejected_invalid_combined_position"
            else:
                combined_error = _position_error_m(
                    location.latitude_deg,
                    location.longitude_deg,
                    combined_position,
                )
                status = "complete"

            results.append({
                "location": location.name,
                "repetition": repetition,
                "latitude_deg": location.latitude_deg,
                "longitude_deg": location.longitude_deg,
                "individual_image_mean_error_m": _mean_position_error_m(
                    location.latitude_deg,
                    location.longitude_deg,
                    positions,
                ),
                "four_pose_error_m": combined_error,
                "accepted_images": len(zeniths),
                "rejected_images": rejected,
                "status": status,
            })

    return Output(tuple(results), tuple(observations))
