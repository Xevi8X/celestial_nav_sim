"""Shared data and numerical helpers for the experiment scripts."""

from __future__ import annotations

import datetime as dt
import hashlib
from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
from scipy.spatial.transform import Rotation

from common import Observer
from sky_render.camera import Camera


__all__ = [
    "Location",
    "angle_difference_deg",
    "assembly_rotation",
    "body_from_camera_observer",
    "body_from_ned",
    "camera_observer_from_body",
    "noisy_attitude",
    "rotation_error_deg",
    "seed",
    "validate_simulation_settings",
]


@dataclass(frozen=True)
class Location:
    name: str
    latitude_deg: float
    longitude_deg: float
    elevation_m: float
    observation_time: dt.datetime


def validate_simulation_settings(
    camera: Camera,
    attitude_noise_std_deg: float,
    locations: tuple[Location, ...],
    repetitions: int,
    random_seed: int,
) -> None:
    """Validate settings shared by the simulated pipeline stages."""

    if not isinstance(camera, Camera) or not camera.is_valid():
        raise ValueError("camera must contain valid tuned settings")
    if (
        isinstance(attitude_noise_std_deg, bool)
        or not isinstance(attitude_noise_std_deg, Real)
        or not np.isfinite(attitude_noise_std_deg)
        or attitude_noise_std_deg < 0.0
    ):
        raise ValueError(
            "attitude_noise_std_deg must be finite and non-negative"
        )
    if not isinstance(locations, tuple) or not locations:
        raise ValueError("locations must be a non-empty tuple")
    if not all(isinstance(location, Location) for location in locations):
        raise TypeError("locations must contain Location values")
    names = tuple(location.name for location in locations)
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError("every location must have a non-empty name")
    if len(set(names)) != len(names):
        raise ValueError("location names must be unique")
    if (
        isinstance(repetitions, bool)
        or not isinstance(repetitions, Integral)
        or repetitions <= 0
    ):
        raise ValueError("repetitions must be a positive integer")
    if isinstance(random_seed, bool) or not isinstance(random_seed, Integral):
        raise ValueError("random_seed must be an integer")


def seed(random_seed: int, *parts: object) -> int:
    """Derive a stable independent seed from an explicit experiment seed."""

    value = "|".join(str(part) for part in (random_seed, *parts))
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2 ** 32)


def body_from_ned(
    roll_deg: float,
    pitch_deg: float,
    yaw_deg: float,
) -> np.ndarray:
    observer = Observer()
    observer.set_orientation(roll_deg, pitch_deg, yaw_deg)
    return np.vstack((
        observer.look_dir,
        observer.look_right,
        -observer.look_up,
    ))


def camera_observer_from_body(
    navigation_to_body: np.ndarray,
    camera_to_body: np.ndarray,
    location: Location,
    timestamp: dt.datetime,
) -> Observer:
    navigation_to_camera = camera_to_body.T @ navigation_to_body
    observer = Observer()
    observer.set_time(timestamp)
    observer.set_location(
        location.latitude_deg,
        location.longitude_deg,
        location.elevation_m,
    )
    observer.set_look_direction(
        look_dir=navigation_to_camera[2],
        look_up=-navigation_to_camera[1],
    )
    return observer


def body_from_camera_observer(
    observer: Observer,
    camera_to_body: np.ndarray,
) -> np.ndarray:
    if observer.observer_matrix is None:
        raise ValueError("camera observer has no solved orientation")
    navigation_to_camera = np.vstack((
        observer.look_right,
        -observer.look_up,
        observer.look_dir,
    ))
    return camera_to_body @ navigation_to_camera


def noisy_attitude(
    actual: np.ndarray,
    random_seed: int,
    standard_deviation_deg: float,
) -> np.ndarray:
    rng = np.random.default_rng(random_seed)
    error = Rotation.from_rotvec(
        rng.normal(0.0, np.radians(standard_deviation_deg), size=3)
    ).as_matrix()
    return error @ actual


def assembly_rotation(angles_deg) -> np.ndarray:
    return Rotation.from_euler("xyz", angles_deg, degrees=True).as_matrix()


def rotation_error_deg(actual: np.ndarray, estimated: np.ndarray) -> float:
    relative = estimated @ actual.T
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def angle_difference_deg(first: float, second: float) -> float:
    return float(abs((first - second + 180.0) % 360.0 - 180.0))
