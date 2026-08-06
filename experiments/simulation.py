"""Small shared pieces used by the four article experiments."""

import datetime as dt
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from common import Observer


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RANDOM_SEED = 20260805
REPETITIONS = 10

# Mean angular residual obtained from the three real-image plate solutions.
ATTITUDE_NOISE_STD_DEG = 0.0022


@dataclass(frozen=True)
class Location:
    name: str
    latitude_deg: float
    longitude_deg: float
    elevation_m: float
    observation_time: dt.datetime


LOCATIONS = (
    Location(
        "Warsaw", 52.2297, 21.0122, 100.0,
        dt.datetime(2026, 1, 15, 23, tzinfo=dt.timezone.utc),
    ),
    Location(
        "New York", 40.7128, -74.0060, 10.0,
        dt.datetime(2026, 1, 16, 5, tzinfo=dt.timezone.utc),
    ),
    Location(
        "Sao Paulo", -23.5505, -46.6333, 760.0,
        dt.datetime(2026, 1, 16, 3, tzinfo=dt.timezone.utc),
    ),
    Location(
        "Sydney", -33.8688, 151.2093, 58.0,
        dt.datetime(2026, 1, 15, 13, tzinfo=dt.timezone.utc),
    ),
)


def seed(*parts: object) -> int:
    text = "|".join(str(part) for part in (RANDOM_SEED, *parts))
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2 ** 32)


def body_from_ned(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
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
    standard_deviation_deg: float = ATTITUDE_NOISE_STD_DEG,
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
