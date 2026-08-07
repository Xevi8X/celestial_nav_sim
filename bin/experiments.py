"""Run and report the complete experiment pipeline."""

from __future__ import annotations

import datetime as dt
import os
from pathlib import Path
import sys

import numpy as np


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_RESULTS_DIRECTORY = _REPOSITORY_ROOT / "results"
_PHOTO_PATHS = tuple(
    _REPOSITORY_ROOT / ".data" / f"image{index}.fit"
    for index in (1, 2, 3)
)
_RANDOM_SEED = 20260805
_REPETITIONS = 10

if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))


def main() -> None:
    os.environ.setdefault(
        "MPLCONFIGDIR",
        str(_RESULTS_DIRECTORY / ".matplotlib"),
    )

    from common import Sky
    from experiments import (
        accelerometer_calibration,
        magnetometer_calibration,
        renderer_tuning,
        static_position,
    )
    from experiments.common import Location, assembly_rotation
    from experiments.reporting import (
        report_accelerometer_calibration,
        report_magnetometer_calibration,
        report_renderer_tuning,
        report_static_position,
    )
    from imu import AccelerometerParameters, MagnetometerParameters

    locations = (
        Location(
            "Warsaw",
            52.2297,
            21.0122,
            100.0,
            dt.datetime(2026, 1, 15, 23, tzinfo=dt.timezone.utc),
        ),
        Location(
            "New York",
            40.7128,
            -74.0060,
            10.0,
            dt.datetime(2026, 1, 16, 5, tzinfo=dt.timezone.utc),
        ),
        Location(
            "Sao Paulo",
            -23.5505,
            -46.6333,
            760.0,
            dt.datetime(2026, 1, 16, 3, tzinfo=dt.timezone.utc),
        ),
        Location(
            "Sydney",
            -33.8688,
            151.2093,
            58.0,
            dt.datetime(2026, 1, 15, 13, tzinfo=dt.timezone.utc),
        ),
    )
    accelerometer_parameters = AccelerometerParameters(
        assembly_matrix=assembly_rotation((0.8, -0.6, 1.0)),
        axis_scale=(1.025, 0.980, 1.012),
        bias=(0.08, -0.05, 0.12),
        noise_stddev=(0.02, 0.02, 0.02),
    )
    magnetometer_parameters = MagnetometerParameters(
        transformation_matrix=np.asarray((
            (1.04, 0.03, -0.02),
            (-0.01, 0.96, 0.025),
            (0.015, -0.02, 1.02),
        )),
        offset=np.asarray((6.0, -4.0, 3.0)),
        noise_stddev=np.asarray((0.15, 0.15, 0.15)),
    )

    renderer = renderer_tuning.run(renderer_tuning.Input(
        photo_paths=_PHOTO_PATHS,
        sky=Sky(magnitude_limit=10),
    ))
    accelerometer = accelerometer_calibration.run(
        accelerometer_calibration.Input(
            camera=renderer.camera,
            attitude_noise_std_deg=renderer.attitude_noise_std_deg,
            true_parameters=accelerometer_parameters,
            locations=locations,
            repetitions=_REPETITIONS,
            random_seed=_RANDOM_SEED,
        )
    )
    magnetometer = magnetometer_calibration.run(
        magnetometer_calibration.Input(
            camera=renderer.camera,
            attitude_noise_std_deg=renderer.attitude_noise_std_deg,
            true_parameters=magnetometer_parameters,
            locations=locations,
            repetitions=_REPETITIONS,
            random_seed=_RANDOM_SEED,
        )
    )
    position = static_position.run(static_position.Input(
        camera=renderer.camera,
        attitude_noise_std_deg=renderer.attitude_noise_std_deg,
        true_accelerometer_parameters=accelerometer_parameters,
        accelerometer_models=accelerometer.models,
        true_magnetometer_parameters=magnetometer_parameters,
        magnetometer_models=magnetometer.models,
        locations=locations,
        repetitions=_REPETITIONS,
        random_seed=_RANDOM_SEED,
    ))

    files = (
        *report_renderer_tuning(
            renderer,
            _RESULTS_DIRECTORY / "renderer_tuning",
        ),
        *report_accelerometer_calibration(
            accelerometer,
            _RESULTS_DIRECTORY / "accelerometer_calibration",
            locations,
        ),
        *report_magnetometer_calibration(
            magnetometer,
            _RESULTS_DIRECTORY / "magnetometer_calibration",
        ),
        *report_static_position(
            position,
            _RESULTS_DIRECTORY / "static_position",
        ),
    )
    for path in files:
        print(path)


if __name__ == "__main__":
    main()
