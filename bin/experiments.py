"""Run and report the complete experiment pipeline."""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import os
from pathlib import Path
import sys

import numpy as np


LOGGER = logging.getLogger(__name__)


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_RESULTS_DIRECTORY = _REPOSITORY_ROOT / "results"
_OUTPUT_CACHE = _RESULTS_DIRECTORY / "experiment_outputs.pkl"
_PHOTO_PATHS = tuple(
    _REPOSITORY_ROOT / ".data" / f"image{index}.fit"
    for index in (1, 2, 3)
)
_RANDOM_SEED = 20260805
_REPETITIONS = 1

if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))


def _run_experiments() -> dict[str, object]:
    from common import Config, Sky
    from experiments import (
        accelerometer_calibration,
        magnetometer_calibration,
        renderer_tuning,
        static_position,
    )
    from experiments.common import Location, assembly_rotation
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
    sky = Sky(magnitude_limit=7)

    LOGGER.info("Starting renderer tuning")
    renderer = renderer_tuning.run(renderer_tuning.Input(
        photo_paths=_PHOTO_PATHS,
        sky=sky,
    ))

    LOGGER.info("Starting accelerometer calibration")
    accelerometer = accelerometer_calibration.run(
        accelerometer_calibration.Input(
            camera=renderer.camera,
            sky=sky,
            camera_to_body=Config.FRD_CAMERA_TO_BODY,
            true_parameters=accelerometer_parameters,
            locations=locations,
            repetitions=_REPETITIONS,
            random_seed=_RANDOM_SEED,
        )
    )

    LOGGER.info("Starting magnetometer calibration")
    magnetometer = magnetometer_calibration.run(
        magnetometer_calibration.Input(
            camera=renderer.camera,
            sky=sky,
            camera_to_body=Config.FRD_CAMERA_TO_BODY,
            true_parameters=magnetometer_parameters,
            locations=locations,
            repetitions=_REPETITIONS,
            random_seed=_RANDOM_SEED,
        )
    )

    LOGGER.info("Starting static-position estimation")
    position = static_position.run(static_position.Input(
        camera=renderer.camera,
        sky=sky,
        camera_to_body=Config.FRD_CAMERA_TO_BODY,
        true_accelerometer_parameters=accelerometer_parameters,
        accelerometer_models=accelerometer.models,
        locations=locations,
        repetitions=_REPETITIONS,
        random_seed=_RANDOM_SEED,
    ))

    return {
        "locations": locations,
        "renderer": renderer,
        "accelerometer": accelerometer,
        "magnetometer": magnetometer,
        "position": position,
    }


def _write_reports(outputs: dict[str, object]) -> tuple[Path, ...]:
    from experiments.reporting import (
        report_accelerometer_calibration,
        report_magnetometer_calibration,
        report_renderer_tuning,
        report_static_position,
    )

    return (
        *report_renderer_tuning(
            outputs["renderer"],
            _RESULTS_DIRECTORY / "renderer_tuning",
        ),
        *report_accelerometer_calibration(
            outputs["accelerometer"],
            _RESULTS_DIRECTORY / "accelerometer_calibration",
            outputs["locations"],
        ),
        *report_magnetometer_calibration(
            outputs["magnetometer"],
            _RESULTS_DIRECTORY / "magnetometer_calibration",
        ),
        *report_static_position(
            outputs["position"],
            _RESULTS_DIRECTORY / "static_position",
        ),
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--from-cache",
        action="store_true",
        help="regenerate reports from results/experiment_outputs.pkl",
    )
    arguments = parser.parse_args(argv)

    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    os.environ.setdefault(
        "MPLCONFIGDIR",
        str(_RESULTS_DIRECTORY / ".matplotlib"),
    )

    from experiments.serialization import (
        load_experiment_outputs,
        save_experiment_outputs,
    )

    if arguments.from_cache:
        LOGGER.info("Loading experiment outputs from %s", _OUTPUT_CACHE)
        outputs = load_experiment_outputs(_OUTPUT_CACHE)
    else:
        outputs = _run_experiments()
        path = save_experiment_outputs(_OUTPUT_CACHE, outputs)
        LOGGER.info("Saved experiment outputs to %s", path)

    for path in _write_reports(outputs):
        LOGGER.info("Wrote %s", path)


if __name__ == "__main__":
    main()
