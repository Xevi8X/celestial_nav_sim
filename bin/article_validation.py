"""Run the four article experiments sequentially."""

import os
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIRECTORY = REPOSITORY_ROOT / "results"

if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


def main():
    os.environ.setdefault(
        "MPLCONFIGDIR",
        str(RESULTS_DIRECTORY / ".matplotlib"),
    )

    from experiments.accelerometer_calibration import (
        AccelerometerCalibrationExperiment,
    )
    from experiments.magnetometer_calibration import (
        MagnetometerCalibrationExperiment,
    )
    from experiments.renderer_tuning import RendererTuningExperiment
    from experiments.static_position import StaticPositionExperiment

    renderer_result, renderer_files = RendererTuningExperiment().run()
    attitude_noise = renderer_result.attitude_noise_std_deg

    calibrations, accelerometer_files = (
        AccelerometerCalibrationExperiment().run(attitude_noise)
    )
    magnetometer_files = MagnetometerCalibrationExperiment().run(
        attitude_noise
    )
    position_files = StaticPositionExperiment().run(
        calibrations=calibrations,
        attitude_noise_std_deg=attitude_noise,
    )

    for path in (
        *renderer_files,
        *accelerometer_files,
        *magnetometer_files,
        *position_files,
    ):
        print(path)


if __name__ == "__main__":
    main()
