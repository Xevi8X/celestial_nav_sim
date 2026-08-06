"""Accelerometer calibration experiment from the article."""

from pathlib import Path

import numpy as np

from imu import Accelerometer, AccelerometerCalibration, AccelerometerParameters

from .reporting import plot_accelerometer_results, write_artifact_csv, write_records_csv
from .simulation import (
    ATTITUDE_NOISE_STD_DEG,
    LOCATIONS,
    REPETITIONS,
    REPOSITORY_ROOT,
    assembly_rotation,
    body_from_ned,
    noisy_attitude,
    rotation_error_deg,
    seed,
)


OUTPUT_DIRECTORY = REPOSITORY_ROOT / "results" / "accelerometer_calibration"

SCALE = (1.025, 0.980, 1.012)
BIAS = (0.08, -0.05, 0.12)
NOISE = (0.02, 0.02, 0.02)
MOUNTING_ROTATION_DEG = (0.8, -0.6, 1.0)
INITIAL_AZIMUTHS_DEG = (0.0, 90.0, 180.0, 270.0)
# Use one explicitly chosen run as the calibration passed to later experiments.
HANDOFF_AZIMUTH_DEG = 0.0
EXPOSURE_INTERVAL_S = 1.0
TRANSITION_TIME_S = 2.0
MATRIX_INDICATOR_LIMIT = 0.04
BIAS_INDICATOR_LIMIT = 0.30
MAXIMUM_CONDITION_NUMBER = 30.0
CONSECUTIVE_UPDATES = 3
MAXIMUM_OBSERVATIONS = 120
YAW_STEP_DEG = 15.0

POSES_DEG = (
    (0.0, 20.0),
    (90.0, 20.0),
    (180.0, 20.0),
    (270.0, 20.0),
    (0.0, 50.0),
    (90.0, 50.0),
    (180.0, 50.0),
    (270.0, 50.0),
    (0.0, 80.0),
    (90.0, 80.0),
    (180.0, 80.0),
    (270.0, 80.0),
)


class AccelerometerCalibrationExperiment:
    """Run the accelerometer maneuver at every location and azimuth."""

    def __init__(self, output_directory=OUTPUT_DIRECTORY):
        self.output_directory = Path(output_directory)

    @staticmethod
    def parameters():
        return AccelerometerParameters(
            assembly_matrix=assembly_rotation(MOUNTING_ROTATION_DEG),
            axis_scale=SCALE,
            bias=BIAS,
            noise_stddev=NOISE,
        )

    def simulate(self, attitude_noise_std_deg=ATTITUDE_NOISE_STD_DEG):
        actual = self.parameters()
        results = []
        updates = []
        calibrations = {}

        for repetition in range(REPETITIONS):
            for location in LOCATIONS:
                for initial_azimuth in INITIAL_AZIMUTHS_DEG:
                    run_seed = seed(
                        "accelerometer",
                        repetition,
                        location.name,
                        initial_azimuth,
                    )
                    sensor = Accelerometer(actual, seed=run_seed)
                    calibration = AccelerometerCalibration(
                        matrix_indicator_limit=MATRIX_INDICATOR_LIMIT,
                        bias_indicator_limit=BIAS_INDICATOR_LIMIT,
                        max_condition_number=MAXIMUM_CONDITION_NUMBER,
                        consecutive_updates=CONSECUTIVE_UPDATES,
                    )
                    elapsed = 0.0
                    estimate = None

                    for observation in range(MAXIMUM_OBSERVATIONS):
                        roll, pitch = POSES_DEG[observation % len(POSES_DEG)]
                        cycle = observation // len(POSES_DEG)
                        yaw = (
                            initial_azimuth + YAW_STEP_DEG * cycle
                        ) % 360.0
                        actual_attitude = body_from_ned(roll, pitch, yaw)
                        reading = sensor.measure(actual_attitude)
                        reference_attitude = noisy_attitude(
                            actual_attitude,
                            seed(run_seed, observation, "attitude"),
                            attitude_noise_std_deg,
                        )
                        estimate = calibration.update(reference_attitude, reading)
                        elapsed += EXPOSURE_INTERVAL_S + TRANSITION_TIME_S

                        condition_number = calibration.condition_number
                        updates.append({
                            "location": location.name,
                            "repetition": repetition,
                            "azimuth_deg": initial_azimuth,
                            "observation": observation,
                            "elapsed_s": elapsed,
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
                        initial_azimuth == HANDOFF_AZIMUTH_DEG
                        and initialized
                        and calibration.converged
                    ):
                        calibrations[(location.name, repetition)] = estimate

        return results, updates, calibrations

    def run(self, attitude_noise_std_deg=ATTITUDE_NOISE_STD_DEG):
        results, updates, calibrations = self.simulate(
            attitude_noise_std_deg
        )
        output = self.output_directory

        locations_path = write_artifact_csv(
            output,
            "simulation_locations",
            ({
                "location": location.name,
                "latitude_deg": location.latitude_deg,
                "longitude_deg": location.longitude_deg,
            } for location in LOCATIONS),
        )
        results_path = write_artifact_csv(
            output,
            "accelerometer_results",
            results,
        )
        figure_path = plot_accelerometer_results(
            results,
            output / "accelerometer_results.eps",
        )
        updates_path = write_records_csv(
            output / "raw" / "accelerometer_updates.csv",
            updates,
        )
        output_files = (
            locations_path,
            results_path,
            figure_path,
            figure_path.with_suffix(".png"),
            updates_path,
        )
        return calibrations, output_files


def main():
    _calibrations, output_files = AccelerometerCalibrationExperiment().run()
    for path in output_files:
        print(path)


if __name__ == "__main__":
    main()
