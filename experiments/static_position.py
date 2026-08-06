"""Static-position experiment from four camera azimuths."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from common import ECEF
from imu import Accelerometer, AccelerometerParameters

from .reporting import (
    plot_position_results,
    write_artifact_csv,
    write_records_csv,
)
from .simulation import (
    ATTITUDE_NOISE_STD_DEG,
    LOCATIONS,
    REPETITIONS,
    REPOSITORY_ROOT,
    assembly_rotation,
    body_from_ned,
    noisy_attitude,
    seed,
)


OUTPUT_DIRECTORY = REPOSITORY_ROOT / "results" / "static_position"

AZIMUTHS_DEG = (0.0, 90.0, 180.0, 270.0)
SAMPLES_PER_POSE = 15
ROLL_DEG = 0.0
PITCH_DEG = 60.0
OUTLIER_THRESHOLD_DEG = 1.0
SINGULAR_HORIZONTAL_NORM = 1e-8

# Standalone runs use the imposed calibration below.  The combined entry point
# replaces it with the estimate produced by the accelerometer experiment.
ACCELEROMETER_SCALE = (1.025, 0.980, 1.012)
ACCELEROMETER_BIAS = (0.08, -0.05, 0.12)
ACCELEROMETER_NOISE = (0.02, 0.02, 0.02)
ACCELEROMETER_MOUNTING_ROTATION_DEG = (0.8, -0.6, 1.0)


class StaticPositionExperiment:
    def __init__(self, output_directory: Path = OUTPUT_DIRECTORY):
        self.output_directory = Path(output_directory)

    @staticmethod
    def accelerometer_parameters() -> AccelerometerParameters:
        return AccelerometerParameters(
            assembly_matrix=assembly_rotation(
                ACCELEROMETER_MOUNTING_ROTATION_DEG
            ),
            axis_scale=ACCELEROMETER_SCALE,
            bias=ACCELEROMETER_BIAS,
            noise_stddev=ACCELEROMETER_NOISE,
        )

    @staticmethod
    def zenith_location(zenith_ecef: np.ndarray):
        vector = np.asarray(zenith_ecef, dtype=float)
        if vector.shape != (3,) or not np.isfinite(vector).all():
            return None

        norm = np.linalg.norm(vector)
        if norm <= np.finfo(float).eps:
            return None
        vector = vector / norm

        horizontal = float(np.hypot(vector[0], vector[1]))
        if horizontal <= SINGULAR_HORIZONTAL_NORM:
            return None

        latitude = np.degrees(np.arctan2(vector[2], horizontal))
        longitude = np.degrees(np.arctan2(vector[1], vector[0]))
        return float(latitude), float(longitude)

    @staticmethod
    def position_error_m(
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

    @classmethod
    def mean_position_error_m(
        cls,
        reference_latitude_deg: float,
        reference_longitude_deg: float,
        positions: Sequence[Sequence[float]],
    ) -> float:
        if not positions:
            raise ValueError("at least one individual position is required")
        errors = [
            cls.position_error_m(
                reference_latitude_deg,
                reference_longitude_deg,
                position,
            )
            for position in positions
        ]
        return float(np.mean(errors))

    def simulate(
        self,
        calibrations=None,
        attitude_noise_std_deg=ATTITUDE_NOISE_STD_DEG,
    ):
        actual_parameters = self.accelerometer_parameters()
        results = []
        observations = []

        for repetition in range(REPETITIONS):
            for location in LOCATIONS:
                correction = actual_parameters
                if calibrations is not None:
                    correction = calibrations[(location.name, repetition)]
                sensor = Accelerometer(
                    actual_parameters,
                    seed=seed("position", repetition, location.name),
                )
                ecef_to_ned = ECEF.ecef_to_ned(
                    location.latitude_deg,
                    location.longitude_deg,
                )
                zeniths = []
                positions = []
                rejected = 0

                for sample, azimuth in enumerate(
                    azimuth
                    for azimuth in AZIMUTHS_DEG
                    for _ in range(SAMPLES_PER_POSE)
                ):
                    attitude = body_from_ned(ROLL_DEG, PITCH_DEG, azimuth)
                    reading = sensor.measure(attitude)
                    try:
                        zenith_body = correction.correct(reading)
                        zenith_body /= np.linalg.norm(zenith_body)
                    except (ValueError, np.linalg.LinAlgError):
                        rejected += 1
                        continue

                    estimated_attitude = noisy_attitude(
                        attitude,
                        seed("position", repetition, location.name, sample),
                        standard_deviation_deg=attitude_noise_std_deg,
                    )
                    zenith_ecef = (
                        ecef_to_ned.T
                        @ estimated_attitude.T
                        @ zenith_body
                    )
                    zenith_ecef /= np.linalg.norm(zenith_ecef)
                    position = self.zenith_location(zenith_ecef)
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
                        "position_error_m": self.position_error_m(
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

                zeniths, positions, outliers = self.reject_outliers(
                    zeniths,
                    positions,
                )
                rejected += outliers
                combined_position = self.zenith_location(np.sum(zeniths, axis=0))

                if combined_position is None:
                    combined_error = None
                    status = "rejected_invalid_combined_position"
                else:
                    combined_error = self.position_error_m(
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
                    "individual_image_mean_error_m": self.mean_position_error_m(
                        location.latitude_deg,
                        location.longitude_deg,
                        positions,
                    ),
                    "four_pose_error_m": combined_error,
                    "accepted_images": len(zeniths),
                    "rejected_images": rejected,
                    "status": status,
                })

        return results, observations

    @staticmethod
    def reject_outliers(zeniths, positions):
        vectors = np.asarray(zeniths)
        center = np.sum(vectors, axis=0)
        center /= np.linalg.norm(center)
        angles = np.degrees(np.arccos(np.clip(vectors @ center, -1.0, 1.0)))
        inliers = angles <= OUTLIER_THRESHOLD_DEG
        if np.count_nonzero(inliers) < 2:
            return vectors, positions, 0
        return (
            vectors[inliers],
            [position for position, keep in zip(positions, inliers) if keep],
            int(np.count_nonzero(~inliers)),
        )

    def run(
        self,
        calibrations=None,
        attitude_noise_std_deg=ATTITUDE_NOISE_STD_DEG,
    ):
        results, observations = self.simulate(
            calibrations,
            attitude_noise_std_deg,
        )
        self.output_directory.mkdir(parents=True, exist_ok=True)

        table = write_artifact_csv(
            self.output_directory,
            "position_results",
            results,
        )
        figure = plot_position_results(
            results,
            self.output_directory / "position_results.eps",
        )
        raw = write_records_csv(
            self.output_directory / "raw" / "position_observations.csv",
            observations,
        )
        return table, figure, figure.with_suffix(".png"), raw


def main():
    for path in StaticPositionExperiment().run():
        print(path)


if __name__ == "__main__":
    main()
