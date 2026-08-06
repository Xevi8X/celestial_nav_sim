"""Magnetometer calibration experiment from the article."""

from pathlib import Path

import numpy as np

from imu import (
    Magnetometer,
    MagnetometerCalibration,
    MagnetometerParameters,
    WMM2025Provider,
    magnetic_heading_deg,
)

from .reporting import (
    plot_magnetometer_results,
    write_artifact_csv,
    write_records_csv,
)
from .simulation import (
    ATTITUDE_NOISE_STD_DEG,
    LOCATIONS,
    REPETITIONS,
    REPOSITORY_ROOT,
    angle_difference_deg,
    body_from_ned,
    noisy_attitude,
    seed,
)


OUTPUT_DIRECTORY = REPOSITORY_ROOT / "results" / "magnetometer_calibration"

TRANSFORMATION_MATRIX = (
    (1.04, 0.03, -0.02),
    (-0.01, 0.96, 0.025),
    (0.015, -0.02, 1.02),
)
OFFSET_UT = (6.0, -4.0, 3.0)
NOISE_UT = (0.15, 0.15, 0.15)

HEADING_STEP_DEG = 15.0
CONSTRAINED_ROLL_DEG = 2.0
CONSTRAINED_PITCH_DEG = 3.0
FULL_SWEEPS_DEG = (
    (30.0, 30.0, 1.0),
    (-30.0, 45.0, -1.0),
    (30.0, 60.0, 1.0),
    (-30.0, 75.0, -1.0),
)

HELD_OUT_ORIENTATIONS = 96
INDICATOR_LIMIT = 0.03
OFFSET_INDICATOR_LIMIT = 0.3
MAXIMUM_CONDITION_NUMBER = 100.0
CONSECUTIVE_UPDATES = 3
REFERENCE_SCALE_UT = 50.0


class MagnetometerCalibrationExperiment:
    """Compare complete and constrained magnetometer calibration maneuvers."""

    def __init__(self, output_directory=OUTPUT_DIRECTORY):
        self.output_directory = Path(output_directory)

    @staticmethod
    def parameters():
        return MagnetometerParameters(
            transformation_matrix=np.asarray(TRANSFORMATION_MATRIX),
            offset=np.asarray(OFFSET_UT),
            noise_stddev=np.asarray(NOISE_UT),
        )

    @staticmethod
    def orientations(constrained):
        headings = np.arange(0.0, 360.0, HEADING_STEP_DEG)
        if constrained:
            for yaw in headings:
                yield (
                    CONSTRAINED_ROLL_DEG,
                    CONSTRAINED_PITCH_DEG,
                    float(yaw),
                )
            return

        for roll, pitch, direction in FULL_SWEEPS_DEG:
            sweep = headings if direction > 0 else headings[::-1]
            for yaw in sweep:
                yield roll, pitch, float(yaw)

    def simulate(self, attitude_noise_std_deg=ATTITUDE_NOISE_STD_DEG):
        wmm = WMM2025Provider()
        actual = self.parameters()
        results = []
        updates = []

        for repetition in range(REPETITIONS):
            for location in LOCATIONS:
                field = wmm.field(
                    location.latitude_deg,
                    location.longitude_deg,
                    location.elevation_m,
                    location.observation_time,
                )
                field_ned = field.ned_uT
                fitted = {}

                for procedure, constrained in (
                    ("Complete pattern", False),
                    ("One constrained turn", True),
                ):
                    run_seed = seed(
                        "magnetometer",
                        repetition,
                        location.name,
                        procedure,
                    )
                    sensor = Magnetometer(actual, seed=run_seed)
                    calibration = MagnetometerCalibration(
                        constrained=constrained,
                        indicator_limit=INDICATOR_LIMIT,
                        offset_indicator_limit=OFFSET_INDICATOR_LIMIT,
                        max_condition_number=MAXIMUM_CONDITION_NUMBER,
                        consecutive_updates=CONSECUTIVE_UPDATES,
                        reference_scale_ut=REFERENCE_SCALE_UT,
                    )

                    accepted = 0
                    for index, (roll, pitch, yaw) in enumerate(
                        self.orientations(constrained)
                    ):
                        actual_attitude = body_from_ned(roll, pitch, yaw)
                        reading = sensor.measure(actual_attitude @ field_ned)
                        estimated_attitude = noisy_attitude(
                            actual_attitude,
                            seed(run_seed, index, "attitude"),
                            attitude_noise_std_deg,
                        )
                        calibration.update(
                            estimated_attitude @ field_ned,
                            reading,
                        )
                        accepted += 1

                        condition_number = calibration.condition_number
                        updates.append({
                            "location": location.name,
                            "repetition": repetition,
                            "procedure": procedure,
                            "observation": index,
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

                    fitted[procedure] = (calibration, accepted)

                evaluations = {
                    procedure: ([], []) for procedure in fitted
                }
                test_rng = np.random.default_rng(seed(
                    "magnetometer-tests",
                    repetition,
                    location.name,
                ))
                test_sensor = Magnetometer(
                    actual,
                    seed=seed(
                        "magnetometer-test-noise",
                        repetition,
                        location.name,
                    ),
                )

                for _ in range(HELD_OUT_ORIENTATIONS):
                    attitude = body_from_ned(
                        test_rng.uniform(-60.0, 60.0),
                        test_rng.uniform(-70.0, 70.0),
                        test_rng.uniform(0.0, 360.0),
                    )
                    reference_body = attitude @ field_ned
                    reading = test_sensor.measure(reference_body)

                    for procedure, (calibration, _accepted) in fitted.items():
                        corrected_body = calibration.correct(reading)
                        vector_error = float(np.linalg.norm(
                            corrected_body - reference_body
                        ))
                        evaluations[procedure][0].append(vector_error)

                        if field.in_blackout_zone:
                            continue
                        corrected_ned = attitude.T @ corrected_body
                        try:
                            heading_error = angle_difference_deg(
                                magnetic_heading_deg(corrected_ned),
                                magnetic_heading_deg(field_ned),
                            )
                        except ValueError:
                            continue
                        evaluations[procedure][1].append(heading_error)

                for procedure, (calibration, accepted) in fitted.items():
                    vector_errors, heading_errors = evaluations[procedure]
                    results.append({
                        "location": location.name,
                        "repetition": repetition,
                        "procedure": procedure,
                        "vector_error_ut": float(np.sqrt(np.mean(
                            np.square(vector_errors)
                        ))),
                        "heading_error_deg": (
                            None if not heading_errors
                            else float(np.sqrt(np.mean(
                                np.square(heading_errors)
                            )))
                        ),
                        "test_orientation_count": len(vector_errors),
                        "accepted_observations": accepted,
                        "rejected_observations": 0,
                        "wmm_blackout_zone": field.in_blackout_zone,
                        "status": (
                            "heading_rejected_wmm_blackout"
                            if field.in_blackout_zone
                            else "converged"
                            if calibration.converged
                            else "maneuver_complete"
                        ),
                    })

        return results, updates

    def write_results(self, results, updates):
        result_csv = write_artifact_csv(
            self.output_directory,
            "magnetometer_results",
            results,
        )
        figure_eps = plot_magnetometer_results(
            results,
            self.output_directory / "magnetometer_results.eps",
        )
        update_csv = write_records_csv(
            self.output_directory / "raw" / "magnetometer_updates.csv",
            updates,
        )
        return (
            result_csv,
            figure_eps,
            figure_eps.with_suffix(".png"),
            update_csv,
        )

    def run(self, attitude_noise_std_deg=ATTITUDE_NOISE_STD_DEG):
        results, updates = self.simulate(attitude_noise_std_deg)
        return self.write_results(results, updates)


def main():
    for path in MagnetometerCalibrationExperiment().run():
        print(path)


if __name__ == "__main__":
    main()
