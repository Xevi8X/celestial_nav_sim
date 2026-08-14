import datetime
import logging
import numpy as np

from celestial_nav import Navigator
from common import ECEF, Observer, Sky
from imu import (
    Accelerometer,
    AccelerometerCalibration,
    AccelerometerParameters,
)
from sky_render import (
    Camera,
    CameraGeometry,
    CameraImageModel,
    ImageFormat,
    Renderer,
)


EXPOSURE_TIME = 0.1
PATTERN_PERIOD = 10.0

AZIMUTH_ZERO = 90.0
AZIMUTH_AMPLITUDE = 5.0
ELEVATION_ZERO = 50.0
ELEVATION_AMPLITUDE = 30.0
ROLL_AMPLITUDE = 60.0

MATRIX_STDDEV_LIMIT = 0.01
BIAS_STDDEV_LIMIT = 0.02
CONDITION_NUMBER_LIMIT = 30.0


LOGGER = logging.getLogger(__name__)


def find_star(sky, observer):
    stars, vectors_ecef = sky.get_stars_ecef(observer)
    vectors_ned = (
        vectors_ecef
        @ ECEF.ecef_to_ned(
            observer.latitude,
            observer.longitude,
        ).T
    )
    vectors_ned /= np.linalg.norm(vectors_ned, axis=1)[:, None]
    azimuth = np.degrees(
        np.arctan2(vectors_ned[:, 1], vectors_ned[:, 0])
    ) % 360.0
    elevation = np.degrees(
        np.arctan2(
            -vectors_ned[:, 2],
            np.hypot(vectors_ned[:, 0], vectors_ned[:, 1]),
        )
    )
    candidates = np.flatnonzero(
        (elevation >= 52.0) & (elevation <= 58.0)
    )
    index = min(candidates, key=lambda i: stars[i].magnitude)
    return stars[index], azimuth[index], elevation[index]


def rotate_observer(time_s):
    phase = 2.0 * np.pi * time_s / PATTERN_PERIOD
    azimuth = AZIMUTH_ZERO + AZIMUTH_AMPLITUDE * np.sin(
        phase + np.pi / 2.0
    )
    elevation = ELEVATION_ZERO + ELEVATION_AMPLITUDE * np.sin(2 * phase)
    roll = ROLL_AMPLITUDE * -np.sin(phase)
    return roll, elevation, azimuth


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    sky = Sky()
    start_time = datetime.datetime(
        2026,
        1,
        1,
        tzinfo=datetime.timezone.utc,
    )
    observer = Observer()
    observer.set_time(start_time)
    observer.set_location(52.2297, 21.0122)
    camera = Camera(
        CameraGeometry(
            fov=35.0,
            width=512,
            height=512,
            image_format=ImageFormat.MONO8,
        ),
        CameraImageModel(
            exposure_time=EXPOSURE_TIME,
            flux=8e5,
            fwhm=2.0,
            sky_background=0.0,
            read_noise=0.0,
        ),
    )
    renderer = Renderer(sky, camera)
    navigator = Navigator(
        sky,
        fov_range=(30, 40),
        star_max_magnitude=7,
    )

    assembly = np.array([
        [1.0, 0.015, -0.010],
        [-0.012, 1.0, 0.018],
        [0.008, -0.014, 1.0],
    ])
    assembly /= np.linalg.norm(assembly, axis=1)[:, None]

    actual = AccelerometerParameters(
        assembly_matrix=assembly,
        axis_scale=np.array([1.025, 0.980, 1.012]),
        bias=np.array([0.08, -0.05, 0.12]),
        noise_stddev=np.full(3, 0.02),
    )
    accelerometer = Accelerometer(actual)
    calibration = AccelerometerCalibration()

    stable_samples = 0
    solved_exposures = 0
    for sample in range(6000):
        time_s = sample * EXPOSURE_TIME
        observer.set_time(
            start_time + datetime.timedelta(seconds=time_s)
        )
        roll, pitch, azimuth = rotate_observer(time_s)
        observer.set_orientation(roll, pitch, azimuth)
        image = renderer.render(observer, noise_seed=sample)

        orientation = navigator.estimate_orientation(
            image,
            time=observer.time,
            latitude_deg=observer.latitude,
            longitude_deg=observer.longitude,
            elevation_m=observer.elevation,
        )

        if orientation.observer_matrix is None:
            LOGGER.warning("No plate solution for sample %d", sample + 1)
            stable_samples = 0
            continue

        reading = accelerometer.measure(
            observer.observer_matrix,
            np.zeros(3),
        )
        estimated = calibration.update(
            orientation.observer_matrix,
            reading,
        )
        variance = calibration.get_variance()
        condition_number = calibration.get_condition_number()
        solved_exposures += 1

        if not np.all(np.isfinite(variance)):
            continue

        matrix_stddev = np.sqrt(variance[:3]).max()
        bias_stddev = np.sqrt(variance[3]).max()

        LOGGER.info(
            "sample=%d time=%.1f roll=%.1f pitch=%.1f azimuth=%.1f "
            "matrix_stddev=%.6f bias_stddev=%.6f condition=%.3f",
            sample + 1,
            time_s,
            roll,
            pitch,
            azimuth,
            matrix_stddev,
            bias_stddev,
            condition_number,
        )

        if (
            matrix_stddev < MATRIX_STDDEV_LIMIT
            and bias_stddev < BIAS_STDDEV_LIMIT
            and condition_number < CONDITION_NUMBER_LIMIT
        ):
            stable_samples += 1
        else:
            stable_samples = 0

        if stable_samples == 3:
            break

        if (sample + 1) % 100 == 0:
            LOGGER.info(
                "%d/%d images solved",
                solved_exposures,
                sample + 1,
            )
    else:
        raise RuntimeError("calibration did not converge")

    np.set_printoptions(precision=6, suppress=True)
    LOGGER.info("Samples used: %d", solved_exposures)
    LOGGER.info("Plate solutions: %d/%d", solved_exposures, sample + 1)
    LOGGER.info("Actual scale: %s", actual.axis_scale)
    LOGGER.info("Estimated scale: %s", estimated.axis_scale)
    LOGGER.info("Actual bias: %s", actual.bias)
    LOGGER.info("Estimated bias: %s", estimated.bias)
    LOGGER.info("Actual noise: %s", actual.noise_stddev)
    LOGGER.info("Estimated noise: %s", estimated.noise_stddev)


if __name__ == "__main__":
    main()
