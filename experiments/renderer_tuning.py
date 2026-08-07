"""Tune a renderer from photographs and render average-model comparisons."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from common import ImageData, Io, Observer, Sky
from sky_render import Camera, CameraGeometry, CameraImageModel, Renderer
from tools import CalibrationConfig, CalibrationError, Calibrator, StepReport


__all__ = ["Input", "Output", "run"]


@dataclass(frozen=True)
class Input:
    """Photographs and sky model used for renderer tuning."""

    photo_paths: tuple[Path, ...]
    sky: Sky


@dataclass(frozen=True)
class Output:
    """Calibrated models and real-versus-rendered comparison data."""

    camera: Camera
    fitted_cameras: tuple[Camera, ...]
    observers: tuple[Observer, ...]
    photos: tuple[Image.Image, ...]
    rendered_photos: tuple[Image.Image, ...]
    histograms: tuple[Mapping[str, object], ...]
    attitude_noise_std_deg: float


def _validate_input(value: Input) -> None:
    if not isinstance(value, Input):
        raise TypeError("input must be a renderer-tuning Input")
    if not isinstance(value.photo_paths, tuple) or not value.photo_paths:
        raise ValueError("photo_paths must be a non-empty tuple")
    if not all(isinstance(path, Path) for path in value.photo_paths):
        raise TypeError("photo_paths must contain Path values")
    if not isinstance(value.sky, Sky):
        raise TypeError("sky must be a Sky")


def _load_photo(path: Path, label: str) -> ImageData:
    if not path.is_file():
        raise FileNotFoundError(f"renderer input does not exist: {path}")

    photo = Io.load_fits(path)
    if photo.latitude_deg is None or photo.longitude_deg is None:
        raise ValueError(f"{label} FITS metadata needs latitude and longitude")
    return photo


def _average_cameras(
    cameras: Sequence[Camera],
    *,
    exposure_s: float | None = None,
) -> Camera:
    if not cameras:
        raise ValueError("at least one camera is required")

    first = cameras[0]
    geometry_key = (
        first.geometry.width,
        first.geometry.height,
        first.geometry.image_format,
    )
    if any(
        (
            camera.geometry.width,
            camera.geometry.height,
            camera.geometry.image_format,
        )
        != geometry_key
        for camera in cameras[1:]
    ):
        raise ValueError("renderer photos must have equal dimensions and format")

    values = np.asarray(
        [
            (
                camera.geometry.fov,
                camera.image_model.flux,
                camera.image_model.fwhm,
                camera.image_model.sky_background,
                camera.image_model.read_noise,
            )
            for camera in cameras
        ],
        dtype=float,
    )
    if not np.isfinite(values).all():
        raise ValueError("cannot average non-finite camera parameters")

    fov, flux, fwhm, sky_background, read_noise = np.mean(values, axis=0)
    return Camera(
        geometry=CameraGeometry(
            fov=float(fov),
            width=first.geometry.width,
            height=first.geometry.height,
            image_format=first.geometry.image_format,
        ),
        image_model=CameraImageModel(
            exposure_time=(
                first.image_model.exposure_time
                if exposure_s is None
                else float(exposure_s)
            ),
            flux=float(flux),
            fwhm=float(fwhm),
            sky_background=float(sky_background),
            read_noise=float(read_noise),
        ),
    )


def _fit_photo(
    photo: ImageData,
    label: str,
    sky: Sky,
    first_fov: float | None,
) -> tuple[Observer, Camera, tuple[StepReport, ...]]:
    calibration_options: dict[str, object] = {
        "exposure_time": photo.exposure_s,
        "time_reference": "start",
    }
    if first_fov is not None:
        # The photographs come from one fixed camera. Restricting only the
        # expensive rough search leaves every final fit independent.
        tolerance = 2.0
        center = round(first_fov / tolerance) * tolerance
        calibration_options.update(
            rough_min_fov=max(0.1, center - tolerance),
            rough_max_fov=min(179.9, center + tolerance),
            rough_star_max_magnitude=10.0,
        )

    calibrator = Calibrator(
        sky,
        config=CalibrationConfig(**calibration_options),
    )
    try:
        observer, camera = calibrator.estimate_camera(
            photo.image,
            photo.observation_time,
            float(photo.latitude_deg),
            float(photo.longitude_deg),
            photo.elevation_m,
        )
    except CalibrationError as error:
        raise CalibrationError(f"{label}/{error.step}", error.reason) from error

    return observer, camera, tuple(calibrator.last_reports)


def _luminance(image: Image.Image) -> np.ndarray:
    values = np.asarray(image, dtype=float)
    if values.ndim == 3:
        values = np.mean(values, axis=2)
    return values


def _histograms(
    photos: Sequence[Image.Image],
    rendered_photos: Sequence[Image.Image],
) -> tuple[Mapping[str, object], ...]:
    if len(photos) != len(rendered_photos):
        raise ValueError("real and rendered photo counts must match")
    rows: list[Mapping[str, object]] = []
    for index, (photo, rendered_photo) in enumerate(
        zip(photos, rendered_photos),
        start=1,
    ):
        real = _luminance(photo)
        rendered = _luminance(rendered_photo)
        high = max(
            float(np.percentile(real, 99.95)),
            float(np.percentile(rendered, 99.95)),
            1.0,
        )
        edges = np.linspace(0.0, high, 65)
        centers = (edges[:-1] + edges[1:]) / 2.0

        for source, image in (("Real", real), ("Rendered", rendered)):
            counts, _ = np.histogram(image, bins=edges)
            rows.extend(
                {
                    "image_id": f"Image {index}",
                    "source": source,
                    "intensity": float(center),
                    "fraction": float(count / max(1, image.size)),
                }
                for center, count in zip(centers, counts)
            )
    return tuple(rows)


def _attitude_noise(reports: Sequence[Sequence[StepReport]]) -> float:
    residuals = [
        float(report.values["rmse_arcsec"]) / 3600.0
        for photo_reports in reports
        for report in photo_reports
        if report.name == "refined_fov"
    ]
    if not residuals:
        raise ValueError("renderer calibration did not report refined-FOV RMSE")
    return float(np.mean(residuals))


def run(input: Input) -> Output:
    """Fit every photograph and compare it with the average camera model."""

    _validate_input(input)

    records = tuple(
        _load_photo(Path(path), f"Image {index}")
        for index, path in enumerate(input.photo_paths, start=1)
    )

    observers: list[Observer] = []
    fitted_cameras: list[Camera] = []
    reports: list[tuple[StepReport, ...]] = []
    first_fov: float | None = None
    for index, photo in enumerate(records, start=1):
        observer, camera, photo_reports = _fit_photo(
            photo,
            f"Image {index}",
            input.sky,
            first_fov,
        )
        observers.append(observer)
        fitted_cameras.append(camera)
        reports.append(photo_reports)
        if first_fov is None:
            first_fov = float(camera.geometry.fov)

    fitted_camera_tuple = tuple(fitted_cameras)
    average_camera = _average_cameras(fitted_camera_tuple)
    rendered_photos = tuple(
        Renderer(
            input.sky,
            _average_cameras(
                fitted_camera_tuple,
                exposure_s=photo.exposure_s,
            ),
        ).render(observer, noise_seed=0)
        for photo, observer in zip(records, observers)
    )
    photos = tuple(photo.image for photo in records)

    return Output(
        camera=average_camera,
        fitted_cameras=fitted_camera_tuple,
        observers=tuple(observers),
        photos=photos,
        rendered_photos=rendered_photos,
        histograms=_histograms(photos, rendered_photos),
        attitude_noise_std_deg=_attitude_noise(reports),
    )
