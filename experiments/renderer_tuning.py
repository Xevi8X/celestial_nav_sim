"""Independent three-photo tuning with average-model comparisons."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from common import ImageData, Io, Observer, Sky
from sky_render import (
    Camera,
    CameraGeometry,
    CameraImageModel,
    Renderer,
)
from tools import CalibrationConfig, CalibrationError, Calibrator, StepReport

from .simulation import ATTITUDE_NOISE_STD_DEG, REPOSITORY_ROOT


PHOTO_PATHS = tuple(
    REPOSITORY_ROOT / ".data" / f"image{index}.fit"
    for index in (1, 2, 3)
)
OUTPUT_DIRECTORY = REPOSITORY_ROOT / "results" / "renderer_tuning"

from .reporting import (
    ARTIFACT_FILENAMES,
    plot_renderer_histograms,
    write_artifact_csv,
    write_records_csv,
    write_renderer_montage,
)


@dataclass(frozen=True)
class ResolvedPhoto:
    label: str
    path: Path
    image: Image.Image
    observation_time: dt.datetime
    exposure_s: float
    latitude_deg: float
    longitude_deg: float
    elevation_m: float


@dataclass(frozen=True)
class RendererFit:
    photo: ResolvedPhoto
    observer: Observer
    camera: Camera
    rendered_image: Image.Image
    reports: Tuple[StepReport, ...]

    @property
    def parameters(self) -> Mapping[str, float]:
        return {
            "field_of_view_deg": float(self.camera.geometry.fov),
            "stellar_flux_counts_per_s": float(self.camera.image_model.flux),
            "point_spread_fwhm_px": float(self.camera.image_model.fwhm),
            "background_counts_per_s": float(
                self.camera.image_model.sky_background
            ),
            "read_noise_counts": float(self.camera.image_model.read_noise),
        }


@dataclass(frozen=True)
class RendererTuningResult:
    fits: Tuple[RendererFit, ...]
    average_camera: Camera

    @property
    def attitude_noise_std_deg(self) -> float:
        residuals = [
            float(report.values["rmse_arcsec"]) / 3600.0
            for fit in self.fits
            for report in fit.reports
            if report.name == "refined_fov"
        ]
        if not residuals:
            return ATTITUDE_NOISE_STD_DEG
        return float(np.mean(residuals))

    def parameter_rows(self) -> Tuple[Mapping[str, object], ...]:
        rows = []
        for fit in self.fits:
            rows.append({"image": fit.photo.label, **fit.parameters})
        rows.append({
            "image": "Average",
            "field_of_view_deg": self.average_camera.geometry.fov,
            "stellar_flux_counts_per_s": self.average_camera.image_model.flux,
            "point_spread_fwhm_px": self.average_camera.image_model.fwhm,
            "background_counts_per_s": (
                self.average_camera.image_model.sky_background
            ),
            "read_noise_counts": self.average_camera.image_model.read_noise,
        })
        return tuple(rows)


def resolve_photo(path: Path, label: str) -> ResolvedPhoto:
    if not path.is_file():
        raise FileNotFoundError(f"renderer input does not exist: {path}")
    record: ImageData = Io.load_fits(path)
    if record.latitude_deg is None or record.longitude_deg is None:
        raise ValueError(f"{label} FITS metadata needs latitude and longitude")
    return ResolvedPhoto(
        label=label,
        path=path,
        image=record.image,
        observation_time=record.observation_time,
        exposure_s=record.exposure_s,
        latitude_deg=float(record.latitude_deg),
        longitude_deg=float(record.longitude_deg),
        elevation_m=float(record.elevation_m),
    )


def average_camera_models(
    cameras: Sequence[Camera],
    *,
    exposure_s: Optional[float] = None,
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
        (camera.geometry.width, camera.geometry.height, camera.geometry.image_format)
        != geometry_key
        for camera in cameras[1:]
    ):
        raise ValueError("renderer photos must have equal dimensions and format")
    values = np.array([
        [
            camera.geometry.fov,
            camera.image_model.flux,
            camera.image_model.fwhm,
            camera.image_model.sky_background,
            camera.image_model.read_noise,
        ]
        for camera in cameras
    ], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("cannot average non-finite camera parameters")
    mean = np.mean(values, axis=0)
    return Camera(
        geometry=CameraGeometry(
            fov=float(mean[0]),
            width=first.geometry.width,
            height=first.geometry.height,
            image_format=first.geometry.image_format,
        ),
        image_model=CameraImageModel(
            exposure_time=(
                first.image_model.exposure_time
                if exposure_s is None else float(exposure_s)
            ),
            flux=float(mean[1]),
            fwhm=float(mean[2]),
            sky_background=float(mean[3]),
            read_noise=float(mean[4]),
        ),
    )


def tune_three_photos(
    paths: Sequence[Path],
    *,
    sky: Optional[Sky] = None,
) -> RendererTuningResult:
    """Fit each photo independently and compare it with the average model."""

    if len(paths) != 3:
        raise ValueError("exactly three photos are required")
    resolved = tuple(
        resolve_photo(Path(path), f"Image {index}")
        for index, path in enumerate(paths, start=1)
    )
    if sky is None:
        sky = Sky(magnitude_limit=10)
    preliminary = []
    first_fov = None
    for photo in resolved:
        calibration_options = {
            "exposure_time": photo.exposure_s,
            "time_reference": "start",
        }
        if first_fov is not None:
            # The files are explicitly required to come from one fixed camera.
            # Reusing the first image's broad FOV classification avoids a very
            # expensive all-sky database search for every subsequent exposure;
            # each image still estimates its own final FOV and photometry.
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
                photo.latitude_deg,
                photo.longitude_deg,
                photo.elevation_m,
            )
        except CalibrationError as error:
            raise CalibrationError(
                f"{photo.label}/{error.step}", error.reason
            ) from error
        if first_fov is None:
            first_fov = float(camera.geometry.fov)
        preliminary.append((
            photo,
            observer,
            camera,
            tuple(calibrator.last_reports),
        ))

    cameras = [item[2] for item in preliminary]
    average_camera = average_camera_models(cameras)
    fits = []
    for photo, observer, camera, reports in preliminary:
        # Flux and background are rates, so the shared average model must use
        # the exposure duration of the real image being compared with it.
        comparison_camera = average_camera_models(
            cameras,
            exposure_s=photo.exposure_s,
        )
        rendered_image = Renderer(sky, comparison_camera).render(
            observer,
            noise_seed=0,
        )
        fits.append(RendererFit(
            photo=photo,
            observer=observer,
            camera=camera,
            rendered_image=rendered_image,
            reports=reports,
        ))
    return RendererTuningResult(tuple(fits), average_camera)


def _parameter_table(tuning: RendererTuningResult):
    definitions = (
        ("Field of view", "field_of_view_deg", "deg"),
        (
            "Stellar flux",
            "stellar_flux_counts_per_s",
            "counts s^-1 for magnitude 0",
        ),
        ("Point-spread width", "point_spread_fwhm_px", "px FWHM"),
        ("Background", "background_counts_per_s", "counts s^-1"),
        ("Read noise", "read_noise_counts", "counts RMS"),
    )
    average = tuning.parameter_rows()[-1]
    return [
        {
            "parameter": label,
            "image1": tuning.fits[0].parameters[field],
            "image2": tuning.fits[1].parameters[field],
            "image3": tuning.fits[2].parameters[field],
            "average": average[field],
            "unit": unit,
        }
        for label, field, unit in definitions
    ]


def _histogram_table(tuning: RendererTuningResult):
    rows = []
    for fit in tuning.fits:
        real = np.asarray(fit.photo.image, dtype=float)
        rendered = np.asarray(fit.rendered_image, dtype=float)
        if real.ndim == 3:
            real = np.mean(real, axis=2)
        if rendered.ndim == 3:
            rendered = np.mean(rendered, axis=2)
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
                    "image_id": fit.photo.label,
                    "source": source,
                    "intensity": float(center),
                    "fraction": float(count / max(1, image.size)),
                }
                for center, count in zip(centers, counts)
            )
    return rows


class RendererTuningExperiment:
    """Fit the three photographs and write only the renderer results."""

    def __init__(self, output_directory: Path = OUTPUT_DIRECTORY):
        self.output_directory = Path(output_directory)

    def run(self):
        tuning = tune_three_photos(PHOTO_PATHS)
        output = self.output_directory
        output.mkdir(parents=True, exist_ok=True)

        files = [write_artifact_csv(
            output,
            "renderer_tuning_results",
            _parameter_table(tuning),
        )]

        montage_rows = (
            {
                "image_id": fit.photo.label,
                "real_image": fit.photo.image,
                "rendered_image": fit.rendered_image,
            }
            for fit in tuning.fits
        )
        png, eps = write_renderer_montage(
            montage_rows,
            output / ARTIFACT_FILENAMES["renderer_image_comparison_png"],
            eps_path=output / ARTIFACT_FILENAMES["renderer_image_comparison_eps"],
        )
        files.extend((png, eps))

        histograms = _histogram_table(tuning)
        histogram_eps = plot_renderer_histograms(
            histograms,
            output / ARTIFACT_FILENAMES["renderer_histogram_comparison"],
        )
        files.extend((histogram_eps, histogram_eps.with_suffix(".png")))
        files.append(write_records_csv(
            output / "raw" / "renderer_histograms.csv",
            histograms,
        ))
        return tuning, tuple(files)


def main():
    _, files = RendererTuningExperiment().run()
    for path in files:
        print(path)


if __name__ == "__main__":
    main()
