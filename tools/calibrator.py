import datetime
import math
import warnings
from dataclasses import dataclass, field
from numbers import Integral, Real
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, label
from scipy.optimize import least_squares, minimize_scalar

from celestial_nav import LostInSpace
from common import ECEF, Observer, Sky
from sky_render import Camera, ImageFormat, Renderer
from sky_render.psf import point_spread_kernel, source_mask_radius

from .calibration_visualization import CalibrationVisualizer


CalibrationPair = Tuple[Observer, Camera]
BoundCalibrationStep = Callable[[Observer, Camera], CalibrationPair]
CatalogId = Any


class CalibrationError(RuntimeError):
    """A calibration failure annotated with the step that failed."""

    def __init__(self, step: str, message: str):
        self.step = step
        self.reason = message
        super().__init__(f"{step}: {message}")


@dataclass(frozen=True)
class CalibrationConfig:
    """Numerical policy for the default calibration pipeline.

    ``time_reference`` defines whether ``time`` passed to
    :meth:`Calibrator.estimate_camera` is the exposure start, midpoint, or end.
    The default is ``"start"``, matching FITS ``DATE-OBS``.  The returned
    observer is timestamped at exposure end, as expected by
    :class:`sky_render.Renderer`.

    With the default one-second exposure, fitted flux and background are
    effective per-image values.  Supply the known exposure to ``Calibrator``
    when rates are needed.
    """

    exposure_time: float = 1.0
    time_reference: str = "start"

    rough_min_fov: float = 5.0
    rough_max_fov: float = 120.0
    rough_star_max_magnitude: float = 7.0
    rough_distortion_bounds: Tuple[float, float] = (-0.5, 0.5)

    refined_fov_tolerance: float = 2.0
    refined_star_max_magnitude: float = 10.0
    refined_distortion_bounds: Tuple[float, float] = (-0.2, 0.2)
    minimum_matches: int = 6
    maximum_false_positive_probability: float = 1e-3
    maximum_astrometric_rmse_px: float = 1.5
    maximum_distortion_displacement_px: float = 1.0

    star_patch_radius: int = 28
    minimum_star_area: int = 3
    minimum_star_snr: float = 5.0
    maximum_star_axis_ratio: float = 2.5
    maximum_centroid_offset_px: float = 3.0
    maximum_photometric_stars: int = 40
    fwhm_bounds: Tuple[float, float] = (0.5, 15.0)
    flux_bounds: Tuple[float, float] = (1e-6, 1e12)

    source_threshold_sigma: float = 4.0
    sky_tile_size: int = 64
    maximum_noise_samples: int = 100000
    maximum_read_noise: float = 100.0

    def __post_init__(self):
        positive = {
            "exposure_time": self.exposure_time,
            "rough_min_fov": self.rough_min_fov,
            "rough_max_fov": self.rough_max_fov,
            "refined_fov_tolerance": self.refined_fov_tolerance,
            "maximum_astrometric_rmse_px": self.maximum_astrometric_rmse_px,
            "maximum_distortion_displacement_px":
                self.maximum_distortion_displacement_px,
            "star_patch_radius": self.star_patch_radius,
            "minimum_star_area": self.minimum_star_area,
            "minimum_star_snr": self.minimum_star_snr,
            "maximum_star_axis_ratio": self.maximum_star_axis_ratio,
            "maximum_centroid_offset_px": self.maximum_centroid_offset_px,
            "maximum_photometric_stars": self.maximum_photometric_stars,
            "source_threshold_sigma": self.source_threshold_sigma,
            "sky_tile_size": self.sky_tile_size,
            "maximum_noise_samples": self.maximum_noise_samples,
            "maximum_read_noise": self.maximum_read_noise,
        }
        for name, value in positive.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be positive")

        integer_fields = {
            "minimum_matches": self.minimum_matches,
            "star_patch_radius": self.star_patch_radius,
            "minimum_star_area": self.minimum_star_area,
            "maximum_photometric_stars": self.maximum_photometric_stars,
            "sky_tile_size": self.sky_tile_size,
            "maximum_noise_samples": self.maximum_noise_samples,
        }
        for name, value in integer_fields.items():
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be an integer")

        if not self.rough_min_fov < self.rough_max_fov:
            raise ValueError("rough_min_fov must be smaller than rough_max_fov")
        if self.time_reference not in ("start", "midpoint", "end"):
            raise ValueError(
                "time_reference must be 'start', 'midpoint', or 'end'"
            )
        if not 0 < self.maximum_false_positive_probability < 1:
            raise ValueError(
                "maximum_false_positive_probability must be between 0 and 1"
            )
        if self.minimum_matches < 4:
            raise ValueError("minimum_matches must be at least 4")

        for name, bounds in (
            ("rough_distortion_bounds", self.rough_distortion_bounds),
            ("refined_distortion_bounds", self.refined_distortion_bounds),
            ("fwhm_bounds", self.fwhm_bounds),
            ("flux_bounds", self.flux_bounds),
        ):
            if (
                len(bounds) != 2
                or not np.isfinite(bounds).all()
                or bounds[0] >= bounds[1]
            ):
                raise ValueError(f"{name} must contain increasing finite bounds")

        if self.fwhm_bounds[0] <= 0 or self.flux_bounds[0] <= 0:
            raise ValueError("FWHM and flux bounds must be positive")


@dataclass(frozen=True)
class StepReport:
    name: str
    values: Mapping[str, object]
    visual: Optional[Image.Image] = None


@dataclass
class StarMeasurement:
    catalog_id: Optional[CatalogId]
    magnitude: float
    catalog_x: float
    catalog_y: float
    centroid_x: float
    centroid_y: float
    area: int
    background: float
    background_sigma: float
    aperture_sum: float
    patch: np.ndarray = field(repr=False)
    source_mask: np.ndarray = field(repr=False)
    fit_mask: np.ndarray = field(repr=False)
    patch_x0: int = field(repr=False)
    patch_y0: int = field(repr=False)


@dataclass
class CalibrationContext:
    sky: Sky
    image: Image.Image
    time: datetime.datetime
    latitude_deg: float
    longitude_deg: float
    elevation_m: float
    config: CalibrationConfig

    image_data: Optional[np.ndarray] = None
    native_data: Optional[np.ndarray] = None
    navigation_image: Optional[Image.Image] = None
    saturated_mask: Optional[np.ndarray] = None
    rough_solution: Optional[LostInSpace.Solution] = None
    refined_solution: Optional[LostInSpace.Solution] = None
    matched_centroids: Optional[np.ndarray] = None
    matched_stars: Optional[np.ndarray] = None
    matched_catalog_ids: Tuple[Optional[CatalogId], ...] = ()
    star_measurements: Tuple[StarMeasurement, ...] = ()
    source_mask: Optional[np.ndarray] = None
    blank_sky_mask: Optional[np.ndarray] = None
    visualization_enabled: bool = False
    reports: List[StepReport] = field(default_factory=list)
    _pending_values: Dict[str, object] = field(default_factory=dict, repr=False)
    _pending_visual: Optional[Image.Image] = field(default=None, repr=False)

    def record(self, **values):
        self._pending_values.update(values)

    def set_visual(self, visual: Image.Image):
        self._pending_visual = visual

    def finish_report(self, name: str) -> StepReport:
        report = StepReport(
            name=name,
            values=dict(self._pending_values),
            visual=self._pending_visual,
        )
        self._pending_values.clear()
        self._pending_visual = None
        self.reports.append(report)
        return report


@dataclass(frozen=True)
class CalibrationStep:
    """Definition of a step that binds run context to a pair-to-pair function."""

    name: str
    binder: Callable[[CalibrationContext], BoundCalibrationStep]

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("calibration step name must be a non-empty string")
        if not callable(self.binder):
            raise TypeError("calibration step binder must be callable")

    def bind(self, context: CalibrationContext) -> BoundCalibrationStep:
        step = self.binder(context)
        if not callable(step):
            raise TypeError(
                f"calibration step binder {self.name!r} did not return a callable"
            )
        return step

    @classmethod
    def from_callable(
        cls,
        name: str,
        step: BoundCalibrationStep,
    ) -> "CalibrationStep":
        if not callable(step):
            raise TypeError("calibration step must be callable")
        return cls(name=name, binder=lambda _context: step)


def otsu_threshold(values: np.ndarray) -> float:
    """Return the exact Otsu threshold for finite scalar samples.

    Unique sample values are used directly rather than assuming an 8-bit
    histogram, so the same implementation works for MONO8, MONO16, and RGB
    luminance cutouts.
    """

    samples = np.asarray(values, dtype=np.float64).ravel()
    samples = samples[np.isfinite(samples)]
    unique, counts = np.unique(samples, return_counts=True)
    if unique.size < 2:
        raise ValueError("Otsu threshold requires at least two distinct values")

    weights_low = np.cumsum(counts, dtype=np.float64)
    weighted_low = np.cumsum(unique * counts, dtype=np.float64)
    total_weight = weights_low[-1]
    total_weighted = weighted_low[-1]

    weights_high = total_weight - weights_low[:-1]
    valid = (weights_low[:-1] > 0) & (weights_high > 0)
    mean_low = weighted_low[:-1] / weights_low[:-1]
    mean_high = (
        total_weighted - weighted_low[:-1]
    ) / weights_high
    between = (
        weights_low[:-1]
        * weights_high
        * (mean_low - mean_high) ** 2
    )
    between[~valid] = -np.inf
    index = int(np.argmax(between))
    return float((unique[index] + unique[index + 1]) / 2)


def _robust_location_scale(values: np.ndarray) -> Tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("cannot estimate location and scale from no samples")
    location = float(np.median(values))
    scale = float(1.4826 * np.median(np.abs(values - location)))
    if not math.isfinite(scale) or scale <= 0:
        scale = max(float(np.std(values)), 1e-6)
    return location, scale


class Calibrator:
    """Estimate an observer attitude and camera through configurable steps.

    Every step has the public ``(observer, camera) -> (observer, camera)``
    contract.  Set ``visualize=True`` to display the final observed-versus-
    calibrated comparison, or provide ``visualization_callback`` to receive
    its :class:`StepReport` and save/embed the ``visual`` PIL image.
    """

    IMAGE_PROPERTIES_STEP = "image_properties"
    ROUGH_FOV_STEP = "rough_fov"
    REFINED_FOV_STEP = "refined_fov"
    STAR_PHOTOMETRY_STEP = "star_photometry"
    SKY_NOISE_STEP = "sky_noise"

    def __init__(
        self,
        sky: Sky,
        steps: Optional[
            Iterable[Union[CalibrationStep, BoundCalibrationStep]]
        ] = None,
        config: Optional[CalibrationConfig] = None,
        exposure_time: Optional[float] = None,
        diagnostic_callback: Optional[
            Callable[[StepReport, Observer, Camera], None]
        ] = None,
        visualize: bool = False,
        visualization_callback: Optional[Callable[[StepReport], None]] = None,
    ):
        if not isinstance(sky, Sky):
            raise TypeError("sky must be a Sky")
        if config is not None and exposure_time is not None:
            raise ValueError("pass config or exposure_time, not both")

        if config is None:
            config = CalibrationConfig(
                exposure_time=1.0 if exposure_time is None else exposure_time
            )
        if not isinstance(config, CalibrationConfig):
            raise TypeError("config must be a CalibrationConfig")
        if diagnostic_callback is not None and not callable(diagnostic_callback):
            raise TypeError("diagnostic_callback must be callable")
        if not isinstance(visualize, bool):
            raise TypeError("visualize must be a bool")
        if (
            visualization_callback is not None
            and not callable(visualization_callback)
        ):
            raise TypeError("visualization_callback must be callable")

        self.sky = sky
        self.config = config
        self.diagnostic_callback = diagnostic_callback
        self.visualize = visualize
        self.visualization_callback = visualization_callback
        self._last_reports: Tuple[StepReport, ...] = ()

        if steps is None:
            self._steps = self._default_steps()
        else:
            self._steps = [
                self._coerce_step(step, index)
                for index, step in enumerate(steps)
            ]
        self._validate_unique_step_names()

    @staticmethod
    def _coerce_step(step, index: int) -> CalibrationStep:
        if isinstance(step, CalibrationStep):
            return step
        if callable(step):
            name = getattr(step, "__name__", "")
            if not name or name == "<lambda>":
                name = f"step_{index + 1}"
            return CalibrationStep.from_callable(name, step)
        raise TypeError(
            "steps must contain CalibrationStep objects or pair-to-pair callables"
        )

    def _default_steps(self) -> List[CalibrationStep]:
        return [
            CalibrationStep(
                self.IMAGE_PROPERTIES_STEP,
                self._bind_image_properties_step,
            ),
            CalibrationStep(self.ROUGH_FOV_STEP, self._bind_rough_fov_step),
            CalibrationStep(
                self.REFINED_FOV_STEP,
                self._bind_refined_fov_step,
            ),
            CalibrationStep(
                self.STAR_PHOTOMETRY_STEP,
                self._bind_star_photometry_step,
            ),
            CalibrationStep(self.SKY_NOISE_STEP, self._bind_sky_noise_step),
        ]

    @property
    def steps(self) -> Tuple[CalibrationStep, ...]:
        return tuple(self._steps)

    @property
    def last_reports(self) -> Tuple[StepReport, ...]:
        return self._last_reports

    def _validate_unique_step_names(self):
        names = [step.name for step in self._steps]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(
                "calibration step names must be unique: "
                + ", ".join(duplicates)
            )

    def add_step(self, name: str, step: BoundCalibrationStep):
        """Append a plain ``(observer, camera) -> (observer, camera)`` step."""
        self.insert_step(len(self._steps), name, step)

    def insert_step(
        self,
        index: int,
        name: str,
        step: BoundCalibrationStep,
    ):
        """Insert a plain pair-to-pair step at ``index``."""
        self.insert_step_definition(
            index,
            CalibrationStep.from_callable(name, step),
        )

    def add_step_definition(self, step: CalibrationStep):
        """Append a context-aware step definition."""
        self.insert_step_definition(len(self._steps), step)

    def insert_step_definition(self, index: int, step: CalibrationStep):
        """Insert a context-aware definition whose binder runs once per image."""
        if not isinstance(step, CalibrationStep):
            raise TypeError("step must be a CalibrationStep")
        if any(existing.name == step.name for existing in self._steps):
            raise ValueError(f"calibration step {step.name!r} already exists")
        self._steps.insert(index, step)

    def remove_step(self, name: str) -> CalibrationStep:
        for index, step in enumerate(self._steps):
            if step.name == name:
                return self._steps.pop(index)
        raise KeyError(name)

    @staticmethod
    def _image_format(image: Image.Image) -> ImageFormat:
        if image.mode == "L":
            return ImageFormat.MONO8
        if image.mode == "RGB":
            return ImageFormat.RGB8
        if image.mode.startswith("I;16"):
            return ImageFormat.MONO16
        if image.mode == "I":
            minimum, maximum = image.getextrema()
            if minimum >= 0 and maximum <= 65535:
                return ImageFormat.MONO16
            raise ValueError(
                "mode I image values must fit the unsigned 16-bit range"
            )
        raise ValueError(f"unsupported image mode: {image.mode}")

    @staticmethod
    def _luminance(image: Image.Image) -> Tuple[np.ndarray, np.ndarray]:
        native = np.asarray(image)
        data = np.asarray(native, dtype=np.float64)
        if data.ndim == 2:
            luminance = data
        elif data.ndim == 3 and data.shape[2] == 3:
            luminance = np.average(
                data,
                axis=2,
                weights=(0.299, 0.587, 0.114),
            )
        else:
            raise ValueError("image must have one grayscale or three RGB channels")
        if not np.isfinite(luminance).all():
            raise ValueError("image contains non-finite values")
        return native, luminance

    @staticmethod
    def _navigation_image(
        data: np.ndarray,
        image_format: ImageFormat,
    ) -> Image.Image:
        """Build tetra3's 8-bit working copy without touching photometry data.

        tetra3's default extraction thresholds behave substantially differently
        on high-bit-depth arrays. ``navigation_scale`` creates its 8-bit
        working copy without changing the native camera-count units used for
        flux, background, and noise.
        """

        scaled = (
            np.asarray(data, dtype=np.float64)
            / image_format.navigation_scale
        )
        return Image.fromarray(np.clip(scaled, 0, 255).astype(np.uint8))

    @staticmethod
    def _validate_request(context: CalibrationContext):
        if not isinstance(context.image, Image.Image):
            raise TypeError("image must be a PIL Image")
        if not isinstance(context.time, datetime.datetime):
            raise TypeError("time must be a datetime")
        if context.time.tzinfo is None:
            raise ValueError("time must be timezone-aware")

        numeric = {
            "latitude_deg": context.latitude_deg,
            "longitude_deg": context.longitude_deg,
            "elevation_m": context.elevation_m,
        }
        for name, value in numeric.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be finite")
        if not -90 <= context.latitude_deg <= 90:
            raise ValueError("latitude_deg must be between -90 and 90")
        if not -180 <= context.longitude_deg <= 180:
            raise ValueError("longitude_deg must be between -180 and 180")

    @staticmethod
    def _exposure_times(
        context: CalibrationContext,
    ) -> Tuple[datetime.datetime, datetime.datetime, datetime.datetime]:
        duration = datetime.timedelta(seconds=context.config.exposure_time)
        if context.config.time_reference == "start":
            start = context.time
            end = start + duration
        elif context.config.time_reference == "midpoint":
            start = context.time - duration / 2
            end = context.time + duration / 2
        else:
            end = context.time
            start = end - duration
        midpoint = start + duration / 2
        return start, midpoint, end

    def _bind_image_properties_step(
        self,
        context: CalibrationContext,
    ) -> BoundCalibrationStep:
        def determine_image_properties(
            observer: Observer,
            camera: Camera,
        ) -> CalibrationPair:
            self._validate_request(context)
            image_format = self._image_format(context.image)
            native, luminance = self._luminance(context.image)

            _start, _midpoint, exposure_end = self._exposure_times(context)
            observer.set_time(exposure_end)
            observer.set_location(
                latitude=context.latitude_deg,
                longitude=context.longitude_deg,
                elevation=context.elevation_m,
            )
            camera.set_image(
                context.image.width,
                context.image.height,
                image_format,
            )
            camera.set_exposure_time(context.config.exposure_time)
            camera.set_noise(
                sky_background=0.0,
                read_noise=0.0,
            )

            context.native_data = native
            context.image_data = luminance
            context.navigation_image = self._navigation_image(
                luminance,
                image_format,
            )
            if native.ndim == 3:
                context.saturated_mask = np.any(
                    native >= image_format.max_value,
                    axis=2,
                )
            else:
                context.saturated_mask = native >= image_format.max_value
            context.record(
                width=camera.geometry.width,
                height=camera.geometry.height,
                image_format=camera.geometry.image_format.name,
                exposure_time=camera.image_model.exposure_time,
                sky_background=camera.image_model.sky_background,
                read_noise=camera.image_model.read_noise,
                time_reference=context.config.time_reference,
                observer_time=observer.time,
            )
            return observer, camera

        return determine_image_properties

    @staticmethod
    def _load_solver(
        min_fov: float,
        max_fov: float,
        star_max_magnitude: float,
    ) -> LostInSpace:
        db_path = LostInSpace.get_db_path(
            min_fov=min_fov,
            max_fov=max_fov,
            star_max_magnitude=star_max_magnitude,
        )
        if not db_path.exists():
            db_path = LostInSpace.generate_db(
                min_fov=min_fov,
                max_fov=max_fov,
                star_max_magnitude=star_max_magnitude,
            )
        return LostInSpace(db_path)

    @staticmethod
    def _apply_solution_orientation(
        context: CalibrationContext,
        observer: Observer,
        solution: LostInSpace.Solution,
    ):
        _start, midpoint, _end = Calibrator._exposure_times(context)
        attitude = (solution.ra, solution.dec, solution.roll)
        if not np.isfinite(attitude).all():
            raise ValueError("plate solution returned a non-finite attitude")
        midpoint_observer = Observer()
        midpoint_observer.set_time(midpoint)
        midpoint_observer.set_location(
            latitude=context.latitude_deg,
            longitude=context.longitude_deg,
            elevation=context.elevation_m,
        )
        direction_ecef, up_ecef = context.sky.radec_to_ecef(
            midpoint_observer,
            solution.ra,
            solution.dec,
            solution.roll,
        )
        ecef_to_ned = ECEF.ecef_to_ned(
            context.latitude_deg,
            context.longitude_deg,
        )
        observer.set_look_direction(
            look_dir=ecef_to_ned @ direction_ecef,
            look_up=ecef_to_ned @ up_ecef,
        )
        if (
            observer.observer_matrix is None
            or not np.isfinite(observer.observer_matrix).all()
        ):
            raise ValueError("plate solution produced an invalid orientation")

    def _bind_rough_fov_step(
        self,
        context: CalibrationContext,
    ) -> BoundCalibrationStep:
        def determine_rough_fov(
            observer: Observer,
            camera: Camera,
        ) -> CalibrationPair:
            if context.navigation_image is None:
                raise ValueError("image properties step has not prepared navigation data")
            config = context.config
            solver = self._load_solver(
                config.rough_min_fov,
                config.rough_max_fov,
                config.rough_star_max_magnitude,
            )
            solution = solver.solve(
                context.navigation_image,
                distortion=config.rough_distortion_bounds,
            )
            if solution is None:
                raise ValueError("could not find a rough plate solution")
            rough_values = (
                solution.fov,
                solution.distortion,
                solution.ra,
                solution.dec,
                solution.roll,
            )
            if not np.isfinite(rough_values).all():
                raise ValueError(
                    "rough plate solution returned non-finite parameters"
                )

            context.rough_solution = solution
            camera.set_fov(float(solution.fov))
            self._apply_solution_orientation(context, observer, solution)
            context.record(
                fov_deg=float(solution.fov),
                distortion=float(solution.distortion),
                matches=int(solution.matches),
            )
            return observer, camera

        return determine_rough_fov

    @staticmethod
    def _astrometric_rmse_px(
        camera: Camera,
        solution: LostInSpace.Solution,
    ) -> float:
        if not camera.is_valid():
            raise ValueError("camera is not valid")
        angle = math.radians(float(solution.rmse) / 3600)
        return float(camera.focal_length * math.tan(angle))

    @staticmethod
    def _distortion_displacement_px(
        camera: Camera,
        solution: LostInSpace.Solution,
        matched_centroids: Optional[np.ndarray] = None,
    ) -> float:
        if not camera.is_valid():
            raise ValueError("camera is not valid")
        geometry = camera.geometry
        k = float(solution.distortion)
        if not math.isfinite(k) or math.isclose(k, 1.0):
            return math.inf

        points = [
            (0.0, 0.0),
            (float(geometry.width), 0.0),
            (0.0, float(geometry.height)),
            (float(geometry.width), float(geometry.height)),
        ]
        if matched_centroids is None:
            matched_centroids, _matched_stars, _catalog_ids = (
                Calibrator._process_matches(solution)
            )
        points.extend(
            (float(x), float(y))
            for y, x in matched_centroids
        )
        center_x = geometry.width / 2
        center_y = geometry.height / 2
        maximum = 0.0
        for x, y in points:
            radius_px = math.hypot(x - center_x, y - center_y)
            radius_normalized = 2 * radius_px / geometry.width
            scale = (
                1 - k * radius_normalized ** 2
            ) / (1 - k)
            maximum = max(maximum, abs(radius_px * (scale - 1)))
        return float(maximum)

    @staticmethod
    def _process_matches(
        solution: LostInSpace.Solution,
    ) -> Tuple[np.ndarray, np.ndarray, Tuple[Optional[CatalogId], ...]]:
        """Validate and normalize tetra3's raw match payload for calibration.

        LostInSpace deliberately preserves tetra3's objects unchanged.  Their
        conversion belongs here, at the first point where numerical arrays and
        aligned catalog rows are actually needed.
        """

        try:
            centroids = np.asarray(
                solution.matched_centroids,
                dtype=np.float64,
            )
            stars = np.asarray(solution.matched_stars, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "refined plate solution returned invalid star matches"
            ) from error

        if centroids.ndim != 2 or centroids.shape[1] != 2:
            raise ValueError(
                "matched_centroids must have shape (matches, 2)"
            )
        if stars.ndim != 2 or stars.shape[1] < 3:
            raise ValueError("matched_stars must have shape (matches, >=3)")
        if centroids.shape[0] != stars.shape[0]:
            raise ValueError("refined plate solution has misaligned star matches")
        if not np.isfinite(centroids).all() or not np.isfinite(stars).all():
            raise ValueError("refined plate solution has non-finite star matches")

        match_count = centroids.shape[0]
        if match_count != int(solution.matches):
            raise ValueError(
                "refined plate solution match count does not agree with "
                "its star-pair payload"
            )
        raw_ids = solution.matched_catalog_ids
        if raw_ids is None:
            catalog_ids: Tuple[Optional[CatalogId], ...] = (None,) * match_count
        else:
            try:
                catalog_ids = tuple(
                    Calibrator._normalize_catalog_id(value)
                    for value in raw_ids
                )
            except TypeError as error:
                raise ValueError(
                    "matched catalog IDs must be an aligned sequence"
                ) from error
            if len(catalog_ids) != match_count:
                raise ValueError(
                    "refined plate solution has misaligned catalog IDs"
                )
        return centroids, stars, catalog_ids

    @staticmethod
    def _normalize_catalog_id(value: object) -> object:
        if isinstance(value, np.ndarray):
            value = value.tolist()
        if isinstance(value, (list, tuple)):
            return tuple(Calibrator._normalize_catalog_id(item) for item in value)
        if isinstance(value, np.integer):
            return int(value)
        return value

    def _validate_refined_solution(
        self,
        camera: Camera,
        solution: LostInSpace.Solution,
        config: CalibrationConfig,
    ) -> Tuple[
        float,
        float,
        np.ndarray,
        np.ndarray,
        Tuple[Optional[CatalogId], ...],
    ]:
        finite_values = (
            solution.ra,
            solution.dec,
            solution.roll,
            solution.fov,
            solution.distortion,
            solution.rmse,
            solution.false_positive_prob,
        )
        if not np.isfinite(finite_values).all():
            raise ValueError("refined plate solution contains non-finite values")
        if solution.matches < config.minimum_matches:
            raise ValueError(
                "refined plate solution has "
                f"{solution.matches} matches; need at least "
                f"{config.minimum_matches}"
            )
        if (
            solution.false_positive_prob
            > config.maximum_false_positive_probability
        ):
            raise ValueError(
                "refined plate solution false-positive probability is "
                f"{solution.false_positive_prob:.3g}"
            )

        camera.set_fov(float(solution.fov))
        rmse_px = self._astrometric_rmse_px(camera, solution)
        if rmse_px > config.maximum_astrometric_rmse_px:
            raise ValueError(
                f"refined plate solution RMSE is {rmse_px:.3f} px"
            )

        centroids, stars, catalog_ids = self._process_matches(solution)
        distortion_px = self._distortion_displacement_px(
            camera,
            solution,
            centroids,
        )
        if distortion_px > config.maximum_distortion_displacement_px:
            raise ValueError(
                "radial distortion would displace image points by up to "
                f"{distortion_px:.3f} px, but Camera is pinhole-only"
            )
        if stars.shape[0] < config.minimum_matches:
            raise ValueError("refined plate solution did not return enough star pairs")
        return rmse_px, distortion_px, centroids, stars, catalog_ids

    def _bind_refined_fov_step(
        self,
        context: CalibrationContext,
    ) -> BoundCalibrationStep:
        def determine_refined_fov(
            observer: Observer,
            camera: Camera,
        ) -> CalibrationPair:
            if context.rough_solution is None:
                raise ValueError("rough FOV step has not produced a solution")
            config = context.config
            tolerance = config.refined_fov_tolerance
            rounded_fov = round(context.rough_solution.fov / tolerance) * tolerance
            min_fov = max(0.1, rounded_fov - tolerance)
            max_fov = min(179.9, rounded_fov + tolerance)
            solver = self._load_solver(
                min_fov,
                max_fov,
                config.refined_star_max_magnitude,
            )
            solution = solver.solve(
                context.navigation_image,
                distortion=config.refined_distortion_bounds,
            )
            if solution is None:
                raise ValueError("could not find a refined plate solution")

            (
                rmse_px,
                distortion_px,
                matched_centroids,
                matched_stars,
                matched_catalog_ids,
            ) = self._validate_refined_solution(
                camera,
                solution,
                config,
            )
            context.refined_solution = solution
            context.matched_centroids = matched_centroids
            context.matched_stars = matched_stars
            context.matched_catalog_ids = matched_catalog_ids
            self._apply_solution_orientation(context, observer, solution)
            context.record(
                fov_deg=float(solution.fov),
                distortion=float(solution.distortion),
                distortion_displacement_px=distortion_px,
                matches=int(solution.matches),
                false_positive_probability=float(
                    solution.false_positive_prob
                ),
                rmse_arcsec=float(solution.rmse),
                rmse_px=rmse_px,
            )
            return observer, camera

        return determine_refined_fov

    @staticmethod
    def _component_near_centroid(
        foreground: np.ndarray,
        expected_y: float,
        expected_x: float,
        maximum_offset: float,
    ) -> Optional[np.ndarray]:
        labels, count = label(foreground, structure=np.ones((3, 3), dtype=bool))
        if count == 0:
            return None

        best_label = None
        best_distance = math.inf
        for component_label in range(1, count + 1):
            yy, xx = np.nonzero(labels == component_label)
            if yy.size == 0:
                continue
            distance = math.hypot(
                float(np.mean(yy)) - expected_y,
                float(np.mean(xx)) - expected_x,
            )
            if distance < best_distance:
                best_distance = distance
                best_label = component_label
        if best_label is None or best_distance > maximum_offset:
            return None
        return labels == best_label

    @staticmethod
    def _star_axis_ratio(
        patch: np.ndarray,
        component: np.ndarray,
        background: float,
    ) -> Tuple[float, float, float]:
        yy, xx = np.nonzero(component)
        weights = np.maximum(patch[component] - background, 0)
        total = float(np.sum(weights))
        if total <= 0 or yy.size < 2:
            return math.inf, math.nan, math.nan
        centroid_y = float(np.dot(weights, yy) / total)
        centroid_x = float(np.dot(weights, xx) / total)
        dy = yy - centroid_y
        dx = xx - centroid_x
        covariance = np.array([
            [
                np.dot(weights, dx * dx) / total,
                np.dot(weights, dx * dy) / total,
            ],
            [
                np.dot(weights, dx * dy) / total,
                np.dot(weights, dy * dy) / total,
            ],
        ])
        eigenvalues = np.linalg.eigvalsh(covariance)
        if eigenvalues[0] <= 1e-9:
            ratio = math.inf
        else:
            ratio = float(math.sqrt(eigenvalues[1] / eigenvalues[0]))
        return ratio, centroid_y, centroid_x

    def _measure_star(
        self,
        context: CalibrationContext,
        centroid: Sequence[float],
        star: Sequence[float],
        catalog_id: Optional[CatalogId],
        other_centroids: np.ndarray,
    ) -> Optional[StarMeasurement]:
        config = context.config
        radius = config.star_patch_radius
        y_tetra, x_tetra = map(float, centroid)
        # tetra3 uses (0.5, 0.5) for the first pixel centre, while Canvas
        # positions integer pixel indices at their centres.
        catalog_y = y_tetra - 0.5
        catalog_x = x_tetra - 0.5
        center_y = int(round(catalog_y))
        center_x = int(round(catalog_x))

        height, width = context.image_data.shape
        y0, y1 = center_y - radius, center_y + radius + 1
        x0, x1 = center_x - radius, center_x + radius + 1
        if y0 < 0 or x0 < 0 or y1 > height or x1 > width:
            return None

        if other_centroids.size:
            distances = np.hypot(
                other_centroids[:, 1] - x_tetra,
                other_centroids[:, 0] - y_tetra,
            )
            distances = distances[distances > 1e-6]
            if distances.size and np.min(distances) < 2 * radius:
                return None

        patch = np.asarray(
            context.image_data[y0:y1, x0:x1],
            dtype=np.float64,
        )
        saturated = context.saturated_mask[y0:y1, x0:x1]
        valid = np.isfinite(patch) & ~saturated
        if np.count_nonzero(valid) < patch.size * 0.8:
            return None
        try:
            threshold = otsu_threshold(patch[valid])
        except ValueError:
            return None

        foreground = (patch > threshold) & valid
        component = self._component_near_centroid(
            foreground,
            expected_y=catalog_y - y0,
            expected_x=catalog_x - x0,
            maximum_offset=config.maximum_centroid_offset_px,
        )
        if component is None:
            return None
        area = int(np.count_nonzero(component))
        if area < config.minimum_star_area or np.any(saturated & component):
            return None

        other_foreground = foreground & ~component
        exclusion = binary_dilation(
            foreground,
            iterations=min(
                radius - 4,
                int(math.ceil(config.fwhm_bounds[1])),
            ),
        )
        background_mask = valid & ~exclusion
        if np.count_nonzero(background_mask) < max(30, patch.size // 5):
            return None
        background, background_sigma = _robust_location_scale(
            patch[background_mask]
        )

        axis_ratio, measured_y, measured_x = self._star_axis_ratio(
            patch,
            component,
            background,
        )
        if (
            not math.isfinite(axis_ratio)
            or axis_ratio > config.maximum_star_axis_ratio
            or not np.isfinite((measured_y, measured_x)).all()
        ):
            return None

        aperture = binary_dilation(component, iterations=2) & valid
        aperture &= ~binary_dilation(other_foreground, iterations=1)
        aperture_sum = float(np.sum(patch[aperture] - background))
        noise_denominator = max(
            background_sigma * math.sqrt(max(np.count_nonzero(aperture), 1)),
            1e-9,
        )
        if aperture_sum <= 0 or aperture_sum / noise_denominator < config.minimum_star_snr:
            return None

        global_centroid_y = y0 + measured_y
        global_centroid_x = x0 + measured_x
        yy, xx = np.indices(patch.shape, dtype=np.float64)
        fit_radius = min(
            radius - 2,
            max(18.0, 3 * math.sqrt(area / math.pi)),
        )
        fit_mask = (
            valid
            & ~binary_dilation(other_foreground, iterations=2)
            & (
                (xx - measured_x) ** 2
                + (yy - measured_y) ** 2
                <= fit_radius ** 2
            )
        )
        if np.count_nonzero(fit_mask) < 20:
            return None

        return StarMeasurement(
            catalog_id=catalog_id,
            magnitude=float(star[2]),
            catalog_x=catalog_x,
            catalog_y=catalog_y,
            centroid_x=global_centroid_x,
            centroid_y=global_centroid_y,
            area=area,
            background=background,
            background_sigma=background_sigma,
            aperture_sum=aperture_sum,
            patch=patch,
            source_mask=foreground,
            fit_mask=fit_mask,
            patch_x0=x0,
            patch_y0=y0,
        )

    @staticmethod
    def _point_spread_function(
        shape: Tuple[int, int],
        centroid_y: float,
        centroid_x: float,
        fwhm: float,
    ) -> np.ndarray:
        yy, xx = np.indices(shape, dtype=np.float64)
        return point_spread_kernel(
            xx,
            yy,
            centroid_x=centroid_x,
            centroid_y=centroid_y,
            fwhm=fwhm,
        )

    def _fit_star_photometry(
        self,
        measurements: Sequence[StarMeasurement],
        camera: Camera,
        config: CalibrationConfig,
    ) -> Tuple[float, float]:
        output_scale = camera.geometry.image_format.output_scale
        exposure = camera.image_model.exposure_time
        flux_estimates = []
        fwhm_estimates = []
        for measurement in measurements:
            magnitude_scale = 10 ** (-0.4 * measurement.magnitude)
            denominator = output_scale * exposure * magnitude_scale
            if denominator > 0:
                flux_estimates.append(measurement.aperture_sum / denominator)

            patch_y = measurement.centroid_y - measurement.patch_y0
            patch_x = measurement.centroid_x - measurement.patch_x0
            yy, xx = np.nonzero(measurement.fit_mask)
            weights = np.maximum(
                measurement.patch[measurement.fit_mask]
                - measurement.background,
                0,
            )
            total = float(np.sum(weights))
            if total > 0:
                variance = float(
                    np.dot(
                        weights,
                        (xx - patch_x) ** 2 + (yy - patch_y) ** 2,
                    )
                    / (2 * total)
                )
                if variance > 0:
                    fwhm_estimates.append(2.355 * math.sqrt(variance))

        if not flux_estimates:
            raise ValueError("could not initialize camera flux")
        initial_flux = float(np.clip(
            np.median(flux_estimates),
            *config.flux_bounds,
        ))
        initial_fwhm = float(np.clip(
            (
                np.median(fwhm_estimates)
                if fwhm_estimates
                else camera.image_model.fwhm
            ),
            *config.fwhm_bounds,
        ))

        def residual(parameters):
            flux = math.exp(parameters[0])
            fwhm = math.exp(parameters[1])
            result = []
            for measurement in measurements:
                patch_y = measurement.centroid_y - measurement.patch_y0
                patch_x = measurement.centroid_x - measurement.patch_x0
                psf = self._point_spread_function(
                    measurement.patch.shape,
                    patch_y,
                    patch_x,
                    fwhm,
                )
                signal = (
                    output_scale
                    * exposure
                    * flux
                    * 10 ** (-0.4 * measurement.magnitude)
                )
                expected = signal * psf
                observed = measurement.patch - measurement.background
                scale = max(measurement.background_sigma, 1.0)
                result.append(
                    (expected - observed)[measurement.fit_mask] / scale
                )
            return np.concatenate(result)

        result = least_squares(
            residual,
            x0=np.log((initial_flux, initial_fwhm)),
            bounds=(
                np.log((config.flux_bounds[0], config.fwhm_bounds[0])),
                np.log((config.flux_bounds[1], config.fwhm_bounds[1])),
            ),
            loss="soft_l1",
            f_scale=1.0,
            x_scale="jac",
            max_nfev=100,
        )
        if not result.success:
            raise RuntimeError(f"star photometry fit failed: {result.message}")
        flux, fwhm = np.exp(result.x)
        return float(flux), float(fwhm)

    def _bind_star_photometry_step(
        self,
        context: CalibrationContext,
    ) -> BoundCalibrationStep:
        def determine_star_photometry(
            observer: Observer,
            camera: Camera,
        ) -> CalibrationPair:
            solution = context.refined_solution
            if solution is None:
                raise ValueError("refined FOV step has not produced star matches")
            if context.image_data is None or context.saturated_mask is None:
                raise ValueError("image properties step has not prepared image data")
            if (
                context.matched_centroids is None
                or context.matched_stars is None
            ):
                raise ValueError(
                    "refined FOV step has not processed star matches"
                )

            all_centroids = context.matched_centroids
            catalog_ids = context.matched_catalog_ids
            measurements = []
            for index, (centroid, star) in enumerate(zip(
                context.matched_centroids,
                context.matched_stars,
            )):
                measurement = self._measure_star(
                    context,
                    centroid,
                    star,
                    catalog_ids[index],
                    all_centroids,
                )
                if measurement is not None:
                    measurements.append(measurement)
                if len(measurements) >= context.config.maximum_photometric_stars:
                    break

            if len(measurements) < 2:
                raise ValueError(
                    "fewer than two isolated, unsaturated matched stars "
                    "passed photometric selection"
                )
            flux, fwhm = self._fit_star_photometry(
                measurements,
                camera,
                context.config,
            )
            camera.set_photometry(flux=flux, fwhm=fwhm)
            context.star_measurements = tuple(measurements)

            source_mask = np.zeros(context.image_data.shape, dtype=bool)
            dilation = source_mask_radius(fwhm)
            for measurement in measurements:
                local_source = binary_dilation(
                    measurement.source_mask,
                    iterations=dilation,
                )
                y0, x0 = measurement.patch_y0, measurement.patch_x0
                y1 = y0 + local_source.shape[0]
                x1 = x0 + local_source.shape[1]
                source_mask[y0:y1, x0:x1] |= local_source
            context.source_mask = source_mask
            context.record(
                selected_stars=len(measurements),
                flux=flux,
                fwhm_px=fwhm,
                median_otsu_area_px=float(np.median(
                    [measurement.area for measurement in measurements]
                )),
            )
            return observer, camera

        return determine_star_photometry

    @staticmethod
    def _all_source_mask(
        data: np.ndarray,
        valid: np.ndarray,
        fwhm: float,
        config: CalibrationConfig,
    ) -> np.ndarray:
        values = data[valid]
        location, scale = _robust_location_scale(values)
        foreground = valid & (
            data > location + config.source_threshold_sigma * scale
        )
        iterations = source_mask_radius(fwhm)
        return binary_dilation(foreground, iterations=iterations)

    @staticmethod
    def _blank_sky_tiles(
        data: np.ndarray,
        valid: np.ndarray,
        tile_size: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        tile_locations = []
        residuals = []
        internal_means = []
        height, width = data.shape
        for y0 in range(0, height, tile_size):
            for x0 in range(0, width, tile_size):
                y1 = min(height, y0 + tile_size)
                x1 = min(width, x0 + tile_size)
                tile_valid = valid[y0:y1, x0:x1]
                if np.count_nonzero(tile_valid) < max(
                    32,
                    tile_valid.size // 2,
                ):
                    continue
                values = data[y0:y1, x0:x1][tile_valid]
                location, scale = _robust_location_scale(values)
                clean = np.abs(values - location) <= 5 * scale
                values = values[clean]
                if values.size < 32:
                    continue
                tile_locations.append(location)
                residuals.append(values - location)
                internal_means.append(np.full(values.size, location))

        if not tile_locations:
            raise ValueError("no uncontaminated blank-sky tiles remain")
        return (
            np.asarray(tile_locations, dtype=np.float64),
            np.concatenate(residuals),
            np.concatenate(internal_means),
        )

    @staticmethod
    def _sample_evenly(
        values: np.ndarray,
        companion: np.ndarray,
        limit: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if values.size <= limit:
            return values, companion
        indices = np.linspace(0, values.size - 1, limit, dtype=int)
        return values[indices], companion[indices]

    def _estimate_read_noise(
        self,
        residual: np.ndarray,
        mean_output: np.ndarray,
        camera: Camera,
        config: CalibrationConfig,
    ) -> Tuple[float, bool]:
        output_scale = camera.geometry.image_format.output_scale
        residual, mean_output = self._sample_evenly(
            residual,
            mean_output,
            config.maximum_noise_samples,
        )
        mean_internal = np.maximum(mean_output / output_scale, 0)
        quantization_variance = 1.0 / 12.0

        def likelihood(read_noise):
            variance = (
                (mean_internal + read_noise ** 2) * output_scale ** 2
                + quantization_variance
            )
            return float(np.mean(
                np.log(variance) + residual ** 2 / variance
            ))

        result = minimize_scalar(
            likelihood,
            bounds=(0.0, config.maximum_read_noise),
            method="bounded",
            options={"xatol": 0.01},
        )
        if not result.success:
            raise RuntimeError(f"read-noise fit failed: {result.message}")

        _, measured_sigma = _robust_location_scale(residual)
        poisson_floor = (
            float(np.median(mean_internal)) * output_scale ** 2
            + quantization_variance
        )
        if measured_sigma ** 2 <= poisson_floor:
            warnings.warn(
                "blank-sky variance is below the Poisson and quantization "
                "floor; read_noise was set to zero and the Camera noise model "
                "cannot reproduce the observed variance",
                RuntimeWarning,
            )
            return 0.0, False
        return float(result.x), True

    def _bind_sky_noise_step(
        self,
        context: CalibrationContext,
    ) -> BoundCalibrationStep:
        def determine_sky_noise(
            observer: Observer,
            camera: Camera,
        ) -> CalibrationPair:
            if context.image_data is None:
                raise ValueError("image properties step has not prepared image data")
            if observer.observer_matrix is None:
                raise ValueError("observer orientation is not configured")

            data = context.image_data
            valid = np.isfinite(data) & ~context.saturated_mask
            ground = camera.ground_mask(observer.observer_matrix)
            valid &= ~ground
            if ground.any() and (~ground).any():
                horizon = (
                    binary_dilation(ground, iterations=3)
                    & binary_dilation(~ground, iterations=3)
                )
                valid &= ~horizon
            valid[[0, -1], :] = False
            valid[:, [0, -1]] = False

            sources = self._all_source_mask(
                data,
                valid,
                camera.image_model.fwhm,
                context.config,
            )
            if context.source_mask is not None:
                sources |= context.source_mask
            valid &= ~sources
            if np.count_nonzero(valid) < 100:
                raise ValueError("fewer than 100 blank-sky pixels remain")
            context.blank_sky_mask = valid.copy()

            if (
                context.native_data is not None
                and context.native_data.ndim == 3
            ):
                channel_results = [
                    self._blank_sky_tiles(
                        np.asarray(
                            context.native_data[:, :, channel],
                            dtype=np.float64,
                        ),
                        valid,
                        context.config.sky_tile_size,
                    )
                    for channel in range(context.native_data.shape[2])
                ]
                tile_locations = np.concatenate([
                    result[0] for result in channel_results
                ])
                residual = np.concatenate([
                    result[1] for result in channel_results
                ])
                mean_output = np.concatenate([
                    result[2] for result in channel_results
                ])
            else:
                tile_locations, residual, mean_output = self._blank_sky_tiles(
                    data,
                    valid,
                    context.config.sky_tile_size,
                )
            sky_output = max(float(np.median(tile_locations)), 0.0)
            sky_background = (
                sky_output
                / camera.geometry.image_format.output_scale
                / camera.image_model.exposure_time
            )
            read_noise, read_noise_identifiable = self._estimate_read_noise(
                residual,
                mean_output,
                camera,
                context.config,
            )
            camera.set_noise(
                sky_background=sky_background,
                read_noise=read_noise,
            )
            context.record(
                fov_deg=camera.geometry.fov,
                flux=camera.image_model.flux,
                fwhm_px=camera.image_model.fwhm,
                sky_background=sky_background,
                read_noise=read_noise,
                read_noise_identifiable=read_noise_identifiable,
                blank_pixels=int(np.count_nonzero(valid)),
                blank_tiles=int(tile_locations.size),
            )
            if context.visualization_enabled:
                rendered = Renderer(context.sky, camera).render(
                    observer,
                    noise_seed=0,
                )
                context.set_visual(CalibrationVisualizer.final_result(
                    context.image,
                    rendered,
                    context._pending_values,
                ))
            return observer, camera

        return determine_sky_noise

    @staticmethod
    def _validate_step_result(
        name: str,
        result,
    ) -> CalibrationPair:
        if not isinstance(result, tuple) or len(result) != 2:
            raise TypeError(
                f"calibration step {name!r} must return (observer, camera)"
            )
        observer, camera = result
        if not isinstance(observer, Observer):
            raise TypeError(
                f"calibration step {name!r} returned a non-Observer"
            )
        if not isinstance(camera, Camera):
            raise TypeError(
                f"calibration step {name!r} returned a non-Camera"
            )
        return observer, camera

    def estimate_camera(
        self,
        image: Image.Image,
        time: datetime.datetime,
        latitude_deg: float,
        longitude_deg: float,
        elevation_m: float = 0.0,
    ) -> CalibrationPair:
        """Run all configured steps and return a complete observer and camera."""

        context = CalibrationContext(
            sky=self.sky,
            image=image,
            time=time,
            latitude_deg=latitude_deg,
            longitude_deg=longitude_deg,
            elevation_m=elevation_m,
            config=self.config,
            visualization_enabled=(
                self.visualize or self.visualization_callback is not None
            ),
        )
        observer = Observer()
        camera = Camera()
        self._last_reports = ()

        for definition in self._steps:
            try:
                bound_step = definition.bind(context)
                observer, camera = self._validate_step_result(
                    definition.name,
                    bound_step(observer, camera),
                )
                report = context.finish_report(definition.name)
                if self.diagnostic_callback is not None:
                    self.diagnostic_callback(report, observer, camera)
            except CalibrationError:
                self._last_reports = tuple(context.reports)
                raise
            except Exception as error:
                self._last_reports = tuple(context.reports)
                raise CalibrationError(definition.name, str(error)) from error

        if not camera.is_valid():
            raise CalibrationError("finalize", "camera is not valid")
        if (
            observer.observer_matrix is None
            or not np.isfinite(observer.observer_matrix).all()
        ):
            raise CalibrationError(
                "finalize",
                "observer attitude was not configured",
            )
        self._last_reports = tuple(context.reports)
        final_report = self._last_reports[-1] if self._last_reports else None
        if final_report is not None and final_report.visual is not None:
            if self.visualization_callback is not None:
                self.visualization_callback(final_report)
            if self.visualize:
                self._show_visualization(final_report)
        return observer, camera

    @staticmethod
    def _show_visualization(report: StepReport):
        import matplotlib.pyplot as plt

        figure, axis = plt.subplots()
        axis.imshow(report.visual)
        axis.set_title(report.name)
        axis.axis("off")
        figure.tight_layout()
        plt.show(block=True)
        plt.close(figure)
