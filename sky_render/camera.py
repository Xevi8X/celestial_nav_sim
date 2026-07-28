from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

import numpy as np


class ImageFormat(Enum):
    MONO8 = auto()
    MONO16 = auto()
    RGB8 = auto()

    @property
    def channels(self):
        return 3 if self is ImageFormat.RGB8 else 1

    @property
    def dtype(self):
        return np.uint16 if self is ImageFormat.MONO16 else np.uint8

    @property
    def max_value(self):
        return 65535 if self is ImageFormat.MONO16 else 255

    @property
    def output_scale(self):
        # Pixel values are physical camera counts. Bit depth changes only the
        # available range and dtype; it is not a radiometric gain.
        return 1.0

    @property
    def navigation_scale(self):
        # tetra3 works on an 8-bit copy. This conversion must not leak into
        # camera flux, background, or noise units.
        return 257.0 if self is ImageFormat.MONO16 else 1.0

    @property
    def monochromatic(self):
        return self is not ImageFormat.RGB8


@dataclass
class CameraGeometry:
    """Parameters that define the image plane and its projection."""

    fov: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    image_format: ImageFormat = ImageFormat.MONO8


@dataclass
class CameraImageModel:
    """Parameters that turn projected sources into pixel values."""

    exposure_time: float = 0.1
    flux: float = 5e5
    fwhm: float = 3.0
    sky_background: float = 3.0
    read_noise: float = 1.0


@dataclass
class Camera:
    """Camera configuration that may be filled incrementally by a calibrator."""

    geometry: CameraGeometry = field(default_factory=CameraGeometry)
    image_model: CameraImageModel = field(default_factory=CameraImageModel)

    def is_valid(self) -> bool:
        geometry = self.geometry
        image_model = self.image_model
        try:
            return (
                isinstance(geometry, CameraGeometry)
                and isinstance(image_model, CameraImageModel)
                and geometry.fov is not None
                and np.isfinite(geometry.fov)
                and 0 < geometry.fov < 180
                and isinstance(geometry.width, (int, np.integer))
                and not isinstance(geometry.width, bool)
                and geometry.width > 0
                and isinstance(geometry.height, (int, np.integer))
                and not isinstance(geometry.height, bool)
                and geometry.height > 0
                and isinstance(geometry.image_format, ImageFormat)
                and np.isfinite((
                    image_model.exposure_time,
                    image_model.flux,
                    image_model.fwhm,
                    image_model.sky_background,
                    image_model.read_noise,
                )).all()
                and image_model.exposure_time > 0
                and image_model.flux > 0
                and image_model.fwhm > 0
                and image_model.sky_background >= 0
                and image_model.read_noise >= 0
            )
        except (AttributeError, TypeError, ValueError):
            return False

    def set_image(
        self,
        width: int,
        height: int,
        image_format: ImageFormat,
    ) -> None:
        self.geometry.width = width
        self.geometry.height = height
        self.geometry.image_format = image_format

    def set_fov(self, fov: float) -> None:
        self.geometry.fov = fov

    def set_exposure_time(self, exposure_time: float) -> None:
        self.image_model.exposure_time = exposure_time

    def set_photometry(self, flux: float, fwhm: float) -> None:
        self.image_model.flux = flux
        self.image_model.fwhm = fwhm

    def set_noise(
        self,
        sky_background: float,
        read_noise: float,
    ) -> None:
        self.image_model.sky_background = sky_background
        self.image_model.read_noise = read_noise

    @property
    def focal_length(self) -> float:
        if not self.is_valid():
            raise ValueError("camera is not valid")
        geometry = self.geometry
        return geometry.width / (
            2 * np.tan(np.radians(geometry.fov) / 2)
        )

    @property
    def camera_matrix(self) -> np.ndarray:
        if not self.is_valid():
            raise ValueError("camera is not valid")
        geometry = self.geometry
        f = self.focal_length
        return np.array([
            [f, 0, (geometry.width - 1) / 2],
            [0, -f, (geometry.height - 1) / 2],
            [0, 0, 1],
        ])

    def ground_mask(self, observer_matrix) -> np.ndarray:
        if not self.is_valid():
            raise ValueError("camera is not valid")
        observer_matrix = np.asarray(observer_matrix, dtype=np.float64)
        if observer_matrix.shape != (3, 3):
            raise ValueError("observer_matrix must have shape (3, 3)")

        geometry = self.geometry
        yy, xx = np.indices(
            (geometry.height, geometry.width),
            dtype=np.float64,
        )
        pixels = np.stack((xx, yy, np.ones_like(xx)), axis=-1)
        rays_camera = pixels @ np.linalg.inv(self.camera_matrix).T
        rays_ned = rays_camera @ observer_matrix

        return rays_ned[:, :, 2] > 0
