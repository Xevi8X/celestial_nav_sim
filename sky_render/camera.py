from dataclasses import dataclass
from enum import Enum, auto

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
        return 257.0 if self is ImageFormat.MONO16 else 1.0

    @property
    def monochromatic(self):
        return self is not ImageFormat.RGB8


@dataclass
class Camera:
    fov: float
    width: int
    height: int

    exposure_time: float = 0.1
    flux: float = 5e5
    fwhm: float = 3.0

    sky_background: float = 3.0
    read_noise: float = 1.0

    image_format: ImageFormat = ImageFormat.MONO8

    def __post_init__(self):
        if not 0 < self.fov < 180:
            raise ValueError("fov must be between 0 and 180 degrees")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be positive")
        if self.fwhm <= 0:
            raise ValueError("fwhm must be positive")
        if self.exposure_time <= 0:
            raise ValueError("exposure_time must be positive")
        if self.flux <= 0:
            raise ValueError("flux must be positive")
        if self.sky_background < 0:
            raise ValueError("sky_background must be non-negative")
        if self.read_noise < 0:
            raise ValueError("read_noise must be non-negative")
        if not isinstance(self.image_format, ImageFormat):
            raise TypeError("image_format must be an ImageFormat")

    @property
    def focal_length(self):
        return self.width / (2 * np.tan(np.radians(self.fov) / 2))

    @property
    def camera_matrix(self):
        f = self.focal_length
        intrinsics = np.array([
            [f, 0, (self.width - 1) / 2],
            [0, -f, (self.height - 1) / 2],
            [0, 0, 1],
        ])
        return intrinsics

    def ground_mask(self, observer_matrix):
        observer_matrix = np.asarray(observer_matrix, dtype=np.float64)
        if observer_matrix.shape != (3, 3):
            raise ValueError("observer_matrix must have shape (3, 3)")

        yy, xx = np.indices((self.height, self.width), dtype=np.float64)
        pixels = np.stack((xx, yy, np.ones_like(xx)), axis=-1)
        rays_camera = pixels @ np.linalg.inv(self.camera_matrix).T
        rays_ned = rays_camera @ observer_matrix

        return rays_ned[:, :, 2] > 0
