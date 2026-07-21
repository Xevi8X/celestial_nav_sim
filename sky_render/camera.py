from dataclasses import dataclass
import numpy as np
from scipy.ndimage import gaussian_filter

@dataclass
class Camera:
    fov: float
    width: int
    height: int

    exposure_time: float = 0.1
    flux: float = 3e5
    fwhm: float = 1.0

    sky_background_e: float = 3.0
    ground_background_e: float = 20.0
    horizon_blur_px: float = 1.0
    read_noise_e: float = 1.0
    
    monochromatic: bool = True

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
        if self.sky_background_e < 0 or self.ground_background_e < 0:
            raise ValueError("background levels must be non-negative")
        if self.horizon_blur_px < 0:
            raise ValueError("horizon_blur_px must be non-negative")
        if self.read_noise_e < 0:
            raise ValueError("read_noise_e must be non-negative")

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

    def horizon_fractions(self, observer_matrix):
        observer_matrix = np.asarray(observer_matrix, dtype=np.float64)
        if observer_matrix.shape != (3, 3):
            raise ValueError("observer_matrix must have shape (3, 3)")

        yy, xx = np.indices((self.height, self.width), dtype=np.float64)
        pixels = np.stack((xx, yy, np.ones_like(xx)), axis=-1)
        rays_camera = pixels @ np.linalg.inv(self.camera_matrix).T
        rays_ned = rays_camera @ observer_matrix

        ground = (rays_ned[:, :, 2] > 0).astype(np.float32)
        if self.horizon_blur_px > 0:
            ground = gaussian_filter(
                ground,
                sigma=self.horizon_blur_px,
                mode="nearest",
            )
        ground = np.clip(ground, 0, 1)
        return 1 - ground, ground
