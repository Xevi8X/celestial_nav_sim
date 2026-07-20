from dataclasses import dataclass
import numpy as np

@dataclass
class Camera:
    fov: float
    width: int
    height: int

    exposure_time: float = 0.1
    flux: float = 3e5
    fwhm: float = 5.0
    
    monochromatic: bool = True

    def __post_init__(self):
        if not 0 < self.fov < 180:
            raise ValueError("fov must be between 0 and 180 degrees")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be positive")
        if self.fwhm <= 0:
            raise ValueError("fwhm must be positive")

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
