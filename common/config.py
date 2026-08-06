from pathlib import Path

import numpy as np

GROUND_BACKGROUND_SCALE = 2.0


class Config:
    CACHE_DIR = Path(__file__).resolve().parents[1] / ".cache"
    EPHEMERIS = "de440s.bsp"
    FLOAT_TOL = 1e-6
    # Camera coordinates are right, down, forward. The camera looks along
    # body forward and image-up points along body up in the FRD body frame.
    FRD_CAMERA_TO_BODY = np.array([
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ])
    GRAVITY = 9.805
    GROUND_BACKGROUND_SCALE = GROUND_BACKGROUND_SCALE
    MAX_EXPOSURE_STEP = 1.0
    MAX_NAVIGATION_ITERATIONS = 10
