from pathlib import Path

GROUND_BACKGROUND_SCALE = 2.0


class Config:
    CACHE_DIR = Path(__file__).resolve().parents[1] / ".cache"
    EPHEMERIS = "de440s.bsp"
    FLOAT_TOL = 1e-6
    GROUND_BACKGROUND_SCALE = GROUND_BACKGROUND_SCALE
    MAX_NAVIGATION_ITERATIONS = 10
