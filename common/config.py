from pathlib import Path

class Config:
    CACHE_DIR = Path(__file__).resolve().parents[1] / ".cache"
    EPHEMERIS = "de440s.bsp"
    FLOAT_TOL = 1e-6