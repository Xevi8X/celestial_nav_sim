from dataclasses import dataclass
from pathlib import Path

import numpy as np
from skyfield.api import Loader, Star, wgs84
from skyfield.data import hipparcos
from skyfield.framelib import itrs

from common import Config
from common import Observer
from common import Rotation

# https://doi.org/10.1016/j.ascom.2018.08.002
DEFAULT_BODIES = (
    (10, "Sun", 695700.0, -27.0), 
    (199, "Mercury", 2439.7, 0.23),
    (299, "Venus", 6051.8, -4.14), 
    (301, "Moon", 1737.4, -12.74),
    (4, "Mars", 3389.5, 0.71), 
    (5, "Jupiter", 69911.0, -2.20),
    (6, "Saturn", 58232.0, 0.46), 
    (7, "Uranus", 25362.0, 5.68),
    (8, "Neptune", 24622.0, 7.78), 
    (9, "Pluto", 1188.3, 15.1),
)

class Sky:
    @dataclass(frozen=True)
    class StarInfo:
        id: int
        magnitude: float

    @dataclass(frozen=True)
    class BodyInfo:
        id: int
        name: str
        radius_km: float
        apparent_magnitude: float

    def __init__(self, magnitude_limit=7.0):
        Path(Config.CACHE_DIR).mkdir(parents=True, exist_ok=True)
        self._loader = Loader(Config.CACHE_DIR)
        self._timescale = self._loader.timescale()

        eph = self._loader(Config.EPHEMERIS)
        self._bodies_info = [Sky.BodyInfo(*body) for body in DEFAULT_BODIES]
        self._bodies = [eph[body.id] for body in self._bodies_info]
        self._earth = eph["earth"]

        with self._loader.open(hipparcos.URL) as file:
            frame = hipparcos.load_dataframe(file)
        frame = frame.dropna(subset=["ra_degrees", "dec_degrees", "magnitude"])
        frame = frame[frame["magnitude"] <= magnitude_limit]
        self._stars_info = [Sky.StarInfo(int(i), float(row["magnitude"])) for i, row in frame.iterrows()]
        self._stars = Star.from_dataframe(frame)

    def to_ecef(self, observer : Observer, objects):
        observer_pos = self._earth + wgs84.latlon(observer.latitude, observer.longitude, observer.elevation)
        observer_in_spacetime = observer_pos.at(self._timescale.from_datetime(observer.time))
        if isinstance(objects, list):
            return np.array([observer_in_spacetime.observe(obj).apparent().frame_xyz(itrs).km for obj in objects])
        return np.asarray(observer_in_spacetime.observe(objects).apparent().frame_xyz(itrs).km).T

    def ecef_to_ned(self, observer : Observer, vectors):
        lat, lon = np.radians([observer.latitude, observer.longitude])
        return vectors @ Rotation.Z(-lon).T @ Rotation.Y(np.pi / 2 + lat).T
    
    def get_stars_ned(self, observer : Observer) -> tuple[list[StarInfo], np.ndarray]:
        return self._stars_info, self.ecef_to_ned(observer, self.to_ecef(observer, self._stars))
    
    def get_bodies_ned(self, observer : Observer) -> tuple[list[BodyInfo], np.ndarray]:
        return self._bodies_info, self.ecef_to_ned(observer, self.to_ecef(observer, self._bodies))
