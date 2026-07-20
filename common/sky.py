from dataclasses import dataclass
from pathlib import Path

import numpy as np
from skyfield.api import Loader, Star, wgs84
from skyfield.data import hipparcos
from skyfield.framelib import itrs

from common import Config, ECEF, Observer

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

    def _get_observer_spacetime(self, observer: Observer):
        observer_location = wgs84.latlon(observer.latitude, observer.longitude, elevation_m=observer.elevation)
        return (self._earth + observer_location).at(self._timescale.from_datetime(observer.time))

    def _get_vector_to_object(self, observer: Observer, obj):
        observer_spacetime = self._get_observer_spacetime(observer)
        apparent = observer_spacetime.observe(obj).apparent()
        alt, az, distance = apparent.altaz('standard')

        alt = alt.radians
        az = az.radians

        ned = np.array([
            np.cos(alt) * np.cos(az),
            np.cos(alt) * np.sin(az),
            -np.sin(alt)
        ])

        ecef_to_ned = ECEF.ecef_to_ned(observer.latitude, observer.longitude)
        return ecef_to_ned.T @ ned * distance.km

    def objects_to_ecef(self, observer : Observer, objects):
        if isinstance(objects, list):
            return np.array([self._get_vector_to_object(observer, obj) for obj in objects])
        return np.asarray(self._get_vector_to_object(observer, objects)).T
    
    def radec_to_ecef(self, observer, ra, dec, roll = 0.0):
        ra, dec, roll = np.radians([ra, dec, roll])

        east = np.array([-np.sin(ra), np.cos(ra), 0.0])
        north = np.array([
            -np.sin(dec) * np.cos(ra),
            -np.sin(dec) * np.sin(ra),
            np.cos(dec)
        ])
        up_icrs = -np.sin(roll) * east + np.cos(roll) * north

        star = Star(
            ra_hours=np.degrees(ra) / 15.0,
            dec_degrees=np.degrees(dec)
        )

        direction = self._get_vector_to_object(observer, star)
        direction /= np.linalg.norm(direction)

        up = itrs.rotation_at(self._timescale.from_datetime(observer.time)) @ up_icrs
        up -= direction * np.dot(direction, up)
        up /= np.linalg.norm(up)

        return direction, up

    def get_stars_ecef(self, observer : Observer) -> tuple[list[StarInfo], np.ndarray]:
        return self._stars_info, self.objects_to_ecef(observer, self._stars)
    
    def get_bodies_ecef(self, observer : Observer) -> tuple[list[BodyInfo], np.ndarray]:
        return self._bodies_info, self.objects_to_ecef(observer, self._bodies)