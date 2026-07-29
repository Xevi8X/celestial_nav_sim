import datetime
import numpy as np
from dataclasses import dataclass
from PIL import Image
from typing import Iterable, Optional, Union

from common import Config, ECEF, Observer, Sky
from .lost_in_space import LostInSpace

class Navigator:
    def __init__(self, sky : Sky, fov_range : tuple[int,int] =(40, 60), star_max_magnitude=7):
        self.sky = sky
        db_path = LostInSpace.get_db_path(min_fov=fov_range[0], max_fov=fov_range[1], star_max_magnitude=star_max_magnitude)
        if not db_path.exists():
            db_path = LostInSpace.generate_db(min_fov=fov_range[0], max_fov=fov_range[1], star_max_magnitude=star_max_magnitude)
        self._lis = LostInSpace(db_path)

    @staticmethod
    def _navigation_image(image: Image.Image) -> Image.Image:
        if image.mode == "I" or image.mode.startswith("I;16"):
            data = np.asarray(image, dtype=np.float64) / 257.0
            return Image.fromarray(np.clip(data, 0, 255).astype(np.uint8))
        return image.convert("L")

    def find_solution(
        self,
        image: Image.Image,
    ) -> Optional[LostInSpace.Solution]:
        solution = self._lis.solve(
            self._navigation_image(image),
            distortion=0.0,
        )
        return solution

    def estimate_orientation(
        self,
        image: Image.Image,
        time: datetime.datetime,
        latitude_deg: float,
        longitude_deg: float,
        elevation_m: float = 0.0,
    ) -> Observer:
        observer = Observer()
        observer.set_time(time)
        observer.set_location(latitude=latitude_deg, longitude=longitude_deg, elevation=elevation_m)
        solution = self.find_solution(image)

        if solution is None:
            return observer

        dir_ecef, up_ecef = self.sky.radec_to_ecef(observer, solution.ra, solution.dec, solution.roll)
        ecef_to_ned = ECEF.ecef_to_ned(observer.latitude, observer.longitude)
        look_dir = ecef_to_ned @ dir_ecef
        look_up = ecef_to_ned @ up_ecef
        observer.set_look_direction(look_dir=look_dir, look_up=look_up)
        return observer

    @dataclass
    class Location:
        latitude_deg: float
        longitude_deg: float

    @dataclass
    class ImageTimeOrientation:
        image: Image.Image
        time: datetime.datetime
        look_dir_ned: np.ndarray
        look_up_ned: np.ndarray

    @dataclass
    class ImageTimeZenit:
        image: Image.Image
        time: datetime.datetime
        zenit_cam: np.ndarray

    def estimate_location_full_orientation(self, data : Union[ImageTimeOrientation, Iterable[ImageTimeOrientation]]) -> Optional[Location]:
        if isinstance(data, self.ImageTimeOrientation):
            data = [data]

        items = []

        for item in data:
            observer = Observer()
            observer.set_time(item.time)
            observer.set_look_direction(look_dir=item.look_dir_ned, look_up=item.look_up_ned)
            observer.set_location(latitude=0.0, longitude=0.0)
            solution = self.find_solution(item.image)
            if solution:
                items.append((observer, solution))

        if not items:
            return None

        lat, lon = 0.0, 0.0

        for _ in range(Config.MAX_NAVIGATION_ITERATIONS):
            ecef_vectors = []
            ned_vectors = []

            for observer, solution in items:
                observer.set_location(latitude=lat, longitude=lon)
                dir_ecef, up_ecef = self.sky.radec_to_ecef(observer, solution.ra, solution.dec, solution.roll)
                ecef_vectors.append(dir_ecef)
                ecef_vectors.append(up_ecef)
                ned_vectors.append(observer.look_dir)
                ned_vectors.append(observer.look_up)

            ecef_vectors = np.array(ecef_vectors)
            ned_vectors = np.array(ned_vectors)
            lat, lon = ECEF.find_location(
                ecef_vectors=ecef_vectors,
                ned_vectors=ned_vectors
            )

            diff = np.linalg.norm(ECEF.north_east_vector(lat, lon, observer.latitude, observer.longitude))
            if diff < 1.0:
                break


        return self.Location(latitude_deg=lat, longitude_deg=lon)

    def estimate_location(self, data: Union[ImageTimeZenit, Iterable[ImageTimeZenit]]) -> Optional[Location]:
        if isinstance(data, self.ImageTimeZenit):
            data = [data]

        items = []

        for item in data:
            zenit_cam = np.asarray(item.zenit_cam, dtype=float)
            if zenit_cam.shape != (3,) or not np.isfinite(zenit_cam).all():
                raise ValueError("zenit_cam must be a finite 3D vector")

            norm = np.linalg.norm(zenit_cam)
            if np.isclose(norm, 0.0, atol=Config.FLOAT_TOL):
                raise ValueError("zenit_cam must be non-zero")

            solution = self.find_solution(item.image)
            if solution:
                observer = Observer()
                observer.set_time(item.time)
                observer.set_location(latitude=0.0, longitude=0.0)
                items.append((observer, solution, zenit_cam / norm))

        if not items:
            return None

        lat, lon = 0.0, 0.0

        for _ in range(Config.MAX_NAVIGATION_ITERATIONS):
            previous_lat, previous_lon = lat, lon
            zenits_ecef = []

            for observer, solution, zenit_cam in items:
                observer.set_location(latitude=lat, longitude=lon)
                direction, up = self.sky.radec_to_ecef(observer, solution.ra, solution.dec, solution.roll)
                right = np.cross(direction, up)
                right /= np.linalg.norm(right)
                up = np.cross(right, direction)

                zenit_ecef = np.column_stack((right, up, direction)) @ zenit_cam
                zenits_ecef.append(zenit_ecef / np.linalg.norm(zenit_ecef))

            zenit_ecef = np.sum(zenits_ecef, axis=0)
            norm = np.linalg.norm(zenit_ecef)
            if np.isclose(norm, 0.0, atol=Config.FLOAT_TOL):
                return None

            x, y, z = zenit_ecef / norm
            lat = np.degrees(np.arctan2(z, np.hypot(x, y)))
            lon = np.degrees(np.arctan2(y, x))

            diff = np.linalg.norm(ECEF.north_east_vector(previous_lat, previous_lon, lat, lon))
            if diff < 1.0:
                break

        return self.Location(latitude_deg=lat, longitude_deg=lon)
