import numpy as np

from common import Config, ECEF, Observer, Sky
from .lost_in_space import LostInSpace

from PIL import Image

class Navigator:
    def __init__(self, sky : Sky, fov_range : tuple[int,int] =(40, 60), star_max_magnitude=7):
        self.sky = sky
        db_path = LostInSpace.get_db_path(min_fov=fov_range[0], max_fov=fov_range[1], star_max_magnitude=star_max_magnitude)
        if not db_path.exists():
            db_path = LostInSpace.generate_db(min_fov=fov_range[0], max_fov=fov_range[1], star_max_magnitude=star_max_magnitude)
        self._lis = LostInSpace(db_path)

    def find_solution(self, image: Image.Image) -> LostInSpace.Solution:
        solution = self._lis.solve(image)
        return solution
    
    def estimate_orientation(self, observer: Observer, image: Image.Image) -> Observer:
        if observer.time is None or observer.latitude is None or observer.longitude is None:
            raise ValueError("Observer time and location must be set before estimating orientation.")
        
        solution = self.find_solution(image)
        observer2 = Observer()
        observer2.set_time(observer.time)
        observer2.set_location(latitude=observer.latitude, longitude=observer.longitude, elevation=observer.elevation)
        dir_ecef, up_ecef = self.sky.radec_to_ecef(observer2, solution.ra, solution.dec, solution.roll)
        ecef_to_ned = ECEF.ecef_to_ned(observer.latitude, observer.longitude)
        look_dir = ecef_to_ned @ dir_ecef
        look_up = ecef_to_ned @ up_ecef
        observer2.set_look_direction(look_dir=look_dir, look_up=look_up)
        return observer2

    def estimate_location(self, observer: Observer, image: Image.Image) -> Observer:
        if observer.time is None or observer.look_dir is None or observer.look_up is None:
            raise ValueError("Observer time and orientation must be set before estimating location.")
        
        solution = self.find_solution(image)
        observer2 = Observer()
        observer2.set_time(observer.time)
        observer2.set_look_direction(look_dir=observer.look_dir, look_up=observer.look_up)
        observer2.set_location(latitude=0.0, longitude=0.0, elevation=0.0)

        for _ in range(Config.MAX_NAVIGATION_ITERATIONS):
            dir_ecef, up_ecef = self.sky.radec_to_ecef(observer2, solution.ra, solution.dec, solution.roll)
            lat, lon = ECEF.find_location(
                ecef_vectors=np.array([dir_ecef, up_ecef]),
                ned_vectors=np.array([observer.look_dir, observer.look_up])
            )
            diff = np.linalg.norm(ECEF.north_east_vector(lat, lon, observer2.latitude, observer2.longitude))
            observer2.set_location(latitude=lat, longitude=lon, elevation=0.0)
            if diff < 1.0:
                break

        return observer2

