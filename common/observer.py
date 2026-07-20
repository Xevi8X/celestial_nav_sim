from dataclasses import dataclass
import datetime

import numpy as np

from common import Config

class Observer:
    def __init__(self):
        self.time : datetime.datetime = None
        self.latitude : float = None
        self.longitude : float = None
        self.elevation : float = 0.0
        self.look_dir : np.ndarray = None
        self.look_up : np.ndarray = None
        self.observer_matrix : np.ndarray = None

    def set_time(self, time : datetime.datetime):
        if time.tzinfo is None:
            raise ValueError("time must be timezone-aware")
        self.time = time

    def set_location(self, latitude : float, longitude : float, elevation : float = 0.0):
        self.latitude = latitude
        self.longitude = longitude
        self.elevation = elevation

    def set_look_direction(self, look_dir, look_up):
        look_dir = np.asarray(look_dir, dtype=float)
        look_up = np.asarray(look_up, dtype=float)
        look_dir_norm = np.linalg.norm(look_dir)
        look_up_norm = np.linalg.norm(look_up)

        if np.isclose(look_dir_norm, 0, Config.FLOAT_TOL) or np.isclose(look_up_norm, 0, Config.FLOAT_TOL):
            raise ValueError("look_dir and look_up must be non-zero vectors")

        self.look_dir = look_dir / look_dir_norm
        self.look_up = look_up / look_up_norm

        right = np.cross(self.look_dir, self.look_up)
        right_norm = np.linalg.norm(right)

        if np.isclose(right_norm, 0, Config.FLOAT_TOL):
            raise ValueError("look_dir and look_up must not be parallel")

        right /= right_norm
        up = np.cross(right, self.look_dir)
        self.observer_matrix = np.array([right, up, self.look_dir])


