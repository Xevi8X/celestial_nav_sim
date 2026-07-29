from dataclasses import dataclass
import datetime

import numpy as np

from common import Config
from common.rotations import Rotation

class Observer:
    def __init__(self):
        self.time : datetime.datetime = None
        self.latitude : float = None
        self.longitude : float = None
        self.elevation : float = None

        self.look_dir : np.ndarray = None
        self.look_up : np.ndarray = None
        self.look_right : np.ndarray = None

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
            self.look_dir = None
            self.look_up = None
            self.look_right = None
            self.observer_matrix = None
            return

        self.look_dir = look_dir / look_dir_norm
        self.look_up = look_up / look_up_norm

        self.look_right = np.cross(self.look_dir, self.look_up)
        right_norm = np.linalg.norm(self.look_right)

        if np.isclose(right_norm, 0, Config.FLOAT_TOL):
            raise ValueError("look_dir and look_up must not be parallel")

        self.look_right /= right_norm
        self.look_up = np.cross(self.look_right, self.look_dir)
        self.observer_matrix = np.array([self.look_right, self.look_up, self.look_dir])

    def set_orientation(self, roll, pitch, yaw):
        roll, pitch, yaw = np.radians([roll, pitch, yaw])

        cy, sy = np.cos(yaw), np.sin(yaw)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cr, sr = np.cos(roll), np.sin(roll)

        self.look_dir = np.array([cp * cy, cp * sy, -sp])
        right = np.array([-sy, cy, 0.0])
        up = np.cross(right, self.look_dir)

        self.look_right = cr * right - sr * up
        self.look_up = sr * right + cr * up

        self.observer_matrix = np.vstack([
            self.look_right,
            self.look_up,
            self.look_dir,
        ])
