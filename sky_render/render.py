import copy
import datetime
import math

import numpy as np
from PIL import Image

from .camera import Camera
from .canvas import Canvas
from common import Config, ECEF, Observer, Sky

RENDER_COLORS = {
    "Sun": (255, 255, 0),
    "Moon": (173, 216, 230),
    "Mercury": (169, 169, 169),
    "Venus": (255, 255, 224),
    "Mars": (255, 0, 0),
    "Jupiter": (255, 165, 0),
    "Saturn": (210, 180, 140),
    "Uranus": (0, 255, 255),
    "Neptune": (0, 0, 255),
    "Pluto": (255, 192, 203),
}


class Renderer:
    def __init__(self, sky: Sky, camera: Camera):
        self._sky = sky
        self._camera = camera

    def set_camera(self, camera: Camera):
        self._camera = camera

    def _draw_sources(
        self,
        canvas: Canvas,
        observer: Observer,
        exposure_time: float,
    ):
        stars_info, stars_ned = self._sky.get_stars_ecef(observer)
        bodies_info, bodies_ned = self._sky.get_bodies_ecef(observer)

        ecef_to_ned = ECEF.ecef_to_ned(observer.latitude, observer.longitude)
        stars_ned = stars_ned @ ecef_to_ned.T
        bodies_ned = bodies_ned @ ecef_to_ned.T

        matrix = observer.observer_matrix.T @ self._camera.camera_matrix.T

        def _project(vectors):
            projected = vectors @ matrix
            with np.errstate(divide="ignore", invalid="ignore"):
                return projected[:, :2] / projected[:, 2, None], projected[:, 2]

        star_xy, star_z = _project(stars_ned)
        body_xy, body_z = _project(bodies_ned)

        def visible(ned, xy, z):
            return (
                (ned[:, 2] < 0) &
                (z > 0) &
                np.isfinite(xy).all(axis=1) &
                (xy[:, 0] >= 0) & (xy[:, 0] < self._camera.width) &
                (xy[:, 1] >= 0) & (xy[:, 1] < self._camera.height)
            )

        stars_visible = visible(stars_ned, star_xy, star_z)
        bodies_visible = visible(bodies_ned, body_xy, body_z)

        for point, info in zip(
            star_xy[stars_visible],
            np.asarray(stars_info, dtype=object)[stars_visible],
        ):
            canvas.draw(point, info.magnitude, exposure_time=exposure_time)

        for point, vector, info in zip(
            body_xy[bodies_visible],
            bodies_ned[bodies_visible],
            np.asarray(bodies_info, dtype=object)[bodies_visible],
        ):
            distance = np.linalg.norm(vector)
            angular_radius = np.arcsin(np.clip(info.radius_km / distance, 0, 1))
            radius_px = self._camera.focal_length * np.tan(angular_radius)
            color = RENDER_COLORS.get(info.name, (255, 255, 255))
            canvas.draw(
                point,
                info.apparent_magnitude,
                radius_px,
                color,
                exposure_time=exposure_time,
            )

    def _render_canvas(self, observer: Observer) -> Canvas:
        canvas = Canvas(self._camera)
        max_exposure_step = Config.MAX_EXPOSURE_STEP
        if not math.isfinite(max_exposure_step) or max_exposure_step <= 0:
            raise ValueError("MAX_EXPOSURE_STEP must be positive")
        sample_count = math.ceil(
            self._camera.exposure_time / max_exposure_step
        )
        sample_exposure = self._camera.exposure_time / sample_count
        start_time = observer.time - datetime.timedelta(
            seconds=self._camera.exposure_time
        )
        for index in range(sample_count):
            sample_observer = copy.copy(observer)
            sample_observer.time = start_time + datetime.timedelta(
                seconds=(index + 0.5) * sample_exposure
            )
            self._draw_sources(canvas, sample_observer, sample_exposure)

        canvas.add_horizon(observer.observer_matrix)

        return canvas

    def render_expected(self, observer: Observer) -> np.ndarray:
        return self._render_canvas(observer).linear_image()

    def render(self, observer: Observer, noise_seed=None) -> Image.Image:
        canvas = self._render_canvas(observer)
        canvas.add_noise(noise_seed)
        return canvas.image()
